# ==============================================================================
# Auto-Compact: timer-driven background capacity compaction.
# Design: docs/plans/DESIGN_auto_compact.md
#
# A single global Timer periodically *requests* compaction by setting each live
# pool's `@atomic _compact_requested` flag (via a WeakRef registry). Each pool's
# owner task *executes* the request — the full `compact!` — at its next `@with_pool`
# ENTRY (the `_current_depth == 1` safepoint), where nothing is borrowed.
#
# Tier 1 (compile-time): `AUTO_COMPACT` gates whether the scope-entry flag check is
# emitted (macros.jl) and whether pools register (task_local_pool.jl). Default ON
# (~0.6 ns/entry); set the Preference false to compile it OUT (full DCE → zero cost).
# Tier 2 (runtime): `enable_auto_compact!` / `disable_auto_compact!`.
# ==============================================================================

using Preferences: @load_preference

"""
    AUTO_COMPACT::Bool

Compile-time master switch (Tier 1) for auto-compaction. Default `true`: the scope-entry
flag check is emitted (a ~0.6 ns inlined check per `@with_pool` entry), new task-local
pools auto-register, and `__init__` auto-starts the background Timer. Set `false` in
`LocalPreferences.toml` (then restart) to compile the feature OUT entirely — the hook is
dead-code-eliminated, restoring zero hot-path cost:

```toml
[AdaptiveArrayPools]
auto_compact = false
```

`enable_auto_compact!` / `disable_auto_compact!` control the background Timer at runtime
(Tier 2).
"""
const AUTO_COMPACT = @load_preference("auto_compact", true)::Bool

# Registry of pools eligible for auto-compaction. `WeakRef` so a pool dies with its
# task; dead refs are swap-removed on each sweep. All access under the lock below.
const _AUTO_COMPACT_REGISTRY = WeakRef[]
const _AUTO_COMPACT_LOCK = ReentrantLock()

# Mutable global config, set by `enable_auto_compact!`, read by `_run_auto_compact!`.
# Mirrors the `compact!` knobs; `factor`/`shrink_to` are `Float64` for stable storage.
mutable struct _AutoCompactConfig
    factor::Float64
    shrink_to::Float64
    min_bytes::Int
    active::Bool
end
const _AUTO_COMPACT_CONFIG = _AutoCompactConfig(10.0, 1.5, 2^20, true)

"""
    register_auto_compact!(pool)

Add `pool` to the auto-compact registry (under lock) so the background Timer's sweep
will flag it. Called from the task-local-pool slow path when `AUTO_COMPACT` is on
(one entry per task-local pool).
"""
function register_auto_compact!(pool::AdaptiveArrayPool)
    lock(() -> push!(_AUTO_COMPACT_REGISTRY, WeakRef(pool)), _AUTO_COMPACT_LOCK)
    return nothing
end

# The Timer callback body: set every live pool's request flag and prune dead refs.
# Takes the timer (unused) so it can be handed straight to `Timer(cb, …)`. The caller
# wraps it in try/catch (Phase 2) — a throwing callback silently kills a Julia Timer.
function _auto_compact_sweep!(_timer)
    lock(_AUTO_COMPACT_LOCK) do
        i = 1
        while i <= length(_AUTO_COMPACT_REGISTRY)
            pool = _AUTO_COMPACT_REGISTRY[i].value
            if pool === nothing
                _AUTO_COMPACT_REGISTRY[i] = _AUTO_COMPACT_REGISTRY[end]
                pop!(_AUTO_COMPACT_REGISTRY)          # dead → swap-remove
            else
                p = pool::AdaptiveArrayPool
                @atomic :monotonic p._compact_requested = true   # request compaction
                i += 1
            end
        end
    end
    return nothing
end

# Owner-side execution: reset the flag BEFORE the work (so a concurrent sweep's re-request
# is not lost), snapshot the config under the lock (it may be rewritten from another thread
# by `enable_auto_compact!`), then run the full `compact!`. Wrapped in try/catch so a
# background-maintenance failure (e.g. OOM allocating the new backing) is logged and the
# cycle skipped — it must never surface as a user-visible error at the `@with_pool` boundary.
# Returns nothing; the summary is discarded (no allocation).
function _run_auto_compact!(pool::AdaptiveArrayPool)
    @atomic :monotonic pool._compact_requested = false
    factor, shrink_to, min_bytes, active = lock(_AUTO_COMPACT_LOCK) do
        cfg = _AUTO_COMPACT_CONFIG
        (cfg.factor, cfg.shrink_to, cfg.min_bytes, cfg.active)
    end
    try
        compact!(pool; factor, shrink_to, min_bytes, active)
    catch e
        @warn "auto-compact failed; skipping this cycle" exception = (e, catch_backtrace()) maxlog = 3
    end
    return nothing
end

# Scope-ENTRY hook body, generated before the checkpoint in every `@with_pool` (gated by
# AUTO_COMPACT). At the `_current_depth == 1` safepoint (entering the outermost scope from
# global), run a flagged pool's compaction. Servicing the request at entry handles EVERY
# exit type of the previous scope — normal / early return / break / continue / exception —
# with a single hook point, and never runs compaction inside a `finally` during unwind.
# `@inline` so the common case (flag clear) is a cheap inlined monotonic read + compare; the
# cold `_run_auto_compact!` stays a non-inlined call. The `::Any` fallback keeps the SHARED
# `@with_pool` codegen safe for non-CPU (GPU) pools — they no-op here; a future phase adds
# their own methods. `:monotonic` suffices: the flag is a one-way eventual-visibility signal
# with no cross-field ordering invariant (cheaper than seq-cst on weak-ordered CPUs).
@inline _maybe_auto_compact!(::Any) = nothing
@inline function _maybe_auto_compact!(pool::AdaptiveArrayPool)
    if (@atomic :monotonic pool._compact_requested) && pool._current_depth == 1
        _run_auto_compact!(pool)
    end
    return nothing
end

# ── Timer lifecycle (Tier 2 runtime control) ─────────────────────────────────

# Handle to the single global auto-compact Timer (`nothing` = stopped).
const _AUTO_COMPACT_TIMER = Ref{Union{Nothing, Timer}}(nothing)

# Timer callback: the sweep wrapped so a throw can't silently kill the Timer (an
# uncaught error in a Julia Timer callback stops it permanently).
function _safe_auto_compact_sweep!(timer)
    try
        _auto_compact_sweep!(timer)
    catch e
        @warn "auto-compact sweep failed (timer continues)" exception = (e, catch_backtrace()) maxlog = 3
    end
    return nothing
end

# Lock-guarded timer lifecycle: the firing callback and concurrent enable!/disable! must not
# interleave a read-modify-write on `_AUTO_COMPACT_TIMER` (→ a leaked or double-closed Timer).
function _start_auto_compact_timer!(interval::Real)
    lock(_AUTO_COMPACT_LOCK) do
        _stop_auto_compact_timer_unlocked!()                      # replace any existing
        _AUTO_COMPACT_TIMER[] = Timer(_safe_auto_compact_sweep!, interval; interval = interval)
    end
    return nothing
end

_stop_auto_compact_timer!() = (lock(_stop_auto_compact_timer_unlocked!, _AUTO_COMPACT_LOCK); nothing)

function _stop_auto_compact_timer_unlocked!()
    t = _AUTO_COMPACT_TIMER[]
    if t !== nothing
        close(t)
        _AUTO_COMPACT_TIMER[] = nothing
    end
    return nothing
end

"""
    enable_auto_compact!(; interval = 30.0, factor = 10, shrink_to = 1.5,
                           min_bytes = 2^20, active = true)

Start (or restart) the background auto-compact `Timer` with the given config. Every
`interval` seconds it flags every registered pool; each pool then runs the full
`compact!` at its next `@with_pool` **entry** (the `_current_depth == 1` safepoint).

Requires the `auto_compact` compile-time Preference (`AUTO_COMPACT`) for the scope-entry
hook and pool auto-registration. With the Preference off this warns and the timer still
runs but stays inert unless you `register_auto_compact!` pools manually.

!!! note "What gets compacted, when, and the zero-alloc trade-off"
    Auto-compaction targets the **task-local pool** (`get_task_local_pool()`) only — the
    pool `@with_pool` uses and the one that auto-registers. A hand-made `AdaptiveArrayPool()`
    is never registered and never auto-compacted; call `compact!` on it directly. The work
    runs at a `@with_pool` **entry** (`_current_depth == 1`), so an idle task with no
    `@with_pool` activity won't compact until it next enters a scope. Because a compaction
    allocates the new right-sized backing(s), a `@with_pool` that services a pending request
    is **not** strictly zero-allocation; set `auto_compact = false` (Preference) for a
    guaranteed-zero-alloc hot path.

See also [`disable_auto_compact!`](@ref), [`auto_compact_enabled`](@ref).
"""
function enable_auto_compact!(;
        interval::Real = 30.0, factor::Real = 10, shrink_to::Real = 1.5,
        min_bytes::Int = 2^20, active::Bool = true,
    )
    AUTO_COMPACT || @warn "enable_auto_compact!: the `auto_compact` Preference is off — the scope-entry hook is compiled out and pools won't auto-register. The timer runs but stays inert unless you register pools manually. Set the Preference and restart to activate." maxlog = 1
    lock(_AUTO_COMPACT_LOCK) do
        cfg = _AUTO_COMPACT_CONFIG
        cfg.factor = Float64(factor)
        cfg.shrink_to = Float64(shrink_to)
        cfg.min_bytes = min_bytes
        cfg.active = active
    end
    _start_auto_compact_timer!(interval)
    return nothing
end

"""
    disable_auto_compact!()

Stop the background auto-compact `Timer`. Idempotent.
"""
disable_auto_compact!() = _stop_auto_compact_timer!()

"""
    auto_compact_enabled() -> Bool

Whether the background auto-compact `Timer` is currently running.
"""
auto_compact_enabled() = _AUTO_COMPACT_TIMER[] !== nothing

# Auto-start the background timer when AUTO_COMPACT is on — but NOT while generating
# precompile output. A live Timer is an open libuv handle, so a precompile worker (e.g.
# precompiling a CUDA/Metal extension, which loads this package and runs `__init__`)
# would hang with "unfinished IO handles with timer events". `jl_generating_output()` is
# nonzero exactly during precompile/sysimage output. Also close the timer at process exit.
function __init__()
    if AUTO_COMPACT && ccall(:jl_generating_output, Cint, ()) == 0
        enable_auto_compact!()
        atexit(disable_auto_compact!)
    end
    return nothing
end
