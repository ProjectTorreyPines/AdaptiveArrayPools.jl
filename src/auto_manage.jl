# ==============================================================================
# Auto-Compact: timer-driven background capacity compaction.
# Design: docs/plans/DESIGN_auto_manage.md
#
# A single global Timer periodically *requests* compaction by setting each live
# pool's `@atomic _compact_requested` flag (via a WeakRef registry). Each pool's
# owner task *executes* the request — the full `compact!` — at its next `@with_pool`
# ENTRY (the `_current_depth == 1` safepoint), where nothing is borrowed.
#
# Tier 1 (compile-time): `AUTO_MANAGE` gates whether the scope-entry flag check is
# emitted (macros.jl) and whether pools register (task_local_pool.jl). Default ON
# (~0.6 ns/entry); set the Preference false to compile it OUT (full DCE → zero cost).
# Tier 2 (runtime): `enable_auto_manage!` / `disable_auto_manage!`.
# ==============================================================================

using Preferences: @load_preference

"""
    AUTO_MANAGE::Bool

Compile-time master switch (Tier 1) for auto-manageion. Default `true`: the scope-entry
flag check is emitted (a ~0.6 ns inlined check per `@with_pool` entry), new task-local
pools auto-register, and `__init__` auto-starts the background Timer. Set `false` in
`LocalPreferences.toml` (then restart) to compile the feature OUT entirely — the hook is
dead-code-eliminated, restoring zero hot-path cost:

```toml
[AdaptiveArrayPools]
auto_manage = false
```

`enable_auto_manage!` / `disable_auto_manage!` control the background Timer at runtime
(Tier 2).
"""
const AUTO_MANAGE = @load_preference("auto_manage", true)::Bool

# Registry of pools eligible for auto-manageion. `WeakRef` so a pool dies with its
# task; dead refs are swap-removed on each sweep. All access under the lock below.
const _AUTO_MANAGE_REGISTRY = WeakRef[]
const _AUTO_MANAGE_LOCK = ReentrantLock()

# Mutable global config, set by `enable_auto_manage!`, read by the sweep / `_run_auto_manage!`.
# Mirrors the `compact!` knobs (`factor`/`shrink_to` `Float64` for stable storage), plus the
# auto-trim cadence `trim_every_ticks` (auto-trim fires every N sweep ticks; 0 = never trim).
mutable struct _AutoManageConfig
    factor::Float64
    shrink_to::Float64
    min_bytes::Int
    active::Bool
    trim_every_ticks::Int
end
const _AUTO_MANAGE_CONFIG = _AutoManageConfig(10.0, 1.5, 2^20, true, 0)

# Sweep tick counter driving the auto-trim cadence. Owned by the single Timer thread
# (incremented/reset only inside the sweep), so a plain `Ref` is sufficient.
const _AUTO_MANAGE_TICK = Ref{Int}(0)

"""
    register_auto_manage!(pool)

Add `pool` to the auto-manage registry (under lock) so the background Timer's sweep
will flag it. Called from the task-local-pool slow path when `AUTO_MANAGE` is on
(one entry per task-local pool). The same registry serves every backend (CPU
`AdaptiveArrayPool`, GPU `MetalAdaptiveArrayPool`/`CuAdaptiveArrayPool`); each carries
its own `@atomic _compact_requested` flag and services it at its own scope entry.

A pool *without* that flag (e.g. `DisabledPool`, or a foreign `AbstractArrayPool`
subtype) is silently skipped: registering one would make the sweep throw on its missing
`_compact_requested` and abort the cycle, starving every pool after it in the registry.
"""
function register_auto_manage!(pool::AbstractArrayPool)
    # Only pools carrying the cross-thread flag can be swept (const-folds per concrete type).
    hasfield(typeof(pool), :_compact_requested) || return nothing
    lock(() -> push!(_AUTO_MANAGE_REGISTRY, WeakRef(pool)), _AUTO_MANAGE_LOCK)
    return nothing
end

# The Timer callback body: set every live pool's request flag and prune dead refs.
# Takes the timer (unused) so it can be handed straight to `Timer(cb, …)`. The caller
# wraps it in try/catch (Phase 2) — a throwing callback silently kills a Julia Timer.
function _auto_manage_sweep!(_timer)
    lock(_AUTO_MANAGE_LOCK) do
        # Auto-trim cadence: fire every `trim_every_ticks` sweeps (0 = never). Computed
        # under the same lock that guards the config so a concurrent `enable!` can't tear it.
        trim_every = _AUTO_MANAGE_CONFIG.trim_every_ticks
        do_trim = false
        if trim_every > 0
            _AUTO_MANAGE_TICK[] += 1
            if _AUTO_MANAGE_TICK[] >= trim_every
                _AUTO_MANAGE_TICK[] = 0
                do_trim = true
            end
        end
        i = 1
        while i <= length(_AUTO_MANAGE_REGISTRY)
            pool = _AUTO_MANAGE_REGISTRY[i].value
            if pool === nothing
                _AUTO_MANAGE_REGISTRY[i] = _AUTO_MANAGE_REGISTRY[end]
                pop!(_AUTO_MANAGE_REGISTRY)          # dead → swap-remove
            else
                p = pool::AbstractArrayPool          # CPU or any GPU backend pool
                @atomic :monotonic p._compact_requested = true   # request compaction (every tick)
                # request auto-trim (every `trim_every` ticks); guard backends without the flag
                if do_trim && hasfield(typeof(p), :_trim_requested)
                    @atomic :monotonic p._trim_requested = true
                end
                i += 1
            end
        end
    end
    return nothing
end

# Owner-side execution: reset the flag BEFORE the work (so a concurrent sweep's re-request
# is not lost), snapshot the config under the lock (it may be rewritten from another thread
# by `enable_auto_manage!`), then run the full `compact!`. Wrapped in try/catch so a
# background-maintenance failure (e.g. OOM allocating the new backing) is logged and the
# cycle skipped — it must never surface as a user-visible error at the `@with_pool` boundary.
# Returns nothing; the summary is discarded (no allocation). Shared across backends:
# `compact!` takes the same factor/shrink_to/min_bytes/active kwargs for every
# `AbstractArrayPool`, so one implementation drives CPU and GPU pools alike.
# Auto-trim one pool: drop each type's cold slot tail down to its recent working-set peak
# (`_ac_peak_n_active`), then reset that peak for the next observation period. Generic over
# `foreach_fixed_slot` + `others`, so it serves every backend once its typed pools carry the
# field. Drops references only (no buffer swap), so `RUNTIME_CHECK` poison on a dropped slot
# survives for any escaped view.
function _auto_trim!(pool::AbstractArrayPool)
    foreach_fixed_slot(pool) do tp
        _trim_to!(tp, tp._ac_peak_n_active)
        tp._ac_peak_n_active = 0
    end
    for tp in values(pool.others)
        _trim_to!(tp, tp._ac_peak_n_active)
        tp._ac_peak_n_active = 0
    end
    return nothing
end

function _run_auto_manage!(pool::AbstractArrayPool)
    # Auto-trim first (when flagged): reclaim the cold tail before compacting the kept slots.
    # The `hasfield` guard const-folds per concrete type → a backend without the flag no-ops.
    # Each action resets its flag BEFORE the work so a concurrent sweep's re-request is kept.
    if hasfield(typeof(pool), :_trim_requested) && (@atomic :monotonic pool._trim_requested)
        @atomic :monotonic pool._trim_requested = false
        try
            _auto_trim!(pool)
        catch e
            @warn "auto-trim failed; skipping this cycle" exception = (e, catch_backtrace()) maxlog = 3
        end
    end
    if (@atomic :monotonic pool._compact_requested)
        @atomic :monotonic pool._compact_requested = false
        factor, shrink_to, min_bytes, active = lock(_AUTO_MANAGE_LOCK) do
            cfg = _AUTO_MANAGE_CONFIG
            (cfg.factor, cfg.shrink_to, cfg.min_bytes, cfg.active)
        end
        try
            compact!(pool; factor, shrink_to, min_bytes, active)
        catch e
            @warn "auto-manage failed; skipping this cycle" exception = (e, catch_backtrace()) maxlog = 3
        end
    end
    return nothing
end

# Scope-ENTRY hook body, generated before the checkpoint in every `@with_pool` (gated by
# AUTO_MANAGE). At the `_current_depth == 1` safepoint (entering the outermost scope from
# global), run a flagged pool's compaction. Servicing the request at entry handles EVERY
# exit type of the previous scope — normal / early return / break / continue / exception —
# with a single hook point, and never runs compaction inside a `finally` during unwind.
# `@inline` so the common case (flag clear) is a cheap inlined monotonic read + compare; the
# cold `_run_auto_manage!` stays a non-inlined call. The `::Any` fallback keeps the SHARED
# `@with_pool` codegen safe for pools that don't opt in (e.g. `DisabledPool`, or a backend
# without its own method) — they no-op here. GPU backends define their own concrete method
# in their extension (e.g. `_maybe_auto_manage!(::MetalAdaptiveArrayPool)`), reusing the
# shared `_run_auto_manage!`. `:monotonic` suffices: the flag is a one-way eventual-visibility
# signal with no cross-field ordering invariant (cheaper than seq-cst on weak-ordered CPUs).
@inline _maybe_auto_manage!(::Any) = nothing
@inline function _maybe_auto_manage!(pool::AdaptiveArrayPool)
    if (@atomic :monotonic pool._compact_requested) && pool._current_depth == 1
        _run_auto_manage!(pool)
    end
    return nothing
end

# ── Timer lifecycle (Tier 2 runtime control) ─────────────────────────────────

# Handle to the single global auto-manage Timer (`nothing` = stopped).
const _AUTO_MANAGE_TIMER = Ref{Union{Nothing, Timer}}(nothing)

# Timer callback: the sweep wrapped so a throw can't silently kill the Timer (an
# uncaught error in a Julia Timer callback stops it permanently).
function _safe_auto_manage_sweep!(timer)
    try
        _auto_manage_sweep!(timer)
    catch e
        @warn "auto-manage sweep failed (timer continues)" exception = (e, catch_backtrace()) maxlog = 3
    end
    return nothing
end

# Lock-guarded timer lifecycle: the firing callback and concurrent enable!/disable! must not
# interleave a read-modify-write on `_AUTO_MANAGE_TIMER` (→ a leaked or double-closed Timer).
function _start_auto_manage_timer!(interval::Real)
    lock(_AUTO_MANAGE_LOCK) do
        _stop_auto_manage_timer_unlocked!()                      # replace any existing
        _AUTO_MANAGE_TIMER[] = Timer(_safe_auto_manage_sweep!, interval; interval = interval)
    end
    return nothing
end

_stop_auto_manage_timer!() = (lock(_stop_auto_manage_timer_unlocked!, _AUTO_MANAGE_LOCK); nothing)

function _stop_auto_manage_timer_unlocked!()
    t = _AUTO_MANAGE_TIMER[]
    if t !== nothing
        close(t)
        _AUTO_MANAGE_TIMER[] = nothing
    end
    return nothing
end

"""
    enable_auto_manage!(; interval = 30.0, factor = 10, shrink_to = 1.5,
                           min_bytes = 2^20, active = true)

Start (or restart) the background auto-manage `Timer` with the given config. Every
`interval` seconds it flags every registered pool; each pool then runs the full
`compact!` at its next `@with_pool` **entry** (the `_current_depth == 1` safepoint).

Requires the `auto_manage` compile-time Preference (`AUTO_MANAGE`) for the scope-entry
hook and pool auto-registration. With the Preference **off**, the `@with_pool` hook is
compiled out, so even though this warns-then-starts the timer (and the timer keeps setting
flags), nothing ever *services* those flags — no automatic compaction happens, and
registering pools manually does not change that. Set the Preference and restart, or call
`compact!` yourself.

!!! note "What gets compacted, when, and the zero-alloc trade-off"
    Auto-compaction targets the **task-local pool** (`get_task_local_pool()`) — the pool
    `@with_pool` uses and the only one that auto-registers. A hand-made `AdaptiveArrayPool()`
    is **not** auto-registered (so not auto-manageed); `register_auto_manage!` it explicitly
    to include it in the sweep, or just call `compact!` on it directly. The work
    runs at a `@with_pool` **entry** (`_current_depth == 1`), so an idle task with no
    `@with_pool` activity won't compact until it next enters a scope. Because a compaction
    allocates the new right-sized backing(s), a `@with_pool` that services a pending request
    is **not** strictly zero-allocation; set `auto_manage = false` (Preference) for a
    guaranteed-zero-alloc hot path.

See also [`disable_auto_manage!`](@ref), [`auto_manage_enabled`](@ref).
"""
function enable_auto_manage!(;
        interval::Real = 30.0, factor::Real = 10, shrink_to::Real = 1.5,
        min_bytes::Int = 2^20, active::Bool = true, trim_interval::Real = 120.0,
    )
    AUTO_MANAGE || @warn "enable_auto_manage!: the `auto_manage` Preference is off — the `@with_pool` hook is compiled out, so the timer's flags are never serviced and no automatic management happens (registering pools does not change this). Set the Preference and restart to activate, or call `compact!`/`trim!` manually." maxlog = 1
    # auto-trim cadence in sweep ticks; non-finite `trim_interval` (e.g. Inf) disables auto-trim.
    trim_every = isfinite(trim_interval) ? max(1, round(Int, trim_interval / interval)) : 0
    lock(_AUTO_MANAGE_LOCK) do
        cfg = _AUTO_MANAGE_CONFIG
        cfg.factor = Float64(factor)
        cfg.shrink_to = Float64(shrink_to)
        cfg.min_bytes = min_bytes
        cfg.active = active
        cfg.trim_every_ticks = trim_every
        _AUTO_MANAGE_TICK[] = 0           # restart the trim cadence from this enable!
    end
    _start_auto_manage_timer!(interval)
    return nothing
end

"""
    disable_auto_manage!()

Stop the background auto-manage `Timer`. Idempotent.
"""
disable_auto_manage!() = _stop_auto_manage_timer!()

"""
    auto_manage_enabled() -> Bool

Whether the background auto-manage `Timer` is currently running.
"""
auto_manage_enabled() = _AUTO_MANAGE_TIMER[] !== nothing

# Auto-start the background timer when AUTO_MANAGE is on — but NOT while generating
# precompile output. A live Timer is an open libuv handle, so a precompile worker (e.g.
# precompiling a CUDA/Metal extension, which loads this package and runs `__init__`)
# would hang with "unfinished IO handles with timer events". `jl_generating_output()` is
# nonzero exactly during precompile/sysimage output. Also close the timer at process exit.
function __init__()
    if AUTO_MANAGE && ccall(:jl_generating_output, Cint, ()) == 0
        enable_auto_manage!()
        atexit(disable_auto_manage!)
    end
    return nothing
end
