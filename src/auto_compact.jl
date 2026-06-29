# ==============================================================================
# Auto-Compact: timer-driven background capacity compaction.
# Design: docs/plans/DESIGN_auto_compact.md
#
# A single global Timer periodically *requests* compaction by setting each live
# pool's `@atomic _compact_requested` flag (via a WeakRef registry). Each pool's
# owner task *executes* the request — the full `compact!` — at its next
# `_current_depth == 1` scope-exit safepoint, where nothing is borrowed.
#
# Tier 1 (compile-time): `AUTO_COMPACT` gates whether the scope-exit flag check is
# emitted (macros.jl) and whether pools register (task_local_pool.jl). Default off
# → zero hot-path cost. Tier 2 (runtime): `enable_auto_compact!` / `disable_auto_compact!`.
# ==============================================================================

using Preferences: @load_preference

"""
    AUTO_COMPACT::Bool

Compile-time master switch (Tier 1) for auto-compaction. When `false` (default), the
scope-exit flag check is not emitted and pools are not registered — zero hot-path
cost. Set in `LocalPreferences.toml` (then restart) to compile it in:

```toml
[AdaptiveArrayPools]
auto_compact = true
```

When on, `enable_auto_compact!` / `disable_auto_compact!` control the background
Timer at runtime (Tier 2).
"""
const AUTO_COMPACT = @load_preference("auto_compact", false)::Bool

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
function register_auto_compact!(pool)
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
                @atomic p._compact_requested = true   # request compaction
                i += 1
            end
        end
    end
    return nothing
end

# Owner-side execution: reset the flag BEFORE the work (so a concurrent sweep's
# re-request is not lost), then run the full configured `compact!`. Called at the
# `_current_depth == 1` safepoint by the macro hook (Phase 3) — the depth guard is the
# caller's. Returns nothing; the summary is discarded (no allocation).
function _run_auto_compact!(pool::AdaptiveArrayPool)
    @atomic pool._compact_requested = false
    cfg = _AUTO_COMPACT_CONFIG
    compact!(
        pool;
        factor = cfg.factor, shrink_to = cfg.shrink_to,
        min_bytes = cfg.min_bytes, active = cfg.active,
    )
    return nothing
end
