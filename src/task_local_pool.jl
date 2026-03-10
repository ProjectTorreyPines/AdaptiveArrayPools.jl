# ==============================================================================
# Task-Local Pool & Configuration
# ==============================================================================

using Preferences: @load_preference

# ==============================================================================
# Pooling Control: 2-Tier Toggle System
#
#   Tier 1: STATIC_POOLING  (compile-time const, master switch)
#           - Set via LocalPreferences.toml ["use_pooling"]
#           - When false: ALL macros produce DisabledPool (zero overhead)
#
#   Tier 2: MAYBE_POOLING   (runtime Ref{Bool}, @maybe_with_pool only)
#           - Toggle at runtime without restart
#           - @with_pool ignores this (always pools when Tier 1 is on)
#
#   Hierarchy: STATIC_POOLING=false gates everything.
#              MAYBE_POOLING only matters when STATIC_POOLING=true.
# ==============================================================================

"""
    STATIC_POOLING::Bool

Compile-time constant (master switch) to completely disable pooling.
When `false`, all macros (`@with_pool`, `@maybe_with_pool`)
generate code that uses `DisabledPool`, causing `acquire!` to fall back
to normal allocation with zero overhead.

This is the **Tier 1** control: when disabled, `MAYBE_POOLING` has no effect.

## Configuration via Preferences.jl

Set in your project's `LocalPreferences.toml`:
```toml
[AdaptiveArrayPools]
use_pooling = false
```

Or programmatically (requires restart):
```julia
using Preferences
Preferences.set_preferences!(AdaptiveArrayPools, "use_pooling" => false)
```

Default: `true`
"""
const STATIC_POOLING = @load_preference("use_pooling", true)::Bool

"""
    MAYBE_POOLING::Ref{Bool}

Runtime toggle for `@maybe_with_pool` macro only (**Tier 2** control).
When `false`, `@maybe_with_pool` will use `DisabledPool`,
causing `acquire!` to allocate normally.

Note: This only affects `@maybe_with_pool`.
`@with_pool` ignores this flag (always uses pooling).

For complete removal of pooling overhead at compile time, use `STATIC_POOLING` instead.

Default: `true`

```julia
MAYBE_POOLING[] = false   # disable @maybe_with_pool at runtime
MAYBE_POOLING[] = true    # re-enable
```
"""
const MAYBE_POOLING = Ref(true)

# Deprecated aliases (backward compat) — same objects, just old names
const USE_POOLING = STATIC_POOLING
const MAYBE_POOLING_ENABLED = MAYBE_POOLING

const _POOL_KEY = :ADAPTIVE_ARRAY_POOL

"""
    get_task_local_pool() -> AdaptiveArrayPool

Retrieves (or creates) the `AdaptiveArrayPool` for the current Task.

Each Task gets its own pool instance via `task_local_storage()`,
ensuring thread safety without locks.

Returns the pool as-is (type `Any` from task_local_storage).
Use `_dispatch_pool_scope` in macro-generated code to narrow to concrete `AdaptiveArrayPool{S}`.
"""
@inline function get_task_local_pool()
    # 1. Fast Path: Try to get existing pool
    # get(dict, key, default) is optimized in Julia Base
    pool = get(task_local_storage(), _POOL_KEY, nothing)

    if pool === nothing
        # 2. Slow Path: Create and store new pool
        # This branch is rarely taken (only once per Task)
        pool = AdaptiveArrayPool()
        task_local_storage(_POOL_KEY, pool)
    end

    return pool::AdaptiveArrayPool
end

# ==============================================================================
# Union Splitting Dispatcher + Safety Level Control
# ==============================================================================
#
# AdaptiveArrayPool{S} is parametric on both modern (≥1.11) and legacy (≤1.10).
# Union splitting narrows to concrete type for dead-code elimination of safety branches.

"""
    _dispatch_pool_scope(f, pool_any)

Union splitting barrier: converts abstract pool → concrete `AdaptiveArrayPool{S}`.

Inside `f`, the pool argument has concrete type, enabling:
- `_safety_level(pool)` → compile-time constant S
- Dead-code elimination of safety branches at S=0
- Zero-allocation try/finally (no Core.Box)

Called from macro-generated code as:
```julia
_dispatch_pool_scope(get_task_local_pool()) do pool
    checkpoint!(pool)
    try ... finally rewind!(pool) end
end
```
"""
@inline function _dispatch_pool_scope(f, pool_any)
    if pool_any isa AdaptiveArrayPool{0}
        return f(pool_any::AdaptiveArrayPool{0})
    elseif pool_any isa AdaptiveArrayPool{1}
        return f(pool_any::AdaptiveArrayPool{1})
    elseif pool_any isa AdaptiveArrayPool{2}
        return f(pool_any::AdaptiveArrayPool{2})
    else
        return f(pool_any::AdaptiveArrayPool{3})
    end
end

"""
    set_safety_level!(level::Int) -> AdaptiveArrayPool

Replace the task-local pool with a new `AdaptiveArrayPool{level}`.

The new pool starts fresh (empty state). Old pool is GC'd.
One-time JIT cost for new `S` specialization.

Also updates `POOL_SAFETY_LV[]` so that `AdaptiveArrayPool()` creates pools
at the new level.

## Example
```julia
set_safety_level!(2)  # Enable full safety (escape detection + poisoning)
# ... run suspicious code ...
set_safety_level!(0)  # Back to zero overhead
```

See also: [`_safety_level`], [`POOL_SAFETY_LV`]
"""
function set_safety_level!(level::Int)
    0 <= level <= 3 || throw(ArgumentError("Safety level must be 0-3; got $level"))
    POOL_SAFETY_LV[] = level
    new_pool = _make_pool(level)
    task_local_storage(_POOL_KEY, new_pool)
    return new_pool
end

# ==============================================================================
# CUDA Pool Stubs (overridden by extension when CUDA is loaded)
# ==============================================================================

"""
    get_task_local_cuda_pool() -> CuAdaptiveArrayPool

Retrieves (or creates) the CUDA pool for the current Task and current GPU device.

Requires CUDA.jl to be loaded. Throws an error if CUDA extension is not available.

See also: [`get_task_local_pool`](@ref) for CPU pools.
"""
function get_task_local_cuda_pool end

"""
    get_task_local_cuda_pools() -> Dict{Int, CuAdaptiveArrayPool}

Returns the dictionary of all CUDA pools for the current task (one per device).

Requires CUDA.jl to be loaded. Throws an error if CUDA extension is not available.
"""
function get_task_local_cuda_pools end
