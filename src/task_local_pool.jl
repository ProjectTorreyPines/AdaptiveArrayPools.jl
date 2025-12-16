# ==============================================================================
# Task-Local Pool & Configuration
# ==============================================================================

using Preferences: @load_preference

"""
    USE_POOLING::Bool

Compile-time constant (master switch) to completely disable pooling.
When `false`, all macros (`@with_pool`, `@maybe_with_pool`)
generate code that uses `nothing` as the pool, causing `acquire!` to fall back
to normal allocation.

This enables zero-overhead when pooling is disabled, as the compiler can
eliminate all pool-related code paths.

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
const USE_POOLING = @load_preference("use_pooling", true)::Bool

"""
    MAYBE_POOLING_ENABLED

Runtime flag for `@maybe_with_pool` macro only.
When `false`, `@maybe_with_pool` will use `nothing` as the pool,
causing `acquire!` to allocate normally.

Note: This only affects `@maybe_with_pool`.
`@with_pool` ignores this flag (always uses pooling).

For complete removal of pooling overhead at compile time, use `USE_POOLING` instead.

Default: `true`
"""
const MAYBE_POOLING_ENABLED = Ref(true)

const _POOL_KEY = :ADAPTIVE_ARRAY_POOL

"""
    get_task_local_pool() -> AdaptiveArrayPool

Retrieves (or creates) the `AdaptiveArrayPool` for the current Task.

Each Task gets its own pool instance via `task_local_storage()`,
ensuring thread safety without locks.
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