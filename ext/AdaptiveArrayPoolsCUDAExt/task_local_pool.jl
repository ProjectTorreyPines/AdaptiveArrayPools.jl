# ==============================================================================
# Task-Local CUDA Pool (Multi-Device Aware)
# ==============================================================================
# Each Task gets one pool per GPU device to prevent cross-device memory access.
# Pools are parameterized by safety level S (CuAdaptiveArrayPool{S}).

const _CU_POOL_KEY = :ADAPTIVE_ARRAY_POOL_CUDA

"""
    get_task_local_cuda_pool() -> CuAdaptiveArrayPool{S}

Retrieves (or creates) the `CuAdaptiveArrayPool` for the current Task and current GPU device.

## Multi-Device Safety
Each pool is bound to a specific GPU device. This function automatically manages
a dictionary of pools (one per device) in task-local storage, ensuring that:
- Device 0's pool is never used on Device 1
- Switching devices (`CUDA.device!(n)`) gets the correct pool

## Implementation
Uses `Dict{Int, CuAdaptiveArrayPool}` in task-local storage, keyed by device ID.
Values are `CuAdaptiveArrayPool{S}` — use `_dispatch_pool_scope` for union splitting.
"""
@inline function AdaptiveArrayPools.get_task_local_cuda_pool()
    # 1. Get or create the pools dictionary
    pools = get(task_local_storage(), _CU_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{Int, CuAdaptiveArrayPool}()
        task_local_storage(_CU_POOL_KEY, pools)
    end

    # 2. Get current device ID (using public API)
    dev_id = CUDA.deviceid(CUDA.device())

    # 3. Get or create pool for this device
    pool = get(pools, dev_id, nothing)
    if pool === nothing
        pool = CuAdaptiveArrayPool()  # Constructor uses POOL_SAFETY_LV[]
        pools[dev_id] = pool
    end

    return pool::CuAdaptiveArrayPool
end

"""
    get_task_local_cuda_pools() -> Dict{Int, CuAdaptiveArrayPool}

Returns the dictionary of all CUDA pools for the current task (one per device).
Useful for diagnostics or bulk operations across all devices.
"""
@inline function AdaptiveArrayPools.get_task_local_cuda_pools()
    pools = get(task_local_storage(), _CU_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{Int, CuAdaptiveArrayPool}()
        task_local_storage(_CU_POOL_KEY, pools)
    end
    return pools
end

# ==============================================================================
# Safety Level Hook (called from set_safety_level! in base)
# ==============================================================================

function AdaptiveArrayPools._set_cuda_safety_level_hook!(level::Int)
    pools = get(task_local_storage(), _CU_POOL_KEY, nothing)
    pools === nothing && return nothing

    # Check that no pool is inside an active scope
    for (dev_id, old_pool) in pools
        old = old_pool::CuAdaptiveArrayPool
        depth = old._current_depth
        depth != 1 && throw(
            ArgumentError(
                "set_safety_level! cannot be called inside an active @with_pool :cuda scope " *
                    "(device=$dev_id, depth=$depth)"
            )
        )
    end

    # Replace all pools (collect keys to avoid mutating Dict during iteration)
    for dev_id in collect(keys(pools))
        old = pools[dev_id]::CuAdaptiveArrayPool
        pools[dev_id] = _make_cuda_pool(level, old)
    end

    return nothing
end
