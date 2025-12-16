# ==============================================================================
# Task-Local CUDA Pool (Multi-Device Aware)
# ==============================================================================
# Each Task gets one pool per GPU device to prevent cross-device memory access.

const _CU_POOL_KEY = :ADAPTIVE_ARRAY_POOL_CUDA

"""
    get_task_local_cuda_pool() -> CuAdaptiveArrayPool

Retrieves (or creates) the `CuAdaptiveArrayPool` for the current Task and current GPU device.

## Multi-Device Safety
Each pool is bound to a specific GPU device. This function automatically manages
a dictionary of pools (one per device) in task-local storage, ensuring that:
- Device 0's pool is never used on Device 1
- Switching devices (`CUDA.device!(n)`) gets the correct pool

## Implementation
Uses `Dict{Int, CuAdaptiveArrayPool}` in task-local storage, keyed by device ID.
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
        pool = CuAdaptiveArrayPool()  # Constructor captures device_id
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
    return pools::Dict{Int, CuAdaptiveArrayPool}
end
