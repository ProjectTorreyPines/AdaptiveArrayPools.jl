# ==============================================================================
# Task-Local Metal Pool (Multi-Device Aware)
# ==============================================================================
# Each Task gets one pool per Metal device to prevent cross-device memory access.
# Pools are parameterized by R (0=off, 1=checks on) via MetalAdaptiveArrayPool{R,S}.

const _METAL_POOL_KEY = :ADAPTIVE_ARRAY_POOL_METAL

"""
    get_task_local_metal_pool() -> MetalAdaptiveArrayPool{R,S}

Retrieves (or creates) the `MetalAdaptiveArrayPool` for the current Task and current Metal device.

## Multi-Device Safety
Each pool is bound to a specific Metal device. This function automatically manages
a dictionary of pools (one per device) in task-local storage, ensuring that:
- Device A's pool is never used on Device B
- Switching devices gets the correct pool

## Implementation
Uses `Dict{UInt64, MetalAdaptiveArrayPool}` in task-local storage, keyed by device hash.
Values are `MetalAdaptiveArrayPool{R,S}` where R is determined by `RUNTIME_CHECK`.
"""
@inline function AdaptiveArrayPools.get_task_local_metal_pool()
    # 1. Get or create the pools dictionary
    pools = get(task_local_storage(), _METAL_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{UInt64, MetalAdaptiveArrayPool}()
        task_local_storage(_METAL_POOL_KEY, pools)
    end

    # 2. Get current device key
    dev = Metal.device()
    dev_key = objectid(dev)

    # 3. Get or create pool for this device
    pool = get(pools, dev_key, nothing)
    if pool === nothing
        pool = MetalAdaptiveArrayPool()  # Uses RUNTIME_CHECK for initial R
        pools[dev_key] = pool
    end

    return pool::MetalAdaptiveArrayPool
end

"""
    get_task_local_metal_pools() -> Dict{UInt64, MetalAdaptiveArrayPool}

Returns the dictionary of all Metal pools for the current task (one per device).
Useful for diagnostics or bulk operations across all devices.
"""
@inline function AdaptiveArrayPools.get_task_local_metal_pools()
    pools = get(task_local_storage(), _METAL_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{UInt64, MetalAdaptiveArrayPool}()
        task_local_storage(_METAL_POOL_KEY, pools)
    end
    return pools
end
