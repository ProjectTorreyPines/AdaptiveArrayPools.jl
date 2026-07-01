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
Uses `Dict{UInt64, MetalAdaptiveArrayPool{RUNTIME_CHECK, METAL_STORAGE}}` in task-local storage, keyed by device hash.
"""
@inline function AdaptiveArrayPools.get_task_local_metal_pool()
    # 1. Get or create the pools dictionary
    pools = get(task_local_storage(), _METAL_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{UInt64, MetalAdaptiveArrayPool{RUNTIME_CHECK, METAL_STORAGE}}()
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
        # Register for auto-management (gated by the compile-time const; mirrors CPU
        # get_task_local_pool). One entry per (task, device) pool. DCE'd when the
        # Preference is off, so no registry traffic on the zero-overhead build.
        AdaptiveArrayPools.AUTO_MANAGE && AdaptiveArrayPools.register_auto_manage!(pool)
    end

    return pool::MetalAdaptiveArrayPool{RUNTIME_CHECK, METAL_STORAGE}
end

"""
    get_task_local_metal_pools() -> Dict{UInt64, MetalAdaptiveArrayPool{RUNTIME_CHECK, METAL_STORAGE}}

Returns the dictionary of all Metal pools for the current task (one per device).
Useful for diagnostics or bulk operations across all devices.
"""
@inline function AdaptiveArrayPools.get_task_local_metal_pools()
    pools = get(task_local_storage(), _METAL_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{UInt64, MetalAdaptiveArrayPool{RUNTIME_CHECK, METAL_STORAGE}}()
        task_local_storage(_METAL_POOL_KEY, pools)
    end
    return pools
end
