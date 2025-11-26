# ==============================================================================
# Global Pool (Task Local Storage) & Configuration
# ==============================================================================

"""
    ENABLE_POOLING

Global flag to enable/disable pooling. When `false`, `@maybe_use_global_pool`
will use `nothing` as the pool, causing `acquire!` to allocate normally.

Default: `true`
"""
const ENABLE_POOLING = Ref(true)

const _POOL_KEY = :ADAPTIVE_ARRAY_POOL

"""
    get_global_pool() -> AdaptiveArrayPool

Retrieves (or creates) the `AdaptiveArrayPool` for the current Task.

Each Task gets its own pool instance via `task_local_storage()`,
ensuring thread safety without locks.
"""
@inline function get_global_pool()
    tls = task_local_storage()
    if haskey(tls, _POOL_KEY)
        return tls[_POOL_KEY]::AdaptiveArrayPool
    else
        pool = AdaptiveArrayPool()
        tls[_POOL_KEY] = pool
        return pool
    end
end
