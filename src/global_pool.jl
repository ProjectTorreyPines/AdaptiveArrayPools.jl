# ==============================================================================
# Global Pool (Task Local Storage) & Configuration
# ==============================================================================

"""
    USE_POOLING::Bool

Compile-time constant (master switch) to completely disable pooling.
When `false`, all macros (`@use_pool`, `@use_global_pool`, `@maybe_use_global_pool`)
generate code that uses `nothing` as the pool, causing `acquire!` to fall back
to normal allocation.

This enables zero-overhead when pooling is disabled, as the compiler can
eliminate all pool-related code paths.

Future: Can be integrated with `Preferences.jl` for per-project configuration.

Default: `true`
"""
const USE_POOLING = true

"""
    MAYBE_POOLING_ENABLED

Runtime flag for `@maybe_use_global_pool` macro only.
When `false`, `@maybe_use_global_pool` will use `nothing` as the pool,
causing `acquire!` to allocate normally.

Note: This only affects `@maybe_use_global_pool`. The other macros
(`@use_pool`, `@use_global_pool`) ignore this flag.

For complete removal of pooling overhead at compile time, use `USE_POOLING` instead.

Default: `true`
"""
const MAYBE_POOLING_ENABLED = Ref(true)

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
