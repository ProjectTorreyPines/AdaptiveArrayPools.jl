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

Uses `get!` for single hash lookup (~30% faster than haskey+getindex).
"""
@inline function get_task_local_pool()
    get!(task_local_storage(), _POOL_KEY) do
        AdaptiveArrayPool()
    end::AdaptiveArrayPool
end
