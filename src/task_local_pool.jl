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

The pool type is `AdaptiveArrayPool{S}` where `S` is determined by
the compile-time constant `RUNTIME_CHECK::Int`. Macro-generated code
type-asserts directly to `AdaptiveArrayPool{RUNTIME_CHECK}`.
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

# ==============================================================================
# Metal Pool Stubs (overridden by extension when Metal is loaded)
# ==============================================================================

"""
    get_task_local_metal_pool() -> MetalAdaptiveArrayPool

Retrieves (or creates) the Metal pool for the current Task and current Metal device.

Requires Metal.jl to be loaded. Throws an error if Metal extension is not available.

See also: [`get_task_local_pool`](@ref) for CPU pools.
"""
function get_task_local_metal_pool end

"""
    get_task_local_metal_pools() -> Dict{UInt64, MetalAdaptiveArrayPool}

Returns the dictionary of all Metal pools for the current task (one per device).

Requires Metal.jl to be loaded. Throws an error if Metal extension is not available.
"""
function get_task_local_metal_pools end
