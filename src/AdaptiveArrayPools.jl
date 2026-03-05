module AdaptiveArrayPools

using Printf

# Public API
export AdaptiveArrayPool, acquire!, unsafe_acquire!, pool_stats, get_task_local_pool
export acquire_view!, acquire_array!  # Explicit naming aliases
export zeros!, ones!, trues!, falses!, similar!, default_eltype  # Convenience functions
export unsafe_zeros!, unsafe_ones!, unsafe_similar!  # Unsafe convenience functions
export Bit  # Sentinel type for BitArray (use with acquire!, trues!, falses!)
export @with_pool, @maybe_with_pool
export USE_POOLING, MAYBE_POOLING_ENABLED, POOL_DEBUG
export checkpoint!, rewind!, reset!
export CACHE_WAYS, set_cache_ways!  # N-way cache configuration
export get_task_local_cuda_pool, get_task_local_cuda_pools  # CUDA (stubs, overridden by extension)

# Extension API (for GPU backends)
export AbstractTypedPool, AbstractArrayPool  # For subtyping
export DisabledPool, DISABLED_CPU, pooling_enabled  # Disabled pool support
# Note: Extensions add methods to _get_pool_for_backend(::Val{:backend}) directly

# Version-specific core types
@static if VERSION >= v"1.11-"
    include("types.jl")
else
    include("legacy/types.jl")
end

# Debugging & validation utilities (needed by macros)
include("utils.jl")

# Version-specific acquisition and bitarray operations
@static if VERSION >= v"1.11-"
    include("acquire.jl")
    include("bitarray.jl")
else
    include("legacy/acquire.jl")
    include("legacy/bitarray.jl")
end

# Convenience functions: zeros!, ones!, similar!
include("convenience.jl")

# Version-specific state management
@static if VERSION >= v"1.11-"
    include("state.jl")
else
    include("legacy/state.jl")
end

# Task-local pool
include("task_local_pool.jl")

# Macros: @with_pool, @maybe_with_pool
include("macros.jl")

end # module
