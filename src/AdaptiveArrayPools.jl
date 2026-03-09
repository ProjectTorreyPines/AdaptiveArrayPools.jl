module AdaptiveArrayPools

using Printf

# Public API
export AdaptiveArrayPool, acquire!, unsafe_acquire!, pool_stats, get_task_local_pool
export acquire_view!, acquire_array!  # Explicit naming aliases
export zeros!, ones!, trues!, falses!, similar!, reshape!, default_eltype  # Convenience functions
export unsafe_zeros!, unsafe_ones!, unsafe_similar!  # Unsafe convenience functions
export Bit  # Sentinel type for BitArray (use with acquire!, trues!, falses!)
export @with_pool, @maybe_with_pool
export STATIC_POOLING, MAYBE_POOLING, POOL_DEBUG, POOL_SAFETY_LV, STATIC_POOL_CHECKS
export PoolEscapeError, EscapePoint, @skip_check_vars
export USE_POOLING, MAYBE_POOLING_ENABLED  # Deprecated aliases (backward compat)
export checkpoint!, rewind!, reset!
export get_task_local_cuda_pool, get_task_local_cuda_pools  # CUDA (stubs, overridden by extension)

# Extension API (for GPU backends)
export AbstractTypedPool, AbstractArrayPool  # For subtyping
export DisabledPool, DISABLED_CPU, pooling_enabled  # Disabled pool support
# Note: Extensions add methods to _get_pool_for_backend(::Val{:backend}) directly

# All includes grouped under a single version branch
@static if VERSION >= v"1.11-"
    include("types.jl")
    include("utils.jl")
    include("acquire.jl")
    include("bitarray.jl")
    include("convenience.jl")
    include("state.jl")
    include("task_local_pool.jl")
    include("debug.jl")
    include("macros.jl")
else
    export CACHE_WAYS, set_cache_ways!  # N-way cache configuration (legacy only)
    include("legacy/types.jl")
    include("utils.jl")
    include("legacy/acquire.jl")
    include("legacy/bitarray.jl")
    include("convenience.jl")
    include("legacy/state.jl")
    include("task_local_pool.jl")
    include("debug.jl")
    include("macros.jl")
end

end # module
