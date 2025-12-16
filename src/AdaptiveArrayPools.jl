module AdaptiveArrayPools

using Printf

# Public API
export AdaptiveArrayPool, acquire!, unsafe_acquire!, pool_stats, get_task_local_pool
export acquire_view!, acquire_array!  # Explicit naming aliases
export zeros!, ones!, similar!, default_eltype  # Convenience functions
export unsafe_zeros!, unsafe_ones!, unsafe_similar!  # Unsafe convenience functions
export @with_pool, @maybe_with_pool
export USE_POOLING, MAYBE_POOLING_ENABLED, POOL_DEBUG
export checkpoint!, rewind!, reset!
export CACHE_WAYS, set_cache_ways!  # N-way cache configuration
export get_task_local_cuda_pool, get_task_local_cuda_pools  # CUDA (stubs, overridden by extension)

# Extension API (for GPU backends)
export AbstractTypedPool, AbstractArrayPool  # For subtyping
export DisabledPool, DISABLED_CPU, pooling_enabled  # Disabled pool support
# Note: Extensions add methods to _get_pool_for_backend(::Val{:backend}) directly

# Core data structures
include("types.jl")

# Debugging & validation utilities (needed by macros)
include("utils.jl")

# Acquisition operations: get_view!, acquire!, unsafe_acquire!, aliases
include("acquire.jl")

# Convenience functions: zeros!, ones!, similar!
include("convenience.jl")

# State management: checkpoint!, rewind!, reset!, empty!
include("state.jl")

# Task-local pool
include("task_local_pool.jl")

# Macros: @with_pool, @maybe_with_pool
include("macros.jl")

end # module
