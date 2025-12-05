module AdaptiveArrayPools

using Printf

export AdaptiveArrayPool, acquire!, unsafe_acquire!, pool_stats, get_task_local_pool
export acquire_view!, acquire_array!  # Explicit naming aliases
export @with_pool, @maybe_with_pool
export USE_POOLING, MAYBE_POOLING_ENABLED, POOL_DEBUG
export checkpoint!, rewind!

# Core data structures
include("types.jl")

# Debugging & validation utilities (needed by macros)
include("utils.jl")

# Core operations: get_view!, acquire!, checkpoint!, rewind!, empty!
include("core.jl")

# Task-local pool
include("task_local_pool.jl")

# Macros: @with_pool, @maybe_with_pool
include("macros.jl")

end # module
