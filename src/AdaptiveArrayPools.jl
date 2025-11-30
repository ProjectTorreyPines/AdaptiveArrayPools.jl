module AdaptiveArrayPools

using Printf

export AdaptiveArrayPool, acquire!, pool_stats, get_global_pool
export @with_pool, @maybe_with_pool, @pool_kwarg
export USE_POOLING, MAYBE_POOLING_ENABLED, POOL_DEBUG

# Note: checkpoint!/rewind! are not exported to keep the public API minimal
# Users can import them: import AdaptiveArrayPools: checkpoint!, rewind!

# Core data structures
include("types.jl")

# Debugging & validation utilities (needed by macros)
include("utils.jl")

# Core operations: get_view!, acquire!, checkpoint!, rewind!, empty!
include("core.jl")

# Global pool (Task Local Storage)
include("global_pool.jl")

# Macros: @with_pool, @maybe_with_pool, @pool_kwarg
include("macros.jl")

end # module
