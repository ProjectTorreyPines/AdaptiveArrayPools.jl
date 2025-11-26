module AdaptiveArrayPools

using Printf

export AdaptiveArrayPool, acquire!, pool_stats
export @use_pool, @maybe_use_pool, @with_pool
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

# Macros: @use_pool, @maybe_use_pool, @with_pool
include("macros.jl")

end # module
