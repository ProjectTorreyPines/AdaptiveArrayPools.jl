module AdaptiveArrayPools

export AdaptiveArrayPool, acquire!, pool_stats
export @use_pool, @use_global_pool, @maybe_use_global_pool
export ENABLE_POOLING, POOL_DEBUG

# Note: mark!/reset!/empty! are NOT exported to avoid conflict with Base
# Users should use: import AdaptiveArrayPools: mark!, reset!, empty!

# Core data structures
include("types.jl")

# Debugging & validation utilities (needed by macros)
include("utils.jl")

# Core operations: get_view!, acquire!, checkpoint!, rewind!, empty!
include("core.jl")

# Global pool (Task Local Storage)
include("global_pool.jl")

# Macros: @use_pool, @use_global_pool, @maybe_use_global_pool
include("macros.jl")

end # module
