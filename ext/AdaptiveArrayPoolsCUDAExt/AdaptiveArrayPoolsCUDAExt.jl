"""
    AdaptiveArrayPoolsCUDAExt

CUDA extension for AdaptiveArrayPools.jl. Provides GPU memory pooling
with the same checkpoint/rewind semantics as CPU pools.

Loaded automatically when `using CUDA` with AdaptiveArrayPools.
"""
module AdaptiveArrayPoolsCUDAExt

using AdaptiveArrayPools
using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool
using Preferences: @load_preference, @set_preferences!

# N-way view cache configuration (CUDA only — CPU ≥1.11 uses slot-first _claim_slot!).
# GPU view/reshape allocates ~80 bytes on CPU heap, so caching still matters.
const CACHE_WAYS = let
    ways = @load_preference("cache_ways", 4)::Int
    if ways < 1 || ways > 16
        @warn "CACHE_WAYS=$ways out of range [1,16], using default 4"
        4
    else
        ways
    end
end
using CUDA

# Type definitions
include("types.jl")

# Dispatch methods (allocate_vector, wrap_array, get_typed_pool!)
include("dispatch.jl")

# GPU-specific get_view! implementation
include("acquire.jl")

# Task-local pool (multi-device aware)
include("task_local_pool.jl")

# State management (checkpoint!, rewind!, reset!, empty!)
include("state.jl")

# Safety: poisoning, escape detection, borrow tracking
include("debug.jl")

# Display & statistics (pool_stats, show)
include("utils.jl")

# Macro support (@with_pool :cuda)
include("macros.jl")

# Convenience functions (Float32 default for zeros!/ones!)
include("convenience.jl")

# Exports (types only - functions are exported from main module)
export CuTypedPool, CuAdaptiveArrayPool
export GPU_FIXED_SLOT_FIELDS
export set_cuda_safety_level!
# get_task_local_cuda_pool, get_task_local_cuda_pools are exported from AdaptiveArrayPools

end # module
