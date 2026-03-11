"""
    AdaptiveArrayPoolsCUDAExt

CUDA extension for AdaptiveArrayPools.jl. Provides GPU memory pooling
with the same checkpoint/rewind semantics as CPU pools.

Loaded automatically when `using CUDA` with AdaptiveArrayPools.
"""
module AdaptiveArrayPoolsCUDAExt

using AdaptiveArrayPools
using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool
using CUDA

# Type definitions
include("types.jl")

# Dispatch methods (allocate_vector, wrap_array, get_typed_pool!)
include("dispatch.jl")

# GPU-specific acquire (arr_wrappers + setfield!, _resize_to_fit!, _reshape_impl!)
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
# get_task_local_cuda_pool, get_task_local_cuda_pools are exported from AdaptiveArrayPools

end # module
