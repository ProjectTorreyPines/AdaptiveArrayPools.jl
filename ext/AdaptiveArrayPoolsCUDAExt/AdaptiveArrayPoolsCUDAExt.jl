"""
    AdaptiveArrayPoolsCUDAExt

CUDA extension for AdaptiveArrayPools.jl. Provides GPU memory pooling
with the same checkpoint/rewind semantics as CPU pools.

Loaded automatically when `using CUDA` with AdaptiveArrayPools.
"""
module AdaptiveArrayPoolsCUDAExt

using AdaptiveArrayPools
using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool, CACHE_WAYS,
                          allocate_vector, wrap_array, get_typed_pool!, get_view!,
                          foreach_fixed_slot, _get_pool_for_backend
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

# Macro support (@with_pool :cuda, @with_cuda_pool)
include("macros.jl")

# Exports
export CuTypedPool, CuAdaptiveArrayPool
export GPU_FIXED_SLOT_FIELDS
export get_task_local_cuda_pool, get_task_local_cuda_pools
export @with_cuda_pool

end # module
