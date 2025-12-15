"""
    AdaptiveArrayPoolsCUDAExt

CUDA extension for AdaptiveArrayPools.jl. Provides GPU memory pooling
with the same checkpoint/rewind semantics as CPU pools.

Loaded automatically when `using CUDA` with AdaptiveArrayPools.
"""
module AdaptiveArrayPoolsCUDAExt

using AdaptiveArrayPools
using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool, CACHE_WAYS,
                          allocate_vector, wrap_array, get_typed_pool!, get_view!
using CUDA

# Type definitions
include("types.jl")

# Dispatch methods (allocate_vector, wrap_array, get_typed_pool!)
include("dispatch.jl")

# GPU-specific get_view! implementation
include("acquire.jl")

# Exports
export CuTypedPool, CuAdaptiveArrayPool
export GPU_FIXED_SLOT_FIELDS

end # module
