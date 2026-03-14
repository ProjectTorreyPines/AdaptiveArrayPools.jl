"""
    AdaptiveArrayPoolsCUDAExt

CUDA extension for AdaptiveArrayPools.jl. Provides GPU memory pooling
with the same checkpoint/rewind semantics as CPU pools.

Loaded automatically when `using CUDA` with AdaptiveArrayPools.
"""
module AdaptiveArrayPoolsCUDAExt

using AdaptiveArrayPools
using CUDA

# GPU pooling requires Julia 1.11+ (setfield!-based Array, arr_wrappers cache).
# On older Julia, the extension loads but provides no functionality.
@static if VERSION >= v"1.11-"

    using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool

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

else
    @warn "AdaptiveArrayPoolsCUDAExt requires Julia 1.11+. GPU pooling is disabled." maxlog = 1
end # @static if

end # module
