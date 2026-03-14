"""
    AdaptiveArrayPoolsMetalExt

Metal extension for AdaptiveArrayPools.jl. Provides GPU memory pooling
with the same checkpoint/rewind semantics as CPU pools.

Loaded automatically when `using Metal` with AdaptiveArrayPools.

Supports Metal.PrivateStorage only. Default element type is Float32.
Explicitly unsupported: Float64, ComplexF64.
"""
module AdaptiveArrayPoolsMetalExt

using AdaptiveArrayPools
using Metal

# GPU pooling requires Julia 1.11+ (setfield!-based Array, arr_wrappers cache).
# On older Julia, the extension loads but provides no functionality.
@static if VERSION >= v"1.11-"

    using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool
    using Metal.GPUArrays

    include("types.jl")
    include("dispatch.jl")
    include("acquire.jl")
    include("task_local_pool.jl")
    include("state.jl")
    include("debug.jl")
    include("utils.jl")
    include("macros.jl")
    include("convenience.jl")

    export MetalTypedPool, MetalAdaptiveArrayPool
    export METAL_FIXED_SLOT_FIELDS

else
    @warn "AdaptiveArrayPoolsMetalExt requires Julia 1.11+. GPU pooling is disabled." maxlog = 1
end # @static if

end # module
