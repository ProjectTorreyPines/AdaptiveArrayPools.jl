# ==============================================================================
# CUDA Dispatch Methods
# ==============================================================================
# Key dispatch points for GPU-specific allocation and type routing.

using AdaptiveArrayPools: allocate_vector, wrap_array, get_typed_pool!

# ==============================================================================
# Allocation Dispatch (single GPU-specific method needed!)
# ==============================================================================

@inline AdaptiveArrayPools.allocate_vector(
    ::AbstractTypedPool{T,CuVector{T}}, n::Int
) where {T} = CuVector{T}(undef, n)

# ==============================================================================
# Array Wrapping Dispatch
# ==============================================================================

# GPU uses reshape which returns CuArray{T,N} via GPUArrays derive()
# (NOT ReshapedArray like CPU - this is simpler for GPU kernels)
@inline AdaptiveArrayPools.wrap_array(
    ::AbstractTypedPool{T,CuVector{T}}, flat_view, dims::NTuple{N,Int}
) where {T,N} = reshape(flat_view, dims)

# ==============================================================================
# get_typed_pool! Dispatches for CuAdaptiveArrayPool
# ==============================================================================

# Fast path: compile-time dispatch for fixed slots
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Float32}) = p.float32
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Float64}) = p.float64
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Float16}) = p.float16
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Int32}) = p.int32
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Int64}) = p.int64
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{ComplexF32}) = p.complexf32
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{ComplexF64}) = p.complexf64
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Bool}) = p.bool

# Slow path: rare types via IdDict (with checkpoint correction!)
@inline function AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{T}) where {T}
    get!(p.others, T) do
        tp = CuTypedPool{T}()
        # CRITICAL: Match CPU behavior - auto-checkpoint new pool if inside @with_pool scope
        # Without this, rewind! would corrupt state for dynamically-created pools
        if p._current_depth > 1
            push!(tp._checkpoint_n_active, 0)  # n_active starts at 0
            push!(tp._checkpoint_depths, p._current_depth)
        end
        tp
    end::CuTypedPool{T}
end
