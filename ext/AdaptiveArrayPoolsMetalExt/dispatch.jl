# ==============================================================================
# Metal Dispatch Methods
# ==============================================================================
# Key dispatch points for Metal-specific allocation and type routing.

using AdaptiveArrayPools: allocate_vector, get_typed_pool!

# ==============================================================================
# Allocation Dispatch
# ==============================================================================

@inline function AdaptiveArrayPools.allocate_vector(
        ::AbstractTypedPool{T, MtlArray{T, 1, S}}, n::Int
    ) where {T, S}
    return MtlArray{T, 1, S}(undef, n)
end

# ==============================================================================
# get_typed_pool! Dispatches for MetalAdaptiveArrayPool
# ==============================================================================

# Fast path: compile-time dispatch for fixed slots
@inline AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{Float32}) = p.float32
@inline AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{Float16}) = p.float16
@inline AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{Int32}) = p.int32
@inline AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{Int64}) = p.int64
@inline AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{ComplexF32}) = p.complexf32
@inline AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{Bool}) = p.bool

# Slow path: rare types via IdDict (with checkpoint correction!)
# Explicitly reject Float64 and ComplexF64 (unsupported by Metal hardware).
@inline function AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{T}) where {T}
    if T === Float64 || T === ComplexF64
        throw(ArgumentError("Metal backend does not support $T"))
    end
    return get!(p.others, T) do
        tp = MetalTypedPool{T, Metal.PrivateStorage}()
        # CRITICAL: Match CPU behavior - auto-checkpoint new pool if inside @with_pool scope
        # Without this, rewind! would corrupt state for dynamically-created pools
        if p._current_depth > 1
            push!(tp._checkpoint_n_active, 0)  # n_active starts at 0
            push!(tp._checkpoint_depths, p._current_depth)
            # Signal that a fallback type was touched so lazy/typed-lazy rewind
            # iterates pool.others (same fix as CPU get_typed_pool!)
            @inbounds p._touched_has_others[p._current_depth] = true
        end
        tp
    end::MetalTypedPool{T, Metal.PrivateStorage}
end
