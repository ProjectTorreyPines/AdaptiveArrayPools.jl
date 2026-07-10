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

# Fixed-slot element types (dedicated struct fields above). Used by trim!(pool, T)
# to recognize always-present types without creating an `others` entry.
const _METAL_FIXED_TYPES = Union{Float32, Float16, Int32, Int64, ComplexF32, Bool}

# Slow path: rare types via IdDict (with checkpoint correction!)
# Explicitly reject Float64 and ComplexF64 (unsupported by Metal hardware).
@inline function AdaptiveArrayPools.get_typed_pool!(p::MetalAdaptiveArrayPool, ::Type{T}) where {T}
    if T === Float64 || T === ComplexF64
        throw(ArgumentError("Metal backend does not support $T"))
    end
    # Memo fast path: same type as the previous slow-path lookup (mirror of CPU
    # src/types.jl's get_typed_pool!; one pointer compare instead of an IdDict lookup).
    p._lookup_memo_type === T && return p._lookup_memo_tp::MetalTypedPool{T, Metal.PrivateStorage}
    tp = get(p.others, T, nothing)
    if tp !== nothing
        tp = tp::MetalTypedPool{T, Metal.PrivateStorage}
        p._lookup_memo_type = T
        p._lookup_memo_tp = tp
        return tp
    end
    # New type — create, register, memoize, and first-touch checkpoint when
    # inside a scope (depth > 1), pushing one depth-tagged stack entry.
    new_tp = MetalTypedPool{T, Metal.PrivateStorage}()
    p.others[T] = new_tp
    p._lookup_memo_type = T
    p._lookup_memo_tp = new_tp
    if p._current_depth > 1
        st = getfield(new_tp, :state)
        push!(st._checkpoint_n_active, 0)  # n_active starts at 0
        push!(st._checkpoint_depths, p._current_depth)
        push!(p._touched_others_states, st)
        push!(p._touched_others_depths, p._current_depth)
        AdaptiveArrayPools._runtime_check(p) && push!(p._touched_others_pools, new_tp)
        # Signal that a fallback type was touched so lazy/typed-lazy rewind
        # iterates the drain path (same fix as CPU get_typed_pool!)
        @inbounds p._touched_has_others[p._current_depth] = true
    end
    return new_tp
end
