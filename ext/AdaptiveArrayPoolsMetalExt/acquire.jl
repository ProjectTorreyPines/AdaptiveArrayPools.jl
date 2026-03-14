# ==============================================================================
# Metal-Specific Acquire Implementation (arr_wrappers + setfield!)
# ==============================================================================
# Mirrors CUDA acquire.jl — fully self-contained, no shared GPU common layer.
#
# Key differences from CUDA:
# - MtlArray uses sizeof(T) for capacity (no aligned_sizeof)
# - MtlArray{T,N,S} carries storage mode S
# - DataRef access via getfield (same .data.rc identity check as CUDA)
#
# ⚠ Depends on MtlArray internal fields (:data, :maxsize, :offset, :dims).
#   Tested with Metal.jl v1.x.
# ==============================================================================

using AdaptiveArrayPools: get_view!, get_array!, allocate_vector, safe_prod,
    _record_type_touch!, _fixed_slot_bit, _checkpoint_typed_pool!,
    _store_arr_wrapper!, _check_pool_growth, _reshape_impl!,
    _acquire_impl!, _acquire_view_impl!, _maybe_record_borrow!,
    _MODE_BITS_MASK

using Metal.GPUArrays: unsafe_free!

# Guard against Metal.jl internal API changes
@static if !(
        ismutabletype(MtlArray) &&
            hasfield(MtlArray, :data) &&
            hasfield(MtlArray, :maxsize) &&
            hasfield(MtlArray, :offset) &&
            hasfield(MtlArray, :dims)
    )
    error("Unsupported Metal.jl version: MtlArray must be mutable with :data, :maxsize, :offset, :dims.")
end

# Verify DataRef has .rc field
let DataRefT = fieldtype(MtlArray{Float32, 1, Metal.PrivateStorage}, :data)
    if !hasfield(DataRefT, :rc)
        error("Unsupported Metal.jl version: DataRef must have :rc field for storage identity check.")
    end
end

# ==============================================================================
# _resize_to_fit! — Capacity-Aware Resize for Metal
# ==============================================================================

"""
    _resize_to_fit!(A::MtlArray{T,1,S}, n::Integer) -> MtlArray{T,1,S}

Resize a MtlVector's logical length, using `setfield!(:dims)` when within capacity.

- `n > capacity`: delegates to `resize!(A, n)` (may grow GPU allocation)
- `n <= capacity, n != length(A)`: `setfield!(:dims)` only — no GPU operation
- `n == length(A)`: no-op

Capacity = `A.maxsize / sizeof(T)`. Since `setfield!(:dims)` preserves
`maxsize`, capacity information is naturally retained across shrink/grow cycles.
"""
@inline function _resize_to_fit!(A::MtlArray{T, 1, S}, n::Integer) where {T, S}
    cap = getfield(A, :maxsize) ÷ sizeof(T)
    if n > cap
        resize!(A, n)
    elseif n != length(A)
        setfield!(A, :dims, (Int(n),))
    end
    return A
end

# ==============================================================================
# _metal_claim_slot! — Capacity-Based Slot Claim
# ==============================================================================

"""
    _metal_claim_slot!(tp::MetalTypedPool{T,S}, total_len::Int) -> Int

Claim the next slot, ensuring the backing vector's GPU buffer has capacity >= `total_len`.
Uses maxsize-based capacity check instead of length check to avoid triggering
Metal.jl's resize! unnecessarily (especially after safety invalidation sets dims=(0,)).
"""
@inline function _metal_claim_slot!(tp::MetalTypedPool{T, S}, total_len::Int) where {T, S}
    tp.n_active += 1
    idx = tp.n_active
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, total_len))
        _check_pool_growth(tp, idx)
    else
        _resize_to_fit!(@inbounds(tp.vectors[idx]), total_len)
    end
    return idx
end

"""
    _metal_claim_slot!(tp::MetalTypedPool{T,S}) -> Int

Claim the next slot without provisioning memory (zero-length backing vector).
Used by `_reshape_impl!` which only needs the slot index for wrapper caching —
the wrapper points to a different array's memory via `setfield!(:data)`.
"""
@inline function _metal_claim_slot!(tp::MetalTypedPool{T, S}) where {T, S}
    tp.n_active += 1
    idx = tp.n_active
    if idx > length(tp.vectors)
        push!(tp.vectors, MtlArray{T, 1, S}(undef, 0))
        _check_pool_growth(tp, idx)
    end
    return idx
end

# ==============================================================================
# _update_metal_wrapper_data! — DataRef Refcount Management
# ==============================================================================

"""
    _update_metal_wrapper_data!(wrapper::MtlArray, source::MtlArray)

Update wrapper's GPU data reference when the source's buffer has changed.
Decrements old refcount, increments new. @noinline: rare path (only on grow
beyond capacity), keep off the hot inlined acquire path.
"""
@noinline function _update_metal_wrapper_data!(wrapper::MtlArray, source::MtlArray)
    unsafe_free!(getfield(wrapper, :data))
    setfield!(wrapper, :data, copy(getfield(source, :data)))
    setfield!(wrapper, :maxsize, getfield(source, :maxsize))
    setfield!(wrapper, :offset, getfield(source, :offset))
    return nothing
end

# ==============================================================================
# _acquire_impl! / _acquire_view_impl! — Direct get_array! Dispatch
# ==============================================================================
# On Metal, both acquire! and acquire_view! go through get_array! directly.
# No view/array distinction — MtlArray is always returned.

"""
    _acquire_impl!(pool::MetalAdaptiveArrayPool, T, n) -> MtlArray{T,1,S}
    _acquire_impl!(pool::MetalAdaptiveArrayPool, T, dims...) -> MtlArray{T,N,S}

Metal override: routes directly to `get_array!` (no view indirection).
"""
@inline function AdaptiveArrayPools._acquire_impl!(pool::MetalAdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    result = get_array!(tp, (n,))
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function AdaptiveArrayPools._acquire_impl!(pool::MetalAdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    result = get_array!(tp, dims)
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function AdaptiveArrayPools._acquire_impl!(pool::MetalAdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _acquire_impl!(pool, T, dims...)
end

"""
    _acquire_view_impl!(pool::MetalAdaptiveArrayPool, T, dims...) -> MtlArray{T,N,S}

Metal override: same as `_acquire_impl!` — Metal has no view/array distinction.
"""
@inline function AdaptiveArrayPools._acquire_view_impl!(pool::MetalAdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    return _acquire_impl!(pool, T, n)
end

@inline function AdaptiveArrayPools._acquire_view_impl!(pool::MetalAdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    return _acquire_impl!(pool, T, dims...)
end

@inline function AdaptiveArrayPools._acquire_view_impl!(pool::MetalAdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _acquire_impl!(pool, T, dims...)
end

# ==============================================================================
# get_view! / get_array! — arr_wrappers + setfield! Based Zero-Alloc
# ==============================================================================
# get_view! delegates to get_array! for backward compat (e.g., direct get_view! calls).
# The main acquire path now bypasses get_view! entirely via _acquire_impl! above.

@inline function AdaptiveArrayPools.get_view!(tp::MetalTypedPool{T, S}, n::Int) where {T, S}
    return get_array!(tp, (n,))
end

@inline function AdaptiveArrayPools.get_view!(tp::MetalTypedPool{T, S}, dims::NTuple{N, Int}) where {T, S, N}
    return get_array!(tp, dims)
end

"""
    get_array!(tp::MetalTypedPool{T,S}, dims::NTuple{N,Int}) -> MtlArray{T,N,S}

Get an N-dimensional `MtlArray` from the pool with `setfield!`-based wrapper reuse.

## Cache Hit (common case, 0-alloc)
1. Look up `arr_wrappers[N][slot]`
2. Check `wrapper.data.rc !== vec.data.rc` — if same GPU buffer, just `setfield!(:dims)`
3. If different (rare: only after grow beyond capacity), update `:data` via refcount management

## Cache Miss (first call per (slot, N))
Creates MtlArray wrapper sharing backing vector's GPU memory via `copy(vec.data)`,
stores in `arr_wrappers[N][slot]` via `_store_arr_wrapper!` (reuses base module helper).
"""
@inline function AdaptiveArrayPools.get_array!(tp::MetalTypedPool{T, S}, dims::NTuple{N, Int}) where {T, S, N}
    total_len = safe_prod(dims)
    slot = _metal_claim_slot!(tp, total_len)
    @inbounds vec = tp.vectors[slot]

    # arr_wrappers lookup (direct index, no hash — same as CPU/CUDA path)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && slot <= length(wrappers)
        wrapper = @inbounds wrappers[slot]
        if wrapper !== nothing
            mtl = wrapper::MtlArray{T, N, S}
            # Check if backing vec's GPU buffer changed (rare: only on grow beyond capacity)
            if getfield(mtl, :data).rc !== getfield(vec, :data).rc
                _update_metal_wrapper_data!(mtl, vec)
            end
            setfield!(mtl, :dims, dims)
            return mtl
        end
    end

    # Cache miss: create wrapper sharing vec's GPU memory
    mtl = MtlArray{T, N, S}(
        copy(getfield(vec, :data)), dims;
        maxsize = getfield(vec, :maxsize),
        offset = getfield(vec, :offset),
    )
    _store_arr_wrapper!(tp, N, slot, mtl)
    return mtl
end

# ==============================================================================
# _reshape_impl! for MtlArray — Zero-Alloc Reshape
# ==============================================================================

"""
    _reshape_impl!(pool::MetalAdaptiveArrayPool, A::MtlArray{T,M,S}, dims::NTuple{N,Int}) -> MtlArray{T,N,S}

Zero-allocation reshape for MtlArray using `setfield!`-based wrapper reuse.

- **Same dimensionality (M == N)**: `setfield!(A, :dims, dims)` — no pool interaction
- **Different dimensionality (M != N)**: Claims a pool slot, reuses cached `MtlArray{T,N,S}`
  wrapper with `setfield!(:dims)` pointing to `A`'s GPU memory.
"""
@inline function AdaptiveArrayPools._reshape_impl!(
        pool::MetalAdaptiveArrayPool, A::MtlArray{T, M, S}, dims::NTuple{N, Int}
    ) where {T, M, S, N}
    for d in dims
        d < 0 && throw(ArgumentError("invalid MtlArray dimensions"))
    end
    total_len = safe_prod(dims)
    length(A) == total_len || throw(
        DimensionMismatch(
            "new dimensions $(dims) must be consistent with array length $(length(A))"
        )
    )

    # 0-D reshape: rare edge case, delegate to Base (arr_wrappers is 1-indexed by N)
    N == 0 && return reshape(A, dims)

    # Same dimensionality: just update dims in-place, no pool interaction
    if M == N
        setfield!(A, :dims, dims)
        return A
    end

    # Different dimensionality: claim slot + reuse cached N-D wrapper
    tp = AdaptiveArrayPools.get_typed_pool!(pool, T)
    _record_type_touch!(pool, T)
    slot = _metal_claim_slot!(tp)

    # Look up cached wrapper (direct index, no hash)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && slot <= length(wrappers)
        wrapper = @inbounds wrappers[slot]
        if wrapper !== nothing
            mtl = wrapper::MtlArray{T, N, S}
            if getfield(mtl, :data).rc !== getfield(A, :data).rc
                _update_metal_wrapper_data!(mtl, A)
            end
            setfield!(mtl, :dims, dims)
            setfield!(mtl, :offset, getfield(A, :offset))
            return mtl
        end
    end

    # Cache miss (first call per slot+N): create wrapper, cache forever
    mtl = MtlArray{T, N, S}(
        copy(getfield(A, :data)), dims;
        maxsize = getfield(A, :maxsize),
        offset = getfield(A, :offset),
    )
    _store_arr_wrapper!(tp, N, slot, mtl)
    return mtl
end

# ==============================================================================
# Metal _record_type_touch! override
# ==============================================================================
# Float16 on Metal: direct struct field with _fixed_slot_bit(Float16)=0.
# We track Float16 via bit 7 (Metal reassignment; CPU uses bit 7 for Bit type,
# absent on GPU). This gives Float16 lazy first-touch checkpointing in bit-14
# (typed lazy) and bit-15 (dynamic) modes, ensuring Case A (not Case B) fires
# at rewind and parent n_active is preserved.

@inline function AdaptiveArrayPools._record_type_touch!(pool::MetalAdaptiveArrayPool, ::Type{T}) where {T}
    depth = pool._current_depth
    b = _fixed_slot_bit(T)
    if b == UInt16(0)
        if T === Float16
            # Float16: Metal direct field tracked via bit 7 (not in pool.others dict).
            b16 = UInt16(1) << 7
            current_mask = @inbounds pool._touched_type_masks[depth]
            # Lazy first-touch checkpoint: bit 14 (typed lazy) OR bit 15 (dynamic), first touch only.
            # Guard: skip if already checkpointed at this depth (prevents double-push).
            if (current_mask & _MODE_BITS_MASK) != 0 && (current_mask & b16) == 0
                if @inbounds(pool.float16._checkpoint_depths[end]) != depth
                    _checkpoint_typed_pool!(pool.float16, depth)
                end
            end
            @inbounds pool._touched_type_masks[depth] = current_mask | b16
        else
            # Genuine others type (UInt8, Int8, etc.) — eagerly snapshotted at scope entry.
            @inbounds pool._touched_has_others[depth] = true
        end
    else
        current_mask = @inbounds pool._touched_type_masks[depth]
        # Lazy first-touch checkpoint for fixed-slot types in bit 14/15 modes.
        # Guard: skip if already checkpointed at this depth (prevents double-push).
        if (current_mask & _MODE_BITS_MASK) != 0 && (current_mask & b) == 0
            tp = AdaptiveArrayPools.get_typed_pool!(pool, T)
            if @inbounds(tp._checkpoint_depths[end]) != depth
                _checkpoint_typed_pool!(tp, depth)
            end
        end
        @inbounds pool._touched_type_masks[depth] = current_mask | b
    end
    return nothing
end
