# ==============================================================================
# CUDA-Specific Acquire Implementation (arr_wrappers + setfield!)
# ==============================================================================
# Mirrors CPU's Julia 1.11+ approach: cached CuArray{T,N} wrappers reused via
# setfield!(:dims) for zero-allocation on cache hit.
#
# Key differences from CPU:
# - CPU Array has :ref (MemoryRef, GC-managed, no refcount)
# - CuArray has :data (DataRef, manual refcount via Threads.Atomic{Int})
# - We minimize refcount overhead via wrapper.data.rc !== vec.data.rc identity
#   check (~2ns). Only update :data when GPU buffer actually changed (rare).
#
# ==============================================================================
# Memory Resize Strategy: _resize_to_fit!
# ==============================================================================
# CUDA.jl resize! behavior varies by version:
#   - v5.9.x: ALWAYS reallocates (no capacity management)
#   - v5.10.x+: capacity check — reallocates only when n > cap or n < cap÷4
#
# _resize_to_fit!(A, n):
#   - n within capacity (maxsize): setfield!(:dims) only — no GPU operation
#   - n beyond capacity: delegates to CUDA.jl resize! (GPU alloc)
#   - n == length(A): no-op
#
# This is strictly better than _resize_without_shrink! (which only optimized
# shrink). _resize_to_fit! also optimizes grow-within-capacity, critical for
# re-acquire after safety invalidation (dims=(0,), maxsize preserved).
#
# ⚠ Depends on CuArray internal fields (:data, :dims, :maxsize, :offset).
#   Tested with CUDA.jl v5.x.
# ==============================================================================

using AdaptiveArrayPools: get_view!, get_array!, allocate_vector, safe_prod,
    _record_type_touch!, _fixed_slot_bit, _checkpoint_typed_pool!,
    _store_arr_wrapper!, _check_pool_growth, _reshape_impl!,
    _MODE_BITS_MASK

using CUDA.GPUArrays: unsafe_free!

# Guard against CUDA.jl internal API changes (tested with v5.x).
@static if !(ismutabletype(CuArray) && hasfield(CuArray, :dims) &&
             hasfield(CuArray, :data) && hasfield(CuArray, :maxsize) &&
             hasfield(CuArray, :offset))
    error("Unsupported CUDA.jl version: CuArray must be mutable with :data, :dims, :maxsize, :offset fields.")
end

# ==============================================================================
# Aligned sizeof (mirrors CUDA.jl internal)
# ==============================================================================

"""Compute aligned element size, matching CUDA.jl's internal `aligned_sizeof`."""
_aligned_sizeof(::Type{T}) where {T} = max(sizeof(T), Base.datatype_alignment(T))

# ==============================================================================
# _resize_to_fit! — Capacity-Aware Resize (superset of _resize_without_shrink!)
# ==============================================================================

"""
    _resize_to_fit!(A::CuVector{T}, n::Integer) -> CuVector{T}

Resize a CuVector's logical length, using `setfield!(:dims)` when within capacity.

- `n > capacity`: delegates to `resize!(A, n)` (may grow GPU allocation)
- `n ≤ capacity, n ≠ length(A)`: `setfield!(:dims)` only — no GPU operation
- `n == length(A)`: no-op

Capacity = `A.maxsize ÷ aligned_sizeof(T)`. Since `setfield!(:dims)` preserves
`maxsize`, capacity information is naturally retained across shrink/grow cycles.
"""
@inline function _resize_to_fit!(A::CuVector{T}, n::Integer) where {T}
    cap = A.maxsize ÷ _aligned_sizeof(T)
    if n > cap
        resize!(A, n)                       # Beyond capacity: delegate to CUDA.jl
    elseif n != length(A)
        setfield!(A, :dims, (Int(n),))      # Within capacity: dims only
    end
    return A
end

# ==============================================================================
# _cuda_claim_slot! — Capacity-Based Slot Claim
# ==============================================================================

"""
    _cuda_claim_slot!(tp::CuTypedPool{T}, total_len::Int) -> Int

Claim the next slot, ensuring the backing vector's GPU buffer has capacity ≥ `total_len`.
Uses maxsize-based capacity check instead of length check to avoid triggering
CUDA.jl's resize! unnecessarily (especially after safety invalidation sets dims=(0,)).
"""
@inline function _cuda_claim_slot!(tp::CuTypedPool{T}, total_len::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, total_len))
        _check_pool_growth(tp, idx)
    else
        # _resize_to_fit! handles all cases:
        # - n > capacity: resize! (GPU alloc)
        # - n != length: setfield!(:dims) — restores length after safety invalidation
        # - n == length: no-op (hot path)
        _resize_to_fit!(@inbounds(tp.vectors[idx]), total_len)
    end
    return idx
end

"""
    _cuda_claim_slot!(tp::CuTypedPool{T}) -> Int

Claim the next slot without provisioning memory (zero-length backing vector).
Used by `_reshape_impl!` which only needs the slot index for wrapper caching —
the wrapper points to a different array's memory via `setfield!(:data)`.
"""
@inline function _cuda_claim_slot!(tp::CuTypedPool{T}) where {T}
    tp.n_active += 1
    idx = tp.n_active
    if idx > length(tp.vectors)
        push!(tp.vectors, CuVector{T}(undef, 0))
        _check_pool_growth(tp, idx)
    end
    return idx
end

# ==============================================================================
# _update_cuda_wrapper_data! — DataRef Refcount Management
# ==============================================================================

"""
    _update_cuda_wrapper_data!(cu::CuArray, source::CuArray)

Update wrapper's GPU data reference when the source's buffer has changed.
Decrements old refcount, increments new. @noinline: rare path (only on grow
beyond capacity), keep off the hot inlined acquire path.
"""
@noinline function _update_cuda_wrapper_data!(cu::CuArray, source::CuArray)
    unsafe_free!(cu.data)
    setfield!(cu, :data, copy(source.data))
    setfield!(cu, :maxsize, source.maxsize)
    setfield!(cu, :offset, 0)
    return nothing
end

# ==============================================================================
# get_view! / get_array! — arr_wrappers + setfield! Based Zero-Alloc
# ==============================================================================

"""
    get_view!(tp::CuTypedPool{T}, n::Int) -> CuArray{T,1}

1D convenience wrapper - delegates to tuple version.
"""
@inline function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, n::Int) where {T}
    return get_view!(tp, (n,))
end

"""
    get_view!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Delegates to `get_array!` — on CUDA, both view and array paths return CuArray.
"""
@inline function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    return get_array!(tp, dims)
end

"""
    get_array!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Get an N-dimensional `CuArray` from the pool with `setfield!`-based wrapper reuse.

## Cache Hit (common case, 0-alloc)
1. Look up `arr_wrappers[N][slot]`
2. Check `wrapper.data.rc !== vec.data.rc` — if same GPU buffer, just `setfield!(:dims)`
3. If different (rare: only after grow beyond capacity), update `:data` via refcount management

## Cache Miss (first call per (slot, N))
Creates CuArray wrapper sharing backing vector's GPU memory via `copy(vec.data)`,
stores in `arr_wrappers[N][slot]` via `_store_arr_wrapper!` (reuses base module helper).
"""
@inline function AdaptiveArrayPools.get_array!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    slot = _cuda_claim_slot!(tp, total_len)
    @inbounds vec = tp.vectors[slot]

    # arr_wrappers lookup (direct index, no hash — same as CPU path)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && slot <= length(wrappers)
        wrapper = @inbounds wrappers[slot]
        if wrapper !== nothing
            cu = wrapper::CuArray{T, N}
            # Check if backing vec's GPU buffer changed (rare: only on grow beyond capacity)
            if cu.data.rc !== vec.data.rc
                _update_cuda_wrapper_data!(cu, vec)
            end
            setfield!(cu, :dims, dims)
            return cu
        end
    end

    # Cache miss: create wrapper sharing vec's GPU memory
    cu = CuArray{T, N}(copy(vec.data), dims; maxsize=vec.maxsize, offset=0)
    _store_arr_wrapper!(tp, N, slot, cu)
    return cu
end

# ==============================================================================
# _reshape_impl! for CuArray — Zero-Alloc Reshape
# ==============================================================================

"""
    _reshape_impl!(pool::CuAdaptiveArrayPool, A::CuArray{T,M}, dims::NTuple{N,Int}) -> CuArray{T,N}

Zero-allocation reshape for CuArray using `setfield!`-based wrapper reuse.

- **Same dimensionality (M == N)**: `setfield!(A, :dims, dims)` — no pool interaction
- **Different dimensionality (M ≠ N)**: Claims a pool slot, reuses cached `CuArray{T,N}`
  wrapper with `setfield!(:dims)` pointing to `A`'s GPU memory.
"""
@inline function AdaptiveArrayPools._reshape_impl!(
        pool::CuAdaptiveArrayPool, A::CuArray{T, M}, dims::NTuple{N, Int}
    ) where {T, M, N}
    for d in dims
        d < 0 && throw(ArgumentError("invalid CuArray dimensions"))
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
    slot = _cuda_claim_slot!(tp)

    # Look up cached wrapper (direct index, no hash)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && slot <= length(wrappers)
        wrapper = @inbounds wrappers[slot]
        if wrapper !== nothing
            cu = wrapper::CuArray{T, N}
            if cu.data.rc !== A.data.rc
                _update_cuda_wrapper_data!(cu, A)
            end
            setfield!(cu, :dims, dims)
            return cu
        end
    end

    # Cache miss (first call per slot+N): create wrapper, cache forever
    cu = CuArray{T, N}(copy(A.data), dims; maxsize=A.maxsize, offset=A.offset)
    _store_arr_wrapper!(tp, N, slot, cu)
    return cu
end

# ==============================================================================
# CUDA _record_type_touch! override (Issue #2 / #2a fix)
# ==============================================================================
# Float16 on CUDA: direct struct field with _fixed_slot_bit(Float16)=0.
# We track Float16 via bit 7 (CUDA reassignment; CPU uses bit 7 for Bit type, absent on GPU).
# This gives Float16 lazy first-touch checkpointing in bit-14 (typed lazy) and bit-15 (dynamic)
# modes, ensuring Case A (not Case B) fires at rewind and parent n_active is preserved.

@inline function AdaptiveArrayPools._record_type_touch!(pool::CuAdaptiveArrayPool, ::Type{T}) where {T}
    depth = pool._current_depth
    b = _fixed_slot_bit(T)
    if b == UInt16(0)
        if T === Float16
            # Float16: CUDA direct field tracked via bit 7 (not in pool.others dict).
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
