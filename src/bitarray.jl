# ==============================================================================
# BitArray Acquisition — Julia 1.11+ (setfield!-based Wrapper Reuse)
# ==============================================================================
#
# This file contains BitArray-specific pool operations for Julia 1.11+.
# Uses setfield!-based wrapper reuse for unlimited dim patterns, 0-alloc.
#
# Key components:
# - Base.zero/one(::Type{Bit}) - Fill value dispatch for Bit sentinel type
# - get_bitarray! - N-D BitArray with setfield!-based caching
# - _acquire_impl! for Bit - Delegates to _unsafe_acquire_impl! for performance
# - _unsafe_acquire_impl! for Bit - Raw BitArray acquisition with caching
# - DisabledPool fallbacks for Bit type
# - empty!(::BitTypedPool) - State management (clearing pool storage)
# - _check_bitchunks_overlap - Safety validation for POOL_DEBUG mode
# - Display helpers: _default_type_name, _vector_bytes, _count_label, _show_type_name
#
# Design Decision: Unified BitArray Return Type
# =============================================
# Unlike regular types where acquire! returns SubArray and unsafe_acquire!
# returns Array, for Bit type BOTH return BitArray{N}. This ensures users
# always get SIMD-optimized performance (~10-100x faster count/sum/bitwise).
# ==============================================================================

# ==============================================================================
# Fill Value Dispatch (BitArray-specific)
# ==============================================================================

# Bit type returns Bool element type for fill operations (zero/one)
@inline Base.zero(::Type{Bit}) = false
@inline Base.one(::Type{Bit}) = true

# ==============================================================================
# BitArray Acquisition (N-D caching with chunks sharing)
# ==============================================================================

"""
    get_bitarray!(tp::BitTypedPool, dims::NTuple{N,Int}) -> BitArray{N}

Get a BitArray{N} that shares `chunks` with the pooled BitVector.

Uses `setfield!`-based wrapper reuse — unlimited dim patterns, 0-alloc after warmup.

## Implementation Notes
- BitVector (N=1): `size()` uses `len` field, `dims` is ignored
- BitArray{N>1}: `size()` uses `dims` field
- All BitArrays share `chunks` with the pool's backing BitVector

## Safety
The returned BitArray is only valid within the `@with_pool` scope.
Do NOT use after the scope ends (use-after-free risk).
"""
function get_bitarray!(tp::BitTypedPool, dims::NTuple{N, Int}) where {N}
    total_len = safe_prod(dims)
    tp.n_active += 1
    idx = tp.n_active

    # 1. Pool expansion needed (new slot)
    if idx > length(tp.vectors)
        pool_bv = BitVector(undef, total_len)
        push!(tp.vectors, pool_bv)

        # Create BitArray sharing chunks
        ba = BitArray{N}(undef, dims)
        ba.chunks = pool_bv.chunks

        # Cache the wrapper
        _store_arr_wrapper!(tp, N, idx, ba)

        # Warn at powers of 2 (possible missing rewind!)
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bytes = sum(_vector_bytes, tp.vectors)
            @warn "BitTypedPool growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
        end

        return ba
    end

    # 2. Ensure pool_bv has correct size
    @inbounds pool_bv = tp.vectors[idx]
    if length(pool_bv) != total_len
        resize!(pool_bv, total_len)
    end

    # 3. Check wrapper cache (direct index, no hash)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && idx <= length(wrappers)
        wrapper = @inbounds wrappers[idx]
        if wrapper !== nothing
            ba = wrapper::BitArray{N}
            # Update fields in-place (all 0-alloc via setfield!)
            setfield!(ba, :len, total_len)
            setfield!(ba, :dims, dims)
            setfield!(ba, :chunks, pool_bv.chunks)
            return ba
        end
    end

    # 4. Cache miss: first call for this (slot, N)
    ba = BitArray{N}(undef, dims)
    ba.chunks = pool_bv.chunks
    _store_arr_wrapper!(tp, N, idx, ba)

    return ba
end

# Convenience: 1D case wraps to tuple
@inline get_bitarray!(tp::BitTypedPool, n::Int) = get_bitarray!(tp, (n,))

# ==============================================================================
# Acquire Implementation (Bit type → delegates to unsafe_acquire for performance)
# ==============================================================================
#
# Unlike other types where acquire! returns SubArray (view-based) and
# unsafe_acquire! returns Array (raw), Bit type always returns BitArray{N}.
# This is because BitArray's SIMD-optimized operations (count, sum, etc.)
# are ~(10x ~ 100x) faster than SubArray equivalents.
#
# The delegation is transparent: users calling acquire!(pool, Bit, dims...) get
# BitArray{N} without needing to know about unsafe_acquire!.

# Bit type: delegates to _unsafe_acquire_impl! for SIMD performance
@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, n::Int)
    return _unsafe_acquire_impl!(pool, Bit, n)
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N}
    return _unsafe_acquire_impl!(pool, Bit, dims...)
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N}
    return _unsafe_acquire_impl!(pool, Bit, dims...)
end

# ==============================================================================
# Unsafe Acquire Implementation (Bit type)
# ==============================================================================

# Bit type: returns BitArray{N} with shared chunks (SIMD optimized, N-D cached)
@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, n::Int)
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    result = get_bitarray!(tp, n)
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N}
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    result = get_bitarray!(tp, dims)
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N}
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    result = get_bitarray!(tp, dims)
    _maybe_record_borrow!(pool, tp)
    return result
end

# ==============================================================================
# DisabledPool Fallbacks (Bit type)
# ==============================================================================

# --- acquire! for DisabledPool{:cpu} with Bit type (returns BitArray) ---
@inline acquire!(::DisabledPool{:cpu}, ::Type{Bit}, n::Int) = BitVector(undef, n)
@inline acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = BitArray{N}(undef, dims)
@inline acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = BitArray{N}(undef, dims)

# --- unsafe_acquire! for DisabledPool{:cpu} with Bit type (returns BitArray) ---
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{Bit}, n::Int) = BitVector(undef, n)
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = BitArray{N}(undef, dims)
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = BitArray{N}(undef, dims)

# ==============================================================================
# State Management — empty!
# ==============================================================================

"""
    empty!(tp::BitTypedPool)

Clear all internal storage for BitTypedPool, releasing all memory.
Restores sentinel values for 1-based sentinel pattern.
"""
function Base.empty!(tp::BitTypedPool)
    empty!(tp.vectors)
    empty!(tp.arr_wrappers)
    tp.n_active = 0
    # Restore sentinel values (1-based sentinel pattern)
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)   # Sentinel: n_active=0 at depth=0
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)     # Sentinel: depth=0 = no checkpoint
    return tp
end

# ==============================================================================
# Safety Validation (POOL_DEBUG mode)
# ==============================================================================

# Check if BitArray chunks overlap with the pool's BitTypedPool storage
function _check_bitchunks_overlap(arr::BitArray, pool::AdaptiveArrayPool, original_val=arr)
    arr_chunks = arr.chunks
    arr_ptr = UInt(pointer(arr_chunks))
    arr_len = length(arr_chunks) * sizeof(UInt64)
    arr_end = arr_ptr + arr_len

    return_site = let rs = pool._pending_return_site; isempty(rs) ? nothing : rs end

    for v in pool.bits.vectors
        v_chunks = v.chunks
        v_ptr = UInt(pointer(v_chunks))
        v_len = length(v_chunks) * sizeof(UInt64)
        v_end = v_ptr + v_len
        if !(arr_end <= v_ptr || v_end <= arr_ptr)
            callsite = _lookup_borrow_callsite(pool, v)
            _throw_pool_escape_error(original_val, Bit, callsite, return_site)
        end
    end
    return nothing
end

# ==============================================================================
# Display Helpers (pool_stats / Base.show)
# ==============================================================================

_default_type_name(::BitTypedPool) = "Bit"
_vector_bytes(v::BitVector) = sizeof(v.chunks)
_count_label(::BitTypedPool) = "bits"
_show_type_name(::BitTypedPool) = "BitTypedPool"
