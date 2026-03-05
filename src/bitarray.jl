# ==============================================================================
# BitArray Acquisition (N-D Cached BitArray API)
# ==============================================================================
#
# This file contains BitArray-specific pool operations, separated from the
# generic Array acquisition code in acquire.jl for maintainability.
#
# Key components:
# - Base.zero/one(::Type{Bit}) - Fill value dispatch for Bit sentinel type
# - get_bitarray! - N-D BitArray with shared chunks and N-way caching
# - _acquire_impl! for Bit - Delegates to _unsafe_acquire_impl! for performance
# - _unsafe_acquire_impl! for Bit - Raw BitArray acquisition with caching
# - DisabledPool fallbacks for Bit type
#
# Design Decision: Unified BitArray Return Type
# =============================================
# Unlike regular types where acquire! returns SubArray and unsafe_acquire!
# returns Array, for Bit type BOTH return BitArray{N}. This design choice is
# intentional for several reasons:
#
# 1. **SIMD Performance**: BitArray operations like `count()`, `sum()`, and
#    bitwise operations are ~(10x ~ 100x) faster than their SubArray equivalents
#    because they use SIMD-optimized chunked algorithms.
#
# 2. **API Simplicity**: Users always get BitArray regardless of which API
#    they call. No need to remember "use unsafe_acquire! for performance".
#
# 3. **N-D Caching**: BitArray{N} can be reused by modifying dims/len fields
#    when ndims matches, achieving 0 allocation on repeated calls. This is
#    unique to BitArray - regular Array cannot modify dims in place.
#
# 4. **Backwards Compatibility**: Code using trues!/falses! just works with
#    optimal performance - these convenience functions return BitVector.
#
# Implementation:
# - _acquire_impl!(pool, Bit, ...) delegates to _unsafe_acquire_impl!
# - get_bitarray! creates BitArray shells sharing pool's chunks
# - N-way cache stores BitArray{N} entries, reused via dims modification
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

Uses `setfield!`-based wrapper reuse (Julia 1.11+). Cached `BitArray{N}` wrappers
are mutated in-place via `setfield!(:dims)`, `setfield!(:len)`, and `setfield!(:chunks)`,
achieving zero allocation for any dimension pattern after warmup.

## Cache Strategy
- **Wrapper exists**: Mutate dims/len/chunks fields in-place (0 bytes)
- **First call per (slot, N)**: Create new BitArray{N}, cache it (~944 bytes)

## Implementation Notes
- BitVector (N=1): `size()` uses `len` field, `dims` is ignored
- BitArray{N>1}: `size()` uses `dims` field
- All BitArrays share `chunks` with the pool's backing BitVector

## Safety
The returned BitArray is only valid within the `@with_pool` scope.
Do NOT use after the scope ends (use-after-free risk).
"""
function get_bitarray!(tp::BitTypedPool, dims::NTuple{N,Int}) where {N}
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
        _store_nd_wrapper!(tp, N, idx, ba)

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

    # 3. Check wrapper cache
    wrappers = get(tp.nd_wrappers, N, nothing)
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
    _store_nd_wrapper!(tp, N, idx, ba)

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

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int,N}) where {N}
    return _unsafe_acquire_impl!(pool, Bit, dims...)
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N,Int}) where {N}
    return _unsafe_acquire_impl!(pool, Bit, dims...)
end

# ==============================================================================
# Unsafe Acquire Implementation (Bit type)
# ==============================================================================

# Bit type: returns BitArray{N} with shared chunks (SIMD optimized, N-D cached)
@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, n::Int)
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    return get_bitarray!(tp, n)
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int,N}) where {N}
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    return get_bitarray!(tp, dims)
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N,Int}) where {N}
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    return get_bitarray!(tp, dims)
end

# ==============================================================================
# DisabledPool Fallbacks (Bit type)
# ==============================================================================

# --- acquire! for DisabledPool{:cpu} with Bit type (returns BitArray) ---
@inline acquire!(::DisabledPool{:cpu}, ::Type{Bit}, n::Int) = BitVector(undef, n)
@inline acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::Vararg{Int,N}) where {N} = BitArray{N}(undef, dims)
@inline acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::NTuple{N,Int}) where {N} = BitArray{N}(undef, dims)

# --- unsafe_acquire! for DisabledPool{:cpu} with Bit type (returns BitArray) ---
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{Bit}, n::Int) = BitVector(undef, n)
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::Vararg{Int,N}) where {N} = BitArray{N}(undef, dims)
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{Bit}, dims::NTuple{N,Int}) where {N} = BitArray{N}(undef, dims)
