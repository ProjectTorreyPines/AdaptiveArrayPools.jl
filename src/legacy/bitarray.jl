# ==============================================================================
# BitArray Acquisition — Legacy (Julia ≤1.10, N-way Set-Associative Cache)
# ==============================================================================
#
# This file contains BitArray-specific pool operations for Julia ≤1.10.
# Uses N-way set-associative cache for N-D BitArray caching.
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
# BitArray Acquisition (N-way set-associative cache, Julia ≤1.10)
# ==============================================================================

"""
    get_bitarray!(tp::BitTypedPool, dims::NTuple{N,Int}) -> BitArray{N}

Get a BitArray{N} that shares `chunks` with the pooled BitVector.

Uses N-way set-associative cache with up to CACHE_WAYS patterns per slot.

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

        # Expand N-way cache (CACHE_WAYS entries per slot)
        for _ in 1:CACHE_WAYS
            push!(tp.nd_arrays, nothing)
            push!(tp.nd_dims, nothing)
            push!(tp.nd_ptrs, UInt(0))
        end
        push!(tp.nd_next_way, 0)

        # Cache in first way
        base = (idx - 1) * CACHE_WAYS + 1
        @inbounds tp.nd_arrays[base] = ba
        @inbounds tp.nd_dims[base] = dims
        @inbounds tp.nd_ptrs[base] = UInt(pointer(pool_bv.chunks))

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
    current_ptr = UInt(pointer(pool_bv.chunks))
    base = (idx - 1) * CACHE_WAYS

    # 3. Check N-way cache for hit
    for k in 1:CACHE_WAYS
        cache_idx = base + k
        @inbounds cached_dims = tp.nd_dims[cache_idx]
        @inbounds cached_ptr = tp.nd_ptrs[cache_idx]

        # Must check isa FIRST for type stability (avoids boxing in == comparison)
        if cached_dims isa NTuple{N,Int} && cached_ptr == current_ptr
            if cached_dims == dims
                # Exact match - return cached BitArray directly (0 alloc)
                return @inbounds tp.nd_arrays[cache_idx]::BitArray{N}
            else
                # Same ndims but different dims - reuse by modifying fields (0 alloc!)
                ba = @inbounds tp.nd_arrays[cache_idx]::BitArray{N}
                ba.len = total_len
                ba.dims = dims
                ba.chunks = pool_bv.chunks
                # Update cache metadata
                @inbounds tp.nd_dims[cache_idx] = dims
                return ba
            end
        end
    end

    # 4. Cache miss - create new BitArray{N}
    ba = BitArray{N}(undef, dims)
    ba.chunks = pool_bv.chunks

    # Round-robin replacement
    @inbounds way_offset = tp.nd_next_way[idx]
    target_idx = base + way_offset + 1
    @inbounds tp.nd_arrays[target_idx] = ba
    @inbounds tp.nd_dims[target_idx] = dims
    @inbounds tp.nd_ptrs[target_idx] = current_ptr
    @inbounds tp.nd_next_way[idx] = (way_offset + 1) % CACHE_WAYS

    return ba
end

# Convenience: 1D case wraps to tuple
@inline get_bitarray!(tp::BitTypedPool, n::Int) = get_bitarray!(tp, (n,))

# ==============================================================================
# Acquire Implementation (Bit type → delegates to unsafe_acquire for performance)
# ==============================================================================

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
