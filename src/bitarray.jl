# ==============================================================================
# BitArray Acquisition (Unified BitVector API)
# ==============================================================================
#
# This file contains BitArray-specific pool operations, separated from the
# generic Array acquisition code in acquire.jl for maintainability.
#
# Key components:
# - Base.zero/one(::Type{Bit}) - Fill value dispatch for Bit sentinel type
# - get_bitvector_wrapper! - SIMD-optimized BitVector with shared chunks
# - _acquire_impl! for Bit - Delegates to _unsafe_acquire_impl! for performance
# - _unsafe_acquire_impl! for Bit - Raw BitVector/BitArray acquisition
# - DisabledPool fallbacks for Bit type
#
# Design Decision: Unified BitVector Return Type
# =============================================
# Unlike regular types where acquire! returns SubArray and unsafe_acquire!
# returns Array, for Bit type BOTH return BitVector. This design choice is
# intentional for several reasons:
#
# 1. **SIMD Performance**: BitVector operations like `count()`, `sum()`, and
#    bitwise operations are ~(10x ~ 100x) faster than their SubArray equivalents
#    because they use SIMD-optimized chunked algorithms.
#
# 2. **API Simplicity**: Users always get BitVector regardless of which API
#    they call. No need to remember "use unsafe_acquire! for performance".
#
# 3. **Semantic Clarity**: The "unsafe" in unsafe_acquire! refers to memory
#    safety concerns (use-after-free risk). BitVector already handles memory
#    efficiently (1 bit per element), so the naming would be misleading.
#
# 4. **Backwards Compatibility**: Code using trues!/falses! just works with
#    optimal performance - these convenience functions now return BitVector.
#
# Implementation:
# - _acquire_impl!(pool, Bit, ...) delegates to _unsafe_acquire_impl!
# - get_bitvector_wrapper! creates BitVector shells sharing pool's chunks
# - N-D requests return reshaped BitArrays (reshape preserves chunk sharing)
# ==============================================================================

# ==============================================================================
# Fill Value Dispatch (BitArray-specific)
# ==============================================================================

# Bit type returns Bool element type for fill operations (zero/one)
@inline Base.zero(::Type{Bit}) = false
@inline Base.one(::Type{Bit}) = true

# ==============================================================================
# BitVector Wrapper (chunks sharing for SIMD performance)
# ==============================================================================

"""
    get_bitvector_wrapper!(tp::BitTypedPool, n::Int) -> BitVector

Get a BitVector that shares `chunks` with the pooled BitVector.

Unlike `get_view!` which returns a `SubArray` (loses SIMD optimizations),
this returns a real `BitVector` with shared chunks, preserving native
BitVector performance (~(10x ~ 100x) faster for `count()`, `sum()`, etc.).

## Implementation
Creates a new BitVector shell and replaces its `chunks` field with the
pooled BitVector's chunks. Uses N-way cache for wrapper reuse.

## Safety
The returned BitVector is only valid within the `@with_pool` scope.
Do NOT use after the scope ends (use-after-free risk).
"""
function get_bitvector_wrapper!(tp::BitTypedPool, n::Int)
    tp.n_active += 1
    idx = tp.n_active

    # 1. Pool expansion needed (new slot)
    if idx > length(tp.vectors)
        pool_bv = BitVector(undef, n)
        push!(tp.vectors, pool_bv)
        push!(tp.views, view(pool_bv, 1:n))
        push!(tp.view_lengths, n)

        # Create wrapper sharing chunks
        wrapper = BitVector(undef, n)
        wrapper.chunks = pool_bv.chunks

        # Expand N-way cache (CACHE_WAYS entries per slot)
        for _ in 1:CACHE_WAYS
            push!(tp.nd_arrays, nothing)
            push!(tp.nd_dims, nothing)
            push!(tp.nd_ptrs, UInt(0))
        end
        push!(tp.nd_next_way, 0)

        # Cache in first way
        base = (idx - 1) * CACHE_WAYS + 1
        @inbounds tp.nd_arrays[base] = wrapper
        @inbounds tp.nd_dims[base] = n
        @inbounds tp.nd_ptrs[base] = UInt(pointer(pool_bv.chunks))

        # Warn at powers of 2 (possible missing rewind!)
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bits = sum(length, tp.vectors)
            @warn "BitTypedPool growing large ($idx arrays, ~$(total_bits ÷ 8) bytes). Missing rewind!()?"
        end

        return wrapper
    end

    # 2. Check N-way cache for hit (cache slots always exist - created with vector slot above)
    @inbounds pool_bv = tp.vectors[idx]
    current_ptr = UInt(pointer(pool_bv.chunks))
    base = (idx - 1) * CACHE_WAYS

    # Linear search across all ways
    for k in 1:CACHE_WAYS
        cache_idx = base + k
        @inbounds cached_n = tp.nd_dims[cache_idx]
        @inbounds cached_ptr = tp.nd_ptrs[cache_idx]

        if cached_n == n && cached_ptr == current_ptr
            return @inbounds tp.nd_arrays[cache_idx]::BitVector
        end
    end

    # 3. Cache miss - resize pool_bv to EXACTLY n elements and create new wrapper
    # Unlike regular arrays where we only grow, BitVector wrappers MUST have exactly
    # the right number of chunks. Otherwise fill!()/count() iterate over all chunks,
    # not just the bits within wrapper.len, causing incorrect behavior.
    if length(pool_bv) != n
        resize!(pool_bv, n)
        @inbounds tp.views[idx] = view(pool_bv, 1:n)
        @inbounds tp.view_lengths[idx] = n
    end

    wrapper = BitVector(undef, n)
    wrapper.chunks = pool_bv.chunks

    # Round-robin replacement
    @inbounds way_offset = tp.nd_next_way[idx]
    target_idx = base + way_offset + 1
    @inbounds tp.nd_arrays[target_idx] = wrapper
    @inbounds tp.nd_dims[target_idx] = n
    @inbounds tp.nd_ptrs[target_idx] = UInt(pointer(pool_bv.chunks))
    @inbounds tp.nd_next_way[idx] = (way_offset + 1) % CACHE_WAYS

    return wrapper
end

# ==============================================================================
# Acquire Implementation (Bit type → delegates to unsafe_acquire for performance)
# ==============================================================================
#
# Unlike other types where acquire! returns SubArray (view-based) and
# unsafe_acquire! returns Array (raw), Bit type always returns BitVector.
# This is because BitVector's SIMD-optimized operations (count, sum, etc.)
# are ~(10x ~ 100x) faster than SubArray equivalents.
#
# The delegation is transparent: users calling acquire!(pool, Bit, n) get
# BitVector without needing to know about unsafe_acquire!.

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

# Bit type: returns BitVector with shared chunks (SIMD optimized)
@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, n::Int)
    tp = get_typed_pool!(pool, Bit)::BitTypedPool
    return get_bitvector_wrapper!(tp, n)
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int,N}) where {N}
    total = safe_prod(dims)
    bv = _unsafe_acquire_impl!(pool, Bit, total)
    return reshape(bv, dims)  # BitArray{N} (Julia's reshape on BitVector returns BitArray)
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N,Int}) where {N}
    return _unsafe_acquire_impl!(pool, Bit, dims...)
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
