# ==============================================================================
# CUDA-Specific Unified get_view! Implementation (N-Way Cache)
# ==============================================================================
# Unlike CPU, GPU views (view(CuVector, 1:n)) return CuVector via GPUArrays derive(),
# NOT SubArray. Similarly, reshape() returns CuArray, not ReshapedArray.
# This allows a single unified implementation for all dimensions.
#
# N-way cache layout (flat vector):
#   views[(slot-1)*CACHE_WAYS + way] for way ∈ 1:CACHE_WAYS
#
# Cache lookup uses simple for loop - measured overhead ~16 bytes (acceptable).
#
# ==============================================================================
# Memory Resize Strategy: _resize_without_shrink!
# ==============================================================================
# GPU vs CPU difference:
#   - CPU Vector: resize!(v, smaller) preserves capacity (pointer unchanged, cheap)
#   - GPU CuVector: resize!(v, smaller) may reallocate when n < capacity÷4
#     (CUDA.jl's 25% threshold triggers pool_alloc + copy + pool_free)
#
# Problem: Pool operations frequently shrink backing vectors:
#   - Safety invalidation: resize!(vec, 0) to invalidate released slots
#   - Acquire path: resize!(vec, smaller_size) when reusing a slot for smaller array
#   Both trigger expensive GPU reallocation via CUDA.jl's 25% threshold.
#
# Solution: _resize_without_shrink!(A, n)
#   - Grow (n > length): delegates to CUDA.jl resize! (may allocate more GPU memory)
#   - Shrink (n < length): setfield!(A, :dims, (n,)) — logical size only, no GPU op
#   - Equal (n == length): no-op
#
# Key property: maxsize is preserved on shrink. When later growing back,
# CUDA.jl computes cap = maxsize ÷ aligned_sizeof(T) and sees n ≤ cap,
# so no reallocation occurs. This is ideal for pool's borrow/return pattern.
#
# ⚠ Depends on CuArray internal fields (:dims, .maxsize). Tested with CUDA.jl v5.x.
# ==============================================================================

using AdaptiveArrayPools: get_view!, get_array!, allocate_vector, safe_prod,
    _record_type_touch!, _fixed_slot_bit, _checkpoint_typed_pool!,
    _MODE_BITS_MASK

# Guard against CUDA.jl internal API changes (tested with v5.x).
# setfield!(:dims) requires CuArray to be mutable and have a :dims field.
@static if !(ismutable(CuArray) && hasfield(CuArray, :dims))
    error("Unsupported CUDA.jl version: expected mutable CuArray with field :dims. _resize_without_shrink! needs updating.")
end

"""
    _resize_without_shrink!(A::CuVector{T}, n::Integer) -> CuVector{T}

Resize a CuVector's logical length without freeing GPU memory on shrink.

- `n > length(A)`: delegates to `resize!(A, n)` (may grow GPU allocation)
- `n == length(A)`: no-op
- `n < length(A)`: only updates `dims` field (GPU memory preserved at `maxsize`)

Avoids CUDA.jl's 25% threshold reallocation on shrink (`n < cap÷4` triggers
`pool_alloc` + `unsafe_copyto!` + `pool_free`), which is expensive for pool
operations like safety invalidation (`resize!(v, 0)`) and acquire-path resizing.
"""
@inline function _resize_without_shrink!(A::CuVector{T}, n::Integer) where {T}
    current = length(A)
    if n > current
        resize!(A, n)                       # grow: delegate to CUDA.jl
    elseif n < current
        setfield!(A, :dims, (Int(n),))      # shrink: dims only, GPU memory preserved
    end
    return A
end

"""
    get_view!(tp::CuTypedPool{T}, n::Int) -> CuVector{T}

1D convenience wrapper - delegates to tuple version.
`(n,)` is stack-allocated (isbits NTuple), so this is zero-allocation when inlined.
"""
@inline function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, n::Int) where {T}
    return get_view!(tp, (n,))
end

"""
    get_view!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Get an N-dimensional view from the pool with unified N-way caching.
Returns cached view on hit (near-zero CPU allocation), creates new on miss.

## N-Way Cache Behavior
- Each slot has CACHE_WAYS (4) cache entries for different dimension patterns
- Cache lookup uses simple for loop (~16 bytes overhead)
- Cache replacement uses round-robin when all ways are occupied

## GPU-Specific Behavior
- GPU `view()` returns `CuVector` (not SubArray)
- GPU `reshape()` returns `CuArray{T,N}` (not ReshapedArray)
- Both allocate ~80 bytes on CPU heap for the wrapper object
- N-way caching eliminates this allocation on cache hit

## Memory Resize Strategy
Backing vectors use `_resize_without_shrink!`: grow delegates to CUDA.jl's
`resize!` (may reallocate), shrink only updates `dims` (GPU memory preserved).
See module header for details.
"""
@inline function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    tp.n_active += 1
    idx = tp.n_active
    total_len = safe_prod(dims)

    # 1. Expand pool if needed (new slot)
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, total_len))
        @inbounds vec = tp.vectors[idx]
        new_view = view(vec, 1:total_len)
        nd_view = N == 1 ? new_view : reshape(new_view, dims)

        # Initialize N-way cache entries for this slot
        for _ in 1:CACHE_WAYS
            push!(tp.views, nothing)
            push!(tp.view_dims, nothing)
        end
        push!(tp.next_way, 1)

        # Store in first way
        base = (idx - 1) * CACHE_WAYS
        @inbounds tp.views[base + 1] = nd_view
        @inbounds tp.view_dims[base + 1] = dims

        # Warn at powers of 2 (512, 1024, 2048, ...) - possible missing rewind!()
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bytes = sum(length, tp.vectors) * sizeof(T)
            @warn "CuTypedPool{$T} growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
        end

        return nd_view
    end

    # 2. N-way cache lookup with for loop
    base = (idx - 1) * CACHE_WAYS
    for k in 1:CACHE_WAYS
        cache_idx = base + k
        @inbounds cached_dims = tp.view_dims[cache_idx]
        if cached_dims isa NTuple{N, Int} && cached_dims == dims
            # Cache hit - return cached view
            return @inbounds tp.views[cache_idx]::CuArray{T, N}
        end
    end

    # 3. Cache miss: create new view, use round-robin replacement
    @inbounds vec = tp.vectors[idx]
    current_len = length(vec)
    if current_len != total_len
        # Resize vector to match requested size (grow or shrink).
        # Uses _resize_without_shrink! to avoid GPU reallocation on shrink.
        _resize_without_shrink!(vec, total_len)
        # CRITICAL: on grow, _resize_without_shrink! delegates to resize! which
        # may reallocate the GPU buffer (pointer change). On shrink, pointer is
        # stable but length changed. Either way, cached views are stale.
        # Must invalidate ALL ways to prevent returning stale/dangling views.
        for k in 1:CACHE_WAYS
            @inbounds tp.views[base + k] = nothing
            @inbounds tp.view_dims[base + k] = nothing
        end
        @inbounds tp.next_way[idx] = 1  # Reset round-robin
    end

    new_view = view(vec, 1:total_len)
    nd_view = N == 1 ? new_view : reshape(new_view, dims)

    # Round-robin replacement (or first way if just flushed)
    @inbounds way = tp.next_way[idx]
    cache_idx = base + way
    @inbounds tp.views[cache_idx] = nd_view
    @inbounds tp.view_dims[cache_idx] = dims
    @inbounds tp.next_way[idx] = (way % CACHE_WAYS) + 1

    return nd_view
end

# ==============================================================================
# CUDA-Specific get_array! - Delegates to unified get_view!
# ==============================================================================

"""
    get_array!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Delegates to `get_view!(tp, dims)` for unified caching.
Used by `unsafe_acquire!` - same zero-allocation behavior as `acquire!`.
"""
@inline function AdaptiveArrayPools.get_array!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    return get_view!(tp, dims)
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
