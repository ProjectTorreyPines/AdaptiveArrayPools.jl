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
# Memory Resize Strategy
# ==============================================================================
# Current: RESIZE TO FIT - backing vectors grow or shrink to match requested size.
# Same behavior as CPU version.
#
# GPU vs CPU difference (verified experimentally):
#   - CPU Vector: resize!(v, smaller) preserves capacity (pointer unchanged)
#   - GPU CuVector: resize!(v, smaller) may reallocate (CUDA.jl uses 25% threshold)
#     However, CUDA memory pool often returns the same block on regrow.
#
# TODO: Potential future optimizations:
#   - CUDA.jl's resize! already uses 25% threshold internally (no realloc if within capacity)
#   - Could use even smaller threshold (e.g., 12.5%) to be more aggressive about shrinking
#   - Could track recent N sizes to make smarter decisions (avoid shrink if sizes fluctuate)
# ==============================================================================

using AdaptiveArrayPools: get_view!, get_nd_view!, get_nd_array!, allocate_vector, safe_prod

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
Backing vectors are resized to match requested size (grow or shrink).
See module header for "lazy shrink" optimization notes.
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
        # Resize vector to match requested size (grow or shrink)
        # Note: CUDA.jl's resize! internally uses 25% threshold - won't reallocate
        #       unless new size exceeds capacity or is <25% of capacity.
        resize!(vec, total_len)
        # CRITICAL: resize! may reallocate the GPU buffer (pointer change).
        # All cached views for this slot now reference the OLD buffer.
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
# CUDA-Specific get_nd_view! - Delegates to unified get_view!
# ==============================================================================

"""
    get_nd_view!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Delegates to `get_view!(tp, dims)` for unified caching.
This override exists for API compatibility with the base package.
"""
@inline function AdaptiveArrayPools.get_nd_view!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    return get_view!(tp, dims)
end

# ==============================================================================
# CUDA-Specific get_nd_array! - Delegates to unified get_view!
# ==============================================================================

"""
    get_nd_array!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Delegates to `get_view!(tp, dims)` for unified caching.
Used by `unsafe_acquire!` - same zero-allocation behavior as `acquire!`.
"""
@inline function AdaptiveArrayPools.get_nd_array!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    return get_view!(tp, dims)
end
