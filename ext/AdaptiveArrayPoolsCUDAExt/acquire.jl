# ==============================================================================
# CUDA-Specific Unified get_view! Implementation
# ==============================================================================
# Unlike CPU, GPU views (view(CuVector, 1:n)) return CuVector via GPUArrays derive(),
# NOT SubArray. Similarly, reshape() returns CuArray, not ReshapedArray.
# This allows a single unified implementation for all dimensions.

using AdaptiveArrayPools: get_view!, get_nd_view!, get_nd_array!, allocate_vector, safe_prod, wrap_array, CACHE_WAYS

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

Get an N-dimensional view from the pool with unified 1-way caching.
Returns cached view on hit (zero CPU allocation), creates new on miss.

## GPU-Specific Behavior
- GPU `view()` returns `CuVector` (not SubArray)
- GPU `reshape()` returns `CuArray{T,N}` (not ReshapedArray)
- Both allocate ~80-96 bytes on CPU heap for the wrapper object
- Caching eliminates this allocation on cache hit
"""
@inline function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    tp.n_active += 1
    idx = tp.n_active
    total_len = safe_prod(dims)

    # 1. Expand pool if needed (new slot)
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, total_len))
        new_view = view(tp.vectors[idx], 1:total_len)
        nd_view = N == 1 ? new_view : reshape(new_view, dims)
        push!(tp.views, nd_view)
        push!(tp.view_dims, dims)

        # Warn at powers of 2 (512, 1024, 2048, ...) - possible missing rewind!()
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bytes = sum(length, tp.vectors) * sizeof(T)
            @warn "CuTypedPool{$T} growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
        end

        return nd_view
    end

    # 2. Cache hit: same dims requested -> return cached view (ZERO CPU ALLOC)
    @inbounds cached_dims = tp.view_dims[idx]
    if cached_dims isa NTuple{N, Int} && cached_dims == dims
        return @inbounds tp.views[idx]::CuArray{T, N}
    end

    # 3. Cache miss: different dims -> update cache
    @inbounds vec = tp.vectors[idx]
    if length(vec) < total_len
        resize!(vec, total_len)
    end

    new_view = view(vec, 1:total_len)
    nd_view = N == 1 ? new_view : reshape(new_view, dims)
    @inbounds tp.views[idx] = nd_view
    @inbounds tp.view_dims[idx] = dims

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
# CUDA-Specific get_nd_array! Implementation (N-way cache)
# ==============================================================================
# Full override needed for type-stability: cache hit returns CuArray{T,N},
# not Array{T,N}. This mirrors the get_view! override pattern.

"""
    get_nd_array!(tp::CuTypedPool{T}, dims::NTuple{N,Int}) -> CuArray{T,N}

Get an N-dimensional `CuArray` from the pool with N-way caching.
"""
@inline function AdaptiveArrayPools.get_nd_array!(tp::CuTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len) # Increments n_active
    slot = tp.n_active

    @inbounds vec = tp.vectors[slot]
    current_ptr = UInt(pointer(vec))

    # Expand cache slots if needed (CACHE_WAYS entries per slot)
    n_slots_cached = length(tp.nd_next_way)
    while slot > n_slots_cached
        for _ in 1:CACHE_WAYS
            push!(tp.nd_arrays, nothing)
            push!(tp.nd_dims, nothing)
            push!(tp.nd_ptrs, UInt(0))
        end
        push!(tp.nd_next_way, 0)
        n_slots_cached += 1
    end

    base = (slot - 1) * CACHE_WAYS

    # Linear Search across all ways (Cache hit = 0 bytes)
    for k in 1:CACHE_WAYS
        cache_idx = base + k
        @inbounds cached_dims = tp.nd_dims[cache_idx]
        @inbounds cached_ptr = tp.nd_ptrs[cache_idx]

        if cached_dims isa NTuple{N, Int} && cached_dims == dims && cached_ptr == current_ptr
            return @inbounds tp.nd_arrays[cache_idx]::CuArray{T,N}
        end
    end

    # Cache Miss - Round-Robin Replacement
    @inbounds way_offset = tp.nd_next_way[slot]
    target_idx = base + way_offset + 1

    arr = wrap_array(tp, flat_view, dims)

    @inbounds tp.nd_arrays[target_idx] = arr
    @inbounds tp.nd_dims[target_idx] = dims
    @inbounds tp.nd_ptrs[target_idx] = current_ptr

    # Update round-robin counter
    @inbounds tp.nd_next_way[slot] = (way_offset + 1) % CACHE_WAYS

    return arr
end
