# ==============================================================================
# CUDA-Specific Unified get_view! Implementation
# ==============================================================================
# Unlike CPU, GPU views (view(CuVector, 1:n)) return CuVector via GPUArrays derive(),
# NOT SubArray. Similarly, reshape() returns CuArray, not ReshapedArray.
# This allows a single unified implementation for all dimensions.

using AdaptiveArrayPools: get_view!, get_nd_view!, allocate_vector, safe_prod

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

