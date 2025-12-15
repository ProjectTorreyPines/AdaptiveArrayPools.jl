# ==============================================================================
# CUDA-Specific get_view! Implementation
# ==============================================================================
# Unlike CPU, GPU views (view(CuVector, 1:n)) return CuVector via GPUArrays derive(),
# NOT SubArray. This means:
# 1. We cannot cache view objects separately (they're just CuVectors)
# 2. View creation is O(1) metadata operation, no GPU allocation
# 3. No benefit from caching - just return fresh view each time

using AdaptiveArrayPools: get_view!, get_nd_array!, allocate_vector, safe_prod, wrap_array, CACHE_WAYS

"""
    get_view!(tp::CuTypedPool{T}, n::Int) -> CuVector{T}

Get a 1D GPU vector view of size `n` from the typed pool.
Returns a fresh view each call (no caching - view creation is O(1) metadata).

## GPU-Specific Behavior
Unlike CPU where views are SubArrays and benefit from caching, GPU views
use GPUArrays' `derive()` mechanism which returns a new CuVector sharing
the same memory buffer. View creation is essentially free (just pointer math).
"""
function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, n::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active

    # 1. Expand pool if needed (new slot)
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, n))
        push!(tp.view_lengths, n)

        # Warn at powers of 2 (512, 1024, 2048, ...) - possible missing rewind!()
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bytes = sum(length, tp.vectors) * sizeof(T)
            @warn "CuTypedPool{$T} growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
        end

        # Return fresh view (no caching - view creates CuVector metadata)
        return view(tp.vectors[idx], 1:n)
    end

    # 2. Check if resize needed
    @inbounds cached_len = tp.view_lengths[idx]
    @inbounds vec = tp.vectors[idx]

    if length(vec) < n
        # WARNING: resize! on CuVector copies old data (wasteful for pools)
        # TODO v1.1: Consider CUDA.unsafe_free! + fresh alloc instead
        resize!(vec, n)
    end

    @inbounds tp.view_lengths[idx] = n

    # Always create fresh view (O(1) metadata, no GPU allocation)
    return view(vec, 1:n)
end

# ==============================================================================
# CUDA-Specific get_nd_array! Implementation
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
