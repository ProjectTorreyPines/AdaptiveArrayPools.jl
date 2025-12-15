# ==============================================================================
# CUDA-Specific get_view! Implementation
# ==============================================================================
# Unlike CPU, GPU views (view(CuVector, 1:n)) return CuVector via GPUArrays derive(),
# NOT SubArray. This means:
# 1. We cannot cache view objects separately (they're just CuVectors)
# 2. View creation is O(1) metadata operation, no GPU allocation
# 3. No benefit from caching - just return fresh view each time

using AdaptiveArrayPools: get_view!, allocate_vector

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
