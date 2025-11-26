# ==============================================================================
# Get View (Internal)
# ==============================================================================

"""
    get_view!(tp::TypedPool{T}, n::Int) -> SubArray

Internal function to get a vector view of size `n` from the typed pool.

## v3: View Caching with SoA
- Cache hit (same size): Returns cached SubArray (zero allocation)
- Cache miss: Creates new view, updates cache
- Uses `view_lengths` for fast Int comparison (no pointer dereference)
"""
function get_view!(tp::TypedPool{T}, n::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active

    # 1. Need to expand pool (new slot)
    if idx > length(tp.vectors)
        push!(tp.vectors, Vector{T}(undef, n))
        new_view = view(tp.vectors[idx], 1:n)
        push!(tp.views, new_view)
        push!(tp.view_lengths, n)
        return new_view
    end

    # 2. Cache hit: same size requested -> return cached view (ZERO ALLOC)
    @inbounds cached_len = tp.view_lengths[idx]
    if cached_len == n
        return @inbounds tp.views[idx]
    end

    # 3. Cache miss: different size -> update cache
    @inbounds vec = tp.vectors[idx]
    if length(vec) < n
        resize!(vec, n)
    end

    new_view = view(vec, 1:n)
    @inbounds tp.views[idx] = new_view
    @inbounds tp.view_lengths[idx] = n

    return new_view
end

# ==============================================================================
# Acquisition API
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> SubArray
    acquire!(pool, Type{T}, dims...) -> ReshapedArray

Acquire a view of an array of type `T` with size `n` or dimensions `dims`.

Returns a `SubArray` (1D) or `ReshapedArray` (multi-dimensional) backed by the pool.
After the enclosing `@use_pool` block ends, the memory is reclaimed for reuse.

## Example
```julia
@use_global_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 1.0
    sum(v)
end
```
"""
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_view!(tp, n)
end

# Multi-dimensional support (Flat Buffer + Reshape)
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    total_len = prod(dims)
    flat_view = acquire!(pool, T, total_len)
    return reshape(flat_view, dims)
end

# Fallback: When pool is `nothing` (e.g. pooling disabled), allocate normally
@inline function acquire!(::Nothing, ::Type{T}, n::Int) where {T}
    Vector{T}(undef, n)
end

@inline function acquire!(::Nothing, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    Array{T, N}(undef, dims)
end

# ==============================================================================
# State Management (v2: Zero-Allocation checkpoint!/rewind!)
# ==============================================================================

"""
    checkpoint!(pool::AdaptiveArrayPool)

Save the current pool state (n_active counters) to internal stacks.

This is called automatically by `@use_pool` and related macros.
After warmup, this function has **zero allocation**.

See also: [`rewind!`](@ref), [`@use_pool`](@ref)
"""
function checkpoint!(pool::AdaptiveArrayPool)
    # Fixed slots - direct field access, no Dict lookup
    push!(pool.float64.saved_stack, pool.float64.n_active)
    push!(pool.float32.saved_stack, pool.float32.n_active)
    push!(pool.int64.saved_stack, pool.int64.n_active)
    push!(pool.int32.saved_stack, pool.int32.n_active)
    push!(pool.complexf64.saved_stack, pool.complexf64.n_active)
    push!(pool.bool.saved_stack, pool.bool.n_active)

    # Others - iterate without allocation (values() returns iterator)
    for p in values(pool.others)
        push!(p.saved_stack, p.n_active)
    end

    return nothing
end

checkpoint!(::Nothing) = nothing

"""
    rewind!(pool::AdaptiveArrayPool)

Restore the pool state (n_active counters) from internal stacks.

Only the counters are restored; allocated memory remains for reuse.
Handles edge case: types added after checkpoint! get their n_active set to 0.

See also: [`checkpoint!`](@ref), [`@use_pool`](@ref)
"""
function rewind!(pool::AdaptiveArrayPool)
    # Fixed slots
    pool.float64.n_active = pop!(pool.float64.saved_stack)
    pool.float32.n_active = pop!(pool.float32.saved_stack)
    pool.int64.n_active = pop!(pool.int64.saved_stack)
    pool.int32.n_active = pop!(pool.int32.saved_stack)
    pool.complexf64.n_active = pop!(pool.complexf64.saved_stack)
    pool.bool.n_active = pop!(pool.bool.saved_stack)

    # Others - handle edge case: new types added after checkpoint!
    for p in values(pool.others)
        if isempty(p.saved_stack)
            # Type was added after checkpoint! - reset to 0
            p.n_active = 0
        else
            p.n_active = pop!(p.saved_stack)
        end
    end

    return nothing
end

rewind!(::Nothing) = nothing

# ==============================================================================
# Pool Clearing
# ==============================================================================

"""
    empty!(tp::TypedPool)

Clear all internal storage of a TypedPool, releasing all memory.
"""
function Base.empty!(tp::TypedPool)
    empty!(tp.vectors)
    empty!(tp.views)
    empty!(tp.view_lengths)
    tp.n_active = 0
    empty!(tp.saved_stack)
    return tp
end

"""
    empty!(pool::AdaptiveArrayPool)

Completely clear the pool, releasing all stored vectors and resetting all state.

This is useful when you want to free memory or start fresh without creating
a new pool instance.

## Example
```julia
pool = AdaptiveArrayPool()
v = acquire!(pool, Float64, 1000)
# ... use v ...
empty!(pool)  # Release all memory
```

## Warning
Any SubArrays previously acquired from this pool become invalid after `empty!`.
"""
function Base.empty!(pool::AdaptiveArrayPool)
    # Fixed slots
    empty!(pool.float64)
    empty!(pool.float32)
    empty!(pool.int64)
    empty!(pool.int32)
    empty!(pool.complexf64)
    empty!(pool.bool)

    # Others - clear all TypedPools then the IdDict itself
    for tp in values(pool.others)
        empty!(tp)
    end
    empty!(pool.others)

    return pool
end

Base.empty!(::Nothing) = nothing
