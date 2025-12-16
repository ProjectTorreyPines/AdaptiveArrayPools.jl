# ==============================================================================
# Allocation Dispatch Points (for extensibility)
# ==============================================================================

# Allocate a new vector (dispatch point for extensions)
@inline allocate_vector(::AbstractTypedPool{T,Vector{T}}, n::Int) where {T} =
    Vector{T}(undef, n)

# Wrap flat view into N-D array (dispatch point for extensions)
@inline function wrap_array(::AbstractTypedPool{T,Vector{T}},
                            flat_view, dims::NTuple{N,Int}) where {T,N}
    unsafe_wrap(Array{T,N}, pointer(flat_view), dims)
end

# ==============================================================================
# Helper: Overflow-Safe Product
# ==============================================================================

"""
    safe_prod(dims::NTuple{N, Int}) -> Int

Compute the product of dimensions with overflow checking.

Throws `OverflowError` if the product exceeds `typemax(Int)`, preventing
memory corruption from integer overflow in `unsafe_wrap` operations.

## Rationale
Without overflow checking, large dimensions like `(10^10, 10^10)` would wrap
around to a small value, causing `unsafe_wrap` to create an array view that
indexes beyond allocated memory.

## Performance
Adds ~0.3-1.2 ns overhead (<1%) compared to unchecked `prod()`, which is
negligible relative to the 100-200 ns cost of the full allocation path.
"""
@inline function safe_prod(dims::NTuple{N, Int}) where {N}
    total = 1
    for d in dims
        total = Base.checked_mul(total, d)
    end
    return total
end

# ==============================================================================
# Get 1D View (Internal - Zero-Allocation Cache)
# ==============================================================================

"""
    get_view!(tp::AbstractTypedPool{T}, n::Int)

Get a 1D vector view of size `n` from the typed pool.
Returns cached view on hit (zero allocation), creates new on miss.
"""
function get_view!(tp::AbstractTypedPool{T}, n::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active

    # 1. Need to expand pool (new slot)
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, n))
        new_view = view(tp.vectors[idx], 1:n)
        push!(tp.views, new_view)
        push!(tp.view_lengths, n)

        # Warn at powers of 2 (512, 1024, 2048, ...) - possible missing rewind!()
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bytes = sum(length, tp.vectors) * sizeof(T)
            @warn "$(nameof(typeof(tp))){$T} growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
        end

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
# Get N-D Array/View (Internal - Zero-Allocation Cache)
# ==============================================================================

"""
    get_nd_array!(tp::AbstractTypedPool{T}, dims::NTuple{N,Int}) -> Array{T,N}

Get an N-dimensional `Array` from the pool with N-way caching.
"""
@inline function get_nd_array!(tp::AbstractTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
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
            return @inbounds tp.nd_arrays[cache_idx]::Array{T, N}
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

"""
    get_nd_view!(tp::AbstractTypedPool{T}, dims::NTuple{N,Int})

Get an N-dimensional view via `reshape` (zero creation cost).
"""
@inline function get_nd_view!(tp::AbstractTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)  # 1D view (cached, 0 alloc)
    return reshape(flat_view, dims)        # ReshapedArray (0 creation cost)
end

# ==============================================================================
# Untracked Acquire Detection
# ==============================================================================

"""
    _mark_untracked!(pool::AbstractArrayPool)

Mark that an untracked acquire has occurred at the current checkpoint depth.
Called by `acquire!` wrapper; macro-transformed calls use `_acquire_impl!` directly.

With 1-indexed _current_depth (starting at 1 for global scope), this always marks
the current scope's _untracked_flags.
"""
@inline function _mark_untracked!(pool::AbstractArrayPool)
    # Always mark (_current_depth >= 1 guaranteed by sentinel)
    @inbounds pool._untracked_flags[pool._current_depth] = true
end

# ==============================================================================
# Internal Implementation Functions (called by macro-transformed code)
# ==============================================================================

"""
    _acquire_impl!(pool, Type{T}, n) -> SubArray{T,1,Vector{T},...}
    _acquire_impl!(pool, Type{T}, dims...) -> ReshapedArray{T,N,...}

Internal implementation of acquire!. Called directly by macro-transformed code
(no untracked marking). User code calls `acquire!` which adds marking.
"""
@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_view!(tp, n)
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_view!(tp, dims)
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _acquire_impl!(pool, T, dims...)
end

# Similar-style
@inline _acquire_impl!(pool::AbstractArrayPool, x::AbstractArray) = _acquire_impl!(pool, eltype(x), size(x))

"""
    _unsafe_acquire_impl!(pool, Type{T}, dims...) -> Array{T,N}

Internal implementation of unsafe_acquire!. Called directly by macro-transformed code.
"""
@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, (n,))
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)
end

# Similar-style
@inline _unsafe_acquire_impl!(pool::AbstractArrayPool, x::AbstractArray) = _unsafe_acquire_impl!(pool, eltype(x), size(x))

# ==============================================================================
# Acquisition API (User-facing with untracked marking)
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> view type
    acquire!(pool, Type{T}, dims...) -> view type
    acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> view type

Acquire a view of an array of type `T` with size `n` or dimensions `dims`.

Returns a view backed by the pool (backend-dependent type):
- **CPU 1D**: `SubArray{T,1,Vector{T},...}` (parent is `Vector{T}`)
- **CPU N-D**: `ReshapedArray{T,N,...}` (zero creation cost)
- **CUDA**: `CuArray{T,N}` (unified N-way cache)

All return types are `StridedArray`, compatible with BLAS and broadcasting.

For type-unspecified paths (struct fields without concrete type parameters),
use [`unsafe_acquire!`](@ref) instead - cached native array instances can be reused.

## Example
```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)      # 1D view
    m = acquire!(pool, Float64, 10, 10)   # 2D view
    v .= 1.0
    m .= 2.0
    sum(v) + sum(m)
end
```

See also: [`unsafe_acquire!`](@ref) for native array access.
"""
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    _mark_untracked!(pool)
    _acquire_impl!(pool, T, n)
end

# Multi-dimensional support (zero-allocation with N-D cache)
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _mark_untracked!(pool)
    _acquire_impl!(pool, T, dims...)
end

# Tuple support: allows acquire!(pool, T, size(A)) where size(A) returns NTuple{N,Int}
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _mark_untracked!(pool)
    _acquire_impl!(pool, T, dims...)
end

# Fallback: When pool is `nothing` (e.g. pooling disabled), allocate normally
@inline function acquire!(::Nothing, ::Type{T}, n::Int) where {T}
    Vector{T}(undef, n)
end

@inline function acquire!(::Nothing, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    Array{T, N}(undef, dims)
end

@inline function acquire!(::Nothing, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    Array{T, N}(undef, dims)
end

# Similar-style convenience methods
"""
    acquire!(pool, x::AbstractArray) -> SubArray

Acquire an array with the same element type and size as `x` (similar to `similar(x)`).

## Example
```julia
A = rand(10, 10)
@with_pool pool begin
    B = acquire!(pool, A)  # Same type and size as A
    B .= A .* 2
end
```
"""
@inline function acquire!(pool::AbstractArrayPool, x::AbstractArray)
    _mark_untracked!(pool)
    _acquire_impl!(pool, eltype(x), size(x))
end

@inline acquire!(::Nothing, x::AbstractArray) = similar(x)

# ==============================================================================
# Unsafe Acquisition API (Raw Arrays)
# ==============================================================================

"""
    unsafe_acquire!(pool, Type{T}, n) -> backend's native array type
    unsafe_acquire!(pool, Type{T}, dims...) -> backend's native array type
    unsafe_acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> backend's native array type

Acquire a native array backed by pool memory.

Returns the backend's native array type:
- **CPU**: `Array{T,N}` (via `unsafe_wrap`)
- **CUDA**: `CuArray{T,N}` (via unified view cache)

For CPU pools, since `Array` instances are mutable references, cached instances can be
returned directly without creating new wrapper objects—ideal for type-unspecified paths.
For CUDA pools, this delegates to the same unified N-way cache as `acquire!`.

## Safety Warning
The returned array is only valid within the `@with_pool` scope. Using it after
the scope ends leads to undefined behavior (use-after-free, data corruption).

**Do NOT call `resize!`, `push!`, or `append!` on returned arrays** - this causes
undefined behavior as the memory is owned by the pool.

## When to Use
- **Type-unspecified paths**: Struct fields without concrete type parameters
  (e.g., `_pooled_chain::PooledChain` instead of `_pooled_chain::PooledChain{M}`)
- FFI calls expecting raw pointers
- APIs that strictly require native array types

## Allocation Behavior
- **CPU**: Cache hit 0 bytes, cache miss ~112 bytes (Array header via `unsafe_wrap`)
- **CUDA**: Cache hit ~0 bytes, cache miss ~80 bytes (CuArray wrapper creation)

## Example
```julia
@with_pool pool begin
    A = unsafe_acquire!(pool, Float64, 100, 100)  # Matrix{Float64} (CPU) or CuMatrix{Float64} (CUDA)
    B = unsafe_acquire!(pool, Float64, 100, 100)
    C = similar(A)  # Regular allocation for result
    mul!(C, A, B)   # BLAS uses A, B directly
end
# A and B are INVALID after this point!
```

See also: [`acquire!`](@ref) for view-based access.
"""
@inline function unsafe_acquire!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, T, n)
end

@inline function unsafe_acquire!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, T, dims...)
end

# Tuple support
@inline function unsafe_acquire!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, T, dims)
end

# Fallback: When pool is `nothing`, allocate normally
@inline function unsafe_acquire!(::Nothing, ::Type{T}, n::Int) where {T}
    Vector{T}(undef, n)
end

@inline function unsafe_acquire!(::Nothing, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    Array{T, N}(undef, dims)
end

@inline function unsafe_acquire!(::Nothing, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    Array{T, N}(undef, dims)
end

# Similar-style convenience methods
"""
    unsafe_acquire!(pool, x::AbstractArray) -> Array

Acquire a raw array with the same element type and size as `x` (similar to `similar(x)`).

## Example
```julia
A = rand(10, 10)
@with_pool pool begin
    B = unsafe_acquire!(pool, A)  # Matrix{Float64}, same size as A
    B .= A .* 2
end
```
"""
@inline function unsafe_acquire!(pool::AbstractArrayPool, x::AbstractArray)
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, eltype(x), size(x))
end

@inline unsafe_acquire!(::Nothing, x::AbstractArray) = similar(x)

# ==============================================================================
# API Aliases
# ==============================================================================

"""
    acquire_view!(pool, Type{T}, dims...)

Alias for [`acquire!`](@ref).

Explicit name emphasizing the return type is a view (`SubArray`/`ReshapedArray`),
not a raw `Array`. Use when you prefer symmetric naming with `acquire_array!`.
"""
const acquire_view! = acquire!

"""
    acquire_array!(pool, Type{T}, dims...)

Alias for [`unsafe_acquire!`](@ref).

Explicit name emphasizing the return type is a raw `Array`.
Use when you prefer symmetric naming with `acquire_view!`.
"""
const acquire_array! = unsafe_acquire!

# Internal implementation aliases (for macro transformation)
const _acquire_view_impl! = _acquire_impl!
const _acquire_array_impl! = _unsafe_acquire_impl!
