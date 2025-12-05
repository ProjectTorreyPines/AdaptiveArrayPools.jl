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
    get_view!(tp::TypedPool{T}, n::Int) -> SubArray{T,1,Vector{T},...}

Internal function to get a 1D vector view of size `n` from the typed pool.

## Cache Hit Conditions
1. Same length requested (`view_lengths[idx] == n`)
2. Slot already exists (`idx <= length(vectors)`)

## Behavior
- **Cache hit**: Returns cached `SubArray` (zero allocation)
- **Cache miss**: Creates new view, updates cache
- **Pool expansion**: Allocates new vector if needed, warns at powers of 2
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

        # Warn at powers of 2 (512, 1024, 2048, ...) - possible missing rewind!()
        if idx >= 512 && (idx & (idx - 1)) == 0
            total_bytes = sum(length, tp.vectors) * sizeof(T)
            @warn "TypedPool{$T} growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
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
    get_nd_array!(tp::TypedPool{T}, dims::NTuple{N,Int}) -> Array{T,N}

Internal function to get an N-dimensional `Array` from the typed pool with caching.
Used by `unsafe_acquire!` directly and as a backing store for `get_nd_view!`.

## Cache Hit Conditions
1. Same dims tuple (`isa NTuple{N, Int} && cached_dims == dims`)
2. Same pointer (backing vector not resized)

## Type Assertion
Uses `::Array{T, N}` for type stability when retrieving from `Vector{Any}`.
"""
@inline function get_nd_array!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len) # Increments n_active
    idx = tp.n_active

    @inbounds vec = tp.vectors[idx]
    current_ptr = UInt(pointer(vec))

    # Expand cache slots if needed
    while idx > length(tp.nd_arrays)
        push!(tp.nd_views, nothing)
        push!(tp.nd_arrays, nothing)
        push!(tp.nd_dims, nothing)
        push!(tp.nd_ptrs, UInt(0))
    end

    # Cache Hit Check
    @inbounds cached_dims = tp.nd_dims[idx]
    @inbounds cached_ptr = tp.nd_ptrs[idx]

    if cached_dims isa NTuple{N, Int} && cached_dims == dims && cached_ptr == current_ptr
        return @inbounds tp.nd_arrays[idx]::Array{T, N}
    end

    # Cache Miss
    arr = unsafe_wrap(Array{T, N}, pointer(flat_view), dims)
    
    @inbounds tp.nd_arrays[idx] = arr
    @inbounds tp.nd_dims[idx] = dims
    @inbounds tp.nd_ptrs[idx] = current_ptr
    @inbounds tp.nd_views[idx] = nothing # Invalidate view cache

    return arr
end

"""
    get_nd_view!(tp::TypedPool{T}, dims::NTuple{N,Int}) -> ReshapedArray{T,N,...}

Internal function to get an N-dimensional view from the typed pool.

Returns a `ReshapedArray` wrapping a 1D view - zero creation cost (no `unsafe_wrap`).
Compiler may optimize away heap allocation via SROA/escape analysis.

## Design Decision
Uses `reshape(1D_view, dims)` instead of `SubArray{Array}` approach:
- Zero `unsafe_wrap` cost (0 bytes vs 112 bytes on cache miss)
- Works with any dimension pattern (no N-way cache limit)
- Simpler implementation

For type-unspecified paths, use `unsafe_acquire!` → `get_nd_array!` instead.
"""
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)  # 1D view (cached, 0 alloc)
    return reshape(flat_view, dims)        # ReshapedArray (0 creation cost)
end

# ==============================================================================
# Acquisition API
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> SubArray{T,1,Vector{T},...}
    acquire!(pool, Type{T}, dims...) -> ReshapedArray{T,N,...}
    acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> ReshapedArray{T,N,...}

Acquire a view of an array of type `T` with size `n` or dimensions `dims`.

Returns a view backed by the pool:
- **1D**: `SubArray{T,1,Vector{T},...}` (parent is `Vector{T}`)
- **N-D**: `ReshapedArray{T,N,...}` (zero creation cost, no `unsafe_wrap`)

Both types are `StridedArray`, compatible with BLAS and broadcasting.

For type-unspecified paths (struct fields without concrete type parameters),
use [`unsafe_acquire!`](@ref) instead - cached Array instances can be reused.

## Example
```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)      # SubArray{Float64,1,...}
    m = acquire!(pool, Float64, 10, 10)   # ReshapedArray{Float64,2,...}
    v .= 1.0
    m .= 2.0
    sum(v) + sum(m)
end
```

See also: [`unsafe_acquire!`](@ref) for raw `Array` access.
"""
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_view!(tp, n)
end

# Multi-dimensional support (zero-allocation with N-D cache)
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_view!(tp, dims)
end

# Tuple support: allows acquire!(pool, T, size(A)) where size(A) returns NTuple{N,Int}
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    acquire!(pool, T, dims...)
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
@inline function acquire!(pool::AdaptiveArrayPool, x::AbstractArray)
    acquire!(pool, eltype(x), size(x))
end

@inline acquire!(::Nothing, x::AbstractArray) = similar(x)

# ==============================================================================
# Unsafe Acquisition API (Raw Arrays)
# ==============================================================================

"""
    unsafe_acquire!(pool, Type{T}, n) -> Vector{T}
    unsafe_acquire!(pool, Type{T}, dims...) -> Array{T,N}
    unsafe_acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> Array{T,N}

Acquire a raw `Array` backed by pool memory.

Since `Array` is already heap-allocated, cached instances can be reused without
wrapper allocation overhead - ideal for type-unspecified paths.

## Safety Warning
The returned array is only valid within the `@with_pool` scope. Using it after
the scope ends leads to undefined behavior (use-after-free, data corruption).

**Do NOT call `resize!`, `push!`, or `append!` on returned arrays** - this causes
undefined behavior as the memory is owned by the pool.

## When to Use
- **Type-unspecified paths**: Struct fields without concrete type parameters
  (e.g., `_pooled_chain::PooledChain` instead of `_pooled_chain::PooledChain{M}`)
- FFI calls expecting raw pointers
- APIs that strictly require `Array` type

## Allocation Behavior
- Cache hit: 0 bytes (cached Array instance reused)
- Cache miss: 112 bytes (Array header creation via `unsafe_wrap`)

## Example
```julia
@with_pool pool begin
    A = unsafe_acquire!(pool, Float64, 100, 100)  # Matrix{Float64}
    B = unsafe_acquire!(pool, Float64, 100, 100)  # Matrix{Float64}
    C = similar(A)  # Regular allocation for result
    mul!(C, A, B)   # BLAS uses A, B directly
end
# A and B are INVALID after this point!
```

See also: [`acquire!`](@ref) for `ReshapedArray` access.
"""
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, (n,))
end

@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)
end

# Tuple support
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)
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
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, x::AbstractArray)
    unsafe_acquire!(pool, eltype(x), size(x))
end

@inline unsafe_acquire!(::Nothing, x::AbstractArray) = similar(x)

# ==============================================================================
# State Management
# ==============================================================================

"""
    checkpoint!(pool::AdaptiveArrayPool)

Save the current pool state (n_active counters) to internal stacks.

This is called automatically by `@with_pool` and related macros.
After warmup, this function has **zero allocation**.

See also: [`rewind!`](@ref), [`@with_pool`](@ref)
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

See also: [`checkpoint!`](@ref), [`@with_pool`](@ref)
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
# Type-Specific State Management (for optimized macros)
# ==============================================================================

"""
    checkpoint!(tp::TypedPool)

Internal method for saving TypedPool state.

!!! warning "Internal API"
    This is an internal implementation detail. For manual pool management,
    use the public API instead:
    ```julia
    checkpoint!(pool, Float64)  # Type-specific checkpoint
    ```

See also: [`checkpoint!(::AdaptiveArrayPool, ::Type)`](@ref), [`rewind!`](@ref)
"""
@inline function checkpoint!(tp::TypedPool)
    push!(tp.saved_stack, tp.n_active)
    nothing
end

"""
    rewind!(tp::TypedPool)

Internal method for restoring TypedPool state.

!!! warning "Internal API"
    This is an internal implementation detail. For manual pool management,
    use the public API instead:
    ```julia
    rewind!(pool, Float64)  # Type-specific rewind
    ```

See also: [`rewind!(::AdaptiveArrayPool, ::Type)`](@ref), [`checkpoint!`](@ref)
"""
@inline function rewind!(tp::TypedPool)
    tp.n_active = pop!(tp.saved_stack)
    nothing
end

"""
    checkpoint!(pool::AdaptiveArrayPool, ::Type{T})

Save state for a specific type only. Used by optimized macros that know
which types will be used at compile time.

~77% faster than full checkpoint! when only one type is used.
"""
@inline function checkpoint!(pool::AdaptiveArrayPool, ::Type{T}) where T
    checkpoint!(get_typed_pool!(pool, T))
end

"""
    rewind!(pool::AdaptiveArrayPool, ::Type{T})

Restore state for a specific type only.
"""
@inline function rewind!(pool::AdaptiveArrayPool, ::Type{T}) where T
    rewind!(get_typed_pool!(pool, T))
end

checkpoint!(::Nothing, ::Type) = nothing
rewind!(::Nothing, ::Type) = nothing

"""
    checkpoint!(pool::AdaptiveArrayPool, types::Type...)

Save state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling.
"""
@generated function checkpoint!(pool::AdaptiveArrayPool, types::Type...)
    exprs = [:(checkpoint!(pool, types[$i])) for i in 1:length(types)]
    quote
        $(exprs...)
        nothing
    end
end

"""
    rewind!(pool::AdaptiveArrayPool, types::Type...)

Restore state for multiple specific types in reverse order.
"""
@generated function rewind!(pool::AdaptiveArrayPool, types::Type...)
    # Reverse order for proper stack unwinding
    exprs = [:(rewind!(pool, types[$i])) for i in length(types):-1:1]
    quote
        $(exprs...)
        nothing
    end
end

checkpoint!(::Nothing, types::Type...) = nothing
rewind!(::Nothing, types::Type...) = nothing

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
    # Clear N-D caches
    empty!(tp.nd_views)
    empty!(tp.nd_arrays)
    empty!(tp.nd_dims)
    empty!(tp.nd_ptrs)
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
