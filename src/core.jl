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
# Acquisition API
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> SubArray{T,1,Vector{T},...}
    acquire!(pool, Type{T}, dims...) -> SubArray{T,N,Array{T,N},...}
    acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> SubArray{T,N,Array{T,N},...}

Acquire a view of an array of type `T` with size `n` or dimensions `dims`.

Returns a `SubArray` backed by the pool. For 1D requests, the parent is `Vector{T}`.
For N-D requests (N >= 2), the parent is `Array{T,N}` created via `unsafe_wrap`.

After the enclosing `@with_pool` block ends, the memory is reclaimed for reuse.

## Example
```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)      # SubArray{Float64,1,Vector{Float64},...}
    m = acquire!(pool, Float64, 10, 10)   # SubArray{Float64,2,Matrix{Float64},...}
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

# Multi-dimensional support (unsafe_wrap + view for concrete Array type)
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    total_len = prod(dims)
    flat_view = acquire!(pool, T, total_len)
    # Create concrete Array{T,N} backed by same memory
    arr = unsafe_wrap(Array{T, N}, pointer(flat_view), dims)
    # Return as SubArray for API consistency (prevents resize!)
    return view(arr, ntuple(_ -> Colon(), Val(N))...)
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

# ==============================================================================
# Unsafe Acquisition API (Raw Arrays)
# ==============================================================================

"""
    unsafe_acquire!(pool, Type{T}, n) -> Vector{T}
    unsafe_acquire!(pool, Type{T}, dims...) -> Array{T,N}
    unsafe_acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> Array{T,N}

Acquire a raw `Array` backed by pool memory.

## Safety Warning
The returned array is only valid within the `@with_pool` scope. Using it after
the scope ends leads to undefined behavior (use-after-free, data corruption).

**Do NOT call `resize!`, `push!`, or `append!` on returned arrays** - this causes
undefined behavior as the memory is owned by the pool.

## Use Cases
- BLAS operations requiring contiguous `Array` (not `SubArray`)
- FFI calls expecting raw pointers
- Performance-critical code where `SubArray` overhead matters

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

See also: [`acquire!`](@ref) for safe `SubArray` access.
"""
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    flat_view = get_view!(tp, n)
    return unsafe_wrap(Vector{T}, pointer(flat_view), n)
end

@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    total_len = prod(dims)
    tp = get_typed_pool!(pool, T)
    flat_view = get_view!(tp, total_len)
    return unsafe_wrap(Array{T, N}, pointer(flat_view), dims)
end

# Tuple support
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    unsafe_acquire!(pool, T, dims...)
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

# ==============================================================================
# State Management (v2: Zero-Allocation checkpoint!/rewind!)
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
    checkpoint!(pool::AdaptiveArrayPool, ::Type{T})

Save state for a specific type only. Used by optimized macros that know
which types will be used at compile time.

~77% faster than full checkpoint! when only one type is used.
"""
@inline function checkpoint!(pool::AdaptiveArrayPool, ::Type{T}) where T
    tp = get_typed_pool!(pool, T)
    push!(tp.saved_stack, tp.n_active)
    nothing
end

"""
    rewind!(pool::AdaptiveArrayPool, ::Type{T})

Restore state for a specific type only.
"""
@inline function rewind!(pool::AdaptiveArrayPool, ::Type{T}) where T
    tp = get_typed_pool!(pool, T)
    tp.n_active = pop!(tp.saved_stack)
    nothing
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
