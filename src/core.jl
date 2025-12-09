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

Internal function to get an N-dimensional `Array` from the typed pool with N-way caching.
Used by `unsafe_acquire!` to cache Array instances and avoid `unsafe_wrap` overhead.

## N-way Set Associative Cache
Each slot can cache up to `CACHE_WAYS` different dimension patterns.
This prevents thrashing when alternating between different array shapes.

## Cache Hit Conditions
1. Same dims tuple (`isa NTuple{N, Int} && cached_dims == dims`)
2. Same pointer (backing vector not resized)

## Type Assertion
Uses `::Array{T, N}` for type stability when retrieving from `Vector{Any}`.
"""
@inline function get_nd_array!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
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

    arr = unsafe_wrap(Array{T, N}, pointer(flat_view), dims)

    @inbounds tp.nd_arrays[target_idx] = arr
    @inbounds tp.nd_dims[target_idx] = dims
    @inbounds tp.nd_ptrs[target_idx] = current_ptr

    # Update round-robin counter
    @inbounds tp.nd_next_way[slot] = (way_offset + 1) % CACHE_WAYS

    return arr
end

"""
    get_nd_view!(tp::TypedPool{T}, dims::NTuple{N,Int}) -> ReshapedArray{T,N,...}

Internal function to get an N-dimensional view from the typed pool.

Returns a `ReshapedArray` wrapping a 1D view - zero creation cost (no `unsafe_wrap`).
`ReshapedArray` is a lightweight, stack-allocated wrapper with minimal overhead.

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
# Untracked Acquire Detection
# ==============================================================================

"""
    _mark_untracked!(pool::AdaptiveArrayPool)

Mark that an untracked acquire has occurred at the current checkpoint depth.
Called by `acquire!` wrapper; macro-transformed calls use `_acquire_impl!` directly.

With 1-indexed _current_depth (starting at 1 for global scope), this always marks
the current scope's _untracked_flags.
"""
@inline function _mark_untracked!(pool::AdaptiveArrayPool)
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
@inline function _acquire_impl!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_view!(tp, n)
end

@inline function _acquire_impl!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_view!(tp, dims)
end

@inline function _acquire_impl!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _acquire_impl!(pool, T, dims...)
end

# Fallback for nothing pool
@inline _acquire_impl!(::Nothing, ::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
@inline _acquire_impl!(::Nothing, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Array{T, N}(undef, dims)
@inline _acquire_impl!(::Nothing, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Array{T, N}(undef, dims)

# Similar-style
@inline _acquire_impl!(pool::AdaptiveArrayPool, x::AbstractArray) = _acquire_impl!(pool, eltype(x), size(x))
@inline _acquire_impl!(::Nothing, x::AbstractArray) = similar(x)

"""
    _unsafe_acquire_impl!(pool, Type{T}, dims...) -> Array{T,N}

Internal implementation of unsafe_acquire!. Called directly by macro-transformed code.
"""
@inline function _unsafe_acquire_impl!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, (n,))
end

@inline function _unsafe_acquire_impl!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)
end

@inline function _unsafe_acquire_impl!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)
end

# Fallback for nothing pool
@inline _unsafe_acquire_impl!(::Nothing, ::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
@inline _unsafe_acquire_impl!(::Nothing, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Array{T, N}(undef, dims)
@inline _unsafe_acquire_impl!(::Nothing, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Array{T, N}(undef, dims)

# Similar-style
@inline _unsafe_acquire_impl!(pool::AdaptiveArrayPool, x::AbstractArray) = _unsafe_acquire_impl!(pool, eltype(x), size(x))
@inline _unsafe_acquire_impl!(::Nothing, x::AbstractArray) = similar(x)

# ==============================================================================
# Acquisition API (User-facing with untracked marking)
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
    _mark_untracked!(pool)
    _acquire_impl!(pool, T, n)
end

# Multi-dimensional support (zero-allocation with N-D cache)
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _mark_untracked!(pool)
    _acquire_impl!(pool, T, dims...)
end

# Tuple support: allows acquire!(pool, T, size(A)) where size(A) returns NTuple{N,Int}
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
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
@inline function acquire!(pool::AdaptiveArrayPool, x::AbstractArray)
    _mark_untracked!(pool)
    _acquire_impl!(pool, eltype(x), size(x))
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

Since `Array` instances are mutable references, cached instances can be returned directly
without creating new wrapper objects—ideal for type-unspecified paths. In contrast,
`ReshapedArray` wraps a view and cannot be meaningfully cached, as each call to `reshape()`
creates a new wrapper.

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
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, T, n)
end

@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, T, dims...)
end

# Tuple support
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
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
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, x::AbstractArray)
    _mark_untracked!(pool)
    _unsafe_acquire_impl!(pool, eltype(x), size(x))
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
    # Increment depth and initialize untracked flag
    pool._current_depth += 1
    push!(pool._untracked_flags, false)
    depth = pool._current_depth

    # Fixed slots - direct field access, no Dict lookup
    _checkpoint_typed_pool!(pool.float64, depth)
    _checkpoint_typed_pool!(pool.float32, depth)
    _checkpoint_typed_pool!(pool.int64, depth)
    _checkpoint_typed_pool!(pool.int32, depth)
    _checkpoint_typed_pool!(pool.complexf64, depth)
    _checkpoint_typed_pool!(pool.bool, depth)

    # Others - iterate without allocation (values() returns iterator)
    for p in values(pool.others)
        _checkpoint_typed_pool!(p, depth)
    end

    return nothing
end

# Internal helper for full checkpoint
@inline function _checkpoint_typed_pool!(tp::TypedPool, depth::Int)
    push!(tp._checkpoint_n_active, tp.n_active)
    push!(tp._checkpoint_depths, depth)
    nothing
end

checkpoint!(::Nothing) = nothing

"""
    rewind!(pool::AdaptiveArrayPool)

Restore the pool state (n_active counters) from internal stacks.
Uses _checkpoint_depths to accurately determine which entries to pop vs restore.

Only the counters are restored; allocated memory remains for reuse.
Handles untracked acquires by checking _checkpoint_depths for accurate restoration.

See also: [`checkpoint!`](@ref), [`@with_pool`](@ref)
"""
function rewind!(pool::AdaptiveArrayPool)
    depth = pool._current_depth

    # Process fixed slots directly (zero allocation)
    _rewind_typed_pool!(pool.float64, depth)
    _rewind_typed_pool!(pool.float32, depth)
    _rewind_typed_pool!(pool.int64, depth)
    _rewind_typed_pool!(pool.int32, depth)
    _rewind_typed_pool!(pool.complexf64, depth)
    _rewind_typed_pool!(pool.bool, depth)

    # Process fallback types
    for tp in values(pool.others)
        _rewind_typed_pool!(tp, depth)
    end

    pop!(pool._untracked_flags)
    pool._current_depth -= 1

    return nothing
end

# Internal helper for full rewind with _checkpoint_depths
# Uses 1-based sentinel pattern: no isempty checks needed (sentinel [0] guarantees non-empty)
@inline function _rewind_typed_pool!(tp::TypedPool, depth::Int)
    if @inbounds tp._checkpoint_depths[end] == depth
        # Checkpointed at current depth → pop both stacks
        pop!(tp._checkpoint_depths)
        tp.n_active = pop!(tp._checkpoint_n_active)
    else
        # Checkpointed at earlier depth → restore without pop
        tp.n_active = @inbounds tp._checkpoint_n_active[end]
    end
    nothing
end

rewind!(::Nothing) = nothing

# ==============================================================================
# Type-Specific State Management (for optimized macros)
# ==============================================================================

"""
    checkpoint!(tp::TypedPool)

Internal method for saving TypedPool state (legacy, uses depth=0).

!!! warning "Internal API"
    This is an internal implementation detail. For manual pool management,
    use the public API instead:
    ```julia
    checkpoint!(pool, Float64)  # Type-specific checkpoint
    ```

See also: [`checkpoint!(::AdaptiveArrayPool, ::Type)`](@ref), [`rewind!`](@ref)
"""
@inline function checkpoint!(tp::TypedPool)
    push!(tp._checkpoint_n_active, tp.n_active)
    push!(tp._checkpoint_depths, 0)  # Legacy depth
    nothing
end

"""
    checkpoint!(tp::TypedPool, depth::Int)

Internal method for saving TypedPool state with depth tracking.
"""
@inline function checkpoint!(tp::TypedPool, depth::Int)
    push!(tp._checkpoint_n_active, tp.n_active)
    push!(tp._checkpoint_depths, depth)
    nothing
end

"""
    rewind!(tp::TypedPool)

Internal method for restoring TypedPool state (pops both stacks).

!!! warning "Internal API"
    This is an internal implementation detail. For manual pool management,
    use the public API instead:
    ```julia
    rewind!(pool, Float64)  # Type-specific rewind
    ```

See also: [`rewind!(::AdaptiveArrayPool, ::Type)`](@ref), [`checkpoint!`](@ref)
"""
@inline function rewind!(tp::TypedPool)
    pop!(tp._checkpoint_depths)
    tp.n_active = pop!(tp._checkpoint_n_active)
    nothing
end

"""
    checkpoint!(pool::AdaptiveArrayPool, ::Type{T})

Save state for a specific type only. Used by optimized macros that know
which types will be used at compile time.

Also updates _current_depth and _untracked_flags for untracked acquire detection.

~77% faster than full checkpoint! when only one type is used.
"""
@inline function checkpoint!(pool::AdaptiveArrayPool, ::Type{T}) where T
    pool._current_depth += 1
    push!(pool._untracked_flags, false)
    checkpoint!(get_typed_pool!(pool, T), pool._current_depth)
end

"""
    rewind!(pool::AdaptiveArrayPool, ::Type{T})

Restore state for a specific type only.
Also updates _current_depth and _untracked_flags.
"""
@inline function rewind!(pool::AdaptiveArrayPool, ::Type{T}) where T
    rewind!(get_typed_pool!(pool, T))
    pop!(pool._untracked_flags)
    pool._current_depth -= 1
end

checkpoint!(::Nothing, ::Type) = nothing
rewind!(::Nothing, ::Type) = nothing

"""
    checkpoint!(pool::AdaptiveArrayPool, types::Type...)

Save state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling. Increments _current_depth once for all types.
"""
@generated function checkpoint!(pool::AdaptiveArrayPool, types::Type...)
    # First increment depth, then checkpoint each type with that depth
    checkpoint_exprs = [:(checkpoint!(get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in 1:length(types)]
    quote
        pool._current_depth += 1
        push!(pool._untracked_flags, false)
        $(checkpoint_exprs...)
        nothing
    end
end

"""
    rewind!(pool::AdaptiveArrayPool, types::Type...)

Restore state for multiple specific types in reverse order.
Decrements _current_depth once after all types are rewound.
"""
@generated function rewind!(pool::AdaptiveArrayPool, types::Type...)
    # Reverse order for proper stack unwinding, rewind TypedPools directly
    rewind_exprs = [:(rewind!(get_typed_pool!(pool, types[$i]))) for i in length(types):-1:1]
    quote
        $(rewind_exprs...)
        pop!(pool._untracked_flags)
        pool._current_depth -= 1
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
Restores sentinel values for 1-based sentinel pattern.
"""
function Base.empty!(tp::TypedPool)
    empty!(tp.vectors)
    empty!(tp.views)
    empty!(tp.view_lengths)
    # Clear N-D Array cache (N-way)
    empty!(tp.nd_arrays)
    empty!(tp.nd_dims)
    empty!(tp.nd_ptrs)
    empty!(tp.nd_next_way)
    tp.n_active = 0
    # Restore sentinel values (1-based sentinel pattern)
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)   # Sentinel: n_active=0 at depth=0
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)     # Sentinel: depth=0 = no checkpoint
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

    # Reset untracked detection state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._untracked_flags)
    push!(pool._untracked_flags, false)       # Sentinel: global scope starts with false

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

# Internal implementation aliases (for macro transformation)
const _acquire_view_impl! = _acquire_impl!
const _acquire_array_impl! = _unsafe_acquire_impl!
