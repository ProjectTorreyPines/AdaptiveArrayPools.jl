# ==============================================================================
# Allocation Dispatch Points (for extensibility)
# ==============================================================================

# Allocate a new vector (dispatch point for extensions)
@inline allocate_vector(::AbstractTypedPool{T, Vector{T}}, n::Int) where {T} =
    Vector{T}(undef, n)

# Wrap flat view into N-D array (dispatch point for extensions)
@inline function wrap_array(
        ::AbstractTypedPool{T, Vector{T}},
        flat_view, dims::NTuple{N, Int}
    ) where {T, N}
    return unsafe_wrap(Array{T, N}, pointer(flat_view), dims)
end

# ==============================================================================
# Helper: Overflow-Safe Product
# ==============================================================================

"""
    safe_prod(dims::NTuple{N, Int}) -> Int

Compute the product of dimensions with overflow checking.

Throws `OverflowError` if the product exceeds `typemax(Int)`, preventing
memory corruption from integer overflow in `unsafe_wrap` operations.

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
# Get N-D Array (N-way Set-Associative Cache, Julia ≤1.10)
# ==============================================================================
#
# On Julia ≤1.10, Array is not a mutable struct, so setfield! cannot be used.
# This provides the N-way cache that stores up to CACHE_WAYS different
# (dims, pointer) patterns per slot via round-robin replacement.

"""
    get_array!(tp::AbstractTypedPool{T}, dims::NTuple{N,Int}) -> Array{T,N}

Get an N-dimensional `Array` from the pool with N-way caching.

Uses a set-associative cache with `CACHE_WAYS` entries per slot (default: 4).
Cache hit (exact dims + pointer match) returns the cached Array at zero cost.
Cache miss creates a new `unsafe_wrap`'d Array (~96 bytes) and stores it via
round-robin replacement.
"""
@inline function get_array!(tp::AbstractTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
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
# Type Touch Recording (for selective rewind)
# ==============================================================================

"""
    _record_type_touch!(pool::AbstractArrayPool, ::Type{T})

Record that type `T` was touched (acquired) at the current checkpoint depth.
Called by `acquire!` and convenience wrappers; macro-transformed calls use
`_acquire_impl!` directly (bypassing this for zero overhead).

For fixed-slot types, sets the corresponding bit in `_touched_type_masks`.
For non-fixed-slot types, sets `_touched_has_others` flag.
"""
@inline function _record_type_touch!(pool::AbstractArrayPool, ::Type{T}) where {T}
    depth = pool._current_depth
    b = _fixed_slot_bit(T)
    if b == UInt16(0)
        @inbounds pool._touched_has_others[depth] = true
    else
        @inbounds pool._touched_type_masks[depth] |= b
    end
    return nothing
end

# CPU-specific override: adds lazy first-touch checkpoint in lazy mode
# and typed-lazy mode.
# _LAZY_MODE_BIT (bit 15) in _touched_type_masks[depth]  ↔  depth entered via _lazy_checkpoint!
# _TYPED_LAZY_BIT (bit 14) in _touched_type_masks[depth]  ↔  depth entered via _typed_lazy_checkpoint!
# On the first acquire of each fixed-slot type T at that depth, we retroactively save
# n_active BEFORE the acquire (current value is still the parent's count), so that
# the subsequent rewind can restore the parent's state correctly.
@inline function _record_type_touch!(pool::AdaptiveArrayPool, ::Type{T}) where {T}
    depth = pool._current_depth
    b = _fixed_slot_bit(T)
    if b == UInt16(0)
        @inbounds pool._touched_has_others[depth] = true
    else
        current_mask = @inbounds pool._touched_type_masks[depth]
        # Lazy checkpoint: lazy mode (bit 15) OR typed lazy mode (bit 14), AND first touch.
        # Guard: skip if already checkpointed at this depth (prevents double-push when a
        # tracked type is also acquired by a helper via acquire! → _record_type_touch!).
        if (current_mask & _MODE_BITS_MASK) != 0 && (current_mask & b) == 0
            tp = get_typed_pool!(pool, T)
            if @inbounds(tp._checkpoint_depths[end]) != depth
                _checkpoint_typed_pool!(tp, depth)
            end
        end
        @inbounds pool._touched_type_masks[depth] = current_mask | b
    end
    return nothing
end

# ==============================================================================
# Internal Implementation Functions (called by macro-transformed code)
# ==============================================================================

"""
    _acquire_impl!(pool, Type{T}, n) -> SubArray{T,1,Vector{T},...}
    _acquire_impl!(pool, Type{T}, dims...) -> ReshapedArray{T,N,...}

Internal implementation of acquire!. Called directly by macro-transformed code
(no type touch recording). User code calls `acquire!` which adds recording.
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
    return _acquire_impl!(pool, T, dims...)
end

# Similar-style
@inline _acquire_impl!(pool::AbstractArrayPool, x::AbstractArray) = _acquire_impl!(pool, eltype(x), size(x))

"""
    _unsafe_acquire_impl!(pool, Type{T}, dims...) -> Array{T,N}

Internal implementation of unsafe_acquire!. Called directly by macro-transformed code.
"""
@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return get_array!(tp, (n,))
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_array!(tp, dims)
end

@inline function _unsafe_acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_array!(tp, dims)
end

# Similar-style
@inline _unsafe_acquire_impl!(pool::AbstractArrayPool, x::AbstractArray) = _unsafe_acquire_impl!(pool, eltype(x), size(x))

# ==============================================================================
# Acquisition API (User-facing with type touch recording)
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> view type
    acquire!(pool, Type{T}, dims...) -> view type
    acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> view type

Acquire a pooled array of type `T` with size `n` or dimensions `dims`.

Returns a pooled array (backend-dependent type):
- **CPU 1D**: `SubArray{T,1,Vector{T},...}` (parent is `Vector{T}`)
- **CPU N-D**: `ReshapedArray{T,N,...}` (zero creation cost)
- **Bit** (`T === Bit`): `BitVector` / `BitArray{N}` (chunks-sharing, SIMD optimized)
- **CUDA**: `CuArray{T,N}` (unified N-way cache)

For CPU numeric arrays, the return types are `StridedArray`, compatible with
BLAS and broadcasting.

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
    _record_type_touch!(pool, T)
    return _acquire_impl!(pool, T, n)
end

# Multi-dimensional support (zero-allocation with N-D cache)
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _acquire_impl!(pool, T, dims...)
end

# Tuple support: allows acquire!(pool, T, size(A)) where size(A) returns NTuple{N,Int}
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _acquire_impl!(pool, T, dims...)
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
    _record_type_touch!(pool, eltype(x))
    return _acquire_impl!(pool, eltype(x), size(x))
end

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
- **Bit** (`T === Bit`): `BitVector` / `BitArray{N}` (chunks-sharing; equivalent to `acquire!`)
- **CUDA**: `CuArray{T,N}` (via unified view cache)

## Safety Warning
The returned array is only valid within the `@with_pool` scope. Using it after
the scope ends leads to undefined behavior (use-after-free, data corruption).

**Do NOT call `resize!`, `push!`, or `append!` on returned arrays** - this causes
undefined behavior as the memory is owned by the pool.

## When to Use
- **Type-unspecified paths**: Struct fields without concrete type parameters
- FFI calls expecting raw pointers
- APIs that strictly require native array types

## Example
```julia
@with_pool pool begin
    A = unsafe_acquire!(pool, Float64, 100, 100)  # Matrix{Float64}
    B = unsafe_acquire!(pool, Float64, 100, 100)
    C = similar(A)  # Regular allocation for result
    mul!(C, A, B)   # BLAS uses A, B directly
end
# A and B are INVALID after this point!
```

See also: [`acquire!`](@ref) for view-based access.
"""
@inline function unsafe_acquire!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    _record_type_touch!(pool, T)
    return _unsafe_acquire_impl!(pool, T, n)
end

@inline function unsafe_acquire!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_acquire_impl!(pool, T, dims...)
end

# Tuple support
@inline function unsafe_acquire!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_acquire_impl!(pool, T, dims)
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
    _record_type_touch!(pool, eltype(x))
    return _unsafe_acquire_impl!(pool, eltype(x), size(x))
end

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

# ==============================================================================
# DisabledPool Fallbacks (pooling disabled with backend context)
# ==============================================================================

# DisabledPool has no internal state to track, so type touch is a no-op.
@inline _record_type_touch!(::DisabledPool, ::Type) = nothing

# --- acquire! for DisabledPool{:cpu} ---
@inline acquire!(::DisabledPool{:cpu}, ::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
@inline acquire!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Array{T, N}(undef, dims)
@inline acquire!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Array{T, N}(undef, dims)
@inline acquire!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)

# --- unsafe_acquire! for DisabledPool{:cpu} ---
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Array{T, N}(undef, dims)
@inline unsafe_acquire!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Array{T, N}(undef, dims)
@inline unsafe_acquire!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
