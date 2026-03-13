# ==============================================================================
# Allocation Dispatch Points (for extensibility)
# ==============================================================================

# Allocate a new vector (dispatch point for extensions)
@inline allocate_vector(::AbstractTypedPool{T, Vector{T}}, n::Int) where {T} =
    Vector{T}(undef, n)

# ==============================================================================
# Helper: Overflow-Safe Product
# ==============================================================================

"""
    safe_prod(dims::NTuple{N, Int}) -> Int

Compute the product of dimensions with overflow checking.

Throws `OverflowError` if the product exceeds `typemax(Int)`, preventing
memory corruption from integer overflow in array creation operations.

## Rationale
Without overflow checking, large dimensions like `(10^10, 10^10)` would wrap
around to a small value, causing array creation to index beyond allocated memory.

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
# Helper: Pool Growth Warning (cold path, kept out of hot loops)
# ==============================================================================

@noinline function _warn_pool_growing(tp::AbstractTypedPool{T}, idx::Int) where {T}
    total_bytes = sum(length, tp.vectors) * sizeof(T)
    @warn "$(nameof(typeof(tp))){$T} growing large ($idx arrays, ~$(Base.format_bytes(total_bytes))). Missing rewind!()?"
    return nothing
end

@inline function _check_pool_growth(tp::AbstractTypedPool, idx::Int)
    # Warn at every power of 2 from 512 onward (512, 1024, 2048, …)
    return if idx >= 512 && (idx & (idx - 1)) == 0
        _warn_pool_growing(tp, idx)
    end
end

# ==============================================================================
# Slot Claim — Shared Primitive for All Acquisition Paths
# ==============================================================================

"""
    _claim_slot!(tp::TypedPool{T}, n::Int) -> Int

Claim the next slot, ensuring the backing vector exists and has capacity >= `n`.
Returns the slot index. This is the shared primitive for all acquisition paths
(`get_view!`, `get_array!`).
"""
@inline function _claim_slot!(tp::TypedPool{T}, n::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, n))
        _check_pool_growth(tp, idx)
    else
        @inbounds vec = tp.vectors[idx]
        if length(vec) < n
            resize!(vec, n)
        end
    end
    return idx
end

"""
    _claim_slot!(tp::TypedPool{T}) -> Int

Claim the next slot without provisioning memory (zero-length backing vector).
Used by `reshape!` which only needs the slot index for `nd_wrapper` caching —
the wrapper points to a different array's memory via `setfield!(:ref)`.
"""
@inline function _claim_slot!(tp::TypedPool{T}) where {T}
    tp.n_active += 1
    idx = tp.n_active
    if idx > length(tp.vectors)
        push!(tp.vectors, Vector{T}(undef, 0))
        _check_pool_growth(tp, idx)
    end
    return idx
end

# ==============================================================================
# Get View (Internal — Always Fresh, SubArray is Stack-Allocated via SROA)
# ==============================================================================

"""
    get_view!(tp::TypedPool{T}, n::Int) -> SubArray{T,1}
    get_view!(tp::TypedPool{T}, dims::NTuple{N,Int}) -> ReshapedArray{T,N}

Get a pooled view from the typed pool.
- **1D**: Returns a fresh `SubArray` (stack-allocated via SROA in compiled code).
- **N-D**: Returns a `ReshapedArray` wrapping a 1D view (zero creation cost).

Always creates fresh views — caching is unnecessary since both `SubArray` and
`ReshapedArray` are small structs that SROA can stack-allocate.

Dispatches on `TypedPool{T}` (not `AbstractTypedPool`) because `_claim_slot!`
is only defined for `TypedPool{T}`. Other subtypes override `get_view!` directly
(e.g., `CuTypedPool`) or use a separate path (e.g., `BitTypedPool` → `get_bitarray!`).
"""
@inline function get_view!(tp::TypedPool{T}, n::Int) where {T}
    idx = _claim_slot!(tp, n)
    return @inbounds view(tp.vectors[idx], 1:n)
end

@inline function get_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    slot = _claim_slot!(tp, total_len)
    return @inbounds reshape(view(tp.vectors[slot], 1:total_len), dims)
end

# ==============================================================================
# reshape! — Zero-Allocation Reshape (setfield!-based, Julia 1.11+)
# ==============================================================================

"""
    _reshape_impl!(pool::AdaptiveArrayPool, A::Array{T,M}, dims::NTuple{N,Int}) -> Array{T,N}

Zero-allocation reshape using `setfield!`-based wrapper reuse (Julia 1.11+).

- **Same dimensionality (M == N)**: `setfield!(A, :size, dims)` — no pool interaction
- **Different dimensionality (M ≠ N)**: Claims a pool slot via `_claim_slot!`,
  reuses cached `Array{T,N}` wrapper with `setfield!(:ref, :size)` pointing to `A`'s memory.
  Automatically reclaimed on `rewind!` via `n_active` restoration.
"""
@inline function _reshape_impl!(pool::AdaptiveArrayPool, A::Array{T, M}, dims::NTuple{N, Int}) where {T, M, N}
    # Reject negative dimensions (match Base.reshape behavior)
    for d in dims
        d < 0 && throw(ArgumentError("invalid Array dimensions"))
    end

    # Validate before claiming slot
    total_len = safe_prod(dims)
    length(A) == total_len || throw(
        DimensionMismatch(
            "new dimensions $(dims) must be consistent with array length $(length(A))"
        )
    )

    # 0-D reshape: rare edge case, delegate to Base (arr_wrappers is 1-indexed by N)
    N == 0 && return reshape(A, dims)

    # Same dimensionality: just update size in-place, no pool interaction
    if M == N
        setfield!(A, :size, dims)
        return A
    end

    # Different dimensionality: claim slot + reuse cached N-D wrapper
    tp = get_typed_pool!(pool, T)
    slot = _claim_slot!(tp)

    # Look up cached wrapper (direct index, no hash)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && slot <= length(wrappers)
        wrapper = @inbounds wrappers[slot]
        if wrapper !== nothing
            arr = wrapper::Array{T, N}
            setfield!(arr, :ref, getfield(A, :ref))
            setfield!(arr, :size, dims)
            return arr
        end
    end

    # Cache miss (first call per slot+N): create wrapper, cache forever
    arr = Array{T, N}(undef, ntuple(_ -> 0, Val(N)))
    setfield!(arr, :ref, getfield(A, :ref))
    setfield!(arr, :size, dims)
    _store_arr_wrapper!(tp, N, slot, arr)
    return arr
end

# ==============================================================================
# Get N-D Array (setfield!-based Wrapper Reuse, Julia 1.11+)
# ==============================================================================
#
# Julia 1.11+ changed Array to mutable struct {ref::MemoryRef{T}, size::NTuple{N,Int}},
# enabling in-place mutation via setfield!. This eliminates N-way cache eviction limits:
# unlimited dimension patterns per slot, 0-alloc after warmup for any dims with same N.

"""
    _store_arr_wrapper!(tp::AbstractTypedPool, N::Int, slot::Int, wrapper)

Store a cached N-D wrapper for the given slot. Creates the per-N Vector if needed.
"""
function _store_arr_wrapper!(tp::AbstractTypedPool, N::Int, slot::Int, wrapper)
    # Grow arr_wrappers vector so index N is valid
    if N > length(tp.arr_wrappers)
        old_len = length(tp.arr_wrappers)
        resize!(tp.arr_wrappers, N)
        for i in (old_len + 1):N
            @inbounds tp.arr_wrappers[i] = nothing
        end
    end
    wrappers = @inbounds tp.arr_wrappers[N]
    if wrappers === nothing
        wrappers = Vector{Any}(nothing, slot)
        @inbounds tp.arr_wrappers[N] = wrappers
    elseif slot > length(wrappers)
        old_len = length(wrappers)
        resize!(wrappers, slot)
        for i in (old_len + 1):slot
            @inbounds wrappers[i] = nothing
        end
    end
    @inbounds wrappers[slot] = wrapper
    return nothing
end

"""
    get_array!(tp::AbstractTypedPool{T,Vector{T}}, dims::NTuple{N,Int}) -> Array{T,N}

Get an N-dimensional `Array` from the pool with `setfield!`-based wrapper reuse.

Uses `_claim_slot!` directly for slot management (independent of view path).
Cache hit: `setfield!(arr, :ref/size)` — 0 allocation.
Cache miss: creates wrapper via `setfield!` pattern, then cached forever.
"""
@inline function get_array!(tp::AbstractTypedPool{T, Vector{T}}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    slot = _claim_slot!(tp, total_len)
    @inbounds vec = tp.vectors[slot]

    # Look up cached wrapper for this dimensionality (direct index, no hash)
    wrappers = N <= length(tp.arr_wrappers) ? (@inbounds tp.arr_wrappers[N]) : nothing
    if wrappers !== nothing && slot <= length(wrappers)
        wrapper = @inbounds wrappers[slot]
        if wrapper !== nothing
            arr = wrapper::Array{T, N}
            # Always update ref: resize! can grow in-place without changing pointer,
            # but the old MemoryRef still has the old (smaller) Memory length.
            # setfield!(:ref) is 0-alloc in compiled code (only 32B at REPL top-level).
            setfield!(arr, :ref, getfield(vec, :ref))
            # Update dimensions (0-alloc: NTuple stored inline in mutable Array)
            setfield!(arr, :size, dims)
            return arr
        end
    end

    # Cache miss: first call for this (slot, N) — create via setfield! pattern
    arr = Array{T, N}(undef, ntuple(_ -> 0, Val(N)))
    setfield!(arr, :ref, getfield(vec, :ref))
    setfield!(arr, :size, dims)
    _store_arr_wrapper!(tp, N, slot, arr)
    return arr
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
    _acquire_impl!(pool, Type{T}, n) -> Array{T,1}
    _acquire_impl!(pool, Type{T}, dims...) -> Array{T,N}

Internal implementation of acquire!. Called directly by macro-transformed code
(no type touch recording). User code calls `acquire!` which adds recording.

Returns raw `Array{T,N}` via cached wrapper reuse (setfield!-based on Julia 1.11+).
"""
@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    result = get_array!(tp, (n,))
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    result = get_array!(tp, dims)
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function _acquire_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    tp = get_typed_pool!(pool, T)
    result = get_array!(tp, dims)
    _maybe_record_borrow!(pool, tp)
    return result
end

# Similar-style
@inline _acquire_impl!(pool::AbstractArrayPool, x::AbstractArray) = _acquire_impl!(pool, eltype(x), size(x))

"""
    _acquire_view_impl!(pool, Type{T}, n) -> SubArray{T,1,Vector{T},...}
    _acquire_view_impl!(pool, Type{T}, dims...) -> ReshapedArray{T,N,...}

Internal implementation of acquire_view!. Called directly by macro-transformed code
(no type touch recording). Returns views into pool-managed vectors.
"""
@inline function _acquire_view_impl!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    result = get_view!(tp, n)
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function _acquire_view_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    result = get_view!(tp, dims)
    _maybe_record_borrow!(pool, tp)
    return result
end

@inline function _acquire_view_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _acquire_view_impl!(pool, T, dims...)
end

# Similar-style
@inline _acquire_view_impl!(pool::AbstractArrayPool, x::AbstractArray) = _acquire_view_impl!(pool, eltype(x), size(x))

# ==============================================================================
# Acquisition API (User-facing with type touch recording)
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> Array{T,1}
    acquire!(pool, Type{T}, dims...) -> Array{T,N}
    acquire!(pool, Type{T}, dims::NTuple{N,Int}) -> Array{T,N}

Acquire a pooled array of type `T` with size `n` or dimensions `dims`.

Returns:
- **CPU**: `Array{T,N}` (zero-alloc via cached wrapper reuse on Julia 1.11+)
- **Bit** (`T === Bit`): `BitVector` / `BitArray{N}` (chunks-sharing, SIMD optimized)
- **CUDA**: `CuArray{T,N}` (unified N-way cache)

The returned array is only valid within the `@with_pool` scope.

## Example
```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)      # Vector{Float64}
    m = acquire!(pool, Float64, 10, 10)   # Matrix{Float64}
    v .= 1.0
    m .= 2.0
    sum(v) + sum(m)
end
```

See also: [`acquire_view!`](@ref) for view-based access.
"""
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct acquire! call>")
    return _acquire_impl!(pool, T, n)
end

# Multi-dimensional support (zero-allocation with N-D cache)
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct acquire! call>")
    return _acquire_impl!(pool, T, dims...)
end

# Tuple support: allows acquire!(pool, T, size(A)) where size(A) returns NTuple{N,Int}
@inline function acquire!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct acquire! call>")
    return _acquire_impl!(pool, T, dims...)
end

# Similar-style convenience methods
"""
    acquire!(pool, x::AbstractArray) -> Array

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
    _set_pending_callsite!(pool, "<direct acquire! call>")
    return _acquire_impl!(pool, eltype(x), size(x))
end

# ==============================================================================
# View Acquisition API
# ==============================================================================

"""
    acquire_view!(pool, Type{T}, n) -> SubArray{T,1,Vector{T},...}
    acquire_view!(pool, Type{T}, dims...) -> ReshapedArray{T,N,...}
    acquire_view!(pool, Type{T}, dims::NTuple{N,Int}) -> view type

Acquire a view into pool-managed memory.

Returns a view (SubArray/ReshapedArray) instead of a raw Array.
Useful when you need stack-allocated wrappers via SROA.

See also: [`acquire!`](@ref) for the default array-returning API.
"""
@inline function acquire_view!(pool::AbstractArrayPool, ::Type{T}, n::Int) where {T}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct acquire_view! call>")
    return _acquire_view_impl!(pool, T, n)
end

@inline function acquire_view!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct acquire_view! call>")
    return _acquire_view_impl!(pool, T, dims...)
end

@inline function acquire_view!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct acquire_view! call>")
    return _acquire_view_impl!(pool, T, dims...)
end

@inline function acquire_view!(pool::AbstractArrayPool, x::AbstractArray)
    _record_type_touch!(pool, eltype(x))
    _set_pending_callsite!(pool, "<direct acquire_view! call>")
    return _acquire_view_impl!(pool, eltype(x), size(x))
end

# ==============================================================================
# API Alias
# ==============================================================================

"""
    acquire_array!(pool, Type{T}, dims...)

Alias for [`acquire!`](@ref).

Explicit name emphasizing the return type is a raw `Array`.
Use when you prefer symmetric naming with `acquire_view!`.
"""
const acquire_array! = acquire!

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

# --- acquire_view! for DisabledPool{:cpu} (no view caching when disabled) ---
@inline acquire_view!(::DisabledPool{:cpu}, ::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
@inline acquire_view!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Array{T, N}(undef, dims)
@inline acquire_view!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Array{T, N}(undef, dims)
@inline acquire_view!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
