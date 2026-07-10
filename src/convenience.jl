# ==============================================================================
# Convenience Functions (zeros!, ones!, similar!)
# ==============================================================================

# ==============================================================================
# Default Element Type
# ==============================================================================

"""
    default_eltype(pool) -> Type

Default element type for convenience functions when type is not specified.
CPU pools default to `Float64`, CUDA pools to `Float32`.

Backends can override this to provide appropriate defaults.
"""
default_eltype(::AbstractArrayPool) = Float64

# ==============================================================================
# zeros! - Acquire zero-initialized arrays from pool
# ==============================================================================

"""
    zeros!(pool, dims...) -> Array
    zeros!(pool, T, dims...) -> Array
    zeros!(pool, dims::Tuple) -> Array
    zeros!(pool, T, dims::Tuple) -> Array

Acquire a zero-initialized array from the pool.

Equivalent to `acquire!(pool, T, dims...)` followed by `fill!(arr, zero(T))`.
Default element type depends on pool backend (CPU: `Float64`, CUDA: `Float32`).
See [`default_eltype`](@ref).

## Example
```julia
@with_pool pool begin
    v = zeros!(pool, 100)              # Vector{Float64}, all zeros
    m = zeros!(pool, Float32, 10, 10)  # Matrix{Float32}, all zeros
end
```

See also: [`ones!`](@ref), [`similar!`](@ref), [`acquire!`](@ref)
"""
@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct zeros! call>")
    return _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct zeros! call>")
    return _zeros_impl!(pool, default_eltype(pool), dims...)
end

@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct zeros! call>")
    return _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct zeros! call>")
    return _zeros_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _zeros_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _acquire_impl!(pool, T, dims...)
    fill!(arr, zero(T))
    return arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _zeros_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    return _zeros_impl!(pool, default_eltype(pool), dims...)
end

# NTuple overloads for macro transformation (handles zeros!(pool, T, size(x)) form)
@inline function _zeros_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _zeros_impl!(pool, T, dims...)
end

@inline function _zeros_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    return _zeros_impl!(pool, default_eltype(pool), dims...)
end

# tp-hoisted forms: macro-transformed code binds `tp = get_typed_pool!(pool, T)`
# once per scope and passes it here, skipping the per-acquire lookup.
@inline function _zeros_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _acquire_impl!(pool, tp, dims...)
    fill!(arr, zero(T))
    return arr
end

@inline function _zeros_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    return _zeros_impl!(pool, tp, dims...)
end

# Bit type specialization: zeros!(pool, Bit, ...) delegates to falses!(pool, ...)
@inline zeros!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = falses!(pool, dims...)
@inline zeros!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = falses!(pool, dims)
@inline _zeros_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = _falses_impl!(pool, dims...)
@inline _zeros_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = _falses_impl!(pool, dims)

# ==============================================================================
# ones! - Acquire one-initialized arrays from pool
# ==============================================================================

"""
    ones!(pool, dims...) -> Array
    ones!(pool, T, dims...) -> Array
    ones!(pool, dims::Tuple) -> Array
    ones!(pool, T, dims::Tuple) -> Array

Acquire a one-initialized array from the pool.

Equivalent to `acquire!(pool, T, dims...)` followed by `fill!(arr, one(T))`.
Default element type depends on pool backend (CPU: `Float64`, CUDA: `Float32`).
See [`default_eltype`](@ref).

## Example
```julia
@with_pool pool begin
    v = ones!(pool, 100)              # Vector{Float64}, all ones
    m = ones!(pool, Float32, 10, 10)  # Matrix{Float32}, all ones
end
```

See also: [`zeros!`](@ref), [`similar!`](@ref), [`acquire!`](@ref)
"""
@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct ones! call>")
    return _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct ones! call>")
    return _ones_impl!(pool, default_eltype(pool), dims...)
end

@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct ones! call>")
    return _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct ones! call>")
    return _ones_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _ones_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _acquire_impl!(pool, T, dims...)
    fill!(arr, one(T))
    return arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _ones_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    return _ones_impl!(pool, default_eltype(pool), dims...)
end

# NTuple overloads for macro transformation (handles ones!(pool, T, size(x)) form)
@inline function _ones_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _ones_impl!(pool, T, dims...)
end

@inline function _ones_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    return _ones_impl!(pool, default_eltype(pool), dims...)
end

# tp-hoisted forms: macro-transformed code binds `tp = get_typed_pool!(pool, T)`
# once per scope and passes it here, skipping the per-acquire lookup.
@inline function _ones_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _acquire_impl!(pool, tp, dims...)
    fill!(arr, one(T))
    return arr
end

@inline function _ones_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    return _ones_impl!(pool, tp, dims...)
end

# Bit type specialization: ones!(pool, Bit, ...) delegates to trues!(pool, ...)
@inline ones!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = trues!(pool, dims...)
@inline ones!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = trues!(pool, dims)
@inline _ones_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = _trues_impl!(pool, dims...)
@inline _ones_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = _trues_impl!(pool, dims)

# ==============================================================================
# trues! - Acquire BitArray filled with true from pool
# ==============================================================================

"""
    trues!(pool, dims...) -> BitArray
    trues!(pool, dims::Tuple) -> BitArray

Acquire a bit-packed boolean array filled with `true` from the pool.

Equivalent to Julia's `trues(dims...)` but using pooled memory.
Uses ~8x less memory than `ones!(pool, Bool, dims...)`.

## Example
```julia
@with_pool pool begin
    bv = trues!(pool, 100)        # BitVector, all true
    bm = trues!(pool, 10, 10)     # BitMatrix, all true
end
```

See also: [`falses!`](@ref), [`ones!`](@ref), [`acquire!`](@ref)
"""
@inline function trues!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, Bit)
    _set_pending_callsite!(pool, "<direct trues! call>")
    return _trues_impl!(pool, dims...)
end
@inline function trues!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, Bit)
    _set_pending_callsite!(pool, "<direct trues! call>")
    return _trues_impl!(pool, dims...)
end

# Internal implementation (for macro transformation)
@inline function _trues_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    arr = _acquire_impl!(pool, Bit, dims...)
    fill!(arr, true)
    return arr
end
@inline _trues_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N} = _trues_impl!(pool, dims...)

# ==============================================================================
# falses! - Acquire BitArray filled with false from pool
# ==============================================================================

"""
    falses!(pool, dims...) -> BitArray
    falses!(pool, dims::Tuple) -> BitArray

Acquire a bit-packed boolean array filled with `false` from the pool.

Equivalent to Julia's `falses(dims...)` but using pooled memory.
Uses ~8x less memory than `zeros!(pool, Bool, dims...)`.

## Example
```julia
@with_pool pool begin
    bv = falses!(pool, 100)       # BitVector, all false
    bm = falses!(pool, 10, 10)    # BitMatrix, all false
end
```

See also: [`trues!`](@ref), [`zeros!`](@ref), [`acquire!`](@ref)
"""
@inline function falses!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, Bit)
    _set_pending_callsite!(pool, "<direct falses! call>")
    return _falses_impl!(pool, dims...)
end
@inline function falses!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, Bit)
    _set_pending_callsite!(pool, "<direct falses! call>")
    return _falses_impl!(pool, dims...)
end

# Internal implementation (for macro transformation)
@inline function _falses_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    arr = _acquire_impl!(pool, Bit, dims...)
    fill!(arr, false)
    return arr
end
@inline _falses_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N} = _falses_impl!(pool, dims...)

# ==============================================================================
# rand! / randn! - Acquire random-initialized arrays from pool
# ==============================================================================
# These extend `Random.rand!` / `Random.randn!` with pool-constructor methods.
# They are NOT type piracy: the pool type (`AbstractArrayPool`) is ours, and the
# signatures are disjoint from Random's (which dispatch on `AbstractArray` first).
#
# Because `_acquire_impl!` returns a genuine `Array{T,N}` (not a view), filling it
# with `Random.rand!(arr)` consumes the RNG stream identically to `rand(T, dims)`
# — so seeding identically reproduces the same sequence.

# Sampleable collection types for the `rand!(pool, S, dims...)` form. Disjoint
# from `Int`/`Type` so it never collides with the dims/typed forms. The trailing
# `d1::Int` (≥1 explicit dim) in the vararg method also prevents a bare Int-tuple
# (used as dims, e.g. `rand!(pool, (8,9))`) from being mistaken for a collection.
const _SampleColl = Union{AbstractArray, Tuple, AbstractSet, AbstractDict, AbstractString}

"""
    rand!(pool, dims...) -> Array
    rand!(pool, T, dims...) -> Array
    rand!(pool, S, dims...) -> Array
    rand!(pool, dims::Tuple) / rand!(pool, T, dims::Tuple) / rand!(pool, S, dims::Tuple)

Acquire a uniform random array from the pool.

- `rand!(pool, [T,] dims...)`: floats are `U[0,1)`, integers full-range.
  Default element type depends on backend (CPU: `Float64`, CUDA: `Float32`).
- `rand!(pool, S, dims...)`: sample from a collection/range `S`
  (e.g. `rand!(pool, 1:6, 10)` for dice). Element type is `eltype(S)`.

Consumes the global RNG identically to `Random.rand!`, so seeding reproduces
the same sequence. Use `Random.seed!` / pass values through `Random` for control.

## Example
```julia
@with_pool pool begin
    v = rand!(pool, 100)              # Vector{Float64}, U[0,1)
    m = rand!(pool, Float32, 8, 8)    # Matrix{Float32}
    d = rand!(pool, 1:6, 10)          # Vector{Int}, each ∈ 1:6
end
```

See also: [`randn!`](@ref), [`zeros!`](@ref), [`acquire!`](@ref)
"""
@inline function rand!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct rand! call>")
    return _rand_impl!(pool, T, dims...)
end
@inline function rand!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct rand! call>")
    return _rand_impl!(pool, default_eltype(pool), dims...)
end
@inline function rand!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct rand! call>")
    return _rand_impl!(pool, T, dims...)
end
@inline function rand!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct rand! call>")
    return _rand_impl!(pool, default_eltype(pool), dims...)
end
# Collection/range form: type touch is recorded inside `_rand_impl!` (the macro
# cannot pre-register `eltype(S)`, so the impl is self-sufficient — see below).
@inline function rand!(pool::AbstractArrayPool, S::_SampleColl, d1::Int, dims::Vararg{Int, N}) where {N}
    _set_pending_callsite!(pool, "<direct rand! call>")
    return _rand_impl!(pool, S, d1, dims...)
end
@inline function rand!(pool::AbstractArrayPool, S::_SampleColl, dims::NTuple{N, Int}) where {N}
    _set_pending_callsite!(pool, "<direct rand! call>")
    return _rand_impl!(pool, S, dims)
end

# Internal implementations (called directly by macro-transformed code).
@inline function _rand_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _acquire_impl!(pool, T, dims...)
    Random.rand!(arr)
    return arr
end
@inline _rand_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N} = _rand_impl!(pool, default_eltype(pool), dims...)
@inline _rand_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = _rand_impl!(pool, T, dims...)
@inline _rand_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N} = _rand_impl!(pool, default_eltype(pool), dims...)

# tp-hoisted forms: macro-transformed code binds `tp = get_typed_pool!(pool, T)`
# once per scope and passes it here, skipping the per-acquire lookup.
@inline function _rand_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool, dims::Vararg{Int, N}) where {N}
    arr = _acquire_impl!(pool, tp, dims...)
    Random.rand!(arr)
    return arr
end
@inline _rand_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool, dims::NTuple{N, Int}) where {N} = _rand_impl!(pool, tp, dims...)
# Collection form self-records its eltype touch: the macro registers the wrong
# (default) type for `rand!(pool, S, dims)` because `S` is not a syntactic type,
# so this impl records `eltype(S)` itself to keep checkpoint/rewind correct.
@inline function _rand_impl!(pool::AbstractArrayPool, S::_SampleColl, d1::Int, dims::Vararg{Int, N}) where {N}
    T = eltype(S)
    _record_type_touch!(pool, T)
    arr = _acquire_impl!(pool, T, d1, dims...)
    Random.rand!(arr, S)
    return arr
end
@inline _rand_impl!(pool::AbstractArrayPool, S::_SampleColl, dims::NTuple{N, Int}) where {N} = _rand_impl!(pool, S, dims...)

"""
    randn!(pool, dims...) -> Array
    randn!(pool, T, dims...) -> Array
    randn!(pool, dims::Tuple) / randn!(pool, T, dims::Tuple)

Acquire a standard-normal `N(0, 1)` random array from the pool. `T` must be a
floating-point (or complex) type. Default element type depends on backend
(CPU: `Float64`). Consumes the global RNG identically to `Random.randn!`.

## Example
```julia
@with_pool pool begin
    g = randn!(pool, 100)              # Vector{Float64}, N(0,1)
    m = randn!(pool, Float32, 8, 8)    # Matrix{Float32}
end
```

See also: [`rand!`](@ref), [`zeros!`](@ref), [`acquire!`](@ref)
"""
@inline function randn!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct randn! call>")
    return _randn_impl!(pool, T, dims...)
end
@inline function randn!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct randn! call>")
    return _randn_impl!(pool, default_eltype(pool), dims...)
end
@inline function randn!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct randn! call>")
    return _randn_impl!(pool, T, dims...)
end
@inline function randn!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    _set_pending_callsite!(pool, "<direct randn! call>")
    return _randn_impl!(pool, default_eltype(pool), dims...)
end

@inline function _randn_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _acquire_impl!(pool, T, dims...)
    Random.randn!(arr)
    return arr
end
@inline _randn_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N} = _randn_impl!(pool, default_eltype(pool), dims...)
@inline _randn_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = _randn_impl!(pool, T, dims...)
@inline _randn_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N} = _randn_impl!(pool, default_eltype(pool), dims...)

# tp-hoisted forms: macro-transformed code binds `tp = get_typed_pool!(pool, T)`
# once per scope and passes it here, skipping the per-acquire lookup.
@inline function _randn_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool, dims::Vararg{Int, N}) where {N}
    arr = _acquire_impl!(pool, tp, dims...)
    Random.randn!(arr)
    return arr
end
@inline _randn_impl!(pool::AbstractArrayPool, tp::AbstractTypedPool, dims::NTuple{N, Int}) where {N} = _randn_impl!(pool, tp, dims...)

# ==============================================================================
# similar! - Acquire arrays with same type/size as template
# ==============================================================================

"""
    similar!(pool, array) -> Array
    similar!(pool, array, T) -> Array
    similar!(pool, array, dims...) -> Array
    similar!(pool, array, T, dims...) -> Array

Acquire an uninitialized array from the pool, using a template array for defaults.

- `similar!(pool, A)`: same element type and size as `A`
- `similar!(pool, A, T)`: element type `T`, same size as `A`
- `similar!(pool, A, dims...)`: same element type as `A`, specified dimensions
- `similar!(pool, A, T, dims...)`: element type `T`, specified dimensions

## Example
```julia
A = rand(10, 10)
@with_pool pool begin
    B = similar!(pool, A)              # Same type and size
    C = similar!(pool, A, Float32)     # Float32, same size
    D = similar!(pool, A, 5, 5)        # Same type, different size
    E = similar!(pool, A, Int, 20)     # Int, 1D
end
```

See also: [`zeros!`](@ref), [`ones!`](@ref), [`acquire!`](@ref)
"""
@inline function similar!(pool::AbstractArrayPool, x::AbstractArray)
    _record_type_touch!(pool, eltype(x))
    _set_pending_callsite!(pool, "<direct similar! call>")
    return _similar_impl!(pool, x)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct similar! call>")
    return _similar_impl!(pool, x, T)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, eltype(x))
    _set_pending_callsite!(pool, "<direct similar! call>")
    return _similar_impl!(pool, x, dims...)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    _set_pending_callsite!(pool, "<direct similar! call>")
    return _similar_impl!(pool, x, T, dims...)
end

# Internal implementation (for macro transformation)
@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray)
    return _acquire_impl!(pool, eltype(x), size(x))
end

@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    return _acquire_impl!(pool, T, size(x))
end

@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int, N}) where {N}
    return _acquire_impl!(pool, eltype(x), dims...)
end

@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    return _acquire_impl!(pool, T, dims...)
end

# ==============================================================================
# reshape! - Reshape arrays using pool's wrapper cache
# ==============================================================================

"""
    reshape!(pool, A, dims...) -> reshaped array
    reshape!(pool, A, dims::Tuple) -> reshaped array

Reshape array `A` to dimensions `dims` using the pool's wrapper cache.

The returned array shares memory with `A` — mutations are visible in both.
The pool provides cached wrapper objects to reduce allocation on repeated calls.

On Julia 1.12+:
- If `ndims(A) == length(dims)` (same dimensionality), `reshape!` mutates `A`
  in-place by changing its size. This differs from `Base.reshape`, which always
  returns a new wrapper.
- For cross-dimensional reshapes (`ndims(A) != length(dims)`), the returned
  `Array` wrapper is taken from the pool's internal cache and may be reused
  after `rewind!` or pool scope exit.

As with all pool-backed objects, the reshaped result must not escape the
surrounding `@with_pool` scope.

On Julia ≤1.11 and CUDA, falls back to `Base.reshape`.

Throws `DimensionMismatch` if `prod(dims) != length(A)`.

## Example
```julia
A = collect(1.0:12.0)
@with_pool pool begin
    B = reshape!(pool, A, 3, 4)   # 12-element vector → 3×4 matrix
    B[1,1] = 999.0                # A[1] is now 999.0
end
```

See also: [`acquire!`](@ref), [`similar!`](@ref)
"""
@inline function reshape!(pool::AbstractArrayPool, A::AbstractArray{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _reshape_impl!(pool, A, dims)
end

@inline function reshape!(pool::AbstractArrayPool, A::AbstractArray{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _reshape_impl!(pool, A, dims)
end

# Internal implementation (fallback: delegates to Base.reshape)
@inline function _reshape_impl!(::AbstractArrayPool, A::AbstractArray, dims::NTuple{N, Int}) where {N}
    for d in dims
        d < 0 && throw(ArgumentError("invalid Array dimensions"))
    end
    return reshape(A, dims)
end

# Vararg forwarding (macro transforms reshape!(pool, A, 3, 4) → _reshape_impl!(pool, A, 3, 4))
@inline _reshape_impl!(pool::AbstractArrayPool, A::AbstractArray, dims::Vararg{Int, N}) where {N} =
    _reshape_impl!(pool, A, dims)


# ==============================================================================
# BackendNotLoadedError - Error for unknown backends
# ==============================================================================

"""
    BackendNotLoadedError <: Exception

Error thrown when a backend-specific operation is attempted but the backend
package is not loaded.

## Example
```julia
@maybe_with_pool :cuda pool begin
    zeros!(pool, 10)  # Throws if CUDA.jl not loaded
end
```
"""
struct BackendNotLoadedError <: Exception
    backend::Symbol
end

function Base.showerror(io::IO, e::BackendNotLoadedError)
    print(io, "Backend :$(e.backend) is not available. ")
    return if e.backend == :cuda
        print(io, "Make sure CUDA.jl is loaded: `using CUDA`")
    else
        print(io, "Make sure the appropriate backend package is loaded.")
    end
end

# ==============================================================================
# DisabledPool Fallbacks (pooling disabled with backend context)
# ==============================================================================

# --- Default Element Type ---
"""
    default_eltype(::DisabledPool{:cpu}) -> Float64

Default element type for disabled CPU pools (matches Julia's `zeros()` default).
"""
default_eltype(::DisabledPool{:cpu}) = Float64

# --- Generic Backend Fallback (throws error) ---
# Catches DisabledPool{:unknown_backend} and similar unhandled backends
@noinline function _throw_backend_not_loaded(backend::Symbol)
    throw(BackendNotLoadedError(backend))
end

# --- zeros! for DisabledPool{:cpu} ---
@inline zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = zeros(T, dims...)
@inline zeros!(p::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = zeros(default_eltype(p), dims...)
@inline zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = zeros(T, dims...)
@inline zeros!(p::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = zeros(default_eltype(p), dims...)

# --- ones! for DisabledPool{:cpu} ---
@inline ones!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = ones(T, dims...)
@inline ones!(p::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = ones(default_eltype(p), dims...)
@inline ones!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = ones(T, dims...)
@inline ones!(p::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = ones(default_eltype(p), dims...)

# --- zeros!/ones! for DisabledPool{:cpu} with Bit type (returns BitArray) ---
@inline zeros!(::DisabledPool{:cpu}, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = falses(dims...)
@inline zeros!(::DisabledPool{:cpu}, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = falses(dims...)
@inline ones!(::DisabledPool{:cpu}, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = trues(dims...)
@inline ones!(::DisabledPool{:cpu}, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = trues(dims...)

# --- trues!/falses! for DisabledPool{:cpu} ---
@inline trues!(::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = trues(dims...)
@inline trues!(::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = trues(dims...)
@inline falses!(::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = falses(dims...)
@inline falses!(::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = falses(dims...)

# --- rand!/randn! for DisabledPool{:cpu} (allocating, plain Array) ---
@inline rand!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = rand(T, dims...)
@inline rand!(p::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = rand(default_eltype(p), dims...)
@inline rand!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = rand(T, dims...)
@inline rand!(p::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = rand(default_eltype(p), dims...)
@inline rand!(::DisabledPool{:cpu}, S::_SampleColl, d1::Int, dims::Vararg{Int, N}) where {N} = rand(S, d1, dims...)
@inline rand!(::DisabledPool{:cpu}, S::_SampleColl, dims::NTuple{N, Int}) where {N} = rand(S, dims)

@inline randn!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Random.randn(T, dims...)
@inline randn!(p::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = Random.randn(default_eltype(p), dims...)
@inline randn!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Random.randn(T, dims...)
@inline randn!(p::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = Random.randn(default_eltype(p), dims...)

# --- similar! for DisabledPool{:cpu} ---
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, dims::Vararg{Int, N}) where {N} = similar(x, dims...)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = similar(x, T, dims...)

# --- reshape! for DisabledPool{:cpu} ---
@inline reshape!(::DisabledPool{:cpu}, A::AbstractArray, dims::Vararg{Int, N}) where {N} = reshape(A, dims...)
@inline reshape!(::DisabledPool{:cpu}, A::AbstractArray, dims::NTuple{N, Int}) where {N} = reshape(A, dims)
