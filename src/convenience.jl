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
    zeros!(pool, dims...) -> view
    zeros!(pool, T, dims...) -> view
    zeros!(pool, dims::Tuple) -> view
    zeros!(pool, T, dims::Tuple) -> view

Acquire a zero-initialized array from the pool.

Equivalent to `acquire!(pool, T, dims...)` followed by `fill!(arr, zero(T))`.
Default element type depends on pool backend (CPU: `Float64`, CUDA: `Float32`).
See [`default_eltype`](@ref).

## Example
```julia
@with_pool pool begin
    v = zeros!(pool, 100)              # Uses default_eltype(pool)
    m = zeros!(pool, Float32, 10, 10)  # Matrix{Float32} view, all zeros
end
```

See also: [`ones!`](@ref), [`similar!`](@ref), [`acquire!`](@ref)
"""
@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    return _zeros_impl!(pool, default_eltype(pool), dims...)
end

@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
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

# Bit type specialization: zeros!(pool, Bit, ...) delegates to falses!(pool, ...)
@inline zeros!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = falses!(pool, dims...)
@inline zeros!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = falses!(pool, dims)
@inline _zeros_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::Vararg{Int, N}) where {N} = _falses_impl!(pool, dims...)
@inline _zeros_impl!(pool::AbstractArrayPool, ::Type{Bit}, dims::NTuple{N, Int}) where {N} = _falses_impl!(pool, dims)

# ==============================================================================
# ones! - Acquire one-initialized arrays from pool
# ==============================================================================

"""
    ones!(pool, dims...) -> view
    ones!(pool, T, dims...) -> view
    ones!(pool, dims::Tuple) -> view
    ones!(pool, T, dims::Tuple) -> view

Acquire a one-initialized array from the pool.

Equivalent to `acquire!(pool, T, dims...)` followed by `fill!(arr, one(T))`.
Default element type depends on pool backend (CPU: `Float64`, CUDA: `Float32`).
See [`default_eltype`](@ref).

## Example
```julia
@with_pool pool begin
    v = ones!(pool, 100)              # Uses default_eltype(pool)
    m = ones!(pool, Float32, 10, 10)  # Matrix{Float32} view, all ones
end
```

See also: [`zeros!`](@ref), [`similar!`](@ref), [`acquire!`](@ref)
"""
@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    return _ones_impl!(pool, default_eltype(pool), dims...)
end

@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
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
    return _trues_impl!(pool, dims...)
end
@inline function trues!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, Bit)
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
    return _falses_impl!(pool, dims...)
end
@inline function falses!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, Bit)
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
# similar! - Acquire arrays with same type/size as template
# ==============================================================================

"""
    similar!(pool, array) -> view
    similar!(pool, array, T) -> view
    similar!(pool, array, dims...) -> view
    similar!(pool, array, T, dims...) -> view

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
    return _similar_impl!(pool, x)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _record_type_touch!(pool, T)
    return _similar_impl!(pool, x, T)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, eltype(x))
    return _similar_impl!(pool, x, dims...)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
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

On Julia 1.11+:
- If `ndims(A) == length(dims)` (same dimensionality), `reshape!` mutates `A`
  in-place by changing its size. This differs from `Base.reshape`, which always
  returns a new wrapper.
- For cross-dimensional reshapes (`ndims(A) != length(dims)`), the returned
  `Array` wrapper is taken from the pool's internal cache and may be reused
  after `rewind!` or pool scope exit.

As with all pool-backed objects, the reshaped result must not escape the
surrounding `@with_pool` scope.

On Julia 1.10 and CUDA, falls back to `Base.reshape`.

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
# unsafe_zeros! - Acquire zero-initialized raw arrays from pool
# ==============================================================================

"""
    unsafe_zeros!(pool, dims...) -> Array
    unsafe_zeros!(pool, T, dims...) -> Array
    unsafe_zeros!(pool, dims::Tuple) -> Array
    unsafe_zeros!(pool, T, dims::Tuple) -> Array

Acquire a zero-initialized raw array (not a view) from the pool.

Equivalent to `unsafe_acquire!(pool, T, dims...)` followed by `fill!(arr, zero(T))`.
Default element type depends on pool backend (CPU: `Float64`, CUDA: `Float32`).
See [`default_eltype`](@ref).

## Example
```julia
@with_pool pool begin
    v = unsafe_zeros!(pool, 100)              # Uses default_eltype(pool)
    m = unsafe_zeros!(pool, Float32, 10, 10)  # Array{Float32}, all zeros
end
```

See also: [`unsafe_ones!`](@ref), [`zeros!`](@ref), [`unsafe_acquire!`](@ref)
"""
@inline function unsafe_zeros!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_zeros_impl!(pool, T, dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    return _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_zeros_impl!(pool, T, dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    return _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _unsafe_zeros_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _unsafe_acquire_impl!(pool, T, dims...)
    fill!(arr, zero(T))
    return arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _unsafe_zeros_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    return _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
end

# NTuple overloads for macro transformation (handles unsafe_zeros!(pool, T, size(x)) form)
@inline function _unsafe_zeros_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _unsafe_zeros_impl!(pool, T, dims...)
end

@inline function _unsafe_zeros_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    return _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
end

# ==============================================================================
# unsafe_ones! - Acquire one-initialized raw arrays from pool
# ==============================================================================

"""
    unsafe_ones!(pool, dims...) -> Array
    unsafe_ones!(pool, T, dims...) -> Array
    unsafe_ones!(pool, dims::Tuple) -> Array
    unsafe_ones!(pool, T, dims::Tuple) -> Array

Acquire a one-initialized raw array (not a view) from the pool.

Equivalent to `unsafe_acquire!(pool, T, dims...)` followed by `fill!(arr, one(T))`.
Default element type depends on pool backend (CPU: `Float64`, CUDA: `Float32`).
See [`default_eltype`](@ref).

## Example
```julia
@with_pool pool begin
    v = unsafe_ones!(pool, 100)              # Uses default_eltype(pool)
    m = unsafe_ones!(pool, Float32, 10, 10)  # Array{Float32}, all ones
end
```

See also: [`unsafe_zeros!`](@ref), [`ones!`](@ref), [`unsafe_acquire!`](@ref)
"""
@inline function unsafe_ones!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_ones_impl!(pool, T, dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    return _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_ones_impl!(pool, T, dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    _record_type_touch!(pool, default_eltype(pool))
    return _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _unsafe_ones_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    arr = _unsafe_acquire_impl!(pool, T, dims...)
    fill!(arr, one(T))
    return arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _unsafe_ones_impl!(pool::AbstractArrayPool, dims::Vararg{Int, N}) where {N}
    return _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
end

# NTuple overloads for macro transformation (handles unsafe_ones!(pool, T, size(x)) form)
@inline function _unsafe_ones_impl!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    return _unsafe_ones_impl!(pool, T, dims...)
end

@inline function _unsafe_ones_impl!(pool::AbstractArrayPool, dims::NTuple{N, Int}) where {N}
    return _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
end

# ==============================================================================
# unsafe_similar! - Acquire raw arrays with same type/size as template
# ==============================================================================

"""
    unsafe_similar!(pool, array) -> Array
    unsafe_similar!(pool, array, T) -> Array
    unsafe_similar!(pool, array, dims...) -> Array
    unsafe_similar!(pool, array, T, dims...) -> Array

Acquire an uninitialized raw array (not a view) from the pool, using a template array for defaults.

- `unsafe_similar!(pool, A)`: same element type and size as `A`
- `unsafe_similar!(pool, A, T)`: element type `T`, same size as `A`
- `unsafe_similar!(pool, A, dims...)`: same element type as `A`, specified dimensions
- `unsafe_similar!(pool, A, T, dims...)`: element type `T`, specified dimensions

## Example
```julia
A = rand(10, 10)
@with_pool pool begin
    B = unsafe_similar!(pool, A)              # Same type and size, raw array
    C = unsafe_similar!(pool, A, Float32)     # Float32, same size
    D = unsafe_similar!(pool, A, 5, 5)        # Same type, different size
end
```

See also: [`similar!`](@ref), [`unsafe_acquire!`](@ref)
"""
@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray)
    _record_type_touch!(pool, eltype(x))
    return _unsafe_similar_impl!(pool, x)
end

@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _record_type_touch!(pool, T)
    return _unsafe_similar_impl!(pool, x, T)
end

@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int, N}) where {N}
    _record_type_touch!(pool, eltype(x))
    return _unsafe_similar_impl!(pool, x, dims...)
end

@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    _record_type_touch!(pool, T)
    return _unsafe_similar_impl!(pool, x, T, dims...)
end

# Internal implementation (for macro transformation)
@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray)
    return _unsafe_acquire_impl!(pool, eltype(x), size(x))
end

@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    return _unsafe_acquire_impl!(pool, T, size(x))
end

@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int, N}) where {N}
    return _unsafe_acquire_impl!(pool, eltype(x), dims...)
end

@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    return _unsafe_acquire_impl!(pool, T, dims...)
end

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

# --- similar! for DisabledPool{:cpu} ---
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, dims::Vararg{Int, N}) where {N} = similar(x, dims...)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = similar(x, T, dims...)

# --- reshape! for DisabledPool{:cpu} ---
@inline reshape!(::DisabledPool{:cpu}, A::AbstractArray, dims::Vararg{Int, N}) where {N} = reshape(A, dims...)
@inline reshape!(::DisabledPool{:cpu}, A::AbstractArray, dims::NTuple{N, Int}) where {N} = reshape(A, dims)

# --- unsafe_zeros! for DisabledPool{:cpu} ---
@inline unsafe_zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = zeros(T, dims...)
@inline unsafe_zeros!(p::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = zeros(default_eltype(p), dims...)
@inline unsafe_zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = zeros(T, dims...)
@inline unsafe_zeros!(p::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = zeros(default_eltype(p), dims...)

# --- unsafe_ones! for DisabledPool{:cpu} ---
@inline unsafe_ones!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = ones(T, dims...)
@inline unsafe_ones!(p::DisabledPool{:cpu}, dims::Vararg{Int, N}) where {N} = ones(default_eltype(p), dims...)
@inline unsafe_ones!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = ones(T, dims...)
@inline unsafe_ones!(p::DisabledPool{:cpu}, dims::NTuple{N, Int}) where {N} = ones(default_eltype(p), dims...)

# --- unsafe_similar! for DisabledPool{:cpu} ---
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray, dims::Vararg{Int, N}) where {N} = similar(x, dims...)
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = similar(x, T, dims...)
