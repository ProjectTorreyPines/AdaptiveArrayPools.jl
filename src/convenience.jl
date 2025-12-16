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
@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _mark_untracked!(pool)
    _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _mark_untracked!(pool)
    _zeros_impl!(pool, default_eltype(pool), dims...)
end

@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _zeros_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _zeros_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    arr = _acquire_impl!(pool, T, dims...)
    fill!(arr, zero(T))
    arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _zeros_impl!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _zeros_impl!(pool, default_eltype(pool), dims...)
end

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
@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _mark_untracked!(pool)
    _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _mark_untracked!(pool)
    _ones_impl!(pool, default_eltype(pool), dims...)
end

@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _ones_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _ones_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    arr = _acquire_impl!(pool, T, dims...)
    fill!(arr, one(T))
    arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _ones_impl!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _ones_impl!(pool, default_eltype(pool), dims...)
end

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
    _mark_untracked!(pool)
    _similar_impl!(pool, x)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _mark_untracked!(pool)
    _similar_impl!(pool, x, T)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int,N}) where {N}
    _mark_untracked!(pool)
    _similar_impl!(pool, x, dims...)
end

@inline function similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _mark_untracked!(pool)
    _similar_impl!(pool, x, T, dims...)
end

# Internal implementation (for macro transformation)
@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray)
    _acquire_impl!(pool, eltype(x), size(x))
end

@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _acquire_impl!(pool, T, size(x))
end

@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int,N}) where {N}
    _acquire_impl!(pool, eltype(x), dims...)
end

@inline function _similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _acquire_impl!(pool, T, dims...)
end

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
@inline function unsafe_zeros!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_zeros_impl!(pool, T, dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _mark_untracked!(pool)
    _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_zeros_impl!(pool, T, dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _unsafe_zeros_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    arr = _unsafe_acquire_impl!(pool, T, dims...)
    fill!(arr, zero(T))
    arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _unsafe_zeros_impl!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _unsafe_zeros_impl!(pool, default_eltype(pool), dims...)
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
@inline function unsafe_ones!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_ones_impl!(pool, T, dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _mark_untracked!(pool)
    _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_ones_impl!(pool, T, dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
end

# Internal implementation (for macro transformation)
@inline function _unsafe_ones_impl!(pool::AbstractArrayPool, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    arr = _unsafe_acquire_impl!(pool, T, dims...)
    fill!(arr, one(T))
    arr
end

# Default type overload for macro transformation (uses default_eltype for backend flexibility)
@inline function _unsafe_ones_impl!(pool::AbstractArrayPool, dims::Vararg{Int,N}) where {N}
    _unsafe_ones_impl!(pool, default_eltype(pool), dims...)
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
    _mark_untracked!(pool)
    _unsafe_similar_impl!(pool, x)
end

@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _mark_untracked!(pool)
    _unsafe_similar_impl!(pool, x, T)
end

@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int,N}) where {N}
    _mark_untracked!(pool)
    _unsafe_similar_impl!(pool, x, dims...)
end

@inline function unsafe_similar!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_similar_impl!(pool, x, T, dims...)
end

# Internal implementation (for macro transformation)
@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray)
    _unsafe_acquire_impl!(pool, eltype(x), size(x))
end

@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}) where {T}
    _unsafe_acquire_impl!(pool, T, size(x))
end

@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray, dims::Vararg{Int,N}) where {N}
    _unsafe_acquire_impl!(pool, eltype(x), dims...)
end

@inline function _unsafe_similar_impl!(pool::AbstractArrayPool, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N}
    _unsafe_acquire_impl!(pool, T, dims...)
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
    if e.backend == :cuda
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
@inline zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = zeros(T, dims...)
@inline zeros!(p::DisabledPool{:cpu}, dims::Vararg{Int,N}) where {N} = zeros(default_eltype(p), dims...)
@inline zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = zeros(T, dims...)
@inline zeros!(p::DisabledPool{:cpu}, dims::NTuple{N,Int}) where {N} = zeros(default_eltype(p), dims...)

# --- ones! for DisabledPool{:cpu} ---
@inline ones!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = ones(T, dims...)
@inline ones!(p::DisabledPool{:cpu}, dims::Vararg{Int,N}) where {N} = ones(default_eltype(p), dims...)
@inline ones!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = ones(T, dims...)
@inline ones!(p::DisabledPool{:cpu}, dims::NTuple{N,Int}) where {N} = ones(default_eltype(p), dims...)

# --- similar! for DisabledPool{:cpu} ---
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, dims::Vararg{Int,N}) where {N} = similar(x, dims...)
@inline similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = similar(x, T, dims...)

# --- unsafe_zeros! for DisabledPool{:cpu} ---
@inline unsafe_zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = zeros(T, dims...)
@inline unsafe_zeros!(p::DisabledPool{:cpu}, dims::Vararg{Int,N}) where {N} = zeros(default_eltype(p), dims...)
@inline unsafe_zeros!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = zeros(T, dims...)
@inline unsafe_zeros!(p::DisabledPool{:cpu}, dims::NTuple{N,Int}) where {N} = zeros(default_eltype(p), dims...)

# --- unsafe_ones! for DisabledPool{:cpu} ---
@inline unsafe_ones!(::DisabledPool{:cpu}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = ones(T, dims...)
@inline unsafe_ones!(p::DisabledPool{:cpu}, dims::Vararg{Int,N}) where {N} = ones(default_eltype(p), dims...)
@inline unsafe_ones!(::DisabledPool{:cpu}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = ones(T, dims...)
@inline unsafe_ones!(p::DisabledPool{:cpu}, dims::NTuple{N,Int}) where {N} = ones(default_eltype(p), dims...)

# --- unsafe_similar! for DisabledPool{:cpu} ---
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray) = similar(x)
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray, dims::Vararg{Int,N}) where {N} = similar(x, dims...)
@inline unsafe_similar!(::DisabledPool{:cpu}, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = similar(x, T, dims...)

# --- Generic DisabledPool fallbacks (unknown backend → error) ---
@inline zeros!(p::DisabledPool{B}, args...) where {B} = _throw_backend_not_loaded(B)
@inline ones!(p::DisabledPool{B}, args...) where {B} = _throw_backend_not_loaded(B)
@inline similar!(p::DisabledPool{B}, args...) where {B} = _throw_backend_not_loaded(B)
@inline unsafe_zeros!(p::DisabledPool{B}, args...) where {B} = _throw_backend_not_loaded(B)
@inline unsafe_ones!(p::DisabledPool{B}, args...) where {B} = _throw_backend_not_loaded(B)
@inline unsafe_similar!(p::DisabledPool{B}, args...) where {B} = _throw_backend_not_loaded(B)

# ==============================================================================
# _impl! Delegators for DisabledPool
# ==============================================================================
# When macros transform zeros!(pool, ...) → _zeros_impl!(pool, ...),
# DisabledPool needs to delegate back to the public API.

@inline _zeros_impl!(p::DisabledPool, args...) = zeros!(p, args...)
@inline _ones_impl!(p::DisabledPool, args...) = ones!(p, args...)
@inline _similar_impl!(p::DisabledPool, args...) = similar!(p, args...)
@inline _unsafe_zeros_impl!(p::DisabledPool, args...) = unsafe_zeros!(p, args...)
@inline _unsafe_ones_impl!(p::DisabledPool, args...) = unsafe_ones!(p, args...)
@inline _unsafe_similar_impl!(p::DisabledPool, args...) = unsafe_similar!(p, args...)
