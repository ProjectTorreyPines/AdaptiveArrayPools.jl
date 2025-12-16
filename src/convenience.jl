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
Default element type is `Float64` when not specified.

## Example
```julia
@with_pool pool begin
    v = zeros!(pool, 100)              # Vector{Float64} view, all zeros
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
    _zeros_impl!(pool, Float64, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _zeros_impl!(pool, T, dims...)
end

@inline function zeros!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _zeros_impl!(pool, Float64, dims...)
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

# Nothing fallback (pooling disabled)
@inline zeros!(::Nothing, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = zeros(T, dims...)
@inline zeros!(::Nothing, dims::Vararg{Int,N}) where {N} = zeros(Float64, dims...)
@inline zeros!(::Nothing, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = zeros(T, dims...)
@inline zeros!(::Nothing, dims::NTuple{N,Int}) where {N} = zeros(Float64, dims...)

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
Default element type is `Float64` when not specified.

## Example
```julia
@with_pool pool begin
    v = ones!(pool, 100)              # Vector{Float64} view, all ones
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
    _ones_impl!(pool, Float64, dims...)
end

@inline function ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _ones_impl!(pool, T, dims...)
end

@inline function ones!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _ones_impl!(pool, Float64, dims...)
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

# Nothing fallback (pooling disabled)
@inline ones!(::Nothing, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = ones(T, dims...)
@inline ones!(::Nothing, dims::Vararg{Int,N}) where {N} = ones(Float64, dims...)
@inline ones!(::Nothing, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = ones(T, dims...)
@inline ones!(::Nothing, dims::NTuple{N,Int}) where {N} = ones(Float64, dims...)

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

# Nothing fallback (pooling disabled)
@inline similar!(::Nothing, x::AbstractArray) = similar(x)
@inline similar!(::Nothing, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline similar!(::Nothing, x::AbstractArray, dims::Vararg{Int,N}) where {N} = similar(x, dims...)
@inline similar!(::Nothing, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = similar(x, T, dims...)

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
Default element type is `Float64` when not specified.

## Example
```julia
@with_pool pool begin
    v = unsafe_zeros!(pool, 100)              # Array{Float64}, all zeros
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
    _unsafe_zeros_impl!(pool, Float64, dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_zeros_impl!(pool, T, dims...)
end

@inline function unsafe_zeros!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _unsafe_zeros_impl!(pool, Float64, dims...)
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

# Nothing fallback (pooling disabled)
@inline unsafe_zeros!(::Nothing, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = zeros(T, dims...)
@inline unsafe_zeros!(::Nothing, dims::Vararg{Int,N}) where {N} = zeros(Float64, dims...)
@inline unsafe_zeros!(::Nothing, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = zeros(T, dims...)
@inline unsafe_zeros!(::Nothing, dims::NTuple{N,Int}) where {N} = zeros(Float64, dims...)

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
Default element type is `Float64` when not specified.

## Example
```julia
@with_pool pool begin
    v = unsafe_ones!(pool, 100)              # Array{Float64}, all ones
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
    _unsafe_ones_impl!(pool, Float64, dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _mark_untracked!(pool)
    _unsafe_ones_impl!(pool, T, dims...)
end

@inline function unsafe_ones!(pool::AbstractArrayPool, dims::NTuple{N,Int}) where {N}
    _mark_untracked!(pool)
    _unsafe_ones_impl!(pool, Float64, dims...)
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

# Nothing fallback (pooling disabled)
@inline unsafe_ones!(::Nothing, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = ones(T, dims...)
@inline unsafe_ones!(::Nothing, dims::Vararg{Int,N}) where {N} = ones(Float64, dims...)
@inline unsafe_ones!(::Nothing, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = ones(T, dims...)
@inline unsafe_ones!(::Nothing, dims::NTuple{N,Int}) where {N} = ones(Float64, dims...)

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

# Nothing fallback (pooling disabled)
@inline unsafe_similar!(::Nothing, x::AbstractArray) = similar(x)
@inline unsafe_similar!(::Nothing, x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
@inline unsafe_similar!(::Nothing, x::AbstractArray, dims::Vararg{Int,N}) where {N} = similar(x, dims...)
@inline unsafe_similar!(::Nothing, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = similar(x, T, dims...)
