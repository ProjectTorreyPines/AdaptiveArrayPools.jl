# ==============================================================================
# CUDA Default Element Type
# ==============================================================================
# CUDA pools default to Float32 (matching CUDA.zeros() behavior).
# All convenience functions (zeros!, ones!, etc.) dispatch through _*_impl!
# which calls default_eltype(pool) for the default type.

"""
    default_eltype(::CuAdaptiveArrayPool) -> Type

Returns `Float32` as the default element type for CUDA pools.
This matches `CUDA.zeros()` behavior.
"""
AdaptiveArrayPools.default_eltype(::CuAdaptiveArrayPool) = Float32

# ==============================================================================
# DisabledPool{:cuda} Fallbacks
# ==============================================================================
# When pooling is disabled but :cuda backend is specified, these methods ensure
# proper CuArray allocation instead of falling back to CPU arrays.

using AdaptiveArrayPools: DisabledPool

"""
    DISABLED_CUDA

Singleton instance for disabled CUDA pooling.
Used by macros when `USE_POOLING=false` with `:cuda` backend.
"""
const DISABLED_CUDA = DisabledPool{:cuda}()

"""
    default_eltype(::DisabledPool{:cuda}) -> Float32

Default element type for disabled CUDA pools (matches CUDA.zeros() default).
"""
AdaptiveArrayPools.default_eltype(::DisabledPool{:cuda}) = Float32

# --- zeros! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.zeros!(::DisabledPool{:cuda}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CUDA.zeros(T, dims...)
@inline AdaptiveArrayPools.zeros!(p::DisabledPool{:cuda}, dims::Vararg{Int,N}) where {N} = CUDA.zeros(AdaptiveArrayPools.default_eltype(p), dims...)
@inline AdaptiveArrayPools.zeros!(::DisabledPool{:cuda}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CUDA.zeros(T, dims...)
@inline AdaptiveArrayPools.zeros!(p::DisabledPool{:cuda}, dims::NTuple{N,Int}) where {N} = CUDA.zeros(AdaptiveArrayPools.default_eltype(p), dims...)

# --- ones! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.ones!(::DisabledPool{:cuda}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CUDA.ones(T, dims...)
@inline AdaptiveArrayPools.ones!(p::DisabledPool{:cuda}, dims::Vararg{Int,N}) where {N} = CUDA.ones(AdaptiveArrayPools.default_eltype(p), dims...)
@inline AdaptiveArrayPools.ones!(::DisabledPool{:cuda}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CUDA.ones(T, dims...)
@inline AdaptiveArrayPools.ones!(p::DisabledPool{:cuda}, dims::NTuple{N,Int}) where {N} = CUDA.ones(AdaptiveArrayPools.default_eltype(p), dims...)

# --- similar! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::CuArray) = CUDA.similar(x)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::CuArray, ::Type{T}) where {T} = CUDA.similar(x, T)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::CuArray, dims::Vararg{Int,N}) where {N} = CUDA.similar(x, dims...)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::CuArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CUDA.similar(x, T, dims...)
# Fallback for non-CuArray inputs (creates CuArray from AbstractArray)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::AbstractArray) = CuArray{eltype(x)}(undef, size(x))
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::AbstractArray, ::Type{T}) where {T} = CuArray{T}(undef, size(x))
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::AbstractArray, dims::Vararg{Int,N}) where {N} = CuArray{eltype(x)}(undef, dims)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:cuda}, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CuArray{T}(undef, dims)

# --- unsafe_zeros! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.unsafe_zeros!(::DisabledPool{:cuda}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CUDA.zeros(T, dims...)
@inline AdaptiveArrayPools.unsafe_zeros!(p::DisabledPool{:cuda}, dims::Vararg{Int,N}) where {N} = CUDA.zeros(AdaptiveArrayPools.default_eltype(p), dims...)
@inline AdaptiveArrayPools.unsafe_zeros!(::DisabledPool{:cuda}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CUDA.zeros(T, dims...)
@inline AdaptiveArrayPools.unsafe_zeros!(p::DisabledPool{:cuda}, dims::NTuple{N,Int}) where {N} = CUDA.zeros(AdaptiveArrayPools.default_eltype(p), dims...)

# --- unsafe_ones! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.unsafe_ones!(::DisabledPool{:cuda}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CUDA.ones(T, dims...)
@inline AdaptiveArrayPools.unsafe_ones!(p::DisabledPool{:cuda}, dims::Vararg{Int,N}) where {N} = CUDA.ones(AdaptiveArrayPools.default_eltype(p), dims...)
@inline AdaptiveArrayPools.unsafe_ones!(::DisabledPool{:cuda}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CUDA.ones(T, dims...)
@inline AdaptiveArrayPools.unsafe_ones!(p::DisabledPool{:cuda}, dims::NTuple{N,Int}) where {N} = CUDA.ones(AdaptiveArrayPools.default_eltype(p), dims...)

# --- unsafe_similar! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::CuArray) = CUDA.similar(x)
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::CuArray, ::Type{T}) where {T} = CUDA.similar(x, T)
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::CuArray, dims::Vararg{Int,N}) where {N} = CUDA.similar(x, dims...)
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::CuArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CUDA.similar(x, T, dims...)
# Fallback for non-CuArray inputs
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::AbstractArray) = CuArray{eltype(x)}(undef, size(x))
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::AbstractArray, ::Type{T}) where {T} = CuArray{T}(undef, size(x))
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::AbstractArray, dims::Vararg{Int,N}) where {N} = CuArray{eltype(x)}(undef, dims)
@inline AdaptiveArrayPools.unsafe_similar!(::DisabledPool{:cuda}, x::AbstractArray, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CuArray{T}(undef, dims)

# --- acquire! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:cuda}, ::Type{T}, n::Int) where {T} = CuVector{T}(undef, n)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:cuda}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CuArray{T,N}(undef, dims)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:cuda}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CuArray{T,N}(undef, dims)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:cuda}, x::CuArray) = CUDA.similar(x)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:cuda}, x::AbstractArray) = CuArray{eltype(x)}(undef, size(x))

# --- unsafe_acquire! for DisabledPool{:cuda} ---
@inline AdaptiveArrayPools.unsafe_acquire!(::DisabledPool{:cuda}, ::Type{T}, n::Int) where {T} = CuVector{T}(undef, n)
@inline AdaptiveArrayPools.unsafe_acquire!(::DisabledPool{:cuda}, ::Type{T}, dims::Vararg{Int,N}) where {T,N} = CuArray{T,N}(undef, dims)
@inline AdaptiveArrayPools.unsafe_acquire!(::DisabledPool{:cuda}, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CuArray{T,N}(undef, dims)
@inline AdaptiveArrayPools.unsafe_acquire!(::DisabledPool{:cuda}, x::CuArray) = CUDA.similar(x)
@inline AdaptiveArrayPools.unsafe_acquire!(::DisabledPool{:cuda}, x::AbstractArray) = CuArray{eltype(x)}(undef, size(x))
