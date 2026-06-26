# ==============================================================================
# Metal Default Element Type
# ==============================================================================
# Metal pools default to Float32 (matching Metal.zeros() behavior).
# All convenience functions (zeros!, ones!, etc.) dispatch through _*_impl!
# which calls default_eltype(pool) for the default type.

"""
    default_eltype(::MetalAdaptiveArrayPool) -> Type

Returns `Float32` as the default element type for Metal pools.
This matches Metal GPU convention.
"""
AdaptiveArrayPools.default_eltype(::MetalAdaptiveArrayPool) = Float32

# ==============================================================================
# DisabledPool{:metal} Fallbacks
# ==============================================================================
# When pooling is disabled but :metal backend is specified, these methods ensure
# proper MtlArray allocation instead of falling back to CPU arrays.

using AdaptiveArrayPools: DisabledPool

"""
    DISABLED_METAL

Singleton instance for disabled Metal pooling.
Used by macros when `STATIC_POOLING=false` with `:metal` backend.
"""
const DISABLED_METAL = DisabledPool{:metal}()

"""
    default_eltype(::DisabledPool{:metal}) -> Float32

Default element type for disabled Metal pools (matches Metal convention).
"""
AdaptiveArrayPools.default_eltype(::DisabledPool{:metal}) = Float32

# --- zeros! for DisabledPool{:metal} ---
@inline AdaptiveArrayPools.zeros!(::DisabledPool{:metal}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = MtlArray(zeros(T, dims...))
@inline AdaptiveArrayPools.zeros!(p::DisabledPool{:metal}, dims::Vararg{Int, N}) where {N} = MtlArray(zeros(AdaptiveArrayPools.default_eltype(p), dims...))
@inline AdaptiveArrayPools.zeros!(::DisabledPool{:metal}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = MtlArray(zeros(T, dims...))
@inline AdaptiveArrayPools.zeros!(p::DisabledPool{:metal}, dims::NTuple{N, Int}) where {N} = MtlArray(zeros(AdaptiveArrayPools.default_eltype(p), dims...))

# --- ones! for DisabledPool{:metal} ---
@inline AdaptiveArrayPools.ones!(::DisabledPool{:metal}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = MtlArray(ones(T, dims...))
@inline AdaptiveArrayPools.ones!(p::DisabledPool{:metal}, dims::Vararg{Int, N}) where {N} = MtlArray(ones(AdaptiveArrayPools.default_eltype(p), dims...))
@inline AdaptiveArrayPools.ones!(::DisabledPool{:metal}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = MtlArray(ones(T, dims...))
@inline AdaptiveArrayPools.ones!(p::DisabledPool{:metal}, dims::NTuple{N, Int}) where {N} = MtlArray(ones(AdaptiveArrayPools.default_eltype(p), dims...))

# --- similar! for DisabledPool{:metal} ---
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::MtlArray) = Metal.similar(x)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::MtlArray, ::Type{T}) where {T} = Metal.similar(x, T)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::MtlArray, dims::Vararg{Int, N}) where {N} = Metal.similar(x, dims...)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::MtlArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Metal.similar(x, T, dims...)
# Fallback for non-MtlArray inputs (creates MtlArray from AbstractArray)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::AbstractArray) = MtlArray{eltype(x)}(undef, size(x))
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::AbstractArray, ::Type{T}) where {T} = MtlArray{T}(undef, size(x))
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::AbstractArray, dims::Vararg{Int, N}) where {N} = MtlArray{eltype(x)}(undef, dims)
@inline AdaptiveArrayPools.similar!(::DisabledPool{:metal}, x::AbstractArray, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = MtlArray{T}(undef, dims)

# --- reshape! for DisabledPool{:metal} ---
@inline AdaptiveArrayPools.reshape!(::DisabledPool{:metal}, A::AbstractArray, dims::Vararg{Int, N}) where {N} = reshape(A, dims...)
@inline AdaptiveArrayPools.reshape!(::DisabledPool{:metal}, A::AbstractArray, dims::NTuple{N, Int}) where {N} = reshape(A, dims)

# --- acquire! for DisabledPool{:metal} ---
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:metal}, ::Type{T}, n::Int) where {T} = MtlVector{T}(undef, n)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:metal}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = MtlArray{T, N}(undef, dims)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:metal}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = MtlArray{T, N}(undef, dims)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:metal}, x::MtlArray) = Metal.similar(x)
@inline AdaptiveArrayPools.acquire!(::DisabledPool{:metal}, x::AbstractArray) = MtlArray{eltype(x)}(undef, size(x))

# --- acquire_view! for DisabledPool{:metal} (no view distinction on GPU) ---
@inline AdaptiveArrayPools.acquire_view!(::DisabledPool{:metal}, ::Type{T}, n::Int) where {T} = MtlVector{T}(undef, n)
@inline AdaptiveArrayPools.acquire_view!(::DisabledPool{:metal}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = MtlArray{T, N}(undef, dims)
@inline AdaptiveArrayPools.acquire_view!(::DisabledPool{:metal}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = MtlArray{T, N}(undef, dims)
@inline AdaptiveArrayPools.acquire_view!(::DisabledPool{:metal}, x::MtlArray) = Metal.similar(x)
@inline AdaptiveArrayPools.acquire_view!(::DisabledPool{:metal}, x::AbstractArray) = MtlArray{eltype(x)}(undef, size(x))

# ==============================================================================
# rand! / randn! (Metal)
# ==============================================================================
# Live pool: the core `_rand_impl!`/`_randn_impl!` are inherited unchanged — they
# call `Random.rand!`/`randn!` on the acquired MtlArray, which Metal.jl runs on the
# GPU. Only the DisabledPool fallbacks and the collection-form rejection are needed.

# --- rand!/randn! for DisabledPool{:metal} (GPU-native Metal RNG) ---
@inline AdaptiveArrayPools.rand!(::DisabledPool{:metal}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Metal.rand(T, dims...)
@inline AdaptiveArrayPools.rand!(p::DisabledPool{:metal}, dims::Vararg{Int, N}) where {N} = Metal.rand(AdaptiveArrayPools.default_eltype(p), dims...)
@inline AdaptiveArrayPools.rand!(::DisabledPool{:metal}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Metal.rand(T, dims...)
@inline AdaptiveArrayPools.rand!(p::DisabledPool{:metal}, dims::NTuple{N, Int}) where {N} = Metal.rand(AdaptiveArrayPools.default_eltype(p), dims...)

@inline AdaptiveArrayPools.randn!(::DisabledPool{:metal}, ::Type{T}, dims::Vararg{Int, N}) where {T, N} = Metal.randn(T, dims...)
@inline AdaptiveArrayPools.randn!(p::DisabledPool{:metal}, dims::Vararg{Int, N}) where {N} = Metal.randn(AdaptiveArrayPools.default_eltype(p), dims...)
@inline AdaptiveArrayPools.randn!(::DisabledPool{:metal}, ::Type{T}, dims::NTuple{N, Int}) where {T, N} = Metal.randn(T, dims...)
@inline AdaptiveArrayPools.randn!(p::DisabledPool{:metal}, dims::NTuple{N, Int}) where {N} = Metal.randn(AdaptiveArrayPools.default_eltype(p), dims...)

# --- Collection/range sampling is CPU-only ---
# GPU RNGs have no efficient arbitrary-collection sampler; `Random.rand!(gpuarray, S)`
# scalar-indexes the device array (disallowed). Reject with a clear error for both the
# live and disabled Metal pools. Overriding `_rand_impl!` covers both the @with_pool
# macro path AND the direct public path (core `rand!(pool, S, …)` delegates to it).
@noinline _metal_no_collection() = throw(
    ArgumentError(
        "rand!(pool, collection/range, dims) is CPU-only: Metal GPU pools have no " *
            "arbitrary-collection sampler. Use rand!(pool, T, dims) for uniform GPU randoms."
    )
)
const _MetalAnyPool = Union{MetalAdaptiveArrayPool, DisabledPool{:metal}}
@inline AdaptiveArrayPools._rand_impl!(::_MetalAnyPool, ::AdaptiveArrayPools._SampleColl, ::Int, ::Vararg{Int, N}) where {N} = _metal_no_collection()
@inline AdaptiveArrayPools._rand_impl!(::_MetalAnyPool, ::AdaptiveArrayPools._SampleColl, ::NTuple{N, Int}) where {N} = _metal_no_collection()
