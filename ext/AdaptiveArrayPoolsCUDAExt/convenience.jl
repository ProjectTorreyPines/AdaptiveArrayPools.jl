# ==============================================================================
# CUDA Convenience Functions (Float32 default)
# ==============================================================================
# Override default-type versions only; explicit type versions use base AbstractArrayPool methods.
# This matches CUDA.zeros() behavior which defaults to Float32.

using AdaptiveArrayPools: _mark_untracked!, _zeros_impl!, _ones_impl!

# ==============================================================================
# zeros! - Float32 default for CUDA (instead of Float64)
# ==============================================================================

@inline function AdaptiveArrayPools.zeros!(pool::CuAdaptiveArrayPool, dims::Vararg{Int})
    _mark_untracked!(pool)
    _zeros_impl!(pool, Float32, dims...)
end

@inline function AdaptiveArrayPools.zeros!(pool::CuAdaptiveArrayPool, dims::Tuple{Vararg{Int}})
    _mark_untracked!(pool)
    _zeros_impl!(pool, Float32, dims...)
end

# ==============================================================================
# ones! - Float32 default for CUDA (instead of Float64)
# ==============================================================================

@inline function AdaptiveArrayPools.ones!(pool::CuAdaptiveArrayPool, dims::Vararg{Int})
    _mark_untracked!(pool)
    _ones_impl!(pool, Float32, dims...)
end

@inline function AdaptiveArrayPools.ones!(pool::CuAdaptiveArrayPool, dims::Tuple{Vararg{Int}})
    _mark_untracked!(pool)
    _ones_impl!(pool, Float32, dims...)
end

# ==============================================================================
# similar! - No override needed
# ==============================================================================
# similar! uses eltype(template_array) as default, which is backend-agnostic.
# The base AbstractArrayPool methods work correctly for CUDA pools.
