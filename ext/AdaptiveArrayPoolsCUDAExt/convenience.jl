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
