# ==============================================================================
# CUDA Macro Support
# ==============================================================================
# Enables @with_pool :cuda syntax for GPU memory pooling.

using AdaptiveArrayPools: _get_pool_for_backend, _pool_type_for_backend

# ==============================================================================
# Backend Registration (Val dispatch - zero overhead)
# ==============================================================================

"""
Register :cuda backend for `@with_pool :cuda` syntax.
Uses Val dispatch for compile-time resolution and full inlining.
"""
@inline AdaptiveArrayPools._get_pool_for_backend(::Val{:cuda}) = get_task_local_cuda_pool()

# ==============================================================================
# Pool Type Registration for Closureless Union Splitting
# ==============================================================================
#
# `_pool_type_for_backend` is called at macro expansion time to determine the
# concrete pool type for closureless `let`/`if isa` chain generation.
# This enables `@with_pool :cuda` to generate `if _raw isa CuAdaptiveArrayPool{0} ...`
# instead of closure-based `_dispatch_pool_scope`.

AdaptiveArrayPools._pool_type_for_backend(::Val{:cuda}) = CuAdaptiveArrayPool
