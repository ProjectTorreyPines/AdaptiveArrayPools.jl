# ==============================================================================
# CUDA Macro Support
# ==============================================================================
# Enables @with_pool :cuda syntax for GPU memory pooling.

using AdaptiveArrayPools: _get_pool_for_backend

# ==============================================================================
# Backend Registration (Val dispatch - zero overhead)
# ==============================================================================

"""
Register :cuda backend for `@with_pool :cuda` syntax.
Uses Val dispatch for compile-time resolution and full inlining.
"""
@inline AdaptiveArrayPools._get_pool_for_backend(::Val{:cuda}) = get_task_local_cuda_pool()
