# ==============================================================================
# CUDA Macro Support
# ==============================================================================
# Enables @with_pool :cuda syntax for GPU memory pooling.

using AdaptiveArrayPools: _get_pool_for_backend, _dispatch_pool_scope

# ==============================================================================
# Backend Registration (Val dispatch - zero overhead)
# ==============================================================================

"""
Register :cuda backend for `@with_pool :cuda` syntax.
Uses Val dispatch for compile-time resolution and full inlining.
"""
@inline AdaptiveArrayPools._get_pool_for_backend(::Val{:cuda}) = get_task_local_cuda_pool()

# ==============================================================================
# Union Splitting for CuAdaptiveArrayPool{S}
# ==============================================================================
#
# The base _dispatch_pool_scope has an `else` fallback for non-CPU pools that
# passes pool_any without type narrowing. This override provides union splitting
# for CUDA pools, enabling compile-time S → dead-code elimination of safety branches.

@inline function AdaptiveArrayPools._dispatch_pool_scope(f, pool_any::CuAdaptiveArrayPool)
    if pool_any isa CuAdaptiveArrayPool{0}
        return f(pool_any::CuAdaptiveArrayPool{0})
    elseif pool_any isa CuAdaptiveArrayPool{1}
        return f(pool_any::CuAdaptiveArrayPool{1})
    elseif pool_any isa CuAdaptiveArrayPool{2}
        return f(pool_any::CuAdaptiveArrayPool{2})
    else
        return f(pool_any::CuAdaptiveArrayPool{3})
    end
end
