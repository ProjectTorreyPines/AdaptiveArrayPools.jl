# ==============================================================================
# CUDA Macro Support
# ==============================================================================
# Enables @with_pool :cuda syntax and provides explicit @with_cuda_pool macro.

using AdaptiveArrayPools: _get_pool_for_backend

# ==============================================================================
# Backend Registration (Val dispatch - zero overhead)
# ==============================================================================

"""
Register :cuda backend for `@with_pool :cuda` syntax.
Uses Val dispatch for compile-time resolution and full inlining.
"""
@inline AdaptiveArrayPools._get_pool_for_backend(::Val{:cuda}) = get_task_local_cuda_pool()

# ==============================================================================
# Explicit @with_cuda_pool Macro (Optional Alias)
# ==============================================================================

"""
    @with_cuda_pool pool expr
    @with_cuda_pool expr

Explicit macro for GPU pooling. Equivalent to `@with_pool :cuda pool expr`.

Useful for users who prefer explicit naming over the unified `@with_pool :cuda` syntax.

## Example
```julia
using AdaptiveArrayPools, CUDA

@with_cuda_pool pool begin
    A = acquire!(pool, Float32, 1000, 1000)
    B = acquire!(pool, Float32, 1000, 1000)
    A .= CUDA.rand(1000, 1000)
    B .= A .* 2
    sum(B)
end
```

See also: [`@with_pool`](@ref)
"""
macro with_cuda_pool(pool_name, expr)
    # Reuse the backend code generation from core
    esc(:($AdaptiveArrayPools.@with_pool :cuda $pool_name $expr))
end

macro with_cuda_pool(expr)
    esc(:($AdaptiveArrayPools.@with_pool :cuda $expr))
end
