# Quick Start

This guide will help you get up and running with AdaptiveArrayPools.jl in minutes.

## Installation

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.add("AdaptiveArrayPools")
```

## Basic Usage

The core workflow is simple:
1. Wrap your function with `@with_pool`
2. Replace allocations with `acquire!` or convenience functions
3. Return computed values (scalars, copies), not the arrays themselves

### Before (Standard Julia)

```julia
function compute(n)
    A = rand(n, n)      # allocates
    B = rand(n, n)      # allocates
    C = A * B           # allocates
    return sum(C)
end

for i in 1:10_000
    compute(100)  # 90k allocations, 2.75 GiB, 31% GC time
end
```

### After (With Pooling)

```julia
using AdaptiveArrayPools, LinearAlgebra, Random

@with_pool pool function compute_pooled(n)
    A = acquire!(pool, Float64, n, n)  # reuses memory
    B = similar!(pool, A)
    C = similar!(pool, A)

    rand!(A); rand!(B)
    mul!(C, A, B)
    return sum(C)
end

compute_pooled(100)  # warmup (first call allocates)
for i in 1:10_000
    compute_pooled(100)  # zero allocations, 0% GC
end
```

## Convenience Functions

Common initialization patterns have shortcuts:

| Function | Equivalent to |
|----------|---------------|
| `zeros!(pool, 10)` | `acquire!` + `fill!(0)` |
| `ones!(pool, Float32, 3, 3)` | `acquire!` + `fill!(1)` |
| `similar!(pool, A)` | `acquire!` matching `eltype(A)`, `size(A)` |

```julia
@with_pool pool function example(n)
    A = zeros!(pool, n, n)        # zero-initialized
    B = ones!(pool, Float32, n)   # Float32 ones
    C = similar!(pool, A)         # same type and size as A
    # ...
end
```

## Return Types

`acquire!` and convenience functions return **view types** (`SubArray`, `ReshapedArray`) that work seamlessly with BLAS/LAPACK:

```julia
A = acquire!(pool, Float64, 10, 10)  # ReshapedArray{Float64,2}
mul!(C, A, B)  # works perfectly with BLAS
```

If you need native `Array` types (FFI, type constraints), use `unsafe_acquire!`:

```julia
A = unsafe_acquire!(pool, Float64, 10, 10)  # Array{Float64,2}
```

## Important Safety Rules

Arrays from the pool are **only valid within the `@with_pool` scope**:

```julia
# DO NOT return pool-backed arrays
@with_pool pool function bad_example()
    A = acquire!(pool, Float64, 10)
    return A  # WRONG - A marked for reuse, data may be overwritten!
end

# Return computed values instead
@with_pool pool function good_example()
    A = acquire!(pool, Float64, 10)
    return sum(A)  # OK - returning a scalar
end
```

For complete safety guidelines, see [Safety Rules](safety-rules.md).

## Next Steps

- [Safety Rules](safety-rules.md) - Complete scope rules and anti-patterns
- [Full API Reference](../reference/api.md) - Complete function and macro reference
- [Configuration](../features/configuration.md) - Preferences and cache tuning
- [Multi-threading](../features/multi-threading.md) - Task/thread safety patterns
- [CUDA Support](../features/cuda-support.md) - GPU backend usage
