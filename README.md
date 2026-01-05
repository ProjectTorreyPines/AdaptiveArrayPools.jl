[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/dev/)
[![CI](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/projecttorreypines/adaptivearraypools.jl/graph/badge.svg?token=ZL0U0OvnL2)](https://codecov.io/github/projecttorreypines/adaptivearraypools.jl)

# AdaptiveArrayPools.jl

**Zero-allocation temporary arrays for Julia.**

A lightweight library that lets you write natural, allocation-style code while automatically reusing memory behind the scenes. Eliminates GC pressure in hot loops without the complexity of manual buffer management.

**Supported backends:**
- **CPU** — `Array`, works out of the box
- **CUDA** — `CuArray`, loads automatically when [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is available

## The Problem

In performance-critical code, temporary array allocations inside loops create massive GC pressure:

```julia
function compute(n)
    A = rand(n, n)      # allocates
    B = rand(n, n)      # allocates
    C = A * B           # allocates
    return sum(C)
end

for i in 1:10_000
    compute(100)  # ⚠️ 90k allocations, 2.75 GiB, 31% GC time
end
```

The traditional fix—passing pre-allocated buffers—works for simple cases but quickly becomes impractical:

- **API pollution**: Every function needs extra buffer arguments, breaking clean interfaces
- **Nested calls**: Buffers must be threaded through entire call stacks, even third-party code
- **Dynamic shapes**: Hard to pre-allocate when array sizes depend on runtime values
- **Package boundaries**: You can't easily pass buffers into library functions you don't control

## The Solution

Wrap your function with `@with_pool` and replace allocations with `acquire!` or convenience functions:

```julia
using AdaptiveArrayPools, LinearAlgebra, Random

@with_pool pool function compute_pooled(n)
    A = acquire!(pool, Float64, n, n)  # reuses memory from pool
    B = similar!(pool, A)
    C = similar!(pool, A)

    rand!(A); rand!(B)
    mul!(C, A, B)
    return sum(C)
end

compute_pooled(100)  # warmup
for i in 1:10_000
    compute_pooled(100) # ✅ Zero allocations, 0% GC
end
```

| Metric | Standard | **AdaptiveArrayPools** | Improvement |
|--------|----------|------------------------|-------------|
| Time | 787 ms | **525 ms** | 1.5× faster |
| Allocations | ⚠️ 90,000 (2.75 GiB) | ✅ **0** | 100% eliminated |
| GC Time | ⚠️ 31% | ✅ **0%** | No GC pauses |

> **CUDA support**: Same API—just use `@with_pool :cuda pool`. See [CUDA Backend](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/usage/cuda).

## How It Works

`@with_pool` automatically manages memory lifecycle for you:

1. **Checkpoint** — Saves current pool state when entering the block
2. **Acquire** — `acquire!` returns arrays backed by pooled memory
3. **Rewind** — When the block ends, all acquired arrays are marked available for reuse

This automatic checkpoint/rewind cycle is what enables zero allocation on repeated calls. You just write normal-looking code with `acquire!` instead of constructors.

`acquire!` returns lightweight views (`SubArray`, `ReshapedArray`) that work seamlessly with BLAS/LAPACK. If you need native `Array` types (FFI, type constraints), use `unsafe_acquire!`—see [API Reference](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/usage/api).

> **Note**: Keeping acquired arrays inside the scope is your responsibility. Return computed values (scalars, copies), not the arrays themselves. See [Safety Guide](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/guide/safety).

**Thread-safe by design**: Each Julia Task gets its own independent pool—no locks needed. See [Multi-Threading](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/advanced/multi-threading) for patterns.

### Convenience Functions

Common initialization patterns have convenience functions:

| Function | Equivalent to |
|----------|---------------|
| `zeros!(pool, 10)` | `acquire!` + `fill!(0)` |
| `ones!(pool, Float32, 3, 3)` | `acquire!` + `fill!(1)` |
| `similar!(pool, A)` | `acquire!` matching `eltype(A)`, `size(A)` |

These return views like `acquire!`. For raw `Array` types, use `unsafe_acquire!` or its convenience variants (`unsafe_zeros!`, `unsafe_ones!`, `unsafe_similar!`). See [API Reference](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/usage/api#convenience-functions).

## Installation

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.add("AdaptiveArrayPools")
```

## Documentation

| Guide | Description |
|-------|-------------|
| [API Reference](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/usage/api) | Complete function and macro reference |
| [CUDA Backend](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/usage/cuda) | GPU-specific usage and examples |
| [Safety Guide](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/guide/safety) | Scope rules and best practices |
| [Multi-Threading](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/advanced/multi-threading) | Task/thread safety patterns |
| [Configuration](https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable/usage/configuration) | Preferences and cache tuning |

## License

[Apache 2.0](LICENSE)

## Contact
Min-Gu Yoo [![Linkedin](https://i.sstatic.net/gVE0j.png)](https://www.linkedin.com/in/min-gu-yoo-704773230) (General Atomics)  yoom@fusion.gat.com