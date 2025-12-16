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
function compute_naive(n)
    A = rand(n, n)      # allocates
    B = rand(n, n)      # allocates
    C = A * B           # allocates
    return sum(C)
end

for i in 1:10_000
    compute_naive(100)  # 91 MiB total, 17% GC time
end
```

The traditional fix—passing pre-allocated buffers through your call stack—works but requires invasive refactoring and clutters your APIs.

## The Solution

Wrap your function with `@with_pool` and use `acquire!` instead of allocation:

```julia
using AdaptiveArrayPools, LinearAlgebra, Random

@with_pool pool function compute_pooled(n)
    A = acquire!(pool, Float64, n, n)  # reuses memory from pool
    B = acquire!(pool, Float64, n, n)
    C = acquire!(pool, Float64, n, n)

    rand!(A); rand!(B)
    mul!(C, A, B)
    return sum(C)
end

compute_pooled(100)  # warmup
for i in 1:10_000
    compute_pooled(100)  # 0 bytes, 0% GC
end
```

| Approach | Memory | GC Time | Code Complexity |
|----------|--------|---------|-----------------|
| Naive allocation | 91 MiB | 17% | Simple |
| Manual buffer passing | 0 | 0% | Complex, invasive refactor |
| **AdaptiveArrayPools** | **0** | **0%** | **Minimal change** |

> **CUDA support**: Same API—just use `@with_pool :cuda pool`. See [CUDA Backend](docs/cuda.md).

## How It Works

`@with_pool` automatically manages memory lifecycle for you:

1. **Checkpoint** — Saves current pool state when entering the block
2. **Acquire** — `acquire!` returns arrays backed by pooled memory
3. **Rewind** — When the block ends, all acquired arrays are recycled for reuse

This automatic checkpoint/rewind cycle is what enables zero allocation on repeated calls. You just write normal-looking code with `acquire!` instead of constructors.

`acquire!` returns lightweight views (`SubArray`, `ReshapedArray`) that work seamlessly with BLAS/LAPACK. If you need native `Array` types (FFI, type constraints), use `unsafe_acquire!`—see [API Reference](docs/api.md).

> **Note**: Keeping acquired arrays inside the scope is your responsibility. Return computed values (scalars, copies), not the arrays themselves. See [Safety Guide](docs/safety.md).

**Thread-safe by design**: Each Julia Task gets its own independent pool, so `@with_pool` inside threaded code is automatically safe:

```julia
Threads.@threads for i in 1:N
    @with_pool pool begin
        a = acquire!(pool, Float64, 100)
        # each thread has its own pool — no race conditions
    end
end
```

## Installation

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.add("AdaptiveArrayPools")
```

## Documentation

| Guide | Description |
|-------|-------------|
| [API Reference](docs/api.md) | Complete function and macro reference |
| [CUDA Backend](docs/cuda.md) | GPU-specific usage and examples |
| [Safety Guide](docs/safety.md) | Scope rules and best practices |
| [Multi-Threading](docs/multi-threading.md) | Task/thread safety patterns |
| [Configuration](docs/configuration.md) | Preferences and cache tuning |

## License

[Apache 2.0](LICENSE)
