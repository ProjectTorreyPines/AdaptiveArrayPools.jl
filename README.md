[![CI](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/projecttorreypines/adaptivearraypools.jl/graph/badge.svg?token=ZL0U0OvnL2)](https://codecov.io/github/projecttorreypines/adaptivearraypools.jl)

# AdaptiveArrayPools.jl

**Zero-allocation array pooling for Julia.**
Reuse temporary arrays to eliminate Garbage Collection (GC) pressure in high-performance hot loops.

## Installation

`AdaptiveArrayPools` is registered with [FuseRegistry](https://github.com/ProjectTorreyPines/FuseRegistry.jl/):

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.Registry.add("General")
Pkg.add("AdaptiveArrayPools")
```

## Quick Start

The `@with_pool` macro automatically manages the pool's lifecycle. It checkpoints the pool state at the start of the block/function and rewinds it at the end, making all acquired arrays available for reuse.

```julia
using AdaptiveArrayPools
using LinearAlgebra

# 1. Define a function that needs temporary arrays
@with_pool pool function heavy_computation(n)
    # Acquire temporary arrays from the pool (returns Views)
    # These are "borrowed" from pre-allocated, auto-managed memory pool
    A_mat = acquire!(pool, Float64, n, n)
    x_vec = acquire!(pool, Float64, n)
    b_vec = acquire!(pool, Float64, n)

    # Use them exactly like normal Arrays
    fill!(A_mat, 1.5)
    fill!(x_vec, 2.0)

    # Perform computation (A * x = b)
    mul!(b_vec, A_mat, x_vec)

    return sum(b_vec)  # ✅ Return scalar - safe
    # Function exit: Pool automatically "rewinds".
    # A, x, and b are now free to be reused by the next call.
end

# 2. Run it in a loop
function run_simulation()
    total = 0.0

    # First call: Pool allocates internal memory
    total += heavy_computation(100)

    # Subsequent calls: Zero allocations!
    # The same memory slots are reused for each iteration.
    for i in 1:1000
        total += heavy_computation(100)
    end
    return total
end
```

## Why Use This?

In high-performance computing, allocating temporary arrays inside a loop creates significant GC pressure, causing stuttering and performance degradation. Manual in-place operations (passing pre-allocated buffers) avoid this but require tedious buffer management and argument passing, making code complex and error-prone.

```julia
using LinearAlgebra, Random
using BenchmarkTools

# ❌ Naive Approach: Allocates new arrays every single call
function compute_naive(n::Int)
    mat1 = rand(n, n) # Allocation!
    mat2 = rand(n, n) # Allocation!

    mat3 = mat1 * mat2 # Allocation!
    return sum(mat3)
end

# ✅ Pooled Approach: Zero allocations in steady state, clean syntax (no manual buffer passing)
@with_pool pool function compute_pooled(n::Int)
    # Get Views from auto-managed pool (No allocation)
    mat1 = acquire!(pool, Float64, n, n)
    mat2 = acquire!(pool, Float64, n, n)
    mat3 = acquire!(pool, Float64, n, n)

    # Use In-place functions without allocations
    Random.rand!(mat1)
    Random.rand!(mat2)
    mul!(mat3, mat1, mat2)
    return sum(mat3)
end

# Naive: Large temporary allocations cause GC pressure
@benchmark compute_naive(2000)
# Time  (mean ± σ):   67.771 ms ±  31.818 ms ⚠️ ┊ GC (mean ± σ):  17.02% ± 18.69%  ⚠️
# Memory estimate: 91.59 MiB ⚠️, allocs estimate: 9.

# Pooled: Zero allocations, no GC pressure
@benchmark compute_pooled(2000)
# Time  (mean ± σ):   57.647 ms ±  3.960 ms ✅ ┊ GC (mean ± σ):  0.00% ± 0.00% ✅
# Memory estimate: 0 bytes ✅, allocs estimate: 0.
```

> **Performance Note:**
> - **vs Manual Pre-allocation**: This library achieves performance comparable to manually passing pre-allocated buffers (in-place operations), but without the boilerplate of managing buffer lifecycles.
> - **Low Overhead**: The overhead of `@with_pool` (including checkpoint/rewind) is typically **tens of nanoseconds** (< 100 ns), making it negligible for most workloads compared to the cost of memory allocation.

## Important: User Responsibility

**Arrays acquired from a pool are only valid within the `@with_pool` scope.**

When `@with_pool` ends, all acquired arrays are "rewound" and their memory becomes available for reuse. Using them after the scope ends leads to **undefined behavior** (data corruption, crashes).

<details>
<summary><b>Safe Patterns</b> (click to expand)</summary>

```julia
@with_pool pool function safe_example(n)
    v = acquire!(pool, Float64, n)
    v .= 1.0

    # ✅ Return computed values (scalars, tuples, etc.)
    return sum(v), length(v)
end

@with_pool pool function safe_copy(n)
    v = acquire!(pool, Float64, n)
    v .= rand(n)

    # ✅ Return a copy if you need the data outside
    return copy(v)
end
```

</details>

<details>
<summary><b>Unsafe Patterns (DO NOT DO THIS)</b> (click to expand)</summary>

```julia
@with_pool pool function unsafe_return(n)
    v = acquire!(pool, Float64, n)
    v .= 1.0
    return v  # ❌ UNSAFE: Returning pool-backed array!
end

result = unsafe_return(100)
# result now points to memory that may be overwritten!

# ❌ Also unsafe: storing in global variables, closures, etc.
global_storage = nothing
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    global_storage = v  # ❌ UNSAFE: escaping via global
end
```

</details>

<details>
<summary><b>Debugging with POOL_DEBUG</b> (click to expand)</summary>

Enable `POOL_DEBUG` to catch direct returns of pool-backed arrays:

```julia
POOL_DEBUG[] = true  # Enable safety checks

@with_pool pool begin
    v = acquire!(pool, Float64, 10)
    v  # Throws ErrorException: "Returning SubArray backed by pool..."
end
```

> **Note:** `POOL_DEBUG` only catches direct returns, not indirect escapes (globals, closures). It's a development aid, not a guarantee.

</details>

## Key Features

- **True Zero Allocation**: Not just array data, but the `SubArray` (View) wrappers are also cached.
- **Low Overhead**: Optimized to have < 100 ns overhead for pool management, suitable for tight inner loops.
- **Task-Local Safety**: `@with_pool` uses `task_local_storage`, making it safe for multi-threaded code.
- **Type Stable**: Optimized for `Float64`, `Int`, and other common types using fixed-slot caching.
- **Non-Intrusive**: If you disable pooling via preferences, `acquire!` compiles down to a standard `Array` allocation.

## Documentation

- [API Reference](docs/api.md) - Macros, functions, and types
- [Runtime Toggle: @maybe_with_pool](docs/maybe_with_pool.md) - Control pooling at runtime
- [Kwarg Injection: @pool_kwarg](docs/pool_kwarg.md) - For composable library functions
- [Configuration](docs/configuration.md) - Preferences.jl integration

## Configuration

You can completely disable pooling at compile-time (removing all overhead) by setting a preference in `LocalPreferences.toml`:

```toml
[AdaptiveArrayPools]
use_pooling = false
```

## License

[Apache 2.0](LICENSE)
