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

```julia
using AdaptiveArrayPools, LinearAlgebra

# 1. Define the hot-loop function with automatic pooling for ZERO-ALLOCATION
@with_pool pool function heavy_computation_step(n)
    # Safe Default: Returns ReshapedArray for N-D (always 0 bytes, prevents resize!)
    A = acquire!(pool, Float64, n, n)
    B = acquire!(pool, Float64, n, n)

    # Power User: Returns raw Matrix{Float64} (only for FFI/type constraints)
    # ⚠️ Must NOT resize! or escape scope
    C = unsafe_acquire!(pool, Float64, n, n)

    # Use them like normal arrays
    fill!(A, 1.0); fill!(B, 2.0)

    # Pass to inner functions as needed
    complex_inner_logic!(C, A, B)

    return sum(C) 
    # ⚠️ Arrays A, B, C must not escape this scope; they become invalid after this function returns!
end

# Standard Julia function (unaware of pooling)
function complex_inner_logic!(C, A, B)
    mul!(C, A, B)
end

# 2. Main application entry point
function main_simulation_loop()
    # ... complex setup logic ...
    
    total = 0.0
    # This loop would normally generate massive GC pressure
    for i in 1:1000
        # ✅ Zero allocation here after the first iteration!
        total += heavy_computation_step(100)
    end
    
    return total
end

# Run simulation
main_simulation_loop()
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
    # Get ReshapedArray views from auto-managed pool (0 bytes allocation)
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

This library prioritizes **zero-overhead performance** over runtime safety checks. Two fundamental rules must be followed:

1. **Scope Rule**: Arrays acquired from a pool are only valid within the `@with_pool` scope.
2. **Task Rule**: Pool objects must not be shared across Tasks (see [Multi-Threading Usage](#multi-threading-usage)).

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
    v  # Throws ErrorException: "Returning pool-backed array..."
end
```

> **Note:** `POOL_DEBUG` only catches direct returns, not indirect escapes (globals, closures). It's a development aid, not a guarantee.

</details>

## Key Features

- **`acquire!` — True Zero Allocation**: Returns lightweight views (`SubArray` for 1D, `ReshapedArray` for N-D) that are created on the stack. **Always 0 bytes**, regardless of dimension patterns or cache state.
- **`unsafe_acquire!` — Cached Allocation**: Returns concrete `Array` types for FFI/type constraints.
  - **1D**: Simple 1:1 cache → always 0 bytes
  - **N-D**: N-way set-associative cache (default: 4-way) → **0 bytes on cache hit**, ~100 bytes on cache miss. Increase `CACHE_WAYS` if you alternate between >4 dimension patterns.
  - Even on cache miss, this is just the `Array` header (metadata)—**actual data memory is always reused from the pool**, making it far more efficient than fresh allocations.
- **Low Overhead**: Optimized to have < 100 ns overhead for pool management, suitable for tight inner loops.
- **Task-Local Isolation**: Each Task gets its own pool via `task_local_storage()`. Thread-safe when `@with_pool` is called within each task's scope (see [Multi-Threading Usage](#multi-threading-usage) below).
- **Type Stable**: Optimized for `Float64`, `Int`, and other common types using fixed-slot caching.
- **Non-Intrusive**: If you disable pooling via preferences, `acquire!` compiles down to a standard `Array` allocation.
- **Flexible API**: Use `acquire!` for safe views (recommended), or `unsafe_acquire!` when concrete `Array` type is required (FFI, type constraints).

## Multi-Threading Usage

AdaptiveArrayPools uses `task_local_storage()` for **task-local isolation**: each Julia Task gets its own independent pool.

```julia
# ✅ SAFE: @with_pool inside @threads
Threads.@threads for i in 1:N
    @with_pool pool begin
        a = acquire!(pool, Float64, 100)
    end
end

# ❌ UNSAFE: @with_pool outside @threads (race condition!)
@with_pool pool Threads.@threads for i in 1:N
    a = acquire!(pool, Float64, 100)  # All threads share one pool!
end
```

| Pattern | Safety |
|---------|--------|
| `@with_pool` inside `@threads` | ✅ Safe |
| `@with_pool` outside `@threads` | ❌ Unsafe |
| Function with `@with_pool` called from `@threads` | ✅ Safe |

> **Important**: Pool objects must not be shared across Tasks. This library does not add locks—correct usage is the user's responsibility.

For detailed explanation including Julia's Task/Thread model and why thread-local pools don't work, see **[Multi-Threading Guide](docs/multi-threading.md)**.

## `acquire!` vs `unsafe_acquire!`

**In most cases, use `acquire!`**. It returns view types (`SubArray` for 1D, `ReshapedArray` for N-D) that are safe and always zero-allocation.

> **Performance Note**: BLAS/LAPACK functions (`mul!`, `lu!`, etc.) are fully optimized for `StridedArray`—there is **no performance difference** between views and raw arrays. Benchmarks show identical throughput.

Use `unsafe_acquire!` **only** when a concrete `Array{T,N}` type is required:
- **FFI/C interop**: External libraries expecting `Ptr{T}` from `Array`
- **Type constraints**: APIs that explicitly require `Matrix{T}` or `Vector{T}`, or type-unstable code where concrete types reduce dispatch overhead

```julia
@with_pool pool begin
    # ✅ Recommended: acquire! for general use (always 0 bytes)
    A = acquire!(pool, Float64, 100, 100)   # ReshapedArray
    B = acquire!(pool, Float64, 100, 100)   # ReshapedArray
    C = acquire!(pool, Float64, 100, 100)   # ReshapedArray
    mul!(C, A, B)  # ✅ BLAS works perfectly with views!

    # ⚠️ Only when concrete Array type is required:
    M = unsafe_acquire!(pool, Float64, 100, 100)  # Matrix{Float64}
    ccall(:some_c_function, Cvoid, (Ptr{Float64},), M)  # FFI needs Array
end
```

| Function | 1D Return | N-D Return | Allocation |
|----------|-----------|------------|------------|
| `acquire!` | `SubArray{T,1}` | `ReshapedArray{T,N}` | Always 0 bytes |
| `unsafe_acquire!` | `SubArray{T,1}` | `Array{T,N}` | 0 bytes (hit) / ~100 bytes header (miss) |

> **Note**: 1D always returns `SubArray` (both functions) with simple 1:1 caching. The N-way cache only applies to **N-D `unsafe_acquire!`**—up to `CACHE_WAYS` (default: 4) dimension patterns per slot; exceeding this causes header-only allocation per miss.

> **Warning**: Both functions return memory only valid within the `@with_pool` scope. Do NOT call `resize!`, `push!`, or `append!` on acquired arrays.

### API Aliases

For explicit naming, you can use these aliases:

```julia
acquire_view!(pool, T, dims...)   # Same as acquire! → returns view types
acquire_array!(pool, T, dims...)  # Same as unsafe_acquire! → returns Array
```

## Documentation

- [API Reference](docs/api.md) - Macros, functions, and types
- [Multi-Threading Guide](docs/multi-threading.md) - Task/Thread model, safe patterns, and design rationale
- [Runtime Toggle: @maybe_with_pool](docs/maybe_with_pool.md) - Control pooling at runtime
- [Configuration](docs/configuration.md) - Preferences.jl integration

## Configuration

Configure AdaptiveArrayPools via `LocalPreferences.toml`:

```toml
[AdaptiveArrayPools]
use_pooling = false  # ⭐ Primary: Disable pooling entirely
cache_ways = 8       # Secondary: N-way cache size (default: 4)
```

### Disabling Pooling (Primary Use Case)

The most important configuration is **`use_pooling = false`**, which completely disables all pooling:

```julia
# With use_pooling = false, acquire! becomes equivalent to:
acquire!(pool, Float64, n, n)  →  Matrix{Float64}(undef, n, n)
```

This is useful for:
- **Debugging**: Isolate pooling-related issues by comparing behavior
- **Benchmarking**: Measure pooling overhead vs direct allocation
- **Gradual adoption**: Add `@with_pool` to code without changing behavior until ready

When disabled, all macros generate `pool = nothing` and `acquire!` falls back to standard allocation with **zero overhead**.

### N-way Cache Tuning (Advanced)

```julia
using AdaptiveArrayPools
set_cache_ways!(8)  # Requires Julia restart
```

Increase `cache_ways` if alternating between >4 dimension patterns per slot.

## License

[Apache 2.0](LICENSE)
