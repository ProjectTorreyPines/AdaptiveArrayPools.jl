# CUDA Backend

AdaptiveArrayPools provides native CUDA support through a package extension that loads automatically when CUDA.jl is available.

## Quick Start

```julia
using AdaptiveArrayPools, CUDA

# Use :cuda backend for GPU arrays
@with_pool :cuda pool function gpu_computation(n)
    A = acquire!(pool, Float64, n, n)  # CuArray view
    B = acquire!(pool, Float64, n, n)  # CuArray view

    fill!(A, 1.0)
    fill!(B, 2.0)

    return sum(A .+ B)
end

# Zero GPU allocation in hot loops
for i in 1:1000
    gpu_computation(100)  # GPU memory reused from pool
end
```

## API

The CUDA backend uses the same API as CPU, with `:cuda` backend specifier:

| Macro/Function | Description |
|----------------|-------------|
| `@with_pool :cuda pool expr` | GPU pool with automatic checkpoint/rewind |
| `acquire!(pool, T, dims...)` | Returns `CuArray` view (always 0 bytes GPU alloc) |
| `unsafe_acquire!(pool, T, dims...)` | Returns raw `CuArray` (for FFI/type constraints) |
| `get_task_local_cuda_pool()` | Returns the task-local CUDA pool |
| `pool_stats(:cuda)` | Print CUDA pool statistics |

## Return Types

| Function | 1D Return | N-D Return |
|----------|-----------|------------|
| `acquire!` | `CuArray{T,1}` (view) | `CuArray{T,N}` (view) |
| `unsafe_acquire!` | `CuArray{T,1}` | `CuArray{T,N}` |

## Allocation Behavior

**GPU Memory**: Always 0 bytes allocation after warmup. The underlying `CuVector` is resized as needed and reused.

**CPU Memory**:
- Cache hit (≤4 dimension patterns per slot): 0 bytes
- Cache miss (>4 patterns): ~100 bytes for wrapper metadata

```julia
# Example: 4 patterns fit in 4-way cache → zero CPU allocation
dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))
for dims in dims_list
    @with_pool :cuda p begin
        A = acquire!(p, Float64, dims...)
        # Use A...
    end
end
```

## Fixed Slot Types

Optimized types with pre-allocated slots (same as CPU):

| Type | Field |
|------|-------|
| `Float64` | `.float64` |
| `Float32` | `.float32` |
| `Float16` | `.float16` |
| `Int64` | `.int64` |
| `Int32` | `.int32` |
| `ComplexF64` | `.complexf64` |
| `ComplexF32` | `.complexf32` |
| `Bool` | `.bool` |

Other types use the fallback dictionary (`.others`).

## Limitations

- **No `@maybe_with_pool :cuda`**: Runtime toggle not supported for CUDA backend
- **Task-local only**: Each Task gets its own CUDA pool, same as CPU
- **Same device**: All arrays in a pool use the same CUDA device

## Example: Matrix Multiplication

```julia
using AdaptiveArrayPools, CUDA, LinearAlgebra

@with_pool :cuda pool function gpu_matmul(n)
    A = acquire!(pool, Float64, n, n)
    B = acquire!(pool, Float64, n, n)
    C = acquire!(pool, Float64, n, n)

    rand!(A); rand!(B)
    mul!(C, A, B)

    return sum(C)
end

# Warmup
gpu_matmul(100)

# Benchmark - zero GPU allocation
using BenchmarkTools
@benchmark gpu_matmul(1000)
```

## Debugging

```julia
# Check pool state
pool_stats(:cuda)

# Output:
# CuAdaptiveArrayPool (device 0)
#   Float64 (fixed) [GPU]
#     slots: 3 (active: 0)
#     elements: 30000 (234.375 KiB)
```
