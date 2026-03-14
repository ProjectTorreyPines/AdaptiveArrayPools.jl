# Metal Backend

AdaptiveArrayPools provides native Apple Silicon GPU support through a package extension that loads automatically when [Metal.jl](https://github.com/JuliaGPU/Metal.jl) is available. Requires Julia 1.11+.

## Quick Start

```julia
using AdaptiveArrayPools, Metal

# Use :metal backend for Apple Silicon GPU arrays
@with_pool :metal pool function gpu_computation(n)
    A = acquire!(pool, Float32, n, n)  # MtlArray
    B = acquire!(pool, Float32, n, n)  # MtlArray

    fill!(A, 1.0f0)
    fill!(B, 2.0f0)

    return sum(A .+ B)
end

# Zero GPU allocation in hot loops
for i in 1:1000
    gpu_computation(100)  # GPU memory reused from pool
end
```

## API

The Metal backend uses the same API as CPU and CUDA, with `:metal` backend specifier:

| Macro/Function | Description |
|----------------|-------------|
| `@with_pool :metal pool expr` | GPU pool with automatic checkpoint/rewind |
| `acquire!(pool, T, dims...)` | Returns `MtlArray` (always 0 bytes GPU alloc) |
| `acquire_view!(pool, T, dims...)` | Returns `MtlArray` (same as `acquire!` on Metal) |
| `get_task_local_metal_pool()` | Returns the task-local Metal pool |
| `pool_stats(:metal)` | Print Metal pool statistics |

## Return Types

| Function | 1D Return | N-D Return |
|----------|-----------|------------|
| `acquire!` | `MtlArray{T,1}` | `MtlArray{T,N}` |
| `acquire_view!` | `MtlArray{T,1}` | `MtlArray{T,N}` |

## Allocation Behavior

**GPU Memory**: Always 0 bytes allocation after warmup. The underlying `MtlVector` is resized as needed and reused.

**CPU-side Wrapper Memory** (for `acquire!` N-D on Metal):
- The Metal backend uses `arr_wrappers`-based direct-index caching for `MtlArray` wrapper reuse
- Each dimensionality `N` has one cached wrapper per slot, reused via `setfield!(:dims)`
- After warmup: **zero CPU-side allocation for any number of dimension patterns** (same `N`)
- Different `N` values each get their own cached wrapper (also zero-alloc after first use)

## Fixed Slot Types

Metal hardware does not support Float64 or ComplexF64. The following types have optimized pre-allocated slots:

| Type | Field |
|------|-------|
| `Float32` | `.float32` |
| `Float16` | `.float16` |
| `Int64` | `.int64` |
| `Int32` | `.int32` |
| `ComplexF32` | `.complexf32` |
| `Bool` | `.bool` |

Other types use the fallback dictionary (`.others`).

!!! note "No Float64/ComplexF64"
    Apple Silicon GPUs do not natively support 64-bit floating point. Use `Float32` or `Float16` instead.

## Limitations

- **No Float64/ComplexF64**: Apple Silicon GPUs do not natively support 64-bit floating point
- **No `@maybe_with_pool :metal`**: Runtime toggle not supported for Metal backend
- **Single-device only**: Tested on single Apple GPU (multi-device untested)
- **Julia 1.11+**: Required for `setfield!`-based Array internals used by GPU extensions
- **Task-local only**: Each Task gets its own Metal pool, same as CPU

## Example: Matrix Computation

```julia
using AdaptiveArrayPools, Metal

@with_pool :metal pool function gpu_compute(n)
    A = acquire!(pool, Float32, n, n)
    B = acquire!(pool, Float32, n, n)
    C = acquire!(pool, Float32, n, n)

    fill!(A, 1.0f0); fill!(B, 2.0f0)
    C .= A .+ B

    return sum(C)
end

# Warmup
gpu_compute(100)

# Benchmark - zero GPU allocation
using BenchmarkTools
@benchmark gpu_compute(1000)
```

## Debugging

```julia
# Check pool state
pool_stats(:metal)

# Output:
# MetalAdaptiveArrayPool
#   Float32 (fixed) [Metal]
#     slots: 3 (active: 0)
#     elements: 30000 (117.188 KiB)
```

## CUDA vs Metal

| Feature | CUDA | Metal |
|---------|------|-------|
| Backend symbol | `:cuda` | `:metal` |
| Array type | `CuArray` | `MtlArray` |
| Float64 support | Yes | No |
| ComplexF64 support | Yes | No |
| Julia requirement | 1.11+ | 1.11+ |
| Safety features | Full | Full |
| Lazy mode | Yes | Yes |
