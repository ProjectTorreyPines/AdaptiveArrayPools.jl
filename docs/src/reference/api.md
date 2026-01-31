# API Reference

## API Summary

### Macros

| Macro | Description |
|-------|-------------|
| `@with_pool name expr` | **Recommended.** Injects a global, task-local pool named `name`. Automatically checkpoints and rewinds. |
| `@maybe_with_pool name expr` | Same as `@with_pool`, but can be toggled on/off at runtime via `MAYBE_POOLING_ENABLED[]`. |

### Functions

| Function | Description |
|----------|-------------|
| `acquire!(pool, T, dims...)` | Returns a view for most `T`: `SubArray{T,1}` for 1D, `ReshapedArray{T,N}` for N-D. For `T === Bit`, returns native `BitVector`/`BitArray{N}`. Cache hit 0 bytes. |
| `acquire!(pool, T, dims::Tuple)` | Tuple overload for `acquire!` (e.g., `acquire!(pool, T, size(x))`). |
| `acquire!(pool, x::AbstractArray)` | Similar-style: acquires array matching `eltype(x)` and `size(x)`. |
| `unsafe_acquire!(pool, T, dims...)` | Returns native `Array`/`CuArray` (CPU: `Vector{T}` for 1D, `Array{T,N}` for N-D). For `T === Bit`, returns native `BitVector`/`BitArray{N}` (equivalent to `acquire!`). Only use for FFI/type constraints. |
| `unsafe_acquire!(pool, x::AbstractArray)` | Similar-style: acquires raw array matching `eltype(x)` and `size(x)`. |
| `checkpoint!(pool)` | Saves the current pool state (stack pointer). |
| `rewind!(pool)` | Restores the pool to the last checkpoint, freeing all arrays acquired since then. |
| `pool_stats(pool)` | Prints detailed statistics about pool usage. |
| `get_task_local_pool()` | Returns the task-local pool instance. |
| `empty!(pool)` | Clears all internal storage, releasing all memory. |

### Convenience Functions

Default element type is `Float64` (CPU) or `Float32` (CUDA).

| Function | Description |
|----------|-------------|
| `zeros!(pool, [T,] dims...)` | Zero-initialized view. Equivalent to `acquire!` + `fill!(0)`. |
| `ones!(pool, [T,] dims...)` | One-initialized view. Equivalent to `acquire!` + `fill!(1)`. |
| `trues!(pool, dims...)` | Bit-packed `BitVector` / `BitArray{N}` filled with `true`. |
| `falses!(pool, dims...)` | Bit-packed `BitVector` / `BitArray{N}` filled with `false`. |
| `similar!(pool, A)` | View matching `eltype(A)` and `size(A)`. |

### Types

| Type | Description |
|------|-------------|
| `AdaptiveArrayPool` | The main pool type. Create with `AdaptiveArrayPool()`. |
| `Bit` | Sentinel type to request packed `BitVector` storage (1 bit/element). |
| `DisabledPool{Backend}` | Sentinel type when pooling is disabled. |

### Configuration & Utilities

| Symbol | Description |
|--------|-------------|
| `USE_POOLING` | Compile-time constant to disable all pooling. |
| `MAYBE_POOLING_ENABLED` | Runtime `Ref{Bool}` for `@maybe_with_pool`. |
| `POOL_DEBUG` | Runtime `Ref{Bool}` to enable safety validation. |
| `set_cache_ways!(n)` | Set N-way cache size. |

---

## Detailed Reference

```@autodocs
Modules = [AdaptiveArrayPools]
Order = [:macro, :function, :type, :constant]
```
