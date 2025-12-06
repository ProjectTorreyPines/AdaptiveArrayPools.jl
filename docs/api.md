# API Reference

## Macros

| Macro | Description |
|-------|-------------|
| `@with_pool name expr` | **Recommended.** Injects a global, task-local pool named `name`. Automatically checkpoints and rewinds. |
| `@maybe_with_pool name expr` | Same as `@with_pool`, but can be toggled on/off at runtime via `MAYBE_POOLING_ENABLED[]`. |

## Functions

| Function | Description |
|----------|-------------|
| `acquire!(pool, T, dims...)` | Returns a view: `SubArray{T,1}` for 1D, `ReshapedArray{T,N}` for N-D. Always 0 bytes. |
| `acquire!(pool, T, dims::Tuple)` | Tuple overload for `acquire!` (e.g., `acquire!(pool, T, size(x))`). |
| `acquire!(pool, x::AbstractArray)` | Similar-style: acquires array matching `eltype(x)` and `size(x)`. |
| `unsafe_acquire!(pool, T, dims...)` | Returns `SubArray{T,1}` for 1D, raw `Array{T,N}` for N-D. Only for FFI/type constraints. |
| `unsafe_acquire!(pool, T, dims::Tuple)` | Tuple overload for `unsafe_acquire!`. |
| `unsafe_acquire!(pool, x::AbstractArray)` | Similar-style: acquires raw array matching `eltype(x)` and `size(x)`. |
| `acquire_view!(pool, T, dims...)` | Alias for `acquire!`. Returns view types. |
| `acquire_array!(pool, T, dims...)` | Alias for `unsafe_acquire!`. Returns Array for N-D. |
| `checkpoint!(pool)` | Saves the current pool state (stack pointer). |
| `checkpoint!(pool, T...)` | Type-specific checkpoint for optimized performance. |
| `rewind!(pool)` | Restores the pool to the last checkpoint, freeing all arrays acquired since then. |
| `rewind!(pool, T...)` | Type-specific rewind for optimized performance. |
| `pool_stats(pool)` | Prints detailed statistics about pool usage. |
| `get_task_local_pool()` | Returns the task-local pool instance. |
| `empty!(pool)` | Clears all internal storage, releasing all memory. |

## Types

| Type | Description |
|------|-------------|
| `AdaptiveArrayPool` | The main pool type. Create with `AdaptiveArrayPool()`. |

## Constants

| Constant | Description |
|----------|-------------|
| `USE_POOLING` | Compile-time constant. Set via `Preferences.jl` to disable all pooling. |
| `MAYBE_POOLING_ENABLED` | Runtime `Ref{Bool}`. Only affects `@maybe_with_pool`. |
| `POOL_DEBUG` | Runtime `Ref{Bool}`. Enable safety validation for debugging. |
| `CACHE_WAYS` | Compile-time constant. N-way cache size for `unsafe_acquire!` (default: 4, range: 1-16). |

## Configuration Functions

| Function | Description |
|----------|-------------|
| `set_cache_ways!(n)` | Set N-way cache size. Requires Julia restart. |

## Safety Notes

Arrays acquired from a pool are **only valid within the `@with_pool` scope**. Do not:
- Return pool-backed arrays from functions
- Store them in global variables
- Capture them in closures that outlive the scope
- Call `resize!`, `push!`, or `append!` on arrays from `unsafe_acquire!`

Use `POOL_DEBUG[] = true` during development to catch direct returns of pool-backed arrays.

## `acquire!` vs `unsafe_acquire!`

| Function | 1D Return | N-D Return | Allocation |
|----------|-----------|------------|------------|
| `acquire!` | `SubArray{T,1}` | `ReshapedArray{T,N}` | Always 0 bytes |
| `unsafe_acquire!` | `SubArray{T,1}` | `Array{T,N}` | 0 bytes (cache hit) / 112 bytes (miss) |

Both share the same underlying pool memory. **Use `acquire!` by default**—BLAS/LAPACK are fully optimized for `StridedArray`, so there's no performance difference.

Use `unsafe_acquire!` only when you need a concrete `Array{T,N}` type (FFI, type signatures, runtime dispatch).

**N-way Cache**: `unsafe_acquire!` caches up to `CACHE_WAYS` (default: 4) different dimension patterns per slot. Exceeding this causes cache eviction.
