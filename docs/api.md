# API Reference

## Macros

| Macro | Description |
|-------|-------------|
| `@with_pool name expr` | **Recommended.** Injects a global, task-local pool named `name`. Automatically checkpoints and rewinds. |
| `@maybe_with_pool name expr` | Same as `@with_pool`, but can be toggled on/off at runtime via `MAYBE_POOLING_ENABLED[]`. |

## Functions

| Function | Description |
|----------|-------------|
| `acquire!(pool, T, dims...)` | Returns a `SubArray{T,N}` view of size `dims` from the pool. |
| `acquire!(pool, T, dims::Tuple)` | Tuple overload for `acquire!` (e.g., `acquire!(pool, T, size(x))`). |
| `acquire!(pool, x::AbstractArray)` | Similar-style: acquires array matching `eltype(x)` and `size(x)`. |
| `unsafe_acquire!(pool, T, dims...)` | Returns a raw `Array{T,N}` backed by pool memory. Use for BLAS, FFI, or hot-path dispatch. |
| `unsafe_acquire!(pool, T, dims::Tuple)` | Tuple overload for `unsafe_acquire!`. |
| `unsafe_acquire!(pool, x::AbstractArray)` | Similar-style: acquires raw array matching `eltype(x)` and `size(x)`. |
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

## Safety Notes

Arrays acquired from a pool are **only valid within the `@with_pool` scope**. Do not:
- Return pool-backed arrays from functions
- Store them in global variables
- Capture them in closures that outlive the scope
- Call `resize!`, `push!`, or `append!` on arrays from `unsafe_acquire!`

Use `POOL_DEBUG[] = true` during development to catch direct returns of pool-backed arrays.

## `acquire!` vs `unsafe_acquire!`

| Function | Return Type | Use Case |
|----------|-------------|----------|
| `acquire!` | `SubArray{T,N}` | General use, prevents accidental `resize!` |
| `unsafe_acquire!` | `Array{T,N}` | BLAS operations, FFI calls, hot-path dispatch |

Both share the same underlying pool memory. Choose based on whether you need a view (safe) or concrete array type (performance).
