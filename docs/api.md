# API Reference

## Macros

| Macro | Description |
|-------|-------------|
| `@use_pool name expr` | **Recommended.** Injects a global, task-local pool named `name`. Automatically checkpoints and rewinds. |
| `@maybe_use_pool name expr` | Same as `@use_pool`, but can be toggled on/off at runtime via `MAYBE_POOLING_ENABLED[]`. |
| `@with_pool pool expr` | Uses a specific pool instance provided by the user. |

## Functions

| Function | Description |
|----------|-------------|
| `acquire!(pool, T, dims...)` | Returns a `SubArray{T}` of size `dims` from the pool. |
| `release!(pool, array)` | Explicitly returns an array to the pool (optional, usually handled by rewind). |
| `checkpoint!(pool)` | Saves the current pool state (stack pointer). |
| `rewind!(pool)` | Restores the pool to the last checkpoint, freeing all arrays acquired since then. |
| `pool_stats(pool)` | Prints detailed statistics about pool usage. |

## Types

| Type | Description |
|------|-------------|
| `AdaptiveArrayPool` | The main pool type. Create with `AdaptiveArrayPool()`. |

## Constants

| Constant | Description |
|----------|-------------|
| `USE_POOLING` | Compile-time constant. Set via `Preferences.jl` to disable all pooling. |
| `MAYBE_POOLING_ENABLED` | Runtime `Ref{Bool}`. Only affects `@maybe_use_pool`. |
| `POOL_DEBUG` | Runtime `Ref{Bool}`. Enable safety validation for debugging. |
