# API Reference

## Macros

| Macro | Description |
|-------|-------------|
| `@with_pool name expr` | **Recommended.** Creates a pool scope with automatic checkpoint/rewind. `name` is bound to the task-local pool. |
| `@with_pool expr` | Same as above, but pool is not named (use when you don't need direct access). |
| `@maybe_with_pool name expr` | Same as `@with_pool`, but can be toggled on/off at runtime via `MAYBE_POOLING_ENABLED[]`. |

## Functions

| Function | Description |
|----------|-------------|
| `acquire!(pool, T, dims...)` | Returns a `SubArray{T}` of size `dims` from the pool. |
| `acquire!(pool, T, dims::Tuple)` | Tuple overload for `acquire!` (e.g., `acquire!(pool, T, size(x))`). |
| `checkpoint!(pool)` | Saves the current pool state (stack pointer). |
| `checkpoint!(pool, T...)` | Type-specific checkpoint for optimized performance. |
| `rewind!(pool)` | Restores the pool to the last checkpoint, freeing all arrays acquired since then. |
| `rewind!(pool, T...)` | Type-specific rewind for optimized performance. |
| `pool_stats(pool)` | Prints detailed statistics about pool usage. |
| `pool_stats()` | Prints statistics for the global task-local pool. |
| `get_global_pool()` | Returns the task-local global pool instance. |
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

## Usage Pattern

```julia
# Define functions that take pool as argument
function compute(n, pool)
    v = acquire!(pool, Float64, n)
    v .= 1.0
    sum(v)
end

# Use @with_pool for lifecycle management
result = @with_pool pool begin
    compute(100, pool)
end
```

## Safety Notes

Arrays acquired from a pool are **only valid within the `@with_pool` scope**. Do not:
- Return pool-backed arrays from functions
- Store them in global variables
- Capture them in closures that outlive the scope

Use `POOL_DEBUG[] = true` during development to catch direct returns of pool-backed arrays.
