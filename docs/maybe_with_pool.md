# @maybe_with_pool

Runtime-toggleable pooling. Users can enable/disable via `MAYBE_POOLING_ENABLED[]`.

## Usage

```julia
# Define a function that takes pool as argument
function compute(n, pool)
    v = acquire!(pool, Float64, n)
    v .= 1.0
    sum(v)
end

# Use @maybe_with_pool for runtime-toggleable pooling
result = @maybe_with_pool pool begin
    compute(10, pool)
end

# Toggle at runtime
MAYBE_POOLING_ENABLED[] = false  # Normal allocation
MAYBE_POOLING_ENABLED[] = true   # Uses pool
```

## When to Use

- Library code where end-users should control pooling behavior
- Debugging: disable pooling to isolate memory issues
- Benchmarking: compare pooled vs non-pooled performance

## How It Works

When `MAYBE_POOLING_ENABLED[] == false`:
- `pool` becomes `nothing`
- `acquire!(nothing, T, dims...)` allocates normally

When `MAYBE_POOLING_ENABLED[] == true`:
- `pool` is the task-local pool
- `acquire!` returns views from the pool

The toggle works at runtime within a single function - no recompilation needed.

## vs @with_pool

| | `@with_pool` | `@maybe_with_pool` |
|---|---|---|
| Runtime toggle | No | Yes |
| Overhead when disabled | None | Branch check |
| Use case | Application code | Library code |

## Safety

Same rules as `@with_pool`: arrays are only valid within the scope. Do not return or store them externally.
