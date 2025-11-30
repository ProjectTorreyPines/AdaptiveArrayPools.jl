# @maybe_with_pool

Runtime-toggleable pooling. Users can enable/disable via `MAYBE_POOLING_ENABLED[]`.

## Usage

```julia
@maybe_with_pool pool function compute(n)
    v = acquire!(pool, Float64, n)
    v .= 1.0
    sum(v)
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

## vs @with_pool

| | `@with_pool` | `@maybe_with_pool` |
|---|---|---|
| Runtime toggle | No | Yes |
| Overhead when disabled | None | Branch check |
| Use case | Application code | Library code |

## Safety

Same rules as `@with_pool`: arrays are only valid within the scope. Do not return or store them externally.
