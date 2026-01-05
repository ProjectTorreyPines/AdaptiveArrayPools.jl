# `@maybe_with_pool`

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
- `pool` becomes `DisabledPool{backend}()` (e.g., `DisabledPool{:cpu}()` or `DisabledPool{:cuda}()`)
- All pool functions (`acquire!`, `zeros!`, etc.) fall back to standard allocation
- Backend context is preserved: `:cuda` → `CuArray`, `:cpu` → `Array`

Use `pooling_enabled(pool)` to check if pooling is active:
```julia
@maybe_with_pool pool begin
    if pooling_enabled(pool)
        # Using pooled memory
    else
        # Using standard allocation (DisabledPool)
    end
end
```

## vs @with_pool

| | `@with_pool` | `@maybe_with_pool` |
|---|---|---|
| Runtime toggle | No | Yes |
| Overhead when disabled | None | Branch check |
| Use case | Application code | Library code |

## Safety

Same rules as `@with_pool`: arrays are only valid within the scope. Do not return or store them externally.
