# @pool_kwarg

Injects `pool` as a keyword argument into a function definition. Unlike `@with_pool`, it does **not** manage checkpoint/rewind automatically - the caller is responsible for pool state management.

## Usage

```julia
# Define a composable function that can use pooling
@pool_kwarg pool function compute_layer(x)
    temp = acquire!(pool, Float64, length(x))
    temp .= x .* 2
    sum(temp)
end

# Call without pool - normal allocation
result = compute_layer([1.0, 2.0, 3.0])

# Call with pool - caller manages lifecycle
pool = AdaptiveArrayPool()
checkpoint!(pool)
result = compute_layer([1.0, 2.0, 3.0]; pool=pool)
rewind!(pool)
```

## When to Use

- **Library internals**: Functions that may be called from pooled or non-pooled contexts
- **Composable layers**: Building blocks that callers combine with their own pool management
- **Performance-critical code**: When you want to checkpoint once at the top level and call multiple pooled functions without intermediate checkpoint/rewind overhead

## How It Works

`@pool_kwarg pool function f(x) ... end` expands to:

```julia
function f(x; pool::Union{AdaptiveArrayPool, Nothing} = nothing)
    ...
end
```

- When `pool === nothing`: `acquire!` allocates normally
- When `pool` is provided: `acquire!` uses the pool

## vs @with_pool

| | `@with_pool` | `@pool_kwarg` |
|---|---|---|
| Checkpoint/rewind | Automatic | Manual (caller's job) |
| Pool source | Global task-local | Passed via kwarg |
| Use case | Top-level entry points | Composable building blocks |

## Example: Layered Computation

```julia
# Define layers with @pool_kwarg
@pool_kwarg pool function layer1(x)
    temp = acquire!(pool, Float64, length(x))
    temp .= x .+ 1
    temp
end

@pool_kwarg pool function layer2(x)
    temp = acquire!(pool, Float64, length(x))
    temp .= x .* 2
    temp
end

# Top-level uses @with_pool for automatic management
@with_pool pool function pipeline(x)
    y = layer1(x; pool=pool)
    z = layer2(y; pool=pool)
    sum(z)
end

# Single checkpoint/rewind for entire pipeline
result = pipeline([1.0, 2.0, 3.0])
```

## Safety

Same rules apply: arrays acquired via `@pool_kwarg` functions are only valid until `rewind!` is called. The caller must ensure proper lifecycle management.
