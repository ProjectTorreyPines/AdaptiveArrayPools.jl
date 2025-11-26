# @with_pool

Use an explicit pool instance. For advanced cases where you manage your own pool.

## Block Mode

```julia
pool = AdaptiveArrayPool()

result = @with_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 1.0
    sum(v)
end  # Arrays returned to pool
```

## Function Mode (kwarg injection)

```julia
@with_pool pool function compute(x)
    temp = acquire!(pool, Float64, length(x))
    temp .= x .* 2
    sum(temp)
end

# Called with or without pool
compute([1.0, 2.0])              # pool=nothing, allocates
compute([1.0, 2.0]; pool=mypool) # uses mypool
```

## When to Use

- Testing with isolated pool instances
- Multi-threaded code with per-thread pools
- Fine-grained control over pool lifecycle

## vs @use_pool

| | `@use_pool` | `@with_pool` |
|---|---|---|
| Pool source | Task-local (automatic) | Explicit instance |
| Setup | None | Create `AdaptiveArrayPool()` |
| Use case | Most code | Advanced/testing |
