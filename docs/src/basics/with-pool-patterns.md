# `@with_pool` Patterns

The `@with_pool` macro provides automatic memory lifecycle management. It supports two usage patterns depending on your needs.

## Pool Name: Choose Any Identifier

The first argument to `@with_pool` is a **variable name** you choose - it doesn't have to be `pool`:

```julia
@with_pool p function foo() ... end
@with_pool mypool function bar() ... end
@with_pool scratch function baz() ... end
```

Use whatever name makes your code clearest.

## Pattern 1: Function Decorator

Wraps an entire function with pool management:

```julia
@with_pool pool function compute(n)
    A = acquire!(pool, Float64, n, n)
    B = zeros!(pool, Float64, n)
    # ... compute ...
    return sum(A) + sum(B)
end

result = compute(100)  # Zero-allocation after warmup
```

**Best for:** Functions that exclusively use pooled arrays, hot-path functions.

## Pattern 2: Block Wrapper

Wraps only a portion of a function:

```julia
function process_data(data)
    n = length(data)

    @with_pool pool begin
        temp = acquire!(pool, Float64, n)
        temp .= data .* 2
        result = sum(temp)
    end  # temp marked for reuse here

    return result * 1.5
end
```

**Best for:** Functions with mixed allocation needs, gradual adoption.

## Pattern Comparison

| Aspect | Function Decorator | Block Wrapper |
|--------|-------------------|---------------|
| Scope | Entire function | begin...end block |
| Syntax | `@with_pool pool function ...` | `@with_pool pool begin ... end` |
| Pool lifetime | Function start to return | Block entry to exit |

## Common Mistakes

```julia
# WRONG: returning the array itself
@with_pool pool function bad()
    v = acquire!(pool, Float64, 100)
    return v  # v marked for reuse after return!
end

# CORRECT: return computed values
@with_pool pool function good()
    v = acquire!(pool, Float64, 100)
    return sum(v)  # Scalar result is safe
end

# CORRECT: return a copy if you need the data
@with_pool pool function also_good()
    v = acquire!(pool, Float64, 100)
    return copy(v)
end
```

## See Also

- [Essential API](api-essentials.md) - Core functions for pool operations
- [Safety Rules](safety-rules.md) - Important scope rules
