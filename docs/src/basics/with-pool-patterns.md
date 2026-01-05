# @with_pool Patterns

The `@with_pool` macro provides automatic memory lifecycle management. It supports two usage patterns depending on your needs.

## Pattern 1: Function Decorator

Wraps an entire function with pool management. The pool is active for the function's full duration.

```julia
@with_pool pool function compute(n)
    A = acquire!(pool, Float64, n, n)
    B = zeros!(pool, Float64, n)

    # ... compute with A and B ...

    return sum(A) + sum(B)  # Return computed values, not arrays
end

# Usage
result = compute(100)  # Zero-allocation after warmup
```

**Best for:**
- Functions that exclusively use pooled arrays
- Hot-path functions called repeatedly
- Clear ownership semantics

## Pattern 2: Block Wrapper

Wraps only a portion of a function. Useful when you need pool arrays for part of the computation.

```julia
function process_data(data)
    # Pre-processing (no pool needed)
    n = length(data)

    @with_pool pool begin
        # Pool is only active inside this block
        temp = acquire!(pool, Float64, n)
        temp .= data .* 2
        result = sum(temp)
    end  # Pool arrays recycled here

    # Post-processing
    return result * 1.5
end
```

**Best for:**
- Functions with mixed allocation needs
- Gradual adoption in existing code
- Fine-grained scope control

## Pattern Comparison

| Aspect | Function Decorator | Block Wrapper |
|--------|-------------------|---------------|
| Scope | Entire function | begin...end block |
| Syntax | `@with_pool pool function ...` | `@with_pool pool begin ... end` |
| Pool lifetime | Function start to return | Block entry to exit |
| Nesting | Functions can call each other | Blocks can be nested |

## Nested Pools

Both patterns support nesting. Each scope maintains independent checkpoint state:

```julia
@with_pool pool function outer(n)
    A = acquire!(pool, Float64, n)

    @with_pool pool begin
        # Inner scope - new checkpoint
        B = acquire!(pool, Float64, n * 2)
        inner_result = sum(B)
    end  # B recycled here

    # A still valid here
    return sum(A) + inner_result
end
```

## Common Mistakes

### Returning pool arrays (wrong)

```julia
@with_pool pool function bad()
    v = acquire!(pool, Float64, 100)
    return v  # v is recycled after return!
end
```

### Correct: return computed values

```julia
@with_pool pool function good()
    v = acquire!(pool, Float64, 100)
    return sum(v)  # Scalar result is safe
end

# Or copy if you need the array
@with_pool pool function also_good()
    v = acquire!(pool, Float64, 100)
    return copy(v)  # Explicit copy is safe
end
```

## See Also

- [Essential API](api-essentials.md) - Core functions for pool operations
- [Safety Rules](safety-rules.md) - Important scope rules
