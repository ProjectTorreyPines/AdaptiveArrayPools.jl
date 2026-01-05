# Advanced Pool Patterns

This page covers advanced usage patterns for experienced users.

## Calling Other `@with_pool` Functions

Each `@with_pool` function manages its own checkpoint. They can call each other freely:

```julia
@with_pool pool function step1(n)
    A = zeros!(pool, Float64, n)
    fill!(A, 1.0)
    return sum(A)
end

@with_pool pool function step2(n)
    B = zeros!(pool, Float64, n)
    fill!(B, 2.0)
    return sum(B)
end

@with_pool pool function pipeline(n)
    a = step1(n)   # step1's arrays marked for reuse when it returns
    b = step2(n)   # step2's arrays marked for reuse when it returns
    C = acquire!(pool, Float64, n)
    fill!(C, a + b)
    return sum(C)
end
```

## Passing Pool as Argument

For complex call hierarchies, use `@with_pool` only at the top level and pass the pool through function arguments:

```julia
# Inner functions receive pool as argument - no @with_pool needed
function compute_step!(pool, data, result)
    temp = acquire!(pool, Float64, length(data))
    temp .= data .* 2
    result[] += sum(temp)
end

function process_chunk!(pool, chunk, result)
    temp = zeros!(pool, Float64, length(chunk))
    compute_step!(pool, chunk, temp)
    result[] += sum(temp)
end

# Only the entry point uses @with_pool
@with_pool pool function main_computation(chunks)
    result = Ref(0.0)
    for chunk in chunks
        process_chunk!(pool, chunk, result)
    end
    return result[]
end
```

**Benefits:**
- Single checkpoint/rewind at top level
- Inner functions are simpler (no macro overhead)
- Pool lifetime is explicit and controlled

## Direct Pool Access in Inner Functions

An alternative to passing pool as argument: inner functions call `get_task_local_pool()` directly, while a top-level `@with_pool` function controls the lifecycle.

```julia
# Inner functions access pool directly - no argument needed
function compute_step!(data, result)
    pool = get_task_local_pool()  # Direct access
    temp = acquire!(pool, Float64, length(data))
    temp .= data .* 2
    result[] += sum(temp)
    # temp NOT released here - stays active
end

function process_chunk!(chunk, accumulator)
    pool = get_task_local_pool()  # Direct access
    buffer = zeros!(pool, Float64, length(chunk))
    compute_step!(chunk, buffer)
    accumulator[] += sum(buffer)
    # buffer NOT released here - stays active
end

# Top-level controls lifecycle with @with_pool
@with_pool pool function main_pipeline(chunks)
    #  checkpoint!() ─────────────────────────────────┐
    accumulator = Ref(0.0)                          # │
    for chunk in chunks                             # │
        process_chunk!(chunk, accumulator)          # │  All arrays from
        # └─ compute_step! allocates temp           # │  inner functions
        # └─ process_chunk! allocates buffer        # │  accumulate here
    end                                             # │
    return accumulator[]                            # │
    #  rewind!() ─────────────────────────────────────┘
    #     └─ ALL arrays (temp, buffer, ...) marked for reuse
end
```

### Memory Flow Visualization

```
main_pipeline(chunks)          Inner Functions
       │
  checkpoint!()
       │
       ├──► process_chunk!()
       │         │
       │         ├──► get_task_local_pool() ──► buffer allocated
       │         │
       │         └──► compute_step!()
       │                   │
       │                   └──► get_task_local_pool() ──► temp allocated
       │
       ├──► process_chunk!()  (next iteration)
       │         └──► ... more allocations ...
       │
       ▼
    rewind!()  ◄─────── ALL arrays marked for reuse
```

### ⚠️ User Responsibility Warning

This pattern requires **you** to guarantee that inner functions are **always** called through a `@with_pool` entry point:

```julia
# SAFE: Called through main_pipeline
main_pipeline(my_chunks)  # ✓ Lifecycle managed

# DANGEROUS: Direct call without @with_pool wrapper
compute_step!(some_data, some_ref)  # ✗ No checkpoint/rewind!
# └─ Arrays allocated but NEVER marked for reuse → pool grows unboundedly
```

**When to use this pattern:**
- Deep call hierarchies where threading pool through every function is tedious
- Performance-critical code where you want to avoid argument passing overhead
- You can enforce that all entry points use `@with_pool`

**When to prefer "Passing Pool as Argument":**
- Functions may be called from various contexts (some pooled, some not)
- Library code where you can't control the caller
- You want explicit documentation of pool dependency in function signatures

## Manual Checkpoint/Rewind

For fine-grained control, use `checkpoint!` and `rewind!` directly:

```julia
function manual_control()
    pool = get_task_local_pool()

    checkpoint!(pool)
    try
        A = acquire!(pool, Float64, 100)
        B = acquire!(pool, Float64, 100)
        # ... compute ...
        return sum(A) + sum(B)
    finally
        rewind!(pool)
    end
end
```

This is what `@with_pool` generates internally. Use manual control when:
- Integrating with existing try/catch blocks
- Conditional checkpoint/rewind logic needed
- Building custom pool management abstractions

## Scope-Only `@with_pool`

You can omit the pool name when inner functions handle their own acquire:

```julia
@with_pool p function step1()
    v = acquire!(p, Float64, 100)
    sum(v)
end

@with_pool p function step2()
    v = acquire!(p, Float64, 200)
    sum(v)
end

# Outer function just provides scope management
@with_pool function orchestrate()
    a = step1()
    b = step2()
    return a + b
end
```

The name-less `@with_pool` still performs checkpoint/rewind but doesn't expose the pool variable. This is useful when you're orchestrating other `@with_pool` functions.

## See Also

- [`@with_pool` Patterns](../basics/with-pool-patterns.md) - Basic usage patterns
- [Safety Rules](../basics/safety-rules.md) - Scope rules
