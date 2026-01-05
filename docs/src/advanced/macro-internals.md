# How `@with_pool` Works

This page explains the internal mechanics of the `@with_pool` macro for advanced users and contributors who want to understand the optimization strategies.

## Overview

The `@with_pool` macro provides automatic lifecycle management with three key optimizations:

1. **Try-Finally Safety** — Guarantees cleanup even on exceptions
2. **Typed Checkpoint/Rewind** — Only saves/restores used types (~77% faster)
3. **Untracked Acquire Detection** — Safely handles `acquire!` calls outside macro visibility

## Basic Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────┐
│  @with_pool pool function foo(x)                            │
│      A = acquire!(pool, Float64, 100)                       │
│      B = similar!(pool, A)                                  │
│      return sum(A) + sum(B)                                 │
│  end                                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────────────────────────┐
              │       Macro Transformation        │
              └───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  function foo(x)                                            │
│      pool = get_task_local_pool()                           │
│      checkpoint!(pool, Float64)     # ← Type-specific       │
│      try                                                    │
│          A = _acquire_impl!(pool, Float64, 100)             │
│          B = _similar_impl!(pool, A)                        │
│          return sum(A) + sum(B)                             │
│      finally                                                │
│          rewind!(pool, Float64)     # ← Type-specific       │
│      end                                                    │
│  end                                                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Points

- **`try-finally`** ensures `rewind!` executes even if an exception occurs
- `acquire!` → `_acquire_impl!` transformation bypasses untracked marking overhead
- Type-specific `checkpoint!(pool, Float64)` is ~77% faster than full checkpoint

## Type Extraction: Static Analysis at Compile Time

The macro analyzes the AST to extract types used in `acquire!` calls:

```julia
# Macro sees these acquire! calls:
@with_pool pool begin
    A = acquire!(pool, Float64, 10, 10)     # → extracts Float64
    B = zeros!(pool, ComplexF64, 100)       # → extracts ComplexF64
    C = similar!(pool, A)                    # → extracts eltype(A) → Float64
end

# Generated code uses typed checkpoint/rewind:
checkpoint!(pool, Float64, ComplexF64)
try
    ...
finally
    rewind!(pool, Float64, ComplexF64)
end
```

### Type Extraction Rules

| Call Pattern | Extracted Type |
|--------------|----------------|
| `acquire!(pool, Float64, dims...)` | `Float64` |
| `acquire!(pool, x)` | `eltype(x)` (if x is external) |
| `zeros!(pool, dims...)` | `default_eltype(pool)` |
| `zeros!(pool, Float32, dims...)` | `Float32` |
| `similar!(pool, x)` | `eltype(x)` |
| `similar!(pool, x, Int64, ...)` | `Int64` |

### When Type Extraction Fails → Full Checkpoint

The macro falls back to full `checkpoint!(pool)` when:

```julia
@with_pool pool begin
    T = eltype(data)                  # T defined locally AFTER checkpoint
    A = acquire!(pool, T, 100)        # Can't use T at checkpoint time!
end
# → Falls back to checkpoint!(pool) / rewind!(pool)

@with_pool pool begin
    local_arr = compute()             # local_arr defined AFTER checkpoint
    B = similar!(pool, local_arr)     # eltype(local_arr) unavailable
end
# → Falls back to checkpoint!(pool) / rewind!(pool)
```

## Untracked Acquire Detection

### The Problem

The macro can only see `acquire!` calls **directly in its AST**. Calls inside helper functions are invisible:

```julia
function helper!(pool)
    return zeros!(pool, Float64, 100)   # Macro can't see this!
end

@with_pool pool begin
    A = acquire!(pool, Int64, 10)       # ← Macro sees this (Int64)
    B = helper!(pool)                    # ← Macro can't see Float64 inside!
end

# If only checkpoint!(pool, Int64), Float64 arrays won't be rewound!
```

### The Solution: `_untracked_flags`

Every `acquire!` call (and convenience functions) marks itself as "untracked":

```julia
# Public API (called from user code outside macro)
@inline function acquire!(pool, ::Type{T}, n::Int) where {T}
    _mark_untracked!(pool)              # ← Sets flag!
    _acquire_impl!(pool, T, n)
end

# Macro-transformed calls skip the marking
# (because macro already knows about them)
_acquire_impl!(pool, T, n)               # ← No flag
```

### Flow Diagram

```
@with_pool pool begin                    State of pool._untracked_flags
    │                                    ─────────────────────────────────
    ├─► checkpoint!(pool, Int64)         depth=2, flag[2]=false
    │
    │   A = _acquire_impl!(...)          (macro-transformed, no flag set)
    │   B = helper!(pool)
    │       └─► zeros!(pool, Float64, N)
    │           └─► _mark_untracked!(pool)  flag[2]=TRUE ←──┐
    │                                                        │
    │   ... more code ...                                    │
    │                                                        │
    └─► rewind! check:                                       │
        if pool._untracked_flags[2]  ─────────────────────────┘
            rewind!(pool)            # Full rewind (safe)
        else
            rewind!(pool, Int64)     # Typed rewind (fast)
        end
end
```

### Why This Works

1. **Macro-tracked calls**: Transformed to `_acquire_impl!` → no flag → typed rewind
2. **Untracked calls**: Use public API → sets flag → triggers full rewind
3. **Result**: Always safe, with optimization when possible

## Nested `@with_pool` Handling

Each `@with_pool` maintains its own checkpoint depth:

```
@with_pool p1 begin                      depth: 1 → 2
    v1 = acquire!(p1, Float64, 10)
    │
    ├─► @with_pool p2 begin              depth: 2 → 3
    │       v2 = acquire!(p2, Int64, 5)
    │       helper!(p2)                  # sets flag[3]=true
    │       sum(v2)
    │   end                              depth: 3 → 2, flag[3] checked
    │
    │   # v1 still valid here!
    sum(v1)
end                                      depth: 2 → 1, flag[2] checked
```

### Depth Tracking Data Structures

```julia
struct AdaptiveArrayPool
    # ... type pools ...
    _current_depth::Int              # Current scope depth (1 = global)
    _untracked_flags::Vector{Bool}   # Per-depth flag array
end

# Initialized with sentinel:
_current_depth = 1                   # Global scope
_untracked_flags = [false]           # Sentinel for depth=1
```

## Performance Impact

| Scenario | Checkpoint Method | Relative Speed |
|----------|-------------------|----------------|
| 1 type, no untracked | `checkpoint!(pool, T)` | **~77% faster** |
| Multiple types, no untracked | `checkpoint!(pool, T1, T2, ...)` | **~50% faster** |
| Any untracked acquire | `checkpoint!(pool)` | Baseline |

The optimization matters most in tight loops with many iterations.

## Code Generation Summary

```julia
# INPUT
@with_pool pool function compute(data)
    A = acquire!(pool, Float64, length(data))
    result = helper!(pool, A)  # May have untracked acquires
    return result
end

# OUTPUT (simplified)
function compute(data)
    pool = get_task_local_pool()

    # Check if parent scope had untracked (for nested pools)
    if pool._untracked_flags[pool._current_depth]
        checkpoint!(pool)                    # Full checkpoint
    else
        checkpoint!(pool, Float64)           # Typed checkpoint
    end

    try
        A = _acquire_impl!(pool, Float64, length(data))
        result = helper!(pool, A)
        return result
    finally
        # Check if untracked acquires occurred in this scope
        if pool._untracked_flags[pool._current_depth]
            rewind!(pool)                    # Full rewind
        else
            rewind!(pool, Float64)           # Typed rewind
        end
    end
end
```

## Key Internal Functions

| Function | Purpose |
|----------|---------|
| `_extract_acquire_types(expr, pool_name)` | AST walk to find types |
| `_filter_static_types(types, local_vars)` | Filter out locally-defined types |
| `_transform_acquire_calls(expr, pool_name)` | Replace `acquire!` → `_acquire_impl!` |
| `_mark_untracked!(pool)` | Set untracked flag for current depth |
| `_generate_typed_checkpoint_call(pool, types)` | Generate `checkpoint!(pool, T...)` |

## See Also

- [Internals](internals.md) — Overview of pool architecture
- [Safety Rules](../guide/safety.md) — Scope rules and best practices
- [Configuration](../usage/configuration.md) — Performance tuning options
