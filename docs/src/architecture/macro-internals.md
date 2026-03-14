# How `@with_pool` Works

This page explains the internal mechanics of the `@with_pool` macro for advanced users and contributors who want to understand the optimization strategies.

## Overview

The `@with_pool` macro provides automatic lifecycle management with three key optimizations:

1. **Direct Rewind (no `try-finally`)** — Enables compiler inlining for ~35-73% less overhead
2. **Typed Checkpoint/Rewind** — Only saves/restores used types (~77% faster)
3. **Untracked Acquire Detection** — Safely handles `acquire!` calls outside macro visibility

## Why No `try-finally`?

Julia's compiler cannot inline functions containing `try-finally`. For `@inline @with_pool`
functions called in hot loops, this means every call pays an exception handler frame cost
(~20-40ns on modern hardware, worse on Julia 1.10 LTS).

`@with_pool` avoids this by inserting `rewind!` directly at every exit point instead:

| Macro | Strategy | Inlinable | Use case |
|-------|----------|:---------:|----------|
| `@with_pool` | Direct rewind at each exit point | Yes | Default — hot paths |
| `@safe_with_pool` | `try-finally` wrapper | No | Exception safety required |

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
│      _entry_depth = pool._current_depth                    │
│      checkpoint!(pool, Float64)     # ← Type-specific       │
│                                                             │
│      A = _acquire_impl!(pool, Float64, 100)                │
│      B = _similar_impl!(pool, A)                           │
│      _result = sum(A) + sum(B)                             │
│                                                             │
│      # Entry depth guard (cleans up leaked inner scopes)    │
│      while pool._current_depth > _entry_depth + 1          │
│          rewind!(pool)                                      │
│      end                                                    │
│      rewind!(pool, Float64)     # ← Type-specific           │
│      return _result                                         │
│  end                                                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Points

- `rewind!` is inserted at **every exit point**: implicit return, explicit `return`, `break`, `continue`
- `acquire!` → `_acquire_impl!` transformation bypasses untracked marking overhead
- Type-specific `checkpoint!(pool, Float64)` is ~77% faster than full checkpoint

### Exit Point Coverage

| Exit type | Handling |
|-----------|----------|
| Implicit return (end of body) | `rewind!` appended before result |
| Explicit `return` | `rewind!` inserted before each `return` statement |
| `break` / `continue` | `rewind!` inserted before each (block form only) |
| `@goto` (internal) | Allowed — stays within pool scope |
| `@goto` (external) | Hard error at macro expansion time |
| Uncaught exception | **Not handled** — use `@safe_with_pool` or `reset!(pool)` |

## Exception Behavior

### `@with_pool` (direct rewind)

Without `try-finally`, uncaught exceptions skip `rewind!`. This is an intentional trade-off:

```julia
# Uncaught exception → pool state invalid
try
    @with_pool pool begin
        acquire!(pool, Float64, 10)
        error("boom")       # rewind! never called
    end
catch
end
# pool._current_depth is wrong here → call reset!(pool)
```

### Entry Depth Guard (nested catch recovery)

When an inner `@with_pool` throws and the outer scope catches, the outer's exit
automatically cleans up leaked inner scopes:

```julia
@with_pool pool function outer()
    v = acquire!(pool, Float64, 10)
    result = try
        @with_pool pool begin
            acquire!(pool, UInt8, 5)
            error("inner boom")   # inner rewind! skipped
        end
    catch
        42                        # pool depth is wrong HERE
    end
    sum(v) + result
    # Entry depth guard runs here → cleans up leaked inner scope
    # Own rewind! runs → outer scope cleaned up
end
```

!!! warning "Catch block limitation"
    Between the inner throw and the outer scope's exit, pool depth is incorrect.
    Do not use pool operations inside the `catch` block.

### `@safe_with_pool` (try-finally)

For code that may throw and needs guaranteed cleanup:

```julia
@safe_with_pool pool begin
    acquire!(pool, Float64, 10)
    risky_operation()          # if this throws, rewind! still runs
end
```

This prevents inlining but guarantees pool cleanup regardless of exceptions.
Use it when:
- The pool body calls functions that may throw
- You need the pool to remain valid after a caught exception
- A custom macro inside the body might generate hidden `return`/`break`/`continue`

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
# ... body with rewind! at each exit ...
rewind!(pool, Float64, ComplexF64)
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

### The Solution: Bitmask-Based Type Touch Tracking

Every `acquire!` call (and convenience functions) records the type touch with type-specific bitmask information:

```julia
# Public API (called from user code outside macro)
@inline function acquire!(pool, ::Type{T}, n::Int) where {T}
    _record_type_touch!(pool, T)         # ← Records type-specific bitmask!
    _acquire_impl!(pool, T, n)
end

# Macro-transformed calls skip the recording
# (because macro already knows about them)
_acquire_impl!(pool, T, n)               # ← No recording
```

Each fixed-slot type maps to a bit in a `UInt16` bitmask via `_fixed_slot_bit(T)`.
Non-fixed-slot types set a separate `_touched_has_others` flag.

### Flow Diagram

```
@with_pool pool begin                    Bitmask state at depth 2
    │                                    ─────────────────────────────
    ├─► checkpoint!(pool, Int64)         masks[2]=0x0000, others[2]=false
    │
    │   A = _acquire_impl!(...)          (macro-transformed, no mark)
    │   B = helper!(pool)
    │       └─► zeros!(pool, Float64, N)
    │           └─► _record_type_touch!(pool, Float64)
    │               masks[2] |= 0x0001 (Float64 bit) ←───┐
    │                                                      │
    │   ... more code ...                                  │
    │                                                      │
    └─► rewind! check:                                     │
        tracked_mask = _tracked_mask_for_types(Int64)      │
        if _can_use_typed_path(pool, tracked_mask) ────────┘
            rewind!(pool, Int64)     # Typed rewind (fast)
        else                         # Float64 not in {Int64} → full
            rewind!(pool)            # Full rewind (safe)
        end
end
```

### Why This Works

1. **Macro-tracked calls**: Transformed to `_acquire_impl!` → no bitmask touch → typed path
2. **External calls**: Use public API → records type-specific bitmask → subset check at rewind
3. **Subset optimization**: If touched types are a subset of tracked types, the typed path is still safe
4. **Result**: Always safe, with finer-grained optimization than a single boolean flag

## Nested `@with_pool` Handling

Each `@with_pool` maintains its own checkpoint depth:

```
@with_pool p1 begin                      depth: 1 → 2
    v1 = acquire!(p1, Float64, 10)
    │
    ├─► @with_pool p2 begin              depth: 2 → 3
    │       v2 = acquire!(p2, Int64, 5)
    │       helper!(p2)                  # marks bitmask at depth 3
    │       sum(v2)
    │   end                              depth: 3 → 2, bitmask checked
    │
    │   # v1 still valid here!
    sum(v1)
end                                      depth: 2 → 1, bitmask checked
```

### Depth Tracking Data Structures

```julia
struct AdaptiveArrayPool
    # ... type pools ...
    _current_depth::Int                     # Current scope depth (1 = global)
    _touched_type_masks::Vector{UInt16}     # Per-depth: which fixed slots were touched
    _touched_has_others::Vector{Bool}       # Per-depth: any non-fixed-slot type touched
end

# Initialized with sentinel:
_current_depth = 1                          # Global scope
_touched_type_masks = [UInt16(0)]           # Sentinel for depth=1
_touched_has_others = [false]               # Sentinel for depth=1
```

## Performance Impact

| Scenario | Checkpoint Method | Relative Speed |
|----------|-------------------|----------------|
| 1 type, no untracked | `checkpoint!(pool, T)` | **~77% faster** |
| Multiple types, no untracked | `checkpoint!(pool, T1, T2, ...)` | **~50% faster** |
| Untracked subset of tracked | `checkpoint!(pool, T...)` | **~77% faster** |
| Unknown untracked types | `checkpoint!(pool)` | Baseline |

The optimization matters most in tight loops with many iterations. The bitmask subset
check allows the typed path even when untracked acquires occur, as long as those types
are already covered by the macro's tracked set.

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
    _entry_depth = pool._current_depth

    # Bitmask subset check: can typed path handle any untracked acquires?
    if _can_use_typed_path(pool, _tracked_mask_for_types(Float64))
        checkpoint!(pool, Float64)           # Typed checkpoint (fast)
    else
        checkpoint!(pool)                    # Full checkpoint (safe)
    end

    A = _acquire_impl!(pool, Float64, length(data))
    _result = helper!(pool, A)

    # Entry depth guard: clean up any leaked inner scopes
    while pool._current_depth > _entry_depth + 1
        rewind!(pool)
    end

    # Own scope rewind
    if _can_use_typed_path(pool, _tracked_mask_for_types(Float64))
        rewind!(pool, Float64)              # Typed rewind (fast)
    else
        rewind!(pool)                       # Full rewind (safe)
    end

    return _result
end
```

## Key Internal Functions

| Function | Purpose |
|----------|---------|
| `_extract_acquire_types(expr, pool_name)` | AST walk to find types |
| `_filter_static_types(types, local_vars)` | Filter out locally-defined types |
| `_transform_acquire_calls(expr, pool_name)` | Replace `acquire!` → `_acquire_impl!` |
| `_transform_return_stmts(expr, ...)` | Insert `rewind!` before each `return` |
| `_transform_break_continue(expr, ...)` | Insert `rewind!` before `break`/`continue` |
| `_check_unsafe_goto(expr)` | Hard error on `@goto` that exits pool scope |
| `_record_type_touch!(pool, T)` | Record type touch in bitmask for current depth |
| `_can_use_typed_path(pool, mask)` | Bitmask subset check for typed vs full path |
| `_tracked_mask_for_types(T...)` | Compile-time bitmask for tracked types |
| `_generate_typed_checkpoint_call(pool, types)` | Generate bitmask-aware checkpoint |

## See Also

- [How It Works](how-it-works.md) — Overview of pool architecture
- [Pool Safety](../safety/overview.md) — Scope rules, compile-time & runtime safety
- [Configuration](../features/configuration.md) — Performance tuning options
