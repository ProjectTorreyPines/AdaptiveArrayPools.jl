# Multi-Threading Guide

AdaptiveArrayPools uses `task_local_storage()` for **task-local isolation**: each Julia Task gets its own independent pool. This design ensures thread safety when used correctly.

## Table of Contents

- [Understanding Julia's Task/Thread Model](#understanding-julias-taskthread-model)
- [How Pools Work with @threads](#how-pools-work-with-threads)
- [Safe Patterns](#safe-patterns)
- [Unsafe Patterns](#unsafe-patterns)
- [Why Task-Local (Not Thread-Local)?](#why-task-local-not-thread-local)
- [User Responsibility](#user-responsibility)

---

## Understanding Julia's Task/Thread Model

Julia uses an **M:N threading model** where multiple Tasks (lightweight coroutines) can run on multiple OS threads.

```
┌─────────────────────────────────────────────────────────────┐
│                     Julia Process                            │
│                                                              │
│  Thread 1              Thread 2              Thread 3        │
│  ┌─────────┐          ┌─────────┐          ┌─────────┐      │
│  │ Task A  │          │ Task C  │          │ Task E  │      │
│  │ (TLS-A) │          │ (TLS-C) │          │ (TLS-E) │      │
│  └─────────┘          └─────────┘          └─────────┘      │
│  ┌─────────┐          ┌─────────┐                           │
│  │ Task B  │          │ Task D  │                           │
│  │ (TLS-B) │          │ (TLS-D) │                           │
│  └─────────┘          └─────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

Key concepts:

| Concept | Description |
|---------|-------------|
| **Thread** | OS-level execution unit. Fixed count at Julia startup. |
| **Task** | Julia's lightweight coroutine (Green Thread). Created dynamically. |
| **task_local_storage()** | Per-Task storage. Each Task has its own isolated TLS. |

### Important: One Thread Can Run Multiple Tasks

A single thread can execute multiple Tasks by switching between them at **yield points** (I/O, `sleep()`, `yield()`, etc.):

```julia
# Both tasks run on Thread 1, interleaved!
task_a = @spawn begin
    println("A start")
    sleep(0.1)      # yield point - switch to Task B
    println("A end")
end

task_b = @spawn begin
    println("B start")
    sleep(0.1)      # yield point - switch back to Task A
    println("B end")
end

# Output (single thread):
# A start
# B start
# A end
# B end
```

---

## How Pools Work with @threads

When you use `Threads.@threads`, Julia distributes iterations across threads. Each thread gets **one Task** that processes its assigned iterations.

```
Threads.@threads for i in 1:100_000   (4 threads)
│
├─ Thread 1: Task-1 → Pool-1
│   └─ Processes i = 1..25,000 (same pool reused for all!)
│
├─ Thread 2: Task-2 → Pool-2
│   └─ Processes i = 25,001..50,000
│
├─ Thread 3: Task-3 → Pool-3
│   └─ Processes i = 50,001..75,000
│
└─ Thread 4: Task-4 → Pool-4
    └─ Processes i = 75,001..100,000

Total: 4 pools created, each reused ~25,000 times
```

### Key Insight

- `@threads` creates **one Task per thread** (not one per iteration!)
- Each Task has its own `task_local_storage()` → its own pool
- Within one `@threads` block, pools are efficiently reused
- Calling `@threads` **multiple times** creates new Tasks → new pools each time

---

## Safe Patterns

### Pattern 1: `@with_pool` Inside `@threads`

```julia
Threads.@threads for i in 1:N
    @with_pool pool begin
        a = acquire!(pool, Float64, 100)
        # ... computation ...
    end  # pool automatically rewinds
end
```

Each thread's Task gets its own pool. Safe and efficient.

### Pattern 2: Function Defined with `@with_pool`

```julia
# Define function with @with_pool
@with_pool pool function inner_work(x)
    tmp = acquire!(pool, Float64, length(x))
    tmp .= x
    return sum(tmp)
end

# Call from @threads - each thread gets its own pool
Threads.@threads for i in 1:N
    result = inner_work(data[i])
end
```

The pool is created per-Task when the function is called, not when defined.

### Pattern 3: Nested Functions

```julia
@with_pool outer_pool function outer_work(data)
    # outer_pool belongs to Main Task
    tmp = acquire!(outer_pool, Float64, 100)

    Threads.@threads for i in 1:length(data)
        # inner_work creates its own pool per thread
        inner_work(data[i])  # Inner pool ≠ outer_pool (safe!)
    end
end
```

Outer and inner pools are completely independent.

---

## Unsafe Patterns

### Pattern 1: `@with_pool` Outside `@threads`

```julia
# ❌ DANGER: Race condition!
@with_pool pool Threads.@threads for i in 1:N
    a = acquire!(pool, Float64, 100)  # All threads share ONE pool!
end
```

**Why it fails**: `pool` is created in the Main Task's TLS. All threads access the same pool simultaneously.

### Pattern 2: Sharing Pool Reference

```julia
# ❌ DANGER: Race condition!
pool = get_global_pool()  # Main Task's pool
Threads.@threads for i in 1:N
    a = acquire!(pool, Float64, 100)  # Shared access!
end
```

### Pattern 3: Passing Pool to `@spawn`

```julia
# ❌ DANGER: Race condition!
@with_pool pool begin
    tasks = [Threads.@spawn begin
        a = acquire!(pool, Float64, 100)  # Multiple tasks, one pool!
    end for _ in 1:4]
    wait.(tasks)
end
```

---

## Why Task-Local (Not Thread-Local)?

You might wonder: "Why not use thread-local pools? They persist across `@threads` calls!"

### The Stack Discipline Problem

AdaptiveArrayPools uses `checkpoint!` and `rewind!` - a **stack-based** allocation system:

```julia
@with_pool pool begin
    checkpoint!(pool)  # Push current state
    a = acquire!(pool, ...)
    b = acquire!(pool, ...)
    # ...
    rewind!(pool)      # Pop and restore state (LIFO!)
end
```

This requires **strict LIFO ordering**: the Task that checkpoints first must rewind last.

### Why Thread-Local Fails with `@spawn`

With `@spawn`, multiple Tasks can interleave on the same thread:

```
Thread 1 (with Thread-Local Pool):

Time →
Task A: checkpoint! ──── acquire! ──── sleep ────────────── rewind!
Task B:        checkpoint! ──── acquire! ──── sleep ──── rewind!
                                                    ↑
                                           A finishes first!
```

**Stack corruption occurs:**

1. Task A: `checkpoint!` → stack = `[0]`
2. Task B: `checkpoint!` → stack = `[0, 1]`
3. Task A: `rewind!` → pops `1` (B's checkpoint!) → stack = `[0]`
4. Task B: `rewind!` → pops `0` (A's checkpoint!) → **WRONG!**

**Result**: B's arrays may be reused while B is still using them → memory corruption.

### Locks Don't Help

Adding locks only prevents **simultaneous access**, not **LIFO violations**. The stack still gets corrupted because Tasks finish in unpredictable order.

### Task-Local: The Only Safe Solution

With Task-local pools:
- Each Task has its own pool
- Each pool has its own stack
- No interleaving possible → LIFO always preserved

---

## User Responsibility

### The Core Rule

> **Pool objects must not be shared across Tasks.**

This library prioritizes **zero-overhead performance** over runtime safety checks. No locks are added because:

1. Locks would defeat the purpose of zero-allocation pooling
2. Even with locks, stack corruption would occur (LIFO violations)

### Quick Reference

| Pattern | Safety | Reason |
|---------|--------|--------|
| `@with_pool` inside `@threads` | ✅ Safe | Each Task gets own pool |
| `@with_pool` outside `@threads` | ❌ Unsafe | All threads share one pool |
| Function with `@with_pool` called from `@threads` | ✅ Safe | Pool created per-Task at call time |
| Passing pool to `@spawn` | ❌ Unsafe | Multiple Tasks access same pool |
| Nested `@with_pool` (outer/inner) | ✅ Safe | Each level has independent pool |

### Debugging Tips

If you encounter unexpected behavior:

1. **Check pool placement**: Is `@with_pool` inside or outside `@threads`?
2. **Check pool sharing**: Is the same pool variable accessed from multiple Tasks?
3. **Enable POOL_DEBUG**: `POOL_DEBUG[] = true` catches some (not all) misuse patterns

---

## Summary

- AdaptiveArrayPools uses **Task-local isolation** for thread safety
- Each Julia Task gets its own independent pool via `task_local_storage()`
- `@threads` creates one Task per thread → pools are reused within the block
- **Always place `@with_pool` inside `@threads`**, not outside
- Thread-local pools are **not an alternative** due to stack discipline requirements
- Correct usage is the user's responsibility (no runtime checks for performance)
