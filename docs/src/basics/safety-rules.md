# Safety Rules

AdaptiveArrayPools achieves zero allocation by reusing memory across calls. This requires understanding one critical rule.

---

## The One Rule

```
+-------------------------------------------------------------+
|                                                             |
|  Pool arrays are ONLY valid within their @with_pool scope   |
|                                                             |
|  When the scope ends, the memory is recycled.               |
|  Using arrays after scope ends = UNDEFINED BEHAVIOR         |
|                                                             |
+-------------------------------------------------------------+
```

### What's Safe

| Pattern | Example | Why It Works |
|---------|---------|--------------|
| Return computed values | `return sum(v)` | Scalar escapes, not the array |
| Return copies | `return copy(v)` | New allocation, independent data |
| Use within scope | `result = A * B` | Arrays valid during computation |

### What's Dangerous

| Pattern | Example | Why It Fails |
|---------|---------|--------------|
| Return array | `return v` | Array recycled after return |
| Store in global | `global_ref = v` | Points to recycled memory |
| Capture in closure | `() -> sum(v)` | v invalid when closure runs |

---

## The Scope Rule in Detail

When `@with_pool` ends, all arrays acquired within that scope are recycled. Using them after the scope ends leads to undefined behavior.

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)

    result = sum(v)  # ✅ compute and return values
    copied = copy(v) # ✅ copy if you need data outside
end
# v is no longer valid here
```

## What NOT to Do

### Don't return pool-backed arrays

```julia
# ❌ Wrong: returning the array itself
@with_pool pool function bad_example()
    v = acquire!(pool, Float64, 100)
    return v  # v will be recycled after this returns!
end

# ✅ Correct: return computed values or copies
@with_pool pool function good_example()
    v = acquire!(pool, Float64, 100)
    return sum(v)  # scalar result
end
```

### Don't store in globals or closures

```julia
# ❌ Wrong: storing in global
global_ref = nothing
@with_pool pool begin
    global_ref = acquire!(pool, Float64, 100)
end
# global_ref now points to recycled memory

# ❌ Wrong: capturing in closure
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    callback = () -> sum(v)  # v captured but will be invalid
end
```

### Don't resize or push! to unsafe_acquire! arrays

```julia
@with_pool pool begin
    v = unsafe_acquire!(pool, Float64, 100)
    # ❌ These break pool memory management:
    # resize!(v, 200)
    # push!(v, 1.0)
    # append!(v, [1.0, 2.0])
end
```

## Debugging with POOL_DEBUG

Enable runtime safety checks during development:

```julia
using AdaptiveArrayPools
AdaptiveArrayPools.POOL_DEBUG[] = true

@with_pool pool function test()
    v = acquire!(pool, Float64, 100)
    return v  # Will warn about returning pool-backed array
end
```

## acquire! vs unsafe_acquire!

| Function | Returns | Best For |
|----------|---------|----------|
| `acquire!` | View types (`SubArray`, `ReshapedArray`) | General use, BLAS/LAPACK |
| `unsafe_acquire!` | Native `Array`/`CuArray` | FFI, type constraints |

Both follow the same scope rules. Use `acquire!` by default—views work with all standard Julia linear algebra operations.

## Thread Safety

Pools are task-local, so each thread automatically gets its own pool:

```julia
# ✅ Safe: each task has independent pool
Threads.@threads for i in 1:N
    @with_pool pool begin
        a = acquire!(pool, Float64, 100)
        # work with a...
    end
end

# ❌ Unsafe: pool created outside threaded region
@with_pool pool begin
    Threads.@threads for i in 1:N
        a = acquire!(pool, Float64, 100)  # race condition!
    end
end
```

See [Multi-Threading](../features/multi-threading.md) for more patterns.
