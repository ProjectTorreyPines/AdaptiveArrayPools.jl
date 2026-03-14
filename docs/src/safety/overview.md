# Overview

Pool arrays are **temporary** — valid only within their `@with_pool` scope. When the scope ends, the memory is marked for reuse.

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    sum(v)       # ✅ return computed values
    copy(v)      # ✅ return owned copies
end
# v is now invalid — its memory may be reused by the next acquire!
```

```julia
@with_pool pool function bad()
    v = acquire!(pool, Float64, 100)
    return v     # ❌ v escapes the pool scope
end
```

**You don't have to catch these yourself.** AdaptiveArrayPools provides a two-layer safety net:

| Layer | What It Catches | Cost |
|-------|----------------|------|
| [**Compile-time**](compile-time.md) | Obvious escapes, structural mutations | Zero (macro analysis) |
| [**Runtime**](runtime.md) (`RUNTIME_CHECK = 1`) | Everything else: opaque escapes, hidden mutations | ~5ns/slot, off by default |

Compile-time analysis is always active. Runtime safety is opt-in for development and fully eliminated in production — zero overhead.

## Quick Reference

### acquire! vs acquire_view!

| Function | Returns | Best For |
|----------|---------|----------|
| `acquire!` | Native `Array{T,N}` (or `BitArray{N}` for `Bit`) | General use, BLAS/LAPACK, FFI |
| `acquire_view!` | View types (`SubArray`, `ReshapedArray`) | When views are preferred |

Both follow the same scope rules. Use `acquire!` by default.

### Thread Safety

Pools are task-local — each thread automatically gets its own pool:

```julia
# ✅ Safe: each task has independent pool
Threads.@threads for i in 1:N
    @with_pool pool begin
        a = acquire!(pool, Float64, 100)
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
