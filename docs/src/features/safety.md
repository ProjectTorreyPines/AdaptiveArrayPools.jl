# Pool Safety

AdaptiveArrayPools catches pool-escape bugs at **two levels**: compile-time (macro analysis) and runtime (configurable safety levels).

## Compile-Time Detection

The `@with_pool` macro statically analyzes your code and **rejects** any expression that would return a pool-backed array. This catches the most common mistakes at zero runtime cost.

```julia
# Direct array escape — caught at macro expansion time
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    v  # ← ERROR: v escapes the pool scope
end
```

This would throw an error message as follows:
```
ERROR: LoadError: PoolEscapeError (compile-time)

  The following variable escapes the @with_pool scope:

    v  ← pool-acquired view

  Declarations:
    [1]  v = acquire!(pool, Float64, 100)  [myfile.jl:2]

  Escaping return:
    [1]  v  [myfile.jl:3]

  Fix: Use collect(v) to return owned copies.
       Or use a regular Julia array (zeros()/Array{T}()) if it must outlive the pool scope.

in expression starting at myfile.jl:1
```

The analyzer tracks aliases, containers, and convenience wrappers:

```julia
# All of these are caught at compile time:
@with_pool pool begin
    v = zeros!(pool, Float64, 10)
    w = v            # alias of pool variable
    t = (1, v)       # tuple wrapping pool array
    w                # ← ERROR
end

@with_pool pool function bad()
    A = acquire!(pool, Float64, 3, 3)
    return A         # ← ERROR (explicit return)
end
```

Safe patterns pass without error:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    sum(v)               # ✅ scalar result
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    collect(v)           # ✅ owned copy
end
```

## Runtime Safety Levels

For bugs the compiler can't catch (e.g., values hidden behind opaque function calls), runtime safety provides configurable protection via the type parameter `S` in `AdaptiveArrayPool{S}`.

### Level Overview

| Level | Name | CPU | CUDA | Overhead |
|-------|------|-----|------|----------|
| **0** | off | No-op (all branches dead-code-eliminated) | Same | Zero |
| **1** | guard | `resize!(v,0)` + `setfield!` invalidation | NaN/sentinel poisoning + cache clear | ~5ns/slot |
| **2** | full | Level 1 + data poisoning + escape detection at scope exit | Level 1 + device-pointer overlap check | Moderate |
| **3** | debug | Level 2 + acquire call-site tracking | Same | Moderate+ |

### Why CPU and CUDA Differ at Level 1

Both achieve the same goal — **make stale references fail loudly** — but use different mechanisms:

| | CPU | CUDA |
|---|-----|------|
| **Strategy** | Structural invalidation | Data poisoning |
| **Mechanism** | `resize!(v, 0)` shrinks backing vector to length 0; `setfield!(:size, (0,))` zeroes the array dimensions | `CUDA.fill!(v, NaN)` / `typemax` / `true` fills backing CuVector with sentinel values |
| **Stale access result** | `BoundsError` (array has length 0) | Reads `NaN` or `typemax` (obviously wrong data) |
| **Why not the other way?** | CPU `resize!` is cheap (~0 cost) | CUDA `resize!` calls `CUDA.Mem.free()` — destroys the pooled VRAM allocation |
| **Cache invalidation** | View length/dims zeroed | N-way view cache entries cleared to `nothing` |

### Setting the Level

```julia
using AdaptiveArrayPools

# Replace current task-local pool (preserves cached arrays, zero-copy)
set_safety_level!(2)

# CUDA pools (all devices)
set_cuda_safety_level!(2)

# Back to zero overhead
set_safety_level!(0)
```

The pool type parameter `S` is a compile-time constant. At `S=0`, the JIT eliminates all safety branches via dead-code elimination — true zero overhead with no `Ref` reads or conditional branches.

### Data Poisoning (Level 2+, CPU)

At Level 1, CPU relies on **structural invalidation** (`resize!` + `setfield!`) which makes stale views throw `BoundsError`. At Level 2+, CPU additionally **poisons** the backing vector data with sentinel values (`NaN`, `typemax`, all-`true` for `BitVector`) *before* structural invalidation. This catches stale access through `unsafe_acquire!` wrappers on Julia 1.10 where `setfield!` on Array is unavailable.

CUDA already poisons at Level 1 (its primary invalidation strategy), so no additional poisoning step is needed at Level 2.

### Escape Detection (Level 2+)

At every `@with_pool` scope exit, the return value is inspected for overlap with pool-backed memory. Recursively checks `Tuple`, `NamedTuple`, `Dict`, `Pair`, `Set`, and `AbstractArray` elements.

Level 3 additionally records each `acquire!` call-site, so the error message pinpoints the exact source line and expression that allocated the escaping array.

### Legacy: `POOL_DEBUG`

`POOL_DEBUG[] = true` triggers Level 2 escape detection regardless of `S`. For new code, prefer `set_safety_level!(2)`.

## Recommended Workflow

```julia
# Development / Testing: catch bugs early
set_safety_level!(2)   # or 3 for call-site info in error messages

# Production: zero overhead
set_safety_level!(0)   # all safety branches eliminated by the compiler
```
