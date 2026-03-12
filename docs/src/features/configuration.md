# Configuration

AdaptiveArrayPools can be configured via `LocalPreferences.toml`:

```toml
[AdaptiveArrayPools]
use_pooling = false      # ⭐ Primary: Disable pooling entirely
runtime_check = 1        # Safety: Enable runtime safety checks
cache_ways = 8           # Advanced: N-way cache size (default: 4)
```

All compile-time preferences require **restarting Julia** to take effect.

## Compile-time: STATIC_POOLING (⭐ Primary)

**The most important configuration.** Completely disable pooling to make `acquire!` behave like standard allocation.

```toml
# LocalPreferences.toml
[AdaptiveArrayPools]
use_pooling = false
```

Or programmatically:

```julia
using Preferences
Preferences.set_preferences!(AdaptiveArrayPools, "use_pooling" => false)
# Restart Julia for changes to take effect
```

When `STATIC_POOLING = false`:
- `pool` becomes `DisabledPool{backend}()` instead of an active pool
- All pool functions fall back to standard allocation
- Backend context is preserved: `:cuda` still returns `CuArray`

```julia
# These become equivalent:
@with_pool pool acquire!(pool, Float64, n, n)  →  Matrix{Float64}(undef, n, n)
@with_pool pool acquire!(pool, Float64, n)     →  Vector{Float64}(undef, n)

# With CUDA backend:
@with_pool :cuda pool zeros!(pool, 100)        →  CUDA.zeros(Float32, 100)
```

Use `pooling_enabled(pool)` to check if pooling is active.

**Use cases:**
- **Debugging**: Compare behavior with/without pooling
- **Benchmarking**: Measure pooling overhead vs direct allocation
- **Gradual adoption**: Add `@with_pool` annotations now, enable pooling later
- **CI/Testing**: Run tests without pooling to isolate issues

All pooling code is **completely eliminated at compile time** (zero overhead).

## Compile-time: RUNTIME_CHECK

Enable runtime safety checks to catch pool-escape bugs. See [Safety](safety.md) for full details.

```toml
# LocalPreferences.toml
[AdaptiveArrayPools]
runtime_check = 1      # enable (0 = off, 1 = on)
# runtime_check = true  # also accepted
```

Or programmatically:

```julia
using Preferences
Preferences.set_preferences!(AdaptiveArrayPools, "runtime_check" => 1)
# Restart Julia for changes to take effect
```

Accepts both `Bool` and `Int` values — internally normalized to `Int`:
- `false` / `0` → off (zero overhead, all safety branches eliminated)
- `true` / `1` → on (poisoning + invalidation + escape detection + borrow tracking)

The safety level is baked into the pool type parameter: `AdaptiveArrayPool{0}` or `AdaptiveArrayPool{1}`. This enables dead-code elimination — at `RUNTIME_CHECK = 0`, all safety branches are completely removed by the compiler.

## Runtime: MAYBE_POOLING

Only affects `@maybe_with_pool`. Toggle without restart.

```julia
MAYBE_POOLING[] = false  # Disable
MAYBE_POOLING[] = true   # Enable (default)
```

## Compile-time: CACHE_WAYS (Julia 1.10 / CUDA only)

Configure the N-way cache size for `unsafe_acquire!`. **On Julia 1.11+ CPU, this setting has no effect** — the `setfield!`-based wrapper reuse supports unlimited dimension patterns with zero allocation.

This setting is relevant for:
- **Julia 1.10** (legacy N-way cache path)
- **CUDA backend** (N-way cache for `CuArray` wrappers)

```toml
# LocalPreferences.toml
[AdaptiveArrayPools]
cache_ways = 8  # Default: 4, Range: 1-16
```

Or programmatically:

```julia
using AdaptiveArrayPools
set_cache_ways!(8)
# Restart Julia for changes to take effect
```

**When to increase**: If your CUDA code or Julia 1.10 code alternates between more than 4 dimension patterns per pool slot, increase `cache_ways` to avoid cache eviction (~100 bytes header per miss).

## Summary

| Setting | Scope | Restart? | Priority | Affects |
|---------|-------|----------|----------|---------|
| `use_pooling` | Compile-time | Yes | ⭐ Primary | All macros, `acquire!` behavior |
| `runtime_check` | Compile-time | Yes | Safety | Poisoning, invalidation, escape detection |
| `cache_ways` | Compile-time | Yes | Advanced | `unsafe_acquire!` N-D caching (Julia 1.10 / CUDA only) |
| `MAYBE_POOLING` | Runtime | No | Optional | `@maybe_with_pool` only |
