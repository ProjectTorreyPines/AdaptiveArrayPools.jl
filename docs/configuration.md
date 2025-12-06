# Configuration

AdaptiveArrayPools can be configured via `LocalPreferences.toml`:

```toml
[AdaptiveArrayPools]
use_pooling = false  # ⭐ Primary: Disable pooling entirely
cache_ways = 8       # Advanced: N-way cache size (default: 4)
```

## Compile-time: USE_POOLING (⭐ Primary)

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

When `USE_POOLING = false`:

```julia
# These become equivalent:
@with_pool pool acquire!(pool, Float64, n, n)  →  Matrix{Float64}(undef, n, n)
@with_pool pool acquire!(pool, Float64, n)     →  Vector{Float64}(undef, n)
```

**Use cases:**
- **Debugging**: Compare behavior with/without pooling
- **Benchmarking**: Measure pooling overhead vs direct allocation
- **Gradual adoption**: Add `@with_pool` annotations now, enable pooling later
- **CI/Testing**: Run tests without pooling to isolate issues

All pooling code is **completely eliminated at compile time** (zero overhead).

## Runtime: MAYBE_POOLING_ENABLED

Only affects `@maybe_with_pool`. Toggle without restart.

```julia
MAYBE_POOLING_ENABLED[] = false  # Disable
MAYBE_POOLING_ENABLED[] = true   # Enable (default)
```

## Runtime: POOL_DEBUG

Enable safety validation to catch direct returns of pool-backed arrays.

```julia
POOL_DEBUG[] = true   # Enable safety checks (development)
POOL_DEBUG[] = false  # Disable (default, production)
```

When enabled, returning a pool-backed array from a `@with_pool` block will throw an error.

## Compile-time: CACHE_WAYS

Configure the N-way cache size for `unsafe_acquire!`. Higher values reduce cache eviction but increase memory per slot.

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

**When to increase**: If your code alternates between more than 4 dimension patterns per pool slot, increase `cache_ways` to avoid cache eviction (~100 bytes header per miss).

> **Scope**: `cache_ways` affects **all `unsafe_acquire!`** calls (including 1D). Only `acquire!` 1D uses simple 1:1 caching.

## Summary

| Setting | Scope | Restart? | Priority | Affects |
|---------|-------|----------|----------|---------|
| `use_pooling` | Compile-time | Yes | ⭐ Primary | All macros, `acquire!` behavior |
| `cache_ways` | Compile-time | Yes | Advanced | `unsafe_acquire!` N-D caching |
| `MAYBE_POOLING_ENABLED` | Runtime | No | Optional | `@maybe_with_pool` only |
| `POOL_DEBUG` | Runtime | No | Debug | Safety validation |
