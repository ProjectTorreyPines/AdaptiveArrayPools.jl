# AdaptiveArrayPools.jl

Zero-allocation array pooling for Julia. Reuse temporary arrays to eliminate GC pressure in hot loops.

## Installation

AdaptiveArrayPools is registered with [FuseRegistry](https://github.com/ProjectTorreyPines/FuseRegistry.jl/):

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.Registry.add("General")
Pkg.add("AdaptiveArrayPools")
```

## Quick Start

```julia
using AdaptiveArrayPools

@use_pool pool function compute(n)
    v = acquire!(pool, Float64, n)  # Get array from pool (no allocation)
    v .= 1.0
    sum(v)
end  # Arrays automatically returned to pool

compute(1000)  # Zero allocations after warm-up
```

## Why Use This?

```julia
# Before: allocates every call
function compute_naive(n)
    v = zeros(Float64, n)  # Allocates!
    v .= 1.0
    sum(v)
end

# After: zero allocations
@use_pool pool function compute_pooled(n)
    v = acquire!(pool, Float64, n)  # Reuses memory
    v .= 1.0
    sum(v)
end
```

## Disable Pooling (Compile-time)

Set in `LocalPreferences.toml` to completely remove pooling overhead:

```toml
[AdaptiveArrayPools]
use_pooling = false
```

When disabled, `acquire!` falls back to normal allocation with zero macro overhead.

## Documentation

- [Runtime Toggle: @maybe_use_pool](docs/maybe_use_pool.md) - Let users control pooling at runtime
- [Explicit Pool: @with_pool](docs/with_pool.md) - Advanced use with custom pool instances
- [Configuration](docs/configuration.md) - Preferences.jl integration and compile-time settings

## API Summary

| Macro | Use Case |
|-------|----------|
| `@use_pool` | Main API - uses global (task-local) pool |
| `@maybe_use_pool` | Runtime toggle via `MAYBE_POOLING_ENABLED[]` |
| `@with_pool` | Explicit pool instance (advanced) |

## License

Apache 2.0
