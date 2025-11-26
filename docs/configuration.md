# Configuration

## Compile-time: USE_POOLING

Completely disable pooling at compile time via `Preferences.jl`.

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
- All macros generate `pool = nothing`
- `acquire!` falls back to normal allocation
- Zero overhead from pooling code

## Runtime: MAYBE_POOLING_ENABLED

Only affects `@maybe_use_pool`. Toggle without restart.

```julia
MAYBE_POOLING_ENABLED[] = false  # Disable
MAYBE_POOLING_ENABLED[] = true   # Enable (default)
```

## Summary

| Setting | Scope | Restart? | Affects |
|---------|-------|----------|---------|
| `USE_POOLING` | Compile-time | Yes | All macros |
| `MAYBE_POOLING_ENABLED` | Runtime | No | `@maybe_use_pool` only |
