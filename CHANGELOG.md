# Changelog

All notable changes to this project are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This package does not yet commit to Semantic Versioning strictness pre-1.0;
version numbers below indicate scope of change, not a stability contract.

## [0.4.0] — 2026-07-10

### BREAKING

`@with_pool` (and its `@maybe_with_pool` / `@safe_with_pool` /
`@safe_maybe_with_pool` variants) now reject three additional "incidental"
tail patterns at **macro-expansion time** — previously these only errored at
runtime (and only when `RUNTIME_CHECK >= 1`):

- a direct acquire-family call as the scope's last expression
- a broadcast-assignment (`x .= v`) tail, where `x`/`v` is pool-backed
- a plain assignment (`y = v`) tail, where the RHS is pool-backed

These join the existing return-looking patterns (bare variable, `return v`,
tuple/vector/NamedTuple literals) that already errored at expansion time.

```julia
# before (compiled and ran; the array was invalid the moment the caller
# touched it, since it had already been rewound):
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    x .= v
end

# after (PoolEscapeError at expansion time). Fix: discard-style scopes must
# end with `nothing`:
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    x .= v
    nothing
end
```

A new `escape_lint` preference is the migration escape hatch for this
breaking change:

```julia
using Preferences
Preferences.set_preferences!("AdaptiveArrayPools", "escape_lint" => "warn")
# "error" (default) — throw PoolEscapeError, same as the pre-existing patterns
# "warn"            — print the same diagnostic via @warn and continue
# "off"             — skip this stage of checking entirely
```

The `RUNTIME_CHECK >= 1` runtime validation is unaffected and remains
authoritative for patterns static analysis cannot see (aliases through
opaque function calls, closures, conditional tails).

### Added

- `Vector{Float64}`-style parametric type literals (curly `acquire!` calls
  with no locally-bound free names, e.g. `acquire!(pool, Vector{Float64}, n)`)
  now take the static typed checkpoint/rewind path instead of demoting to the
  dynamic/lazy path. Curly types built from a locally-bound name
  (`Vector{T}` where `T` is assigned earlier in the same scope) still demote,
  since the concrete type isn't known until runtime.
- `escape_lint` preference (`"error"` default | `"warn"` | `"off"`), loaded
  once at package load as the compile-time constant `ESCAPE_LINT`, controlling
  the severity of the three new incidental-tail patterns above.

### Performance

- Typed scopes now hoist `get_typed_pool!` to a single lookup per static type
  per scope: each static type's `TypedPool` is bound once right after
  `checkpoint!` and threaded through new `_*_impl!(pool, tp, dims...)`
  method variants, removing the per-`acquire!` fallback-registry lookup
  (fixed-slot types were already zero-cost field loads; this closes the gap
  for fallback/dynamic types with repeated same-type acquires in one scope).

### Fixed

- `_transform_acquire_calls` no longer recurses into a nested
  `@with_pool`/`@maybe_with_pool`/`@safe_with_pool`/`@safe_maybe_with_pool`
  macrocall. Previously, transforming an outer scope's body would rewrite
  `acquire!` calls inside a syntactically nested inner `@with_pool` block
  before the inner macro ever ran its own expansion pass, causing the inner
  scope's type-extraction to find no `acquire!` calls and silently demote to
  the dynamic/lazy path regardless of what the inner scope actually did.
