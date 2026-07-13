# Runtime Safety (`RUNTIME_CHECK`)

**Off by default** — zero overhead in production. Enable it during development to catch what [compile-time detection](compile-time.md) misses.

## Enabling

Set the `runtime_check` preference and **restart Julia**:

```toml
# LocalPreferences.toml
[AdaptiveArrayPools]
runtime_check = 1     # enable all safety checks
```

Or programmatically:

```julia
using Preferences
Preferences.set_preferences!(AdaptiveArrayPools, "runtime_check" => 1)
# Restart Julia for changes to take effect
```

!!! warning "Restart Required"
    `RUNTIME_CHECK` is a **compile-time constant** baked into the pool type (`AdaptiveArrayPool{S}`). At `S=0`, the JIT eliminates all safety branches completely — no `Ref` reads, no conditional branches, no overhead.

## What It Catches

When `RUNTIME_CHECK = 1`, every `@with_pool` scope exit triggers:

### 1. Data Poisoning

Released arrays are filled with detectable sentinel values:

| Element Type | Poison Value |
|-------------|-------------|
| `Float64`, `Float32`, `Float16` | `NaN` |
| `Int64`, `Int32`, etc. | `typemax(T)` |
| `ComplexF64`, `ComplexF32` | `NaN + NaN*im` |
| `Bool` | `true` |

### 2. Structural Invalidation

Stale references are made to fail loudly (`BoundsError`):

| | CPU | CUDA / Metal |
|---|-----|------|
| **Mechanism** | `resize!(v, 0)` + `setfield!(:size, (0,))` | `_resize_to_fit!(v, 0)` (GPU memory preserved) |
| **Why different?** | CPU `resize!` is cheap | GPU `resize!` would free device memory |

### 3. Escape Detection

The return value is recursively inspected for overlap with pool-backed memory (`Tuple`, `NamedTuple`, `Dict`, `Pair`, `Set`, `AbstractArray`).

```julia
# Throws PoolRuntimeEscapeError at scope exit
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    sneaky(v)  # returns v through opaque call — caught here
end
```

### 4. Borrow Tracking

Each `acquire!` call-site is recorded, so error messages pinpoint the exact source:

```
PoolEscapeError (runtime, RUNTIME_CHECK >= 1)

    Array{Float64, 1}
      ← backed by Float64 pool memory, will be reclaimed at scope exit
      ← acquired at src/solver.jl:42
        v = acquire!(pool, Float64, n)

  Fix: end the block with nothing if the value is discarded, collect() to return an owned copy, or compute a scalar.
```

The `nothing` ending is the most common fix: a scope's **last expression is its return value**, so a block that ends on a pool-backed array leaks it by accident. End with `nothing` (or a scalar) when the array is only used for side effects.

### 5. Mutation Detection

Detects structural mutations that escaped compile-time analysis by comparing wrapper state against backing storage at rewind:

| Backend | Detection Method |
|---------|-----------------|
| CPU (1.12+) | `MemoryRef` identity + length divergence |
| CUDA | `DataRef` identity + length divergence |
| Metal | `DataRef` identity + length divergence |

Emits a one-time advisory `@warn` (`maxlog=1`). The pool **self-heals** on next `acquire!` — no data corruption, only pooling benefits are temporarily lost.

## Guarantees & Limitations

**Capacity is retained.** Poisoning and structural invalidation mark a slot's
*current* contents as stale — they do **not** shrink the pool. The slot keeps its
allocated capacity and is reused on the next `acquire!`; `compact!` skips
still-poisoned slots. So `RUNTIME_CHECK=1` never trades away pooling's memory
reuse — it only makes stale references fail loudly.

**Allocation.** `S=0` (the production default) is zero-allocation — the checks
are dead-code-eliminated. `S=1` is a development tool and *does* allocate: borrow
tracking lazily builds an `IdDict` and records a callsite string per `acquire!`
so escape/mutation errors can name the source. The poison fills and rewind
comparisons overwrite existing buffers without allocating, but don't measure a
zero-GC hot path under `S=1` — use `S=0`.

**What it cannot catch.** Escape detection inspects the *return value* at scope
exit — it cannot prove ownership through arrays captured by closures, stored in
globals that outlive the scope, or aliased behind opaque calls that never surface
in the return value. These are undecidable statically; `S=1` catches the
return-position cases, and the [compile-time analysis](compile-time.md) handles
the common syntactic ones (function-form/`return` escapes error at expansion;
block-tail escapes are replaced by an `EscapedPoolArray` guard that traps on
first use even at `S=0`).

**Test hygiene.** For tests that run under both `S=0` and `S=1`: keep asserts
S-adaptive (don't assert poison values unless `S≥1`), and end `@with_pool` blocks
with `nothing` unless you deliberately return an owned value.

## Recommended Workflow

```toml
# Development / Testing:
[AdaptiveArrayPools]
runtime_check = 1     # catch bugs early

# Production:
[AdaptiveArrayPools]
runtime_check = 0     # zero overhead (default)
```
