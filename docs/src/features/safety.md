# Pool Safety

AdaptiveArrayPools catches pool-escape bugs at **two levels**: compile-time (macro analysis) and runtime (configurable via `RUNTIME_CHECK`).

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

## Runtime Safety (`RUNTIME_CHECK`)

For bugs the compiler can't catch (e.g., values hidden behind opaque function calls), runtime safety provides configurable protection via the type parameter `S` in `AdaptiveArrayPool{S}`.

### Binary System

| `RUNTIME_CHECK` | State | What Happens | Overhead |
|:-:|-------|--------------|----------|
| **0** | off | All safety branches dead-code-eliminated | **Zero** |
| **1** | on | Poisoning + structural invalidation + escape detection + borrow tracking | ~5ns/slot |

`RUNTIME_CHECK` is a **compile-time constant** — not a runtime toggle. At `RUNTIME_CHECK = 0`, the JIT eliminates all safety branches completely. No `Ref` reads, no conditional branches, no overhead whatsoever.

### Enabling Runtime Safety

Set the `runtime_check` preference in `LocalPreferences.toml` and **restart Julia**:

```toml
# LocalPreferences.toml
[AdaptiveArrayPools]
runtime_check = 1     # enable all safety checks
# runtime_check = true  # also accepted (normalized to 1 internally)
```

Or programmatically:

```julia
using Preferences
Preferences.set_preferences!(AdaptiveArrayPools, "runtime_check" => 1)
# Restart Julia for changes to take effect
```

!!! warning "Restart Required"
    `RUNTIME_CHECK` is baked into the pool type at compile time (`AdaptiveArrayPool{S}`). Changing the preference **requires restarting Julia** — it cannot be toggled at runtime.

### What `RUNTIME_CHECK = 1` Enables

When safety is on, `@with_pool` scope exit triggers the following protections:

#### 1. Data Poisoning

Released arrays are filled with detectable sentinel values **before** structural invalidation:

| Element Type | Poison Value | Detection |
|-------------|-------------|-----------|
| `Float64`, `Float32`, `Float16` | `NaN` | `isnan(x)` returns `true` |
| `Int64`, `Int32`, etc. | `typemax(T)` | Obviously wrong value |
| `ComplexF64`, `ComplexF32` | `NaN + NaN*im` | `isnan(real(x))` |
| `Bool` | `true` | All-true is suspicious |
| Other types | `zero(T)` | Generic fallback |

#### 2. Structural Invalidation

After poisoning, stale references are made to fail loudly:

| | CPU | CUDA |
|---|-----|------|
| **Mechanism** | `resize!(v, 0)` shrinks backing vector; `setfield!(:size, (0,))` zeroes array dimensions | `_resize_to_fit!(v, 0)` shrinks logical length (GPU memory preserved) |
| **Stale access** | `BoundsError` (array has length 0) | `BoundsError` (logical length 0); poisoned data visible on re-acquire |
| **arr_wrapper** | Dimensions set to `(0,)` / `(0,0)` | Same |
| **Why different?** | CPU `resize!` is cheap (~0 cost) | CUDA `resize!` would call `CUDA.Mem.free()` — destroys pooled VRAM |

#### 3. Escape Detection

At every `@with_pool` scope exit, the return value is inspected for overlap with pool-backed memory. Recursively checks `Tuple`, `NamedTuple`, `Dict`, `Pair`, `Set`, and `AbstractArray` elements.

```julia
# Throws PoolRuntimeEscapeError at scope exit
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    opaque_function(v)  # returns v through opaque call
end
```

#### 4. Borrow Tracking

Each `acquire!` call-site is recorded, so escape error messages pinpoint the exact source line and expression that allocated the escaping array:

```
PoolEscapeError (runtime) — pool-backed array escaping @with_pool scope

  Leaked value:  SubArray{Float64, 1}
  Backing type:  Float64

  acquired at:   src/solver.jl:42
                 v = acquire!(pool, Float64, n)

  Enable RUNTIME_CHECK >= 1 to detect pool escapes at runtime.
```

## Recommended Workflow

```toml
# Development / Testing (LocalPreferences.toml):
[AdaptiveArrayPools]
runtime_check = 1     # catch bugs early — restart Julia after changing

# Production:
[AdaptiveArrayPools]
runtime_check = 0     # zero overhead — all safety branches eliminated
```
