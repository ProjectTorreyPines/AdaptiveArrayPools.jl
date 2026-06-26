# Design: `rand!` / `randn!` pool constructors

**Status:** Approved design, pending implementation
**Branch:** `feature/rand-randn-pool-constructors`
**Author:** Min-Gu Yoo

## 1. Goal

Add random-array convenience constructors to the pool API, peers of `zeros!` /
`ones!`:

```julia
@with_pool pool begin
    v = rand!(pool, 100)            # Vector{Float64}, U[0,1)
    m = rand!(pool, Float32, 8, 8)  # Matrix{Float32}, U[0,1)
    d = rand!(pool, 1:6, 10)        # Vector{Int}, each ∈ 1:6  (dice)
    g = randn!(pool, 100)           # Vector{Float64}, N(0,1)
end
```

These acquire a right-sized array from the pool and fill it in place with random
values, preserving the package's zero-allocation guarantee inside `@with_pool`.

## 2. Key decisions (settled during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Dependency | **`Random` as a hard `[deps]`** | `Random` is a stdlib (in the sysimage); `using Random` adds ~0 load cost. The usual reason to use a weakdep/extension (avoid forcing a heavy dep — cf. CUDA/Metal) does not apply. |
| Why not an extension | The convenience functions are **wired into `@with_pool` at 3 sites**; an extension would force a core stub + `GlobalRef` to a conditionally-defined impl and yield cryptic `MethodError`s when `Random` isn't loaded. Hard-dep keeps `_rand_impl!` next to `_zeros_impl!`. |
| Namespace | **`import Random: rand!, randn!` then `export rand!, randn!`** (re-export) | Makes `rand!` a true peer of `zeros!` — works with just `using AdaptiveArrayPools`. Re-exporting the *same binding* means **no conflict warning** if the user also does `using Random`. |
| Distributions (v1) | **`rand!` (uniform) + `randn!` (normal)** | Covers the overwhelming majority of use. `randexp!` deferred (low demand, trivial to add later via same machinery). |
| Collection/range form | **Include `rand!(pool, S, dims)`** (CPU-only) | This is the real "random integers in a range" case (`rand!(pool, 1:6, n)`). More useful than full-range `rand!(pool, Int, n)`. |
| GPU parity | **Deferred to a follow-up PR** | Live GPU pools inherit the core impl via dispatch (see §7), so it is cheap, but Metal's `randn!`/`rand!` MPS coverage needs verification. Kept out of v1 to ship CPU cleanly. |

## 3. Why this is "wide but shallow"

The implementation body is ~2 lines per function (`acquire!` → `Random.rand!`).
The work is **parallelism with `zeros!`**: every site that already knows about
`zeros!` must learn `rand!` / `randn!`. This is a consequence of the package
centralizing checkpoint / escape-analysis / type-registration logic in the
macro — new convenience functions are cheap to write but must be *registered* in
the macro. The spec below is therefore primarily a **checklist of mirror-sites**.

## 4. Public API surface (CPU, v1)

Mirrors `zeros!`'s four forms, plus the collection form for `rand!`:

```julia
# Uniform
rand!(pool, dims...)                 # default_eltype(pool)
rand!(pool, T::Type, dims...)        # typed
rand!(pool, dims::NTuple{N,Int})     # tuple form (for macro size(x) handling)
rand!(pool, T::Type, dims::NTuple{N,Int})
rand!(pool, S, dims...)              # sample from collection S; eltype = eltype(S)
rand!(pool, S, dims::NTuple{N,Int})

# Normal (T must be AbstractFloat; non-float T inherits Random.randn!'s error)
randn!(pool, dims...)
randn!(pool, T::Type, dims...)
randn!(pool, dims::NTuple{N,Int})
randn!(pool, T::Type, dims::NTuple{N,Int})
```

**Default eltype** comes from `default_eltype(pool)` (CPU: `Float64`), so
`rand!(pool, 10)` → `U[0,1)` Float64 and `randn!(pool, 10)` → `N(0,1)` Float64.

**Collection form semantics**: `rand!(pool, S, dims...)` acquires an array of
`eltype(S)` and fills it via `Random.rand!(arr, S)`. Examples:
`rand!(pool, 1:6, 10)` (Int dice), `rand!(pool, ('a','b','c'), 5)` (Char).

## 5. Components & files to change

### 5.1 `Project.toml`
- Add `Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"` to `[deps]`.
- Add `Random = "1"` to `[compat]`.

### 5.2 `src/AdaptiveArrayPools.jl` (main module)
- `import Random: rand!, randn!` (brings names in scope for extending + re-export).
- `import Random` (so `rand`, `randn` allocating fallbacks are reachable for `DisabledPool`).
- Add `rand!, randn!` to the convenience-functions `export` line (next to `zeros!, ones!, ...`).

### 5.3 `src/convenience.jl` (next to `zeros!` / `_zeros_impl!`)
Public wrappers (each does `_record_type_touch!` + `_set_pending_callsite!` + impl,
identical structure to `zeros!`):
- `rand!(pool::AbstractArrayPool, ...)` — all six forms above.
- `randn!(pool::AbstractArrayPool, ...)` — four forms.

Internal impls (backend-agnostic — rely on `Random.rand!`/`randn!` dispatch on the array type):
- `_rand_impl!(pool, T, dims...)  = (arr = _acquire_impl!(pool, T, dims...); Random.rand!(arr); arr)`
- `_rand_impl!(pool, dims...)     = _rand_impl!(pool, default_eltype(pool), dims...)`
- `_rand_impl!(pool, S, dims...)  = (arr = _acquire_impl!(pool, eltype(S), dims...); Random.rand!(arr, S); arr)`  *(collection)*
- NTuple overloads for each.
- `_randn_impl!(...)` analogous to `_rand_impl!` minus the collection form, using `Random.randn!`.

**Dispatch disambiguation (collection vs typed vs default):** the collection
argument `S` is constrained to sampleable, non-`Type`, non-`Integer` collections —
`S::Union{AbstractArray, Tuple, AbstractRange, AbstractSet, AbstractString, AbstractDict}`
— so `rand!(pool, 1:6, 10)` picks the collection method while `rand!(pool, 10)`
and `rand!(pool, Int, 10)` pick the dims/typed methods. This mirrors how Base
separates `rand(coll, n)` from `rand(T, n)`.

### 5.4 `src/convenience.jl` — `DisabledPool{:cpu}` fallbacks
Mirror the `zeros!` disabled-pool methods (`convenience.jl:439-442`):
- `rand!(::DisabledPool{:cpu}, T, dims...) = rand(T, dims...)`
- `rand!(p::DisabledPool{:cpu}, dims...)   = rand(default_eltype(p), dims...)`
- `rand!(::DisabledPool{:cpu}, S, dims...) = rand(S, dims...)`  *(collection)*
- `randn!(::DisabledPool{:cpu}, T, dims...) = randn(T, dims...)`, default + NTuple forms.

### 5.5 `src/macros.jl` — the three mirror-sites
1. **Type extraction** `_extract_acquire_types` (~L1143-1189): add `:rand!` / `:randn!`.
   - Typed form: read the type token like `zeros!`/`ones!`.
   - Default form: use `default_eltype(pool)` like `zeros!`.
   - **Collection form (NEW code path):** emit `eltype(S)` as a runtime expression
     (constant-folds for literal ranges) rather than reading a literal token.
2. **Call → impl transform** (~L1483-1537, GlobalRefs at L1491-1492, impl-name list at L1584):
   - Add `_RAND_IMPL_REF = GlobalRef(@__MODULE__, :_rand_impl!)` and `_RANDN_IMPL_REF`.
   - Map `:rand! → _RAND_IMPL_REF`, `:randn! → _RANDN_IMPL_REF` (both the `fn ==` and `QuoteNode` branches).
   - Add `:_rand_impl!`, `:_randn_impl!` to the impl-name list (~L1584).
   - The transform only swaps the function name; the collection arg passes through unchanged.
3. **Escape-detection symbol lists** (~L1943-1973, and the lists feeding L2032/L2371):
   - Add `:rand!`, `:randn!` so `x = rand!(pool, 10); return x` triggers `PoolEscapeError`,
     matching `zeros!` behavior.

## 6. Error handling

- `randn!(pool, Int, n)` — inherits `Random.randn!`'s error for non-float eltypes
  (no special handling; documented).
- Collection form with an eltype the pool can't store — surfaces through the
  normal `acquire!` type path (`_record_type_touch!(pool, eltype(S))`).
- Escape of a pool-backed random array — `PoolEscapeError`, via the macro escape
  lists (§5.5.3), identical to `zeros!`.

## 7. GPU (Metal + CUDA) — implemented

`CuAdaptiveArrayPool` / `MetalAdaptiveArrayPool <: AbstractArrayPool`, and the GPU
extensions do **not** override `_rand_impl!`/`_randn_impl!` for live pools — they
**inherit the core impl**, which works on GPU because `Random.rand!`/`randn!` are
overloaded for `CuArray`/`MtlArray` and run on-device. **Verified on Metal**
(Apple Silicon): `Random.rand!`/`randn!` on `MtlArray` are GPU-native for
`Float32`/`Float16`/`Int32`. So the live GPU pool gets the typed/default forms for
free, exactly like `zeros!`.

GPU-specific code (per backend, in the extension):
- `DisabledPool{:cuda}`/`{:metal}` fallbacks (`Metal.rand`/`Metal.randn`,
  `CUDA.rand`/`CUDA.randn`).
- **Collection-form rejection**: `rand!(pool, S, dims)` is **CPU-only** — GPU RNGs have
  no arbitrary-collection sampler (`Random.rand!(gpuarray, 1:6)` scalar-indexes the
  device array, which is disallowed). Overriding `_rand_impl!(::GPUPool, ::_SampleColl, …)`
  raises a clear `ArgumentError` (covers the macro path and the direct path, live + disabled).

Tests: `test/metal/test_random.jl` (run locally — **34 pass**) and
`test/cuda/test_random.jl` (mirror, to run on a CUDA machine).

## 8. Testing (`test/test_convenience.jl`, mirroring the `zeros!` testsets)

- `rand!`: length/size & eltype correct for default, typed (`Float32`, `Int`), and
  multi-dim forms; **all values ∈ [0,1)** for float eltypes.
- `rand!(pool, 1:6, n)`: `eltype == Int`, **all values ∈ 1:6**; a Char/tuple collection case.
- `randn!`: length/size & eltype correct; values finite (statistical sanity, not exact equality).
- **Zero-allocation** inside `@with_pool` (reuse the existing alloc-test helper / `test_zero_allocation.jl` pattern).
- **Escape detection**: `x = rand!(pool, 10); return x` triggers `PoolEscapeError`
  (mirror the existing `test_compile_escape.jl` style).
- **`DisabledPool` fallback**: returns a plain `Array` of correct type/size.
- **Determinism**: seed via `Random.seed!` (no `rng` arg in v1) and assert reproducible values.
- **Aqua** (`test_aqua.jl`): confirm no new ambiguities, no piracy flag (methods are
  on our `AbstractArrayPool` type → not piracy), deps/compat consistent with the new `Random` entry.

## 9. Documentation

- Docstrings on `rand!` / `randn!` mirroring the `zeros!` docstring (signatures,
  default-eltype note, collection-form example, `See also`).
- README / docs convenience-function section: add `rand!` / `randn!` next to `zeros!` / `ones!`.

## 10. Explicitly out of scope (YAGNI)

- `randexp!` (trivial later add).
- `rng`-argument forms (`rand!(pool, rng, dims)`) — would complicate macro type extraction.
- `randperm!` / `randcycle!` / `shuffle!` (different semantics — permutation/in-place on a user array).
- `bitrand` / `rand!(pool, Bit, …)` — tied to the deferred `Bit`/`BitArray` handling.
- GPU collection sampling.
