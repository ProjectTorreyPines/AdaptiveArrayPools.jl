# Inactive Slot Trimming for AdaptiveArrayPools

- **Date:** 2026-06-28
- **Status:** Revised design — pending user review (not yet implemented)
- **Scope:** New manual `trim!` API. CPU + Metal in first cut; CUDA deferred.
- **Author:** design session

---

## 1. Motivation

AdaptiveArrayPools keeps backing buffers after scopes exit so later `acquire!`
calls can reuse memory and stay allocation-free on the hot path. This is the
right default for normal workloads.

The downside is memory retention after an unusual workload. A rare large input
can grow one or more slots far beyond the usual working set. After the scope
rewinds, those slots are no longer active, but the pool still keeps strong
references to their backing buffers and cached wrappers. Julia's GC cannot
collect those buffers while the pool still references them.

We want a manual operation that says:

> Keep the arrays currently in use. Drop the inactive pooled buffers that are
> only being retained for possible future reuse.

This is not an immediate OS/VRAM release guarantee. The goal is to release the
pool-held references so the old buffers become GC-eligible once no user code
still references them.

## 2. Core Concept: Active vs Inactive Slots

Each typed pool has:

- **active slots:** `1:tp.n_active`
  - Currently claimed by the live call stack.
  - Must remain valid.
- **inactive retained slots:** `(tp.n_active + 1):length(tp.vectors)`
  - Already released by `rewind!` or previous scope exit.
  - Kept only as a reuse cache.
  - Safe for a manual cleanup operation to detach from the pool.

`trim!` operates only on inactive retained slots. It never touches active slots.

This makes the API easier to reason about than "deep" or "level" terminology:
the persistent thing is the typed pool's slot array, not a durable level object.

## 3. Proposed API

```julia
trim!(pool::AdaptiveArrayPool; force_gc::Bool = false)
trim!(pool::AdaptiveArrayPool, ::Type{T}; force_gc::Bool = false)
trim!(; force_gc::Bool = false)
```

Metal extension:

```julia
trim!(pool::MetalAdaptiveArrayPool; force_gc::Bool = false)
trim!(pool::MetalAdaptiveArrayPool, ::Type{T}; force_gc::Bool = false)
```

Disabled pools:

```julia
trim!(::DisabledPool; force_gc::Bool = false)
trim!(::DisabledPool, ::Type{T}; force_gc::Bool = false)
```

Return a small summary:

```julia
(;
    slots_released::Int,
    wrappers_released::Int,
    estimated_bytes_released::Int,
    gc_triggered::Bool,
)
```

`force_gc=false` is the default. When `true`, `trim!` calls `GC.gc()` after
detaching inactive references. This still does not promise immediate OS or GPU
memory return; it only asks Julia to run collection after the pool stops holding
the buffers.

## 4. Semantics

For each affected typed pool:

1. Let `keep = tp.n_active`.
2. Estimate storage held by inactive slots for the return summary.
3. Drop backing-buffer references by truncating `tp.vectors` to `keep`.
4. Drop cached wrapper references for inactive slots by truncating each
   wrapper-cache vector to `keep`.
5. Leave checkpoint stacks, depth state, typed-pool objects, and live slots
   unchanged.

Illustrative core:

```julia
function _trim_inactive_typed_pool!(tp)
    keep = tp.n_active
    old_len = length(tp.vectors)

    released = old_len - keep
    bytes = _inactive_storage_bytes(tp, keep + 1, old_len)

    resize!(tp.vectors, keep)
    _trim_inactive_wrappers!(tp, keep)

    return (; slots_released = released,
              wrappers_released = ...,
              estimated_bytes_released = bytes)
end
```

No replacement vectors are allocated. If a later workload needs more slots, the
existing `acquire!` cold path will allocate the needed backing buffers at the
actual requested size.

This avoids speculative "right-sizing" or reserve heuristics. A cleanup API
should not guess future sizes or allocate warm reserves on the user's behalf.

## 5. Relationship to Existing Operations

| Function | `n_active` | Backing buffers | Wrapper caches | Depth/checkpoint state |
|---|---:|---|---|---|
| `rewind!` | restored by one scope | keep all | keep all | pop one scope |
| `reset!` | set to 0 | keep all | keep all | reset to sentinel |
| `empty!` | set to 0 | drop all | drop all | reset to sentinel |
| `trim!` | unchanged | drop inactive only | drop inactive only | unchanged |

Important distinction:

- `trim!` is not `reset!`. If slots are still active, `trim!` preserves them.
- If user code has allowed `n_active` to grow at top level, call `reset!(pool)`
  first if the intent is to mark those slots inactive, then call `trim!(pool)`.
- At normal top level after `@with_pool` scopes have rewound, `n_active` is
  usually `0`, so `trim!(pool)` releases all retained backing buffers while
  keeping the pool object and lightweight structural state.

## 6. Why Not Allocate Smaller Replacement Buffers?

An earlier idea was to replace a huge inactive buffer with a smaller new buffer
such as `1.5x` or `2x` the expected normal size.

That is not part of this design.

Reasons:

- It performs a new allocation during a memory-reclamation call.
- The "normal future size" is only a guess without hot-path usage tracking.
- If the user actually needs that slot again, the existing `acquire!` path will
  allocate the right size when it has the real request.
- The simplest and most honest behavior is: detach inactive references now,
  allocate later only if needed.

## 7. Implementation Notes

### 7.1 Backing Buffers

CPU typed pools can detach inactive backing buffers with:

```julia
resize!(tp.vectors, tp.n_active)
```

This removes references to the inactive `Vector`s. The old vectors become
GC-eligible once no other references exist.

For `BitTypedPool`, the same outer-vector truncation works; the dropped
`BitVector`s own their chunk arrays.

For Metal, the same structural operation removes Julia references to inactive
`MtlArray`s. Actual VRAM return still depends on Julia GC and Metal.jl's memory
pool behavior, matching the caveat already documented for `empty!`.

### 7.2 Wrapper Caches

Dropping `tp.vectors` is not enough when cached wrappers still reference the old
backing storage.

For each cached wrapper vector in `tp.arr_wrappers`, truncate it to `tp.n_active`
or clear inactive entries before truncation:

```julia
for wrappers in tp.arr_wrappers
    wrappers === nothing && continue
    old_len = length(wrappers)
    released += max(0, old_len - tp.n_active)
    resize!(wrappers, min(length(wrappers), tp.n_active))
end
```

The wrapper cache remains structurally reusable. Future `acquire!` calls past
the current active region will take the existing cold path and create fresh
wrappers if needed.

### 7.3 Byte Accounting

`estimated_bytes_released` is an estimate of storage that the pool stopped
retaining, not proof that RSS/VRAM immediately dropped.

Suggested helpers:

- CPU `Vector`: `Base.summarysize(v)` because `resize!(v, 0)` can leave capacity
  retained even when `length(v) == 0`.
- `BitVector`: `Base.summarysize(v)` for the same capacity reason.
- Metal `MtlArray`: use GPU allocation size (`sizeof(v)` or `maxsize`-based
  helper consistent with existing Metal stats), plus CPU wrapper size only if
  useful for diagnostics.

Avoid `length(v) * sizeof(T)` for released slots in safety mode: those vectors
may have logical length `0` while still retaining large capacity.

## 8. Safety Contract

`trim!` preserves active slots and their wrappers. Arrays currently in use by
the live call stack remain valid.

Inactive retained slots are different: they have already been released by pool
scope management. User code should not rely on arrays/views from those slots.
After `trim!`, any pool-owned references to those inactive buffers are gone.

In runtime-check mode (`S >= 1`), released slots have already been invalidated
by `rewind!`/`reset!` where applicable. `trim!` does not need to add new hot-path
safety metadata.

`trim!` should be idempotent:

- Calling it twice in a row releases nothing on the second call.
- Calling it inside a nested computation keeps currently active slots.
- Calling it at top level after normal rewind releases retained buffers.

## 9. Backend Structure

Generic core in `src/state.jl`:

- `_trim_inactive_typed_pool!(tp::AbstractTypedPool)`
- `_trim_inactive_wrappers!(tp, keep::Int)`
- `_inactive_storage_bytes(tp, first::Int, last::Int)`

Per-pool summaries are accumulated inline in the `trim!(pool)` loops over the
fixed slots and `others` (no separate merge helper is needed).

CPU pool entry points:

- `trim!(pool::AdaptiveArrayPool; force_gc=false)`
- `trim!(pool::AdaptiveArrayPool, ::Type{T}; force_gc=false)`
- `trim!(; force_gc=false)` for `get_task_local_pool()`

Metal extension entry points:

- Thin methods in `ext/AdaptiveArrayPoolsMetalExt/state.jl`
- Reuse generic typed-pool helpers where field layout matches.
- Add Metal-specific storage-byte helper if needed.

CUDA:

- Deferred until CPU + Metal behavior is proven.
- Expected to mirror Metal structurally.

Legacy (Julia < 1.12):

- The legacy pool (`src/legacy/`) uses an N-way set-associative cache with a
  different field layout (`nd_arrays`/`nd_dims`/… instead of `arr_wrappers`), so
  real reclamation is not implemented there.
- `trim!` is still **defined and exported** on the legacy path as a NO-OP that
  returns a zero summary and warns once (`@warn ... maxlog=1`). This keeps the
  public API callable across the full `[compat] julia = "1.10"` range, so
  dependent packages can `import`/call `trim!` on any supported Julia version
  without a load-time or runtime `UndefVarError`. Real reclamation requires
  Julia 1.12+. (Adding a real legacy implementation later would be non-breaking.)

## 10. Phase Plan (TDD)

Execute phases one at a time.

### Phase 1 — CPU `trim!`

**Goal:** Implement `trim!` for CPU pools and typed pools.

**Test strategy:**

- Active slots are preserved.
- Inactive `tp.vectors` entries are removed.
- Inactive wrapper-cache entries are removed.
- `reset!(pool); trim!(pool)` drops all retained CPU buffers.
- `trim!(pool, T)` affects only that type.
- `DisabledPool` returns a zero summary.
- `force_gc=true` sets `gc_triggered=true`.

**Tasks:**

- [ ] RED: Add CPU tests in `test/test_state.jl` or a new targeted test file.
- [ ] GREEN: Add generic typed-pool trimming helpers and CPU API methods.
- [ ] REFACTOR: Share summary aggregation helpers and docstrings.

**Quality gate:**

- [ ] Targeted CPU state tests pass.
- [ ] Existing allocation/zero-allocation tests for normal acquire paths still pass.
- [ ] `trim!` does not add code to hot acquire or automatic rewind paths.

**Rollback:** Revert `src/state.jl`, export changes, and new tests.

### Phase 2 — Safety Mode + Wrapper Reference Verification

**Goal:** Prove trimming actually detaches pool-held references, including
cached wrappers.

**Test strategy:**

- Create cached `Array`/`BitArray` wrappers for inactive slots, call `trim!`,
  and assert wrapper cache length is truncated.
- In `S=1`, released slots that were invalidated to length zero are still
  counted using capacity-aware byte accounting.
- Subsequent `acquire!` after trim self-heals by allocating fresh backing storage
  and wrapper entries.

**Tasks:**

- [ ] RED: Add tests for wrapper-cache truncation and S=1 byte estimates.
- [ ] GREEN: Fix byte accounting and wrapper-cache cleanup details.
- [ ] REFACTOR: Keep helper names and docstrings aligned with "inactive slots".

**Quality gate:**

- [ ] Targeted CPU safety tests pass.
- [ ] No stale wrapper retains old backing storage through the pool cache.

**Rollback:** Revert helper changes and tests from this phase.

### Phase 3 — Metal Parity

**Goal:** Add the same inactive-slot trimming API to Metal pools.

**Test strategy:**

- Active Metal slots are preserved.
- Inactive Metal backing `MtlArray` references are removed.
- Inactive Metal wrapper-cache entries are removed.
- `force_gc=true` calls Julia GC after references are detached.
- Return summary uses Metal-appropriate storage estimates.

**Tasks:**

- [ ] RED: Add Metal trim tests under `test/metal/`.
- [ ] GREEN: Add thin Metal pool entry points and byte helper overrides.
- [ ] REFACTOR: Keep CPU/Metal implementation split consistent with `reset!` and `empty!`.

**Quality gate:**

- [ ] Targeted Metal tests pass on the M1 machine.
- [ ] CPU tests from Phases 1-2 remain green.

**Rollback:** Revert Metal extension changes and Metal tests.

### Phase 4 — Public Docs

**Goal:** Document `trim!` as the manual inactive-slot cleanup operation.

**Test strategy:**

- Doctest or docs build if available.
- API reference includes `trim!`.
- User docs clearly distinguish `trim!`, `reset!`, and `empty!`.

**Tasks:**

- [ ] RED: Add docs references that fail until API exists.
- [ ] GREEN: Update API docs, examples, and exported function list.
- [ ] REFACTOR: Tighten wording around GC eligibility vs immediate memory return.

**Quality gate:**

- [ ] Docs build succeeds.
- [ ] No public docs imply immediate OS/VRAM release.

**Rollback:** Revert docs and export/reference updates.

## 11. Deferred Ideas

These are intentionally out of scope for the first implementation:

- Outlier-only trimming (`policy=:largest`, `min_bytes`, `ratio`).
- Dry-run reporting.
- Per-type or per-backend default thresholds.
- Hysteresis or usage tracking.
- Automatic periodic trimming.
- Replacement/reserve buffers for "right-sized" warm reuse.

All of these add policy. The first version should be a simple primitive:

> Drop inactive retained references; preserve active working set.

## 12. Risks

| Risk | Impact | Likelihood | Mitigation | Detection |
|---|---:|---:|---|---|
| Cached wrappers keep old buffers alive | High | Medium | Truncate wrapper caches with `tp.vectors` | Wrapper-cache reference tests |
| Byte summary reports zero for resized inactive buffers | Medium | High in `S=1` | Use capacity-aware estimates (`summarysize`, Metal allocation size) | S=1 byte-accounting tests |
| Users expect immediate RSS/VRAM drop | Medium | Medium | Document GC eligibility and `force_gc` limits | Docs review |
| `trim!` confused with `reset!` | Medium | Medium | Keep `n_active` unchanged and document `reset!; trim!` pattern | API docs/tests |
| Metal memory pool behavior differs from CPU | Medium | Medium | Match `empty!` caveats; test reference removal, not immediate VRAM | Metal tests |

## 13. Open Items

- Final summary field names: `slots_released` vs `inactive_slots_released`,
  `estimated_bytes_released` vs `bytes_detached`.
- Whether zero-argument `trim!()` should stay CPU-only or eventually accept a
  backend selector.
- Whether `force_gc=true` should call `GC.gc()` once or use a stronger collection
  pattern. Default remains `false`.
