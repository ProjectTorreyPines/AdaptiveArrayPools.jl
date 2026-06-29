# Capacity Compaction for AdaptiveArrayPools (`compact!`)

- **Date:** 2026-06-28
- **Status:** Implemented (CPU + Metal + CUDA), Julia 1.12+. Shipped on top of the
  `trim!` work. The Timer-driven auto mode (§9) remains future work.
- **Scope:** Manual `compact!` API that shrinks over-allocated backing buffers in
  place. CPU + Metal + CUDA parity; an optional Timer-driven auto mode is deferred.
- **Author:** design session

---

## 1. Motivation

`trim!` (the sibling feature) drops the pool's references to **inactive** slots —
it reduces the *number of retained slots*. It deliberately never touches active
slots, so it cannot help when the bloated buffer is one the caller is still
holding.

The remaining gap: a slot's backing buffer keeps the **high-water-mark capacity**
it ever needed. A rare large input grows a slot to (say) 1,000,000 elements; every
later `acquire!` on that slot only uses 100, but the buffer stays at 1,000,000
because `resize!` down never frees capacity (verified: `resize!(v, 100)` leaves
`capacity == 1_000_000`). That over-allocation is invisible to `trim!` whenever
the slot is active.

`compact!` fills that gap:

> For buffers whose allocated capacity is far larger than the size currently in
> use, shrink the capacity back down to a small multiple of the used size —
> reclaiming the outlier bloat while keeping a right-sized buffer for reuse.

This is a **cold-path** operation: called rarely (manually, or eventually on a
slow timer), not in the hot `acquire!`/`rewind!` loop. The design therefore
favors simplicity and safety over maximal reclamation.

## 2. Relationship to `trim!` — two orthogonal axes

| | `trim!` (exists) | `compact!` (this doc) |
| --- | --- | --- |
| Reduces | number of retained slots (structural) | over-allocated capacity of a slot's buffer |
| Touches | inactive slots only | retained buffers, **including active ones** |
| Safety | always safe (no live references) | must re-sync live wrappers (see §7) |
| Verb family | `reset!` / `empty!` / `trim!` | adds `compact!` |

They are independent: `trim!` removes the tail; `compact!` compacts what remains.
Keeping them as **separate functions** (rather than a `trim!(; shrink_factor=…)`
kwarg) keeps each operation single-purpose and preserves `trim!`'s "always safe"
guarantee. The opt-in is layered: calling `compact!` opts into capacity compaction
of inactive slots (Tier 1, safe); `compact!(…; active=true)` further opts into
compacting active slots (Tier 2, §7).

## 3. Core concept: capacity vs logical length

For a backing vector `v = tp.vectors[slot]`:

- **used** = the slot's **current logical extent**, recorded per-slot in
  `tp.slot_extents[slot]` by `_claim_slot!` (`_slot_used`). This is the size of the
  most recent claim and covers **both** acquisition paths — `acquire!` (Array
  wrapper) *and* `acquire_view!` (which returns an **uncached** `SubArray`/
  `ReshapedArray`). **Not** `length(v)`: `_claim_slot!` only ever *grows* the backing
  (`if length(vec) < n; resize!`, `acquire.jl:73`), so `length(v)` is the **high-water
  mark**. An earlier design read `used` from the cached wrappers only; that
  **under-counted a live view's extent** (views cache no wrapper) and let
  `compact!(active=true)` shrink a backing below a held view → OOB. `slot_extents`
  fixes this (and, being a `Vector{Int}`, makes `_slot_used` return a concrete `Int`,
  keeping the gate arithmetic allocation-free). (Verified by TDD.)
- **capacity** = allocated buffer length (`length(getfield(v, :ref).mem)` on 1.12;
  wrapped in `_slot_capacity`).

`compact!` shrinks `capacity` toward `used`; it never changes `used` (held wrappers
keep their `:size`, and `target ≥ used` so every wrapper stays in bounds).

A slot is a `compact!` candidate when **both**:

```
capacity ≥ factor × used                      # ratio gate: "≥10× bloated"
(capacity − target) × sizeof(T) ≥ min_bytes    # absolute gate: real reclaim
    where target = ceil(shrink_to × used)
```

The absolute byte gate prevents pointless reallocation of small buffers (e.g. a
1000-capacity / 50-used buffer is 20× bloated but only reclaims ~7.6 KB).

## 4. API (as implemented)

Mirror `trim!`'s surface for consistency:

```julia
compact!(pool::AdaptiveArrayPool;
         factor::Real    = 10,      # shrink only if capacity ≥ factor × used
         shrink_to::Real = 1.5,     # new capacity = ceil(shrink_to × used)
         min_bytes::Int  = 2^20,    # …and only if ≥ this many bytes are reclaimed
         active::Bool    = true,    # default: also compact ACTIVE slots (§7); false = inactive-only
         force_gc::Bool  = false)
compact!(pool::AdaptiveArrayPool, ::Type{T}; kwargs...)        # single type
compact!(pool::AdaptiveArrayPool, types::Type...; kwargs...)   # varargs (mirrors trim!)
compact!(; kwargs...)                                          # task-local CPU pool
```

GPU extensions, `DisabledPool`, and the legacy (< 1.12) path mirror `trim!`'s
overload set exactly (legacy = warn-once no-op). Returns the same shape of
summary as `trim!`, plus a compaction-specific field:

```julia
(; slots_compacted, bytes_reclaimed, gc_triggered)
```

(Reuse the `_trim_summary` pattern — a single `_compact_summary` helper — so every
overload returns one concrete, inferrable `NamedTuple`. See the `trim!` PR's
type-stability work for why this matters.)

## 5. Shrink mechanism — in-place backing shrink (preserve object identity)

**Decision (revised):** shrink the backing vector **in place** — keep the same
`Vector` object and swap only its internal `Memory` for a new, smaller one. Do
**not** replace the slot object (`tp.vectors[slot] = nv`).

Use an **explicit `Memory` swap**, not `sizehint!`. `sizehint!` is documented as an
*advisory hint* — it is not guaranteed to shrink the buffer (the amount is
implementation/version-defined), so it cannot be the mechanism for a reclamation
API. The explicit swap is deterministic:

```julia
target = ceil(Int, shrink_to * used)          # target ≥ used
nv = Vector{T}(undef, target)                  # new small buffer (guaranteed size)
copyto!(nv, 1, v, 1, used)                     # preserve the live `used` elements
setfield!(v, :size, (used,))                   # logical length stays `used`
setfield!(v, :ref,  getfield(nv, :ref))        # v keeps identity; its Memory is now nv's
# then re-sync every cached wrapper for this slot (see §7)
```

This is the same `setfield!(:ref/:size)` the pool already uses for wrapper reuse,
so it is 1.12+-only (legacy < 1.12 → no-op, like `trim!`).

Why keep the object and swap `:ref`, rather than "allocate a new vector and assign
the slot" (the earlier draft)? A smoke test
(`scratchpad/manual_refswap.jl`) confirmed the difference for live
`acquire_view!` views:

- **Replace object** (`tp.vectors[slot] = nv`): a `SubArray`'s `parent` still points
  at the *old* vector object → the view does **not** follow → broken.
- **Swap `:ref` in place**: the vector object keeps its identity and only its `:ref`
  changes; a `SubArray` reads through `parent` on every access, so it **follows the
  new buffer automatically** (verified: identity kept, `capacity 1M→5`, view reads
  the new buffer, old `Memory` GC-collected).

So **both** wrappers (re-synced, §7) **and** views (follow `parent`) survive.

A bounds note: a view is only valid up to the slot's current `used` length, so
shrinking capacity toward `used` never invalidates a view that was valid at the
current size (a stale wider view was already out of bounds before compaction).

**Does the old buffer actually get freed?** Yes — verified. The §5 `Memory` swap
re-points `v.ref` at a new smaller `Memory`; a `WeakRef` to the old `Memory` is dead
after `GC.gc()`, confirming it is reclaimed. One catch:
each `Array` **wrapper carries its own `:ref`** still pointing at the old `Memory`,
which pins it. So full reclamation requires **both** the in-place shrink **and** the
wrapper re-sync (§7.1) — skipping the re-sync leaves the wrapper stale *and* keeps
the old buffer alive. (`acquire_view!` views do not pin the old buffer: they read
through `parent`, which now points at the new one.)

Note this is a different operation from `trim!`. `trim!` shrinks the *outer slot
array* (`resize!(tp.vectors, n_active)`), dropping whole inactive backing-`Vector`
objects — it reduces the slot **count** and returns those buffers 100%. `compact!`
keeps the slot and shrinks its buffer's **capacity** in place. They are
complementary (remove unused slots vs. right-size the buffers of slots still in
use), not two spellings of one thing.

## 6. Modular layering (for the future auto-manager)

The reusable unit is the per-slot primitive; everything else composes it:

```
_maybe_compact_slot!(tp, slot, factor, shrink_to, min_bytes) -> reclaimed::Int
    │   one slot: gate check → (if candidate) in-place shrink + wrapper re-sync
    │   returns bytes reclaimed (0 if skipped)
    ▼
compact!(pool; …) -> summary
    │   scan all typed pools' retained slots, call the primitive, accumulate
    ▼
auto-manager (Phase C):
    Timer fires → sets a "compaction requested" flag (does NOT touch buffers)
    → the next pool touch point on the user's task runs the deferred compaction
```

The auto-manager writes **no new reclamation logic** — it only *schedules* a call
to `compact!`/`_maybe_compact_slot!`, which then runs synchronously on the user's
task at the next touch point (§7.3). The `factor`/`min_bytes` gates make it
naturally "leave the common case alone, only touch true outliers."

That touch point may have `n_active > 0`, so the auto path **can** compact active
slots (Tier 2), not just inactive ones — safely, because the in-place swap +
re-sync runs single-threaded at a moment the user is inside a pool call rather than
mid-array-access. (This is the whole point of cooperative deferral: it lets the
timer reach the long-lived *active* outliers the user cares about, without the data
race a direct background mutation would cause.)

## 7. Safety

### 7.1 Two tiers — and why all the risk lives in Tier 2

The entire safety story splits cleanly by slot state:

- **Tier 1 — inactive slots (default).** No live user reference exists. Shrinking
  is unconditionally safe; the only bookkeeping is clearing the slot's cached
  wrappers (they get rebuilt on the next `acquire!`). A plain `compact!(pool)` does
  only this.
- **Tier 2 — active slots (opt-in, `active=true`).** The user is currently holding
  the array from this slot, so the swap must re-sync the wrapper (§7.1) and is only
  done at a point where the user is not mid-access — either an explicit mid-scope
  manual call, or the auto path's cooperative deferral (§7.3). With the in-place
  strategy (§5) both wrappers and views stay valid, so the gating concern here is
  **concurrency, not references**.

**Why Tier 2 is possible at all** (the non-obvious part): the array `acquire!`
returns is **not a copy** — it *is* the pool's own cached wrapper object, the same
one stored in `tp.arr_wrappers[N][slot]`. (Verified: `a === arr_wrappers[1][1]`.)
So the pool is not blind to "what the user holds" — it handed out its own objects
and still references every one of them. The wrapper is a thin handle that only
*points* at the backing buffer via a mutable `:ref`; the data lives in the buffer.
So after the in-place backing shrink (§5), the pool re-points the wrapper:

```julia
# v = tp.vectors[slot]; after the §5 in-place Memory swap
for wrappers in tp.arr_wrappers           # one per dimensionality N
    w = wrappers === nothing ? continue : wrappers[slot]
    w === nothing && continue
    setfield!(w, :ref, getfield(v, :ref))    # re-point to v's new (smaller) Memory
    # :size unchanged — logical dims are preserved
end
```

The wrapper carries its **own** `:ref` (a copy taken at creation), so it does not
follow `v`'s in-place change on its own and must be re-synced. The user holds the
*same* `w` object, so repointing its `:ref` redirects the user's array — no
reassignment in user code. This is the **same `setfield!(:ref)` the pool already
performs on every `acquire!`**, so the mechanism is load-bearing, not new.

### 7.2 Views survive too — under the in-place strategy

With the §5 in-place shrink, escaped `acquire_view!` results are **no longer a
hazard**. A `SubArray` / `ReshapedArray` holds its `parent` — the backing **vector
object** — and reads through it on every access. Because in-place shrink keeps that
object's identity and only swaps its `:ref`, the view transparently follows to the
new buffer (verified in the smoke test). Both reference kinds survive:

| held reference | how it survives an in-place compaction |
| --- | --- |
| `acquire!` / `reshape!` `Array` wrapper | pool re-syncs its own `:ref` (§7.1) |
| `acquire_view!` `SubArray` / `ReshapedArray` | follows `parent`'s in-place `:ref` swap automatically |

This is exactly why §5 chose in-place over replace-object: the *replace-object*
alternative (`tp.vectors[slot] = nv`) strands views on the old object (stale read +
pinned old buffer). Staying in-place is a **correctness requirement** for Tier 2,
not a stylistic preference.

The only residual view caveat is bounds: never shrink capacity below the slot's
current `used` length — which the policy never does (`target ≥ used`).

### 7.3 Concurrency — the real constraint for auto (Phase C)

References are solved (§7.1–7.2). The genuine wall for **active** compaction is
concurrency: a `Timer` callback that reallocates a buffer while the user's task is
reading/writing it is a data race no reference trick can fix — lost writes,
use-after-free on the freed old `Memory`, a torn `:ref`. This is independent of
wrappers vs views.

**The only safe auto design is cooperative deferral:**

- The `Timer` does **not** touch buffers. It sets a "compaction requested" flag
  (plus the policy).
- The actual shrink runs **synchronously on the user's task** at the next pool
  touch point (`acquire!` / `checkpoint!` / `rewind!`, or an explicit
  `maybe_compact!(pool)` the user can sprinkle into a long compute). At that instant
  the user is inside a pool call, not mid-array-access, so the in-place swap +
  re-sync is single-threaded-safe — and it can compact **active** slots, not just
  inactive ones.
- Honest limitation: a buffer held across a long compute with **no** intervening
  pool call cannot be compacted until the next touch point. Reclamation is delayed,
  never unsafe. With `Threads.nthreads() > 1`, stricter synchronization is required
  if pool calls can race across threads.

So the user's goal — "a timer that reclaims oversized **active** outliers" — is
achievable: the timer **schedules** the compaction for the next safe touch point
rather than performing it directly.

## 8. Backend parity

- **CPU (`Vector{T}`):** the explicit `Memory` swap of §5. CPU's native `resize!`
  *never* reallocates on shrink (capacity is retained forever), so the explicit
  swap is the only way to actually return the memory.
- **CUDA / Metal:** backing is `CuVector` / `MtlArray`. Same shape — allocate a
  smaller **device** buffer, copy `used` elements device-to-device, swap the
  `:data`/DataRef + `:maxsize` + `:dims` (keeping the array object's identity, as on
  CPU), then re-sync the GPU wrappers. Mind the GPU refcount rules the `trim!` work
  established (Metal passes the `DataRef` directly; CUDA's constructor takes
  ownership). Crucially, this must **not** route through the pool's
  `_resize_to_fit!`, which deliberately *preserves* the GPU allocation
  (`maxsize`) for reuse — `compact!` is exactly the op that overrides that and frees
  it. Metal is hardware-testable on Apple Silicon; CUDA via CI.

  Note the symmetry: CUDA.jl's native `resize!` (v5.10+) already reallocates a
  device buffer when it shrinks below `capacity ÷ 4` (`acquire.jl:18`). That is the
  same idea as `compact!`'s `factor` gate — "realloc once `used` drops below a
  fraction of capacity" — just with a fixed 25% threshold instead of our
  configurable `factor` (e.g. `factor=10` ⇒ realloc below 10%). `compact!` brings
  that capacity-aware behavior to CPU (which has none) and makes the threshold a
  policy knob on both backends.
- **Legacy (< 1.12):** defined no-op that warns once and returns a zero summary,
  exactly like legacy `trim!` (the `setfield!(:ref)` mechanism is 1.12+-only).

## 9. Phasing

1. **Phase A — CPU `compact!`** + the `_maybe_compact_slot!` primitive, full TDD:
   capacity actually reclaimed, `used` data preserved, live wrapper still valid
   after compaction, gates respected (ratio + min_bytes), no-op when nothing is
   bloated, type-stable summary, varargs/per-type/DisabledPool/legacy parity.
2. **Phase B — Metal + CUDA parity** (Metal validated on hardware).
3. **Phase C — auto mode (deferred; out of scope for the first PR).** When taken
   up: an external policy decides *when* and flips a "compaction requested" flag;
   the actual compaction runs at a safe pool-scope boundary — concretely, the end
   of a `@with_pool` block (the `rewind!`/scope-finalization the macro already
   emits), not an arbitrary instant. That point is single-threaded-safe and may have
   `n_active > 0`, so it can also reach active outliers (§7.3). Driving it from a
   `Timer` is one possible policy source, but the **trigger is decoupled from the
   execution point** — the timer only flags; the macro boundary executes.

## 10. Open questions / decisions still to make

- **Default `min_bytes`** — `2^20` (1 MiB) is a starting guess; tune against a
  benchmark before settling.
- **Default `factor` / `shrink_to`** — `factor=10`, `shrink_to=1.5` are starting
  guesses; CUDA.jl's native 25% (`÷4`) is a reference point for `factor`.
- **Whether `compact!` should also drop fully-inactive slots** (i.e. subsume
  `trim!`) or stay strictly capacity-only and compose with `trim!`. Current lean:
  stay capacity-only; users call `trim!(pool); compact!(pool)` for both axes.
- **Per-type policy** — whether `factor`/`shrink_to` should ever be settable
  per element type, or stay global per call. Default: global per call (simpler).
