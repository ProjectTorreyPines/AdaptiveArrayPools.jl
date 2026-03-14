# Pool-Backed Array Structural Mutation: Safety Analysis & Review Request

## Decision (Resolved)

**Runtime: warning (`@warn ... maxlog=1`).** Compile-time: hard error (`PoolMutationError`).

Rationale: `resize!/push!/pop!` on pool-backed arrays does NOT cause data corruption
or undefined behavior. The pool is self-healing — `acquire!` at the same slot always
resets the wrapper's memory reference. The only consequences are:
- Pooling benefits (zero-alloc reuse) may be lost
- Temporary extra memory retention (GPU: VRAM) until next acquire at the same slot

This is analogous to standard Julia aliasing behavior, not a safety violation.
Runtime error would be disproportionate; warning with `maxlog=1` is appropriate.

---

## Architecture Overview

AdaptiveArrayPools.jl is a Julia array pooling library. The core idea:

```
checkpoint!(pool)       # save current allocation state
v = acquire!(pool, T, n)  # get a T[n] array from pool (zero-alloc on reuse)
# ... use v ...
rewind!(pool)           # release all arrays acquired since checkpoint
```

### How arrays are stored

```
pool.float64 :: TypedPool{Float64}
  .vectors    :: Vector{Vector{Float64}}   # backing vectors (the actual storage)
  .arr_wrappers :: Vector{Union{Nothing, Vector{Any}}}  # cached N-D Array wrappers
  .n_active   :: Int                       # how many slots are currently "borrowed"
```

### What `acquire!` returns

`acquire!` does NOT return `tp.vectors[slot]` directly.
It returns a **cached wrapper** that shares the same underlying memory:

```julia
# CPU acquire path (simplified from src/acquire.jl)
function get_array!(tp, dims::NTuple{N,Int})
    slot = _claim_slot!(tp, total_len)
    vec = tp.vectors[slot]              # backing vector

    # Cache hit: reuse existing wrapper
    wrapper = arr_wrappers[N][slot]
    if wrapper !== nothing
        arr = wrapper::Array{T,N}
        setfield!(arr, :ref, getfield(vec, :ref))   # point to backing vec's Memory
        setfield!(arr, :size, dims)                  # set dimensions
        return arr                                    # WRAPPER returned, not vec
    end

    # Cache miss: create new wrapper sharing vec's memory
    arr = Array{T,N}(undef, zeros...)
    setfield!(arr, :ref, getfield(vec, :ref))
    setfield!(arr, :size, dims)
    _store_arr_wrapper!(tp, N, slot, arr)
    return arr
end
```

**Key invariant**: `wrapper.ref.mem === vec.ref.mem` — they share the same `Memory{T}` object.

On GPU (CUDA/Metal), the pattern is analogous but uses `DataRef` instead of `MemoryRef`:
- CUDA: `wrapper.data.rc === vec.data.rc`
- Metal: `getfield(wrapper, :data).rc === getfield(vec, :data).rc`

### What `rewind!` does

```julia
# Invalidation at rewind time (src/state.jl, simplified)
function _invalidate_released_slots!(tp::TypedPool{T}, old_n_active, S)
    new_n = tp.n_active

    # Step 1: Check for structural mutation (BEFORE any invalidation)
    _check_wrapper_mutation!(tp, new_n, old_n_active)

    # Step 2: Poison released vectors (NaN for floats, typemax for ints)
    _poison_released_vectors!(tp, old_n_active)

    # Step 3: Resize backing vectors to length 0
    for i in (new_n+1):old_n_active
        resize!(tp.vectors[i], 0)
    end

    # Step 4: Zero out cached wrapper dimensions
    for wrapper in released_wrappers
        setfield!(wrapper::Array, :size, (0, 0, ...))
    end
end
```

---

## The Question: What happens when user does `resize!(wrapper, 100_000)`?

The wrapper is the object returned by `acquire!`. The backing vector is `tp.vectors[slot]`.

### Case A: `resize!` beyond Memory capacity (reallocation)

```
Before:
  tp.vectors[slot]     → Memory_A (backing, pool-managed)
  arr_wrappers[1][slot] → Memory_A (wrapper, returned to user)

After resize!(wrapper, 100_000):
  tp.vectors[slot]     → Memory_A (unchanged — pool doesn't know about resize)
  arr_wrappers[1][slot] → Memory_B (NEW memory allocated by Julia's resize!)
```

The wrapper now has its own independent memory. **No memory overlap. No data corruption.**

### Case B: `resize!` within Memory capacity (no reallocation)

Julia may change only the logical length without reallocating. In this case:
- Same `Memory{T}` object, wrapper just has different logical bounds
- Elements beyond original length are uninitialized but within valid allocated memory
- No corruption, but reading beyond `vec_len` returns uninitialized data

### Case C: `push!` that triggers reallocation

Same as Case A. After enough pushes, Julia allocates new Memory.
The wrapper becomes independent from the backing vector.

---

## Self-Healing Mechanism

**The pool automatically recovers on the next `acquire!` at the same slot.**

### CPU path (src/acquire.jl:244-251)

```julia
# Cache hit: ALWAYS overwrites wrapper's MemoryRef
arr = wrapper::Array{T, N}
setfield!(arr, :ref, getfield(vec, :ref))   # ← forces wrapper back to backing vec
setfield!(arr, :size, dims)
return arr
```

CPU is unconditional — `setfield!(:ref)` runs every cache hit, regardless of mutation.
This means even if the wrapper pointed to Memory_B, next acquire resets it to Memory_A.

### GPU path (ext/.../acquire.jl)

```julia
# Cache hit: conditional update
if getfield(mtl, :data).rc !== getfield(vec, :data).rc
    _update_metal_wrapper_data!(mtl, vec)    # unsafe_free! old, copy new DataRef
end
setfield!(mtl, :dims, dims)
return mtl
```

GPU also self-heals, but conditionally (only when DataRef diverged).
`_update_metal_wrapper_data!` properly decrements the old refcount and increments the new.

---

## Potential Concerns & Edge Cases

### 1. GPU VRAM temporary leak

After `resize!(wrapper, 100_000)` on GPU:
- The cached wrapper in `arr_wrappers[N][slot]` holds a DataRef to the resized GPU buffer
- This DataRef keeps the GPU memory alive until the next `acquire!` at the same slot
- Between the resize and next acquire, VRAM is "leaked" (held by cached wrapper)

**Severity**: Low. Temporary, self-resolving. Proportional to the resized size.
**Mitigation**: Next acquire at same slot calls `_update_wrapper_data!` which frees it.

### 2. Invalidation path: dims zeroing on already-diverged wrapper

At rewind, `_invalidate_released_slots!` does:
```julia
setfield!(wrapper::Array, :size, (0, 0, ...))
```

If the wrapper was resized and now points to Memory_B:
- This zeros the wrapper's dims but doesn't affect Memory_B's data
- Memory_B is still referenced by the wrapper's MemoryRef/DataRef
- On CPU: Memory_B will be GC'd when wrapper ref is overwritten at next acquire
- On GPU: Same, but `_update_wrapper_data!` explicitly calls `unsafe_free!`

**No issue here** — dims zeroing is purely logical, doesn't affect memory management.

### 3. Concurrent/task-local pools

If multiple tasks share a pool (not recommended but possible):
- Task A resizes wrapper → wrapper diverges
- Task B rewinds → mutation check fires, warns
- No data corruption because wrapper and backing are independent

**No additional risk** from mutation in concurrent scenarios.

### 4. resize! WITHIN capacity but beyond backing vector length

```julia
v = acquire!(pool, Float64, 10)    # wrapper and vec both length 10
# Memory capacity might be >> 10 (from a previous larger acquire)
resize!(v, 50)                     # within capacity, no reallocation
```

Now wrapper has length 50 from the same Memory, but backing vec still has length 10.
Elements 11-50 of the wrapper are valid memory (within allocated Memory) but were not
initialized by the pool. They contain whatever was in Memory from the previous cycle.

**Severity**: Low. Not UB — Julia Memory objects are fully allocated. Data is unpredictable
but not dangerous (it's old pool data, not out-of-bounds memory).

### 5. Could `resize!` on wrapper corrupt the backing vector?

**No.** `resize!` on wrapper can only:
- Reallocate wrapper's Memory (Case A) → backing vector unaffected
- Change wrapper's logical bounds within existing Memory (Case B) → backing vector's
  MemoryRef still points to the same range

The backing vector's MemoryRef is a separate Julia object. `resize!` on the wrapper
cannot modify `tp.vectors[slot]`'s MemoryRef fields.

### 6. Could `push!` on wrapper corrupt the backing vector?

**No.** Same reasoning. `push!` may trigger `resize!` internally (Case A) or grow
within capacity (Case B). Neither modifies the backing vector.

### 7. What about `pop!`, `deleteat!`, `splice!`, `empty!`?

These only modify the wrapper (shrink or rearrange elements). They cannot affect
the backing vector because the backing vector is a separate Julia object with its
own MemoryRef. Even in the shared-Memory case (Case B), these operations only
change the wrapper's logical bounds, not the backing vector's.

### 8. What if user does `resize!(wrapper, 0)` then `resize!(wrapper, original_size)`?

- First resize to 0: no reallocation (within capacity), just dims change
- Second resize back: no reallocation (still within capacity), dims restored
- Backing vector is unaffected throughout
- At rewind: no mutation detected (same Memory, same length as backing)

This is actually a **non-issue** — pool wouldn't even warn about it.

---

## Summary: Risk Assessment

| Scenario | Data Corruption | UB | VRAM Leak | Functional Impact |
|----------|:-:|:-:|:-:|:--|
| resize! beyond capacity | No | No | Temporary (GPU) | Pool self-heals on next acquire |
| resize! within capacity | No | No | No | Wrapper reads stale pool data |
| push! with reallocation | No | No | Temporary (GPU) | Pool self-heals on next acquire |
| pop!/deleteat!/splice! | No | No | No | None — wrapper shrinks |
| Concurrent mutation | No | No | Temporary (GPU) | Warning at rewind |

**No scenario causes data corruption, undefined behavior, or permanent resource leak.**

The pool is **self-healing by design**: the acquire path always resets the wrapper's
memory reference to the backing vector, regardless of what happened to the wrapper
between acquire and rewind.

---

## Current Implementation

### Compile-time (macro expansion)

```julia
# src/macros.jl — PoolMutationError
@with_pool pool begin
    v = acquire!(pool, Float64, 10)
    resize!(v, 100)   # ← caught at macro expansion → PoolMutationError (hard error)
end
```

This is a **hard error** at compile time. Zero runtime cost. Catches definite
anti-patterns before code runs. User can `copy()` to opt out.

### Runtime (rewind time)

```julia
# src/debug.jl — _check_wrapper_mutation!
# Called from _invalidate_released_slots! before poison/invalidation
@noinline function _check_wrapper_mutation!(tp::TypedPool{T}, new_n, old_n)
    for i in (new_n+1):old_n
        vec = tp.vectors[i]
        for wrapper in released_wrappers_at_slot_i
            # Check 1: Memory identity diverged
            if wrapper.ref.mem !== vec.ref.mem
                @warn "..." maxlog=1
                return
            end
            # Check 2: wrapper length exceeds backing
            if prod(wrapper.size) > length(vec)
                @warn "..." maxlog=1
                return
            end
        end
    end
end
```

This is a **warning** with `maxlog=1` (one-time advisory per session).
Only runs when safety level S >= 1. Early return after first detection.

---

## Options Under Consideration

### Option 1: Keep as warning (current)
- **Pro**: Non-disruptive. Pool self-heals. No false-positive breakage.
- **Pro**: `maxlog=1` means minimal noise.
- **Con**: User might not notice the warning.

### Option 2: Elevate to error (throw)
- **Pro**: Forces user to fix the anti-pattern immediately.
- **Con**: Throwing during `rewind!` would skip cleanup of remaining TypedPools.
  - Could partially fix with try/catch in rewind loop, but adds complexity.
- **Con**: The mutation is actually harmless — pool self-heals.
- **Con**: May break valid code where user intentionally resizes (rare but possible).

### Option 3: Remove runtime check entirely
- **Pro**: Simplest. No code, no maintenance.
- **Pro**: The mutation is demonstrably harmless.
- **Con**: User gets no feedback about potential performance anti-pattern.
- **Con**: GPU VRAM temporary leak goes unnoticed.

### Option 4: Make it configurable (warn at S=1, error at S=2)
- **Pro**: Flexible — users can choose their strictness level.
- **Con**: AdaptiveArrayPool has binary S (0 or 1), no S=2 level.
- **Con**: Adding S=2 increases type parameter complexity.

---

## Review Questions

1. **Is there any edge case we missed** where `resize!/push!` on a pool-backed wrapper
   could cause data corruption, undefined behavior, or a permanent resource leak?

2. **Given the self-healing mechanism**, is the runtime check (warning or error)
   justified, or is it over-engineering?

3. **If we keep the check**, should it be:
   - A warning (current) — informational, non-blocking
   - An error — strict, but risks disrupting rewind cleanup
   - Something else (e.g., debug log, opt-in only)

4. **The compile-time check (PoolMutationError) is a hard error.** Given that
   mutation is actually harmless, should this also be downgraded to a warning?
   Or is compile-time strictness still valuable as a "lint" for anti-patterns?

---

## Appendix: Key Source Files

| File | Description |
|------|-------------|
| `src/acquire.jl:234-261` | CPU `get_array!` — wrapper creation/reuse with `setfield!(:ref)` |
| `src/state.jl:258-314` | `_invalidate_released_slots!` — poison + resize + dims zeroing |
| `src/debug.jl:352-456` | `_check_wrapper_mutation!` — CPU runtime mutation detection |
| `src/macros.jl:2448-2660` | `PoolMutationError` + compile-time `_check_structural_mutation` |
| `ext/.../acquire.jl` | GPU `get_array!` — wrapper creation with DataRef identity check |
| `ext/.../debug.jl` | GPU `_check_wrapper_mutation!` — DataRef-based detection |
