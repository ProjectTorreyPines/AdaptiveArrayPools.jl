# Naming & Refactoring Proposal: Post-Optimization Cleanup

> **Status**: Brainstorm / RFC
> **Context**: After Phase 3 (dynamic-selective) and Phase 5 (typed-fallback) optimizations,
> many internal function names still reflect the original "untracked acquire detection" mental
> model. This document proposes renaming to match the evolved architecture.

---

## 1. Current Architecture: Three Execution Modes

The `@with_pool` macro generates one of three checkpoint/rewind strategies:

| Mode | Checkpoint | Acquire | Rewind | When |
|------|-----------|---------|--------|------|
| **Typed** | `checkpoint!(pool, T...)` | `_acquire_impl!(pool, T, ...)` | `rewind!(pool, T...)` | All types statically known |
| **Dynamic-Selective** | `_depth_only_checkpoint!(pool)` | `acquire!(pool, T, ...)` *(unchanged)* | `_dynamic_selective_rewind!(pool)` | Types only known at runtime |
| **Full** (manual) | `checkpoint!(pool)` | `acquire!(pool, T, ...)` | `rewind!(pool)` | User calls manually |

Additionally, the **Typed** mode has a runtime fallback:
- If `_can_use_typed_path()` is false → `_typed_checkpoint_with_lazy!` + `_typed_selective_rewind!`

---

## 2. Naming Tensions

### 2.1 `_mark_untracked!` — The Core Irony

**Current name**: `_mark_untracked!`
**What it does**: Records type usage in bitmask. Triggers lazy checkpoint on first touch.

The word "untracked" is **doubly misleading**:
1. The function **tracks** type usage (sets bitmask bits)
2. The data it records is used to **selectively rewind** (the opposite of "untracked")

The original semantics: "mark that this acquire happened in a path the macro doesn't track."
The actual semantics now: "record that type T was touched at this depth, and lazily checkpoint if needed."

#### Candidates

| Candidate | Pros | Cons |
|-----------|------|------|
| `_record_type_touch!` | "touch" captures first-touch/lazy-checkpoint semantics; action-oriented | Doesn't convey the bitmask mechanism |
| `_track_type_usage!` | Most literal description of what happens | "track" is overloaded (macro "tracks" types too) |
| `_notify_acquire!` | Observer-pattern feel; captures side-effect (lazy checkpoint) | Too generic; doesn't convey type-specificity |
| `_register_type!` | Clean, idiomatic ("register X in a registry") | Doesn't convey the "at this depth" scoping |
| `_touch_type!` | Shortest; "touch" is a Unix/DB idiom for "first access triggers action" | Might be too terse for complex semantics |
| `_mark_type_used!` | Simple and accurate | Still has "mark" which is vague |

### 2.2 `_acquire_impl!` — The "Fast Path" Naming

**Current name**: `_acquire_impl!`
**What it does**: Core acquire logic without type tracking. Called by macro-transformed code.

The `_impl!` suffix is conventional but **non-descriptive**. It doesn't convey *why* this variant
exists (to skip tracking overhead when the macro already knows the types).

#### Candidates

| Candidate | Pros | Cons |
|-----------|------|------|
| `_acquire_direct!` | "direct" = no intermediary tracking step | Might imply "direct memory access" |
| `_acquire_bare!` | "bare" = stripped of wrapper logic | Non-standard terminology |
| `_acquire_core!` | "core" = the essential operation | Generic; doesn't explain *why* it's separate |
| `_acquire_scoped!` | "scoped" = macro already manages this scope | Misleading — the function itself isn't scoped |
| Keep `_acquire_impl!` | Well-understood `_impl` convention in Julia | Doesn't explain the tracking bypass |

### 2.3 `_untracked_fixed_masks` / `_untracked_has_others` — Field Names

**Current names**: Pool fields storing per-depth bitmask data.
**What they store**: Which types were acquired at each depth (for selective rewind).

These fields are the **runtime type tracking** data structure, yet named "untracked."

#### Candidates

| Candidate | Pros | Cons |
|-----------|------|------|
| `_touched_fixed_masks` / `_touched_has_others` | "touched" matches first-touch semantics | Might confuse with "dirty" bit patterns |
| `_used_fixed_masks` / `_used_has_others` | Simplest, most literal | Too generic |
| `_acquired_fixed_masks` / `_acquired_has_others` | Directly describes the event (acquire happened) | Slightly long |
| `_runtime_fixed_masks` / `_runtime_has_others` | Contrasts with "compile-time" tracked types | Doesn't describe *what* is tracked |
| Keep current names | Consistency with existing code, comments, tests | Perpetuates the "untracked" confusion |

### 2.4 Mode-Specific Functions — Consistency

The three modes don't follow a consistent naming pattern:

```
Typed:             checkpoint!(pool, T...)     / rewind!(pool, T...)
  + fallback:      _typed_checkpoint_with_lazy!(pool, T...) / _typed_selective_rewind!(pool, mask)
Dynamic-Selective: _depth_only_checkpoint!(pool) / _dynamic_selective_rewind!(pool)
Full:              checkpoint!(pool)           / rewind!(pool)
```

**Observation**: The "typed" fallback functions have long compound names that mix the *mode*
(`typed`) with the *mechanism* (`with_lazy`, `selective`).

#### Possible Consistent Scheme

Option A — Mode prefix:
```
_typed_checkpoint!          → checkpoint!(pool, T...)    (already clean)
_typed_lazy_checkpoint!     → _typed_checkpoint_with_lazy!(pool, T...)
_typed_selective_rewind!    → _typed_selective_rewind!(pool, mask)  (already clean)
_dynamic_checkpoint!        → _depth_only_checkpoint!(pool)
_dynamic_rewind!            → _dynamic_selective_rewind!(pool)
```

Option B — Mechanism suffix:
```
_checkpoint_typed!          (checkpoint the typed pools)
_checkpoint_lazy!           (checkpoint with lazy first-touch)
_checkpoint_depth_only!     (only increment depth)
_rewind_typed!              (rewind typed pools)
_rewind_selective!          (rewind based on bitmask)
_rewind_dynamic!            (dynamic bitmask rewind)
```

---

## 3. Holistic Renaming Proposals

### Proposal A: "Touch" Metaphor (First-Touch Semantics)

The architecture's key insight is **first-touch tracking**: when a type is first used at a depth,
it gets recorded (and lazily checkpointed). The "touch" metaphor captures this cleanly.

```
# Type tracking
_mark_untracked!(pool, T)        → _touch_type!(pool, T)

# Pool fields
_untracked_fixed_masks           → _touched_fixed_masks
_untracked_has_others            → _touched_has_others

# Acquire internals (keep _impl convention)
_acquire_impl!(pool, T, n)      → (keep as is, or _acquire_core!)
_unsafe_acquire_impl!(...)      → (keep as is, or _unsafe_acquire_core!)

# Dynamic mode (rename for symmetry)
_depth_only_checkpoint!(pool)    → _lazy_checkpoint!(pool)       # "lazy" = defers to first touch
_dynamic_selective_rewind!(pool) → _lazy_rewind!(pool)           # symmetric with checkpoint

# Typed fallback (simplify)
_typed_checkpoint_with_lazy!     → _typed_lazy_checkpoint!       # adjective before noun
_typed_selective_rewind!         → _typed_lazy_rewind!           # symmetric pair

# Macro generators (follow function names)
_generate_dynamic_selective_checkpoint_call → _generate_lazy_checkpoint_call
_generate_dynamic_selective_rewind_call    → _generate_lazy_rewind_call

# Guards
_can_use_typed_path              → (keep as is — already clear)
_tracked_mask_for_types          → (keep as is — already clear)
```

**Pros**: Concise, consistent metaphor, captures the core mechanism.
**Cons**: "lazy" is overloaded in CS (lazy evaluation, lazy initialization).

### Proposal B: "Record/Direct" Pair (Action-Based)

Focus on what each function *does* as an action:

```
# Type tracking
_mark_untracked!(pool, T)        → _record_type_touch!(pool, T)

# Pool fields
_untracked_fixed_masks           → _acquired_type_masks
_untracked_has_others            → _acquired_has_others

# Acquire internals
_acquire_impl!(pool, T, n)      → _acquire_direct!(pool, T, n)     # "direct" = no recording
_unsafe_acquire_impl!(...)      → _unsafe_acquire_direct!(...)

# All convenience _impl! follow:
_zeros_impl!                     → _zeros_direct!
_ones_impl!                      → _ones_direct!
_similar_impl!                   → _similar_direct!

# Dynamic mode
_depth_only_checkpoint!(pool)    → _deferred_checkpoint!(pool)       # "deferred" = save later
_dynamic_selective_rewind!(pool) → _deferred_selective_rewind!(pool) # rewind what was deferred

# Typed fallback
_typed_checkpoint_with_lazy!     → _typed_deferred_checkpoint!       # typed + deferred for extras
_typed_selective_rewind!         → (keep — already descriptive)

# Macro generators
_generate_dynamic_selective_*    → _generate_deferred_*
```

**Pros**: Very descriptive, each name tells you exactly what happens.
**Cons**: Longer names, "deferred" is less intuitive than "lazy."

### Proposal C: "Scope" Metaphor (Inside/Outside Macro Scope)

Frame the naming around the key architectural distinction: code inside `@with_pool` scope
(macro-managed) vs outside (self-tracking):

```
# Type tracking — called from "outside scope" or "dynamic scope"
_mark_untracked!(pool, T)        → _track_type!(pool, T)

# Pool fields
_untracked_fixed_masks           → _scope_type_masks      # per-scope tracking
_untracked_has_others            → _scope_has_others

# Acquire internals — used by "in-scope" (macro-managed) code
_acquire_impl!(pool, T, n)      → _scoped_acquire!(pool, T, n)   # "scoped" = macro handles tracking
_unsafe_acquire_impl!(...)      → _scoped_unsafe_acquire!(...)

# Dynamic mode
_depth_only_checkpoint!(pool)    → _open_scope!(pool)              # "open" a new tracking scope
_dynamic_selective_rewind!(pool) → _close_scope!(pool)             # "close" and rewind the scope

# Typed fallback
_typed_checkpoint_with_lazy!     → _open_typed_scope_with_fallback!
_typed_selective_rewind!         → _close_typed_scope_with_fallback!

# Guards
_can_use_typed_path              → _scope_is_typed_only
```

**Pros**: Captures the architectural mental model cleanly.
**Cons**: "scope" semantics might clash with Julia's lexical scoping concepts.

### Proposal D: Minimal Rename (Conservative)

Only rename the most confusing items, keep everything else:

```
# The one truly misleading name:
_mark_untracked!(pool, T)        → _record_type_touch!(pool, T)

# The confusing fields:
_untracked_fixed_masks           → _touched_type_masks
_untracked_has_others            → _touched_has_others

# Everything else stays as-is
_acquire_impl!                   → (keep)
_depth_only_checkpoint!          → (keep)
_dynamic_selective_rewind!       → (keep)
_typed_checkpoint_with_lazy!     → (keep)
_typed_selective_rewind!         → (keep)
```

**Pros**: Minimal churn, only fixes the genuinely confusing names.
**Cons**: Misses the opportunity for holistic consistency.

---

## 4. Cross-Cutting Concerns

### 4.1 Public API — Should NOT Change

These are stable public APIs and should **never** be renamed:
- `acquire!`, `unsafe_acquire!`, `acquire_view!`, `acquire_array!`
- `checkpoint!`, `rewind!`, `reset!`, `empty!`
- `zeros!`, `ones!`, `trues!`, `falses!`, `similar!`
- `@with_pool`, `@maybe_with_pool`
- `get_task_local_pool`

### 4.2 CUDA Extension Parity

Any rename must be mirrored in:
- `ext/AdaptiveArrayPoolsCUDAExt/types.jl`
- `ext/AdaptiveArrayPoolsCUDAExt/state.jl`
- `ext/AdaptiveArrayPoolsCUDAExt/acquire.jl`

### 4.3 Test Impact

Renaming internal functions affects:
- `test/test_macro_internals.jl` (directly calls `_depth_only_checkpoint!`, `_dynamic_selective_rewind!`, etc.)
- `test/test_state.jl` (checkpoint/rewind tests)
- `test/test_macroexpand.jl` (checks expanded code contains specific function names)
- Any benchmarks referencing internal functions

### 4.4 `Bit` 15 / Bit 14 Constants

Currently the mode flags are raw hex literals (`0x8000`, `0x4000`). A related cleanup:
```julia
const _DYNAMIC_MODE_BIT   = UInt16(0x8000)  # bit 15
const _LAZY_MODE_BIT      = UInt16(0x4000)  # bit 14
const _MODE_BITS_MASK     = UInt16(0xC000)  # bits 14-15
const _TYPE_BITS_MASK     = UInt16(0x00FF)  # bits 0-7
```

These constants would replace scattered magic numbers throughout `state.jl` and `acquire.jl`.

### 4.5 The `_impl!` Convention — Keep or Replace?

The `_impl!` suffix is a **widely understood Julia convention** (e.g., `Base._similar_impl`).
Replacing it with `_direct!`, `_core!`, or `_scoped!` trades familiarity for specificity.

Arguments for keeping `_impl!`:
- Julia developers immediately understand it as "internal implementation"
- No ambiguity about the function's role as a building block
- Grep-friendly: `_*_impl!` finds all implementation functions

Arguments for replacing:
- `_impl!` doesn't explain *why* the split exists (tracking bypass)
- New developers might not realize the critical difference between `acquire!` and `_acquire_impl!`

---

## 5. Recommended Changes (for Discussion)

### Tier 1: High Impact, Low Risk (Do First)

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `_mark_untracked!` | `_record_type_touch!` | Most misleading name; "touch" captures first-touch + lazy checkpoint |
| `_untracked_fixed_masks` | `_touched_type_masks` | Field stores which types were *touched*, not which are "untracked" |
| `_untracked_has_others` | `_touched_has_others` | Consistent with above |

### Tier 2: Medium Impact, Medium Risk

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `_depth_only_checkpoint!` | `_lazy_checkpoint!` | "lazy" captures the deferred-to-first-touch semantics |
| `_dynamic_selective_rewind!` | `_lazy_rewind!` | Symmetric with `_lazy_checkpoint!` |
| `_typed_checkpoint_with_lazy!` | `_typed_lazy_checkpoint!` | Cleaner word order |
| Magic numbers `0x8000`, `0x4000`, `0xC000`, `0x00FF` | Named constants (see 4.4) | Self-documenting code |

### Tier 3: Low Impact, Higher Risk (Optional)

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `_acquire_impl!` | Keep as `_acquire_impl!` | Julia convention, well-understood |
| `_generate_dynamic_selective_*` | `_generate_lazy_*` | Follows Tier 2 rename |
| `_typed_selective_rewind!` | `_typed_lazy_rewind!` | Consistent with pair |

---

## 6. Alternative: Do Nothing

**Case for not renaming**: The current names work. They're documented. Tests pass.
"Untracked" has a clear historical meaning in the codebase, and commit history explains
the evolution. Renaming has a nonzero risk of introducing bugs (missed references,
CUDA extension drift) and makes git blame harder to follow.

**Counter-argument**: The package is pre-1.0 and has few external users. Now is the
cheapest time to fix naming before the API surface solidifies.

---

## 7. Open Questions

1. **Should `_impl!` functions be renamed?** They work fine as a convention, but
   `_acquire_direct!` or `_acquire_core!` would be more self-documenting.

2. **Is "lazy" the right word?** In Julia, `lazy` is associated with `Lazy.jl` and
   lazy evaluation. "Deferred" is more precise but longer.

3. **Should mode names be formalized?** Currently modes are described in comments
   as "typed", "dynamic-selective", "full". Should there be an enum or named constants?

4. **How deep should the rename go?** Renaming `_mark_untracked!` alone fixes 80%
   of the confusion. Is a holistic rename worth the churn?

5. **Should the bitmask `_touched_type_masks` also track mode bits?** Currently
   bits 0-7 = types, bits 14-15 = mode flags, all in the same field. Should mode
   flags be a separate field for clarity?

---

## Appendix: Complete Current → Proposed Mapping (Proposal A: "Touch/Lazy")

```
# === Type Tracking ===
_mark_untracked!(pool, T)                    → _record_type_touch!(pool, T)

# === Pool Fields ===
_untracked_fixed_masks                       → _touched_type_masks
_untracked_has_others                        → _touched_has_others

# === Acquire (keep _impl convention) ===
_acquire_impl!(pool, T, n)                   → (no change)
_unsafe_acquire_impl!(pool, T, n)            → (no change)
_zeros_impl!(pool, T, dims...)               → (no change)
_ones_impl!(pool, T, dims...)                → (no change)
_similar_impl!(pool, T, dims...)             → (no change)

# === Dynamic-Selective Mode ===
_depth_only_checkpoint!(pool)                → _lazy_checkpoint!(pool)
_dynamic_selective_rewind!(pool)             → _lazy_rewind!(pool)

# === Typed Fallback ===
_typed_checkpoint_with_lazy!(pool, T...)     → _typed_lazy_checkpoint!(pool, T...)
_typed_selective_rewind!(pool, mask)         → _typed_lazy_rewind!(pool, mask)

# === Selective Rewind Helper ===
_selective_rewind_fixed_slots!(pool, mask)   → (no change — already descriptive)

# === Guards ===
_can_use_typed_path(pool, mask)              → (no change)
_tracked_mask_for_types(T...)                → (no change)

# === Macro Generators ===
_generate_dynamic_selective_checkpoint_call  → _generate_lazy_checkpoint_call
_generate_dynamic_selective_rewind_call      → _generate_lazy_rewind_call
_generate_typed_checkpoint_call              → (no change)
_generate_typed_rewind_call                  → (no change)

# === Constants (new) ===
(raw 0x8000)                                 → _LAZY_MODE_BIT (or _DYNAMIC_MODE_BIT)
(raw 0x4000)                                 → _TYPED_LAZY_BIT
(raw 0xC000)                                 → _MODE_BITS_MASK
(raw 0x00FF)                                 → _TYPE_BITS_MASK
```

### Files Affected

| File | Changes |
|------|---------|
| `src/types.jl` | Field renames: `_untracked_*` → `_touched_*` |
| `src/acquire.jl` | `_mark_untracked!` → `_record_type_touch!` |
| `src/state.jl` | Mode functions, field references, constants |
| `src/macros.jl` | Generator function renames, field references |
| `src/convenience.jl` | `_mark_untracked!` calls |
| `src/task_local_pool.jl` | (unlikely changes) |
| `src/utils.jl` | (unlikely changes) |
| `ext/AdaptiveArrayPoolsCUDAExt/*.jl` | Mirror all renames |
| `test/test_macro_internals.jl` | Direct calls to renamed functions |
| `test/test_state.jl` | Field references |
| `test/test_macroexpand.jl` | String matching on expanded names |
| `test/test_allocation.jl` | (unlikely changes) |
