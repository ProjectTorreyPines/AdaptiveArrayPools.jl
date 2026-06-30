# Automatic Memory Management

A pool keeps buffers around so the next `acquire!` is allocation-free — that reuse is the
whole point. But over a long-running program two kinds of slack accumulate:

- **Over-sized slots** — a slot that once held a large array keeps that large backing buffer
  even after it is reused for small arrays (the buffer only ever grows).
- **Unused slots** — a type or concurrency level that was needed at a peak but isn't part of
  the recent working set still has its slots retained.

`auto_manage` reclaims both in the background, GC-style, so you don't have to call
`compact!` / `trim!` by hand. **It is on by default.**

## What it does

One background `Timer` drives two complementary actions, on independent cadences:

| Action          | Reclaims                                            | Default cadence |
|-----------------|-----------------------------------------------------|-----------------|
| **auto-compact** | shrinks an over-allocated slot's backing **in place** | every `compact_interval` (30 s) |
| **auto-trim**   | drops **whole slots** beyond the recent working-set peak | every `trim_interval` (120 s) |

They act on different axes — one shrinks the buffer *within* a slot, the other drops *whole* slots:

```text
auto-compact — shrink ONE slot's bloated backing buffer (the slot itself stays)

    before   ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░     ▓ live data   ░ wasted capacity  (bloated)
    after    ▓▓▓░                           reallocated to ~1.5× the live size;
                                            arrays you're still holding follow the move

auto-trim — drop whole SLOTS that fell out of the recent working set

    before   [1][2][3][4][5]                grew to a past peak of 5 concurrent arrays…
    after    [1][2]                         …recent peak is 2  →  slots 3–5 freed entirely
```

A slot is auto-compacted only when it is genuinely bloated — backing capacity
`≥ compact_bloat_factor ×` its last use **and** the reclaim is `≥ compact_min_bytes` (1 MiB) — so a
steady-state pool is left untouched. Auto-trim drops each type's slots down to the largest
concurrency seen in the last period (a type unused for the whole period trims to zero).

## When (and where) the work happens

The `Timer` thread only ever sets a flag; it never touches pool data. Your **owner task**
performs the actual reclamation at the next `@with_pool` **entry**, at the top-level
safepoint (`depth == 1`) where nothing is borrowed:

```julia
@with_pool p begin          # ← if a request is pending, it is serviced HERE, before your code
    x = acquire!(p, Float64, 1000)
    # ...
end
```

This means reclamation never runs mid-computation and never on the background thread, so it
**cannot change any result** — it only changes memory layout. An idle task that never enters a
`@with_pool` scope is simply never serviced (and a finished task's pool is freed by the GC).

!!! note "Targets the task-local pool"
    Auto-manage acts on the **task-local pool** (`get_task_local_pool()`) — the one `@with_pool`
    uses and the only one auto-registered. A hand-made `AdaptiveArrayPool()` is not auto-managed;
    call `compact!` / `trim!` on it directly, or register it explicitly.

## Controlling it

The usual way to configure auto-manage is **once, in `LocalPreferences.toml`** — the timer reads
these `auto_manage_*` keys at startup, so you never need to write any code:

```toml
# LocalPreferences.toml  (restart Julia to take effect)
[AdaptiveArrayPools]
auto_manage = true                    # master on/off (compile-time)
auto_manage_compact_interval = 30.0   # seconds — how often to auto-compact
auto_manage_trim_interval    = 120.0  # seconds — how often to auto-trim (Inf disables it)
# advanced compaction tuning (rarely needed):
auto_manage_compact_bloat_factor  = 10      # compact a slot at ≥ this × its live size
auto_manage_compact_target_ratio  = 1.5     # shrink it down to this × live size
auto_manage_compact_min_bytes     = 1048576 # skip if it would reclaim less
```

The same knobs are available at **runtime** (same names, minus the package-implied prefix) for
ad-hoc tuning or in startup code — these override the Preference defaults:

```julia
enable_auto_manage!(; compact_interval = 30.0, trim_interval = 120.0,
                      compact_bloat_factor = 10, compact_target_ratio = 1.5,
                      compact_min_bytes = 2^20)

enable_auto_manage!(; trim_interval = Inf)   # compact-only: disable auto-trim
disable_auto_manage!()                       # stop the background timer (this session)
auto_manage_enabled()                        # → Bool
```

`disable_auto_manage!()` only stops the timer for the current session; the feature re-arms on
the next Julia start. To turn it off permanently, set the `auto_manage` preference (below).

## Performance & turning it off

The always-on cost is a single sub-nanosecond atomic flag read per `@with_pool` entry plus one
`Int` comparison per `acquire!` — both dead-code-eliminated when the feature is compiled off.

The one caveat: a `@with_pool` entry that *services* a pending request allocates the new
right-sized buffer(s), so that particular entry is **not** zero-allocation. For a guaranteed
zero-allocation hot path (e.g. a hard-real-time loop or an allocation benchmark), compile the
feature out via the `auto_manage` preference (see [Configuration](@ref)):

```toml
# LocalPreferences.toml  (restart Julia to take effect)
[AdaptiveArrayPools]
auto_manage = false
```

With the preference off, the `@with_pool` hook is eliminated entirely — `enable_auto_manage!`
becomes a no-op-with-warning, and you reclaim memory only when you call `compact!` / `trim!`
yourself.

## Manual control

`auto_manage` is just a policy over the manual primitives, which you can always call directly:

- `compact!(pool)` — shrink over-allocated backing buffers now.
- `trim!(pool)` — drop inactive retained buffers now (see [Essential API](@ref)).

## Backends & versions

CPU, CUDA, and Metal, on Julia 1.12+. On Julia ≤ 1.11 the whole feature is a defined,
exported **no-op** (`AUTO_MANAGE == false`, `auto_manage_enabled() == false`), so code using the
API runs unchanged across the full supported Julia range.
