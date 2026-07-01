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

The reclaiming is done by two primitives — `compact!` and `trim!` — and `auto_manage` simply
runs them for you on a timer. First, what each one does to a pool:

```text
compact — shrink ONE slot's over-grown backing buffer down to what it actually holds

    before   ███░░░░░░░░░░░░░░░░░░░░░░░░   █ live data   ░ over-allocated slack
    after    ███░                          buffer reallocated to ~1.5× the live data
                                           (the slot stays; arrays you hold follow the move)

trim — free the retained slots that sit BEYOND the ones currently in use

    before   [█][█][░][░][░]               █ in use (2)   ░ idle, still retained (3)
    after    [█][█]                        the idle tail is freed; the in-use slots stay
```

So `compact` works *within* a slot (a smaller buffer); `trim` drops *whole* slots. `auto_manage`
runs both on a background `Timer` — a little more smartly than calling them by hand:

| Runs | Action | …automatically, but only where it pays off |
|------|--------|--------------------------------------------|
| every `compact_interval` (**30 s**) | **auto-compact** | `compact!` restricted to genuinely bloated slots: capacity `≥ compact_bloat_factor ×` the live size **and** `≥ compact_min_bytes` (1 MiB) to reclaim — a steady-state pool is left untouched |
| every `trim_interval` (**120 s**) | **auto-trim** | `trim!` down to the **recent working-set peak** (the widest concurrency seen since the last trim, not just the instantaneous in-use count) — so a pool that briefly widens isn't thrashed; a type unused all period trims to zero |

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
