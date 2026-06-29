using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!

# ==============================================================================
# compact! — capacity compaction (shrink over-allocated backing buffers in place)
# Design: docs/design/capacity_compaction.md  |  Plan: docs/plans/PLAN_compact.md
#
# Model: a slot's backing Vector length is the HIGH-WATER mark (largest ever
# acquired; _claim_slot! only grows). The CURRENT use is recorded per-slot in
# `tp.slot_extents` by _claim_slot! — covering BOTH acquire! (Array) and acquire_view!
# (uncached SubArray/ReshapedArray). So a slot is bloated when backing capacity ≫
# slot_extents[slot]; compact! shrinks the backing to ~shrink_to×(that extent) in
# place and re-syncs wrappers (held views follow via preserved Vector identity).
# ==============================================================================

# Allocated backing capacity (Memory length).
_cap(v) = length(getfield(v, :ref).mem)

# Build slot 1 (Float64) bloated: acquire `big` (grows backing to high-water), rewind,
# re-acquire `small` on the same slot (backing stays high-water; wrapper size = small).
# Returns (pool, wrapper-of-size-small).
function _bloated_pool(big::Int, small::Int)
    pool = AdaptiveArrayPool{0}()
    checkpoint!(pool)
    a = acquire!(pool, Float64, big)
    a .= 0.0
    rewind!(pool)
    checkpoint!(pool)
    b = acquire!(pool, Float64, small)
    return pool, b
end

# Build slot 1 (Float64) bloated AND INACTIVE: grow backing to `big` (high-water),
# rewind, re-acquire `small` (this sets the cached wrapper's size to `small`), then
# rewind again so the slot is inactive (n_active == 0) while its backing capacity
# stays at the `big` high-water mark. An inactive slot is only "bloated" when its
# LAST logical size (the cached wrapper size) is far below its retained capacity —
# `acquire(big) → rewind` alone leaves wrapper size == capacity (not bloated).
function _inactive_bloated_pool(big::Int, small::Int)
    pool = AdaptiveArrayPool{0}()
    checkpoint!(pool)
    a = acquire!(pool, Float64, big)
    a .= 0.0
    rewind!(pool)
    checkpoint!(pool)
    b = acquire!(pool, Float64, small)
    b .= 0.0
    rewind!(pool)
    return pool
end

# Concrete public summary shape (mirror of the `trim!` summary test).
const _COMPACT_SUMMARY = @NamedTuple{
    slots_compacted::Int, bytes_reclaimed::Int, gc_triggered::Bool,
}
const _ZERO_COMPACT = (; slots_compacted = 0, bytes_reclaimed = 0, gc_triggered = false)

@testset "compact!" begin

    @testset "primitive: in-place shrink + wrapper re-sync + view follow + GC" begin
        pool, b = _bloated_pool(1_000_000, 100)
        b .= 1.0:100.0
        tp = pool.float64
        @test length(b) == 100                       # wrapper = current use
        @test length(tp.vectors[1]) == 1_000_000     # backing = high-water (the bloat)
        @test _cap(tp.vectors[1]) >= 1_000_000

        vw = view(tp.vectors[1], 1:100)              # view into the slot's backing object

        reclaimed = AdaptiveArrayPools._maybe_compact_slot!(tp, 1, 10, 1.5, 2^20)

        @test reclaimed > 0
        @test length(tp.vectors[1]) == 150           # backing shrunk to ceil(1.5 * 100)
        @test _cap(tp.vectors[1]) == 150
        @test b == collect(1.0:100.0)                # wrapper re-synced: data intact
        b[1] = 99.0
        @test tp.vectors[1][1] == 99.0               # wrapper writes into the new buffer
        @test collect(vw)[2] == 2.0                  # view follows parent to the new buffer
        rewind!(pool)
    end

    # GC reclaim in its own function scope so intermediate strong refs (the wrapper,
    # the captured Memory, the view) leave scope before collection — otherwise
    # conservative stack roots keep the old buffer alive and the test is flaky.
    function _old_mem_weakref()
        pool, b = _bloated_pool(1_000_000, 100)
        b .= 1.0:100.0
        tp = pool.float64
        wr = WeakRef(getfield(tp.vectors[1], :ref).mem)
        AdaptiveArrayPools._maybe_compact_slot!(tp, 1, 10, 1.5, 2^20)
        return wr, pool                              # keep `pool` alive; only the OLD buffer should die
    end

    @testset "primitive: old backing buffer is GC-reclaimed after compaction" begin
        wr, pool = _old_mem_weakref()
        GC.gc(true); GC.gc(true)
        @test wr.value === nothing                   # old 1M Memory reclaimed
        @test pool.float64.n_active == 1             # pool still alive (not a trivial pass)
    end

    @testset "primitive gates: ratio and absolute-bytes" begin
        # ratio gate: capacity only 5x the wrapper size (< factor 10) -> skip
        # (min_bytes = 0 isolates the ratio gate).
        pool, _ = _bloated_pool(500, 100)
        tp = pool.float64
        cap0 = _cap(tp.vectors[1])
        @test AdaptiveArrayPools._maybe_compact_slot!(tp, 1, 10, 1.5, 0) == 0
        @test _cap(tp.vectors[1]) == cap0            # untouched
        rewind!(pool)

        # bytes gate: 100x bloated but reclaim ~8 KB < min_bytes (1 MiB) -> skip
        pool2, _ = _bloated_pool(1000, 10)
        tp2 = pool2.float64
        cap2 = _cap(tp2.vectors[1])
        @test AdaptiveArrayPools._maybe_compact_slot!(tp2, 1, 10, 1.5, 2^20) == 0
        @test _cap(tp2.vectors[1]) == cap2           # untouched
        rewind!(pool2)
    end

    # ── Phase 2: public compact! — Tier 1 (inactive slots only) ────────────────

    @testset "Tier 1: compact!(pool) reclaims inactive bloat" begin
        pool = _inactive_bloated_pool(1_000_000, 100)
        tp = pool.float64
        @test tp.n_active == 0                       # slot 1 is inactive…
        @test _cap(tp.vectors[1]) >= 1_000_000       # …but retains high-water capacity

        s = compact!(pool)

        @test s.slots_compacted == 1
        @test s.bytes_reclaimed >= (1_000_000 - 150) * sizeof(Float64)
        @test s.gc_triggered == false
        @test _cap(tp.vectors[1]) == 150             # shrunk to ceil(1.5 * 100)
    end

    @testset "default (active=true) compacts active bloat; active=false opts out" begin
        # compact!'s purpose is to reclaim from arrays still in use, so the default
        # reaches active slots too. The factor gate (10×) keeps it from touching
        # normally-sized arrays (covered by the next testset).
        pool, b = _bloated_pool(1_000_000, 100)      # slot 1 ACTIVE (b held), cap 1M, used 100
        b .= 1.0:100.0
        tp = pool.float64
        @test tp.n_active == 1
        cap0 = _cap(tp.vectors[1])

        # active=false explicitly opts out → active slot untouched
        s_off = compact!(pool; active = false)
        @test s_off.slots_compacted == 0
        @test _cap(tp.vectors[1]) == cap0

        # default compacts the held active slot in place; held wrapper follows
        s = compact!(pool)
        @test s.slots_compacted == 1
        @test _cap(tp.vectors[1]) == 150
        @test b == collect(1.0:100.0)                # data intact + re-synced
        b[1] = 99.0
        @test tp.vectors[1][1] == 99.0               # writes into the new buffer
        rewind!(pool)
    end

    @testset "default leaves a normally-sized active array alone (factor gate)" begin
        # Safety net for the active=true default: a slot whose capacity ≈ its use
        # (not 10× bloated) is never compacted, so live arrays are not churned.
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool)
        a = acquire!(pool, Float64, 100)             # capacity ≈ use, NOT bloated
        a .= 5.0
        cap0 = _cap(pool.float64.vectors[1])

        s = compact!(pool)                           # default active=true, but factor gate skips

        @test s.slots_compacted == 0
        @test _cap(pool.float64.vectors[1]) == cap0
        @test a == fill(5.0, 100)                    # untouched, still valid
        rewind!(pool)
    end

    @testset "per-type compact!(pool, T) targets only T; guard never creates a pool" begin
        pool = _inactive_bloated_pool(1_000_000, 100)
        # also build an inactive-bloated Int64 slot (Int64 is a fixed slot)
        checkpoint!(pool); ai = acquire!(pool, Int64, 1_000_000); ai .= 0; rewind!(pool)
        checkpoint!(pool); bi = acquire!(pool, Int64, 100); bi .= 0; rewind!(pool)

        s = compact!(pool, Float64)                  # compact only Float64
        @test s.slots_compacted == 1
        @test _cap(pool.float64.vectors[1]) == 150
        @test _cap(pool.int64.vectors[1]) >= 1_000_000   # Int64 untouched

        # Guard: a never-used fallback type must NOT be registered in `others`.
        @test !haskey(pool.others, UInt8)
        n_before = length(pool.others)
        s2 = compact!(pool, UInt8)
        @test s2 == _ZERO_COMPACT
        @test length(pool.others) == n_before
        @test !haskey(pool.others, UInt8)            # still not created
    end

    @testset "compact! return value is type-stable (no Any-typed fields)" begin
        # Mirrors the trim! stability test: the summary is a public return value, so
        # its fields must infer concrete even though the full-pool form folds over
        # fixed slots (@generated) AND the `others` Vector{Any} (dynamic dispatch).
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Int64, 50)
        acquire!(pool, UInt8, 10)                    # exercises the `others` path
        rewind!(pool)

        rt = only(Base.return_types(compact!, (typeof(pool),)))
        @test isconcretetype(rt)
        @test rt == _COMPACT_SUMMARY
        @test only(Base.return_types(compact!, (typeof(pool), Type{Float64}))) == _COMPACT_SUMMARY

        @test (@inferred compact!(pool)) isa _COMPACT_SUMMARY
        @test (@inferred compact!(pool, Float64)) isa _COMPACT_SUMMARY
    end

    @testset "no-op when nothing is bloated; force_gc sets the flag" begin
        pool = AdaptiveArrayPool{0}()
        s = compact!(pool)
        @test s == _ZERO_COMPACT
        @test compact!(pool; force_gc = true).gc_triggered == true
        @test compact!(pool; force_gc = false).gc_triggered == false
    end

    @testset "DisabledPool zero summary; compact!() uses the task-local pool" begin
        s = compact!(DISABLED_CPU)
        @test s == _ZERO_COMPACT
        @test compact!(DISABLED_CPU, Float64) == _ZERO_COMPACT

        # No-arg form routes to the task-local pool: on an empty one it returns the
        # zero summary, and force_gc threads through the no-arg overload.
        pool = get_task_local_pool()
        empty!(pool)
        @test compact!() == _ZERO_COMPACT
        @test compact!(; force_gc = true).gc_triggered == true
        empty!(pool)                                 # cleanup (task-local is shared)
    end

    @testset "self-heal: re-acquire after compacting an inactive slot works" begin
        pool = _inactive_bloated_pool(1_000_000, 100)
        s = compact!(pool)
        @test s.slots_compacted == 1
        @test _cap(pool.float64.vectors[1]) == 150

        # Re-acquire the compacted slot within the shrunk capacity (reuse), then at a
        # size that forces a regrow — both must yield correct, writable arrays whose
        # writes land in the backing (wrapper :ref re-synced through the grow).
        checkpoint!(pool)
        a = acquire!(pool, Float64, 120)             # 120 <= 150: reuse shrunk buffer
        a .= 3.0
        @test length(a) == 120
        @test all(a .== 3.0)
        rewind!(pool)

        checkpoint!(pool)
        b = acquire!(pool, Float64, 5_000)           # 5000 > 150: backing must regrow
        b .= 7.0
        @test length(b) == 5_000
        @test all(b .== 7.0)
        b[1] = 9.0
        @test pool.float64.vectors[1][1] == 9.0      # write lands in the regrown backing
        rewind!(pool)
    end

    # ── Phase 3: Tier 2 (active=true opt-in) + Type... varargs ─────────────────

    @testset "Tier 2: active=true compacts a slot the user is still holding" begin
        pool, b = _bloated_pool(1_000_000, 100)      # slot 1 is ACTIVE (b held), cap 1M
        b .= 1.0:100.0
        tp = pool.float64
        @test tp.n_active == 1
        @test _cap(tp.vectors[1]) >= 1_000_000

        vw = view(b, 1:50)                           # view INTO the held active wrapper

        # active=false (default) must leave the active slot alone.
        s0 = compact!(pool; active = false)
        @test s0.slots_compacted == 0
        @test _cap(tp.vectors[1]) >= 1_000_000

        # active=true shrinks the held slot in place.
        s = compact!(pool; active = true)
        @test s.slots_compacted == 1
        @test s.bytes_reclaimed >= (1_000_000 - 150) * sizeof(Float64)
        @test _cap(tp.vectors[1]) == 150             # ceil(1.5 * 100)

        # The user's held wrapper survives: data intact, re-synced, still writable.
        @test b == collect(1.0:100.0)
        b[1] = 99.0
        @test tp.vectors[1][1] == 99.0               # write lands in the new buffer
        @test collect(vw)[2] == 2.0                  # view of the held wrapper follows
        vw[1] = -1.0
        @test b[1] == -1.0                           # view write reaches the wrapper
        rewind!(pool)
    end

    @testset "active=true also still compacts inactive slots (scan = 1:end)" begin
        # active=true is "compact everything"; inactive bloat is included too.
        pool = _inactive_bloated_pool(1_000_000, 100)
        s = compact!(pool; active = true)
        @test s.slots_compacted == 1
        @test _cap(pool.float64.vectors[1]) == 150
    end

    @testset "varargs compact!(pool, T...) compacts each listed type" begin
        pool = _inactive_bloated_pool(1_000_000, 100)            # Float64 inactive bloat
        checkpoint!(pool); ai = acquire!(pool, Int64, 1_000_000); ai .= 0; rewind!(pool)
        checkpoint!(pool); bi = acquire!(pool, Int64, 100); bi .= 0; rewind!(pool)

        s = compact!(pool, Float64, Int64)
        @test s.slots_compacted == 2
        @test _cap(pool.float64.vectors[1]) == 150
        @test _cap(pool.int64.vectors[1]) == 150

        # Type-stable: the varargs form returns the same concrete summary.
        @test only(Base.return_types(compact!, (typeof(pool), Type{Float64}, Type{Int64}))) ==
            _COMPACT_SUMMARY
        @test (@inferred compact!(pool, Float64, Int64)) isa _COMPACT_SUMMARY  # 2nd call: no-op

        # A never-used fallback type in the list is skipped, never created.
        pool2 = AdaptiveArrayPool{0}()
        @test !haskey(pool2.others, UInt16)
        s2 = compact!(pool2, Float64, UInt16)
        @test s2.slots_compacted == 0
        @test !haskey(pool2.others, UInt16)
    end

    @testset "varargs honors active=true and force_gc" begin
        pool, b = _bloated_pool(1_000_000, 100)                 # active Float64 bloat
        b .= 0.0
        s = compact!(pool, Float64; active = true, force_gc = true)
        @test s.slots_compacted == 1
        @test s.gc_triggered == true
        @test _cap(pool.float64.vectors[1]) == 150
        rewind!(pool)
    end

    @testset "DisabledPool varargs is a zero-summary no-op" begin
        @test compact!(DISABLED_CPU, Float64, Int64) == _ZERO_COMPACT
        @test compact!(DISABLED_CPU, Float64, Int64; active = true) == _ZERO_COMPACT
    end

    # ── Regression: active=true must not shrink below a live acquire_view! extent ──
    # A slot can carry a SMALL cached Array wrapper (from a prior acquire!) AND a
    # LARGER live view (from acquire_view!, which returns an uncached SubArray /
    # ReshapedArray). `_slot_used` only scans the cached Array wrappers, so it
    # under-reports the live extent and `compact!(active=true)` could shrink the
    # backing below the view → out-of-bounds. Assertions are length-only (never index
    # the view) so a buggy run reports a failure instead of segfaulting the runner.
    @testset "active=true respects a live 1-D acquire_view! extent" begin
        pool = AdaptiveArrayPool{0}()
        # slot 1: small cached Array wrapper (100) over a large high-water backing (1M)
        checkpoint!(pool); a = acquire!(pool, Float64, 1_000_000); a .= 0.0; rewind!(pool)
        checkpoint!(pool); a2 = acquire!(pool, Float64, 100); a2 .= 0.0; rewind!(pool)
        tp = pool.float64
        @test length(tp.arr_wrappers[1][1]) == 100       # small cached wrapper
        @test _cap(tp.vectors[1]) >= 1_000_000

        checkpoint!(pool)
        v = acquire_view!(pool, Float64, 1000)           # large uncached view on the slot
        @test length(v) == 1000
        @test tp.n_active == 1                           # slot 1 is ACTIVE

        compact!(pool; active = true)

        # SAFE INVARIANT: the backing must still cover the live view (no OOB).
        @test _cap(tp.vectors[1]) >= length(v)
        @test length(tp.vectors[1]) >= length(v)
        rewind!(pool)
    end

    @testset "active=true respects a live N-D acquire_view! (ReshapedArray) extent" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool); a = acquire!(pool, Float64, 1_000_000); a .= 0.0; rewind!(pool)
        checkpoint!(pool); a2 = acquire!(pool, Float64, 100); a2 .= 0.0; rewind!(pool)
        tp = pool.float64

        checkpoint!(pool)
        vw = acquire_view!(pool, Float64, 40, 25)        # ReshapedArray, extent 1000
        @test length(vw) == 1000
        @test tp.n_active == 1

        compact!(pool; active = true)

        @test _cap(tp.vectors[1]) >= length(vw)
        @test length(tp.vectors[1]) >= length(vw)
        rewind!(pool)
    end

    @testset "extreme kwargs never throw (shrink target clamped to capacity)" begin
        pool = _inactive_bloated_pool(1_000_000, 100)
        # shrink_to so large that ceil(Int, shrink_to*used) would overflow Int —
        # must be a safe no-op (target clamped to capacity → reclaim 0 → skipped),
        # not an InexactError.
        s = compact!(pool; shrink_to = 1.0e300)
        @test s.slots_compacted == 0
        @test _cap(pool.float64.vectors[1]) >= 1_000_000   # untouched
        # shrink_to larger than capacity/used ratio (but not overflowing) also no-ops.
        @test compact!(pool; shrink_to = 1.0e6).slots_compacted == 0
    end

end
