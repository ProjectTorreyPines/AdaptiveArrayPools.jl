using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!

# ==============================================================================
# compact! — capacity compaction (shrink over-allocated backing buffers in place)
# Design: docs/design/capacity_compaction.md  |  Plan: docs/plans/PLAN_compact.md
#
# Model: a slot's backing Vector length is the HIGH-WATER mark (largest ever
# acquired; _claim_slot! only grows). The CURRENT use is the wrapper's size. So a
# slot is bloated when backing capacity ≫ current wrapper size. compact! shrinks the
# backing to ~shrink_to×(wrapper size) in place and re-syncs wrappers.
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

end
