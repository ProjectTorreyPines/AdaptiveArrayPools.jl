using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!, trim!, checkpoint!, rewind!

# ==============================================================================
# trim! — manual inactive-slot trimming
#
# Concept: trim! drops the pool's strong references to INACTIVE retained slots
# (indices n_active+1 : end) while preserving ACTIVE slots (1 : n_active).
# ==============================================================================

@testset "trim!" begin

    @testset "keeps active slots, drops deeper inactive (CPU, S=0)" begin
        pool = AdaptiveArrayPool{0}()   # S=0: retained vectors keep their length

        checkpoint!(pool)               # depth 2
        acquire!(pool, Float64, 100)    # slot 1 (stays active)
        acquire!(pool, Float64, 50)     # slot 2 (stays active)

        checkpoint!(pool)               # depth 3
        acquire!(pool, Float64, 25)     # slot 3
        acquire!(pool, Float64, 10)     # slot 4
        rewind!(pool)                   # back to depth 2: n_active → 2, slots 3,4 retained

        @test pool.float64.n_active == 2
        @test length(pool.float64.vectors) == 4   # inactive slots still retained

        summary = trim!(pool)

        # Active slots preserved; inactive backing vectors dropped
        @test pool.float64.n_active == 2
        @test length(pool.float64.vectors) == 2
        @test summary.slots_released == 2
        @test summary.gc_triggered == false
    end

    @testset "active slots are byte-for-byte untouched (data + identity)" begin
        pool = AdaptiveArrayPool{0}()

        checkpoint!(pool)               # depth 2 — these stay ACTIVE
        a = acquire!(pool, Float64, 8)
        a .= 1.0:8.0                    # known data in active slot 1
        b = acquire!(pool, Float64, 4)
        b .= 100.0                      # known data in active slot 2

        checkpoint!(pool)               # depth 3 — become INACTIVE on rewind
        acquire!(pool, Float64, 1000)
        acquire!(pool, Float64, 2000)
        rewind!(pool)                   # n_active → 2; slots 3,4 retained (inactive)

        # Identity of the ACTIVE backing vectors before trimming.
        v1 = pool.float64.vectors[1]
        v2 = pool.float64.vectors[2]

        summary = trim!(pool)

        # Inactive dropped; active count intact.
        @test summary.slots_released == 2
        @test pool.float64.n_active == 2
        @test length(pool.float64.vectors) == 2

        # ACTIVE backing vectors are the SAME objects — not replaced or resized.
        @test pool.float64.vectors[1] === v1
        @test pool.float64.vectors[2] === v2

        # ACTIVE arrays' data is byte-for-byte intact and they stay usable.
        @test a == collect(1.0:8.0)
        @test all(==(100.0), b)
        a[1] = 42.0                     # still a live, writable view into the pool
        @test pool.float64.vectors[1][1] == 42.0
    end

    @testset "drops inactive wrapper-cache entries (memory-pin fix)" begin
        pool = AdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Float64, 4, 5)   # 2-D → arr_wrappers[2][1]
        acquire!(pool, Float64, 3, 3)   # 2-D → arr_wrappers[2][2]
        rewind!(pool)                   # n_active → 0; slots + wrappers retained

        tp = pool.float64
        @test tp.n_active == 0
        @test length(tp.arr_wrappers) >= 2
        @test tp.arr_wrappers[2] !== nothing
        @test length(tp.arr_wrappers[2]) == 2   # two cached wrappers retained

        summary = trim!(pool)

        # Every wrapper-cache vector truncated to the active count (0 here):
        # a cached Array holds a :ref into the old buffer, pinning it in memory.
        for w in tp.arr_wrappers
            w === nothing && continue
            @test length(w) <= tp.n_active
        end
        @test summary.wrappers_released >= 2
    end

    @testset "per-type trim!(pool, T) affects only that type" begin
        pool = AdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Int64, 200)
        rewind!(pool)                   # both retained, n_active == 0 each

        @test length(pool.float64.vectors) == 1
        @test length(pool.int64.vectors) == 1

        summary = trim!(pool, Float64)

        @test length(pool.float64.vectors) == 0   # trimmed
        @test length(pool.int64.vectors) == 1     # untouched
        @test summary.slots_released == 1
    end

    @testset "DisabledPool returns a zero summary" begin
        s = trim!(DISABLED_CPU)
        @test s.slots_released == 0
        @test s.wrappers_released == 0
        @test s.estimated_bytes_released == 0
        @test s.gc_triggered == false

        s2 = trim!(DISABLED_CPU, Float64)
        @test s2.slots_released == 0
    end

    @testset "zero-arg trim!() operates on the task-local pool" begin
        pool = get_task_local_pool()
        empty!(pool)                    # clean slate (task-local is shared)

        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        rewind!(pool)
        @test length(pool.float64.vectors) == 1

        summary = trim!()               # no-arg → task-local pool
        @test length(pool.float64.vectors) == 0
        @test summary.slots_released >= 1

        empty!(pool)                    # cleanup to avoid cross-test contamination
    end

    @testset "force_gc=true sets gc_triggered (regression)" begin
        pool = AdaptiveArrayPool{0}()
        @test trim!(pool; force_gc = true).gc_triggered == true
        @test trim!(pool; force_gc = false).gc_triggered == false

        # per-type form: force_gc through the real-trim path
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        rewind!(pool)
        s = trim!(pool, Float64; force_gc = true)
        @test s.slots_released == 1
        @test s.gc_triggered == true
    end

    @testset "reset!(pool); trim!(pool) releases all retained buffers (regression)" begin
        pool = AdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Float64, 50)
        # (no rewind: simulate top-level n_active growth)
        @test pool.float64.n_active == 2

        reset!(pool)                    # mark all slots inactive (n_active → 0)
        @test pool.float64.n_active == 0
        @test length(pool.float64.vectors) == 2   # reset! preserves buffers

        trim!(pool)                     # now release them
        @test length(pool.float64.vectors) == 0
    end

    @testset "byte accounting" begin
        @testset "S=0: estimate reflects retained buffer sizes" begin
            pool = AdaptiveArrayPool{0}()
            checkpoint!(pool)
            acquire!(pool, Float64, 10_000)   # 80_000 bytes of backing storage
            rewind!(pool)
            s = trim!(pool)
            @test s.slots_released == 1
            @test s.estimated_bytes_released >= 80_000
        end

        @testset "S=1: capacity-aware estimate counts buffers shrunk to length 0" begin
            pool = AdaptiveArrayPool{1}()
            checkpoint!(pool)
            acquire!(pool, Float64, 10_000)
            rewind!(pool)                     # S=1: inactive slot resize!'d to length 0
            @test length(pool.float64.vectors[1]) == 0   # logical length 0...
            s = trim!(pool)
            @test s.slots_released == 1
            @test s.estimated_bytes_released >= 80_000    # ...but retained capacity counted
        end
    end

    @testset "self-heal: acquire! after trim! allocates fresh and works" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool)
        acquire!(pool, Float64, 1000)
        rewind!(pool)
        trim!(pool)
        @test length(pool.float64.vectors) == 0

        checkpoint!(pool)
        v = acquire!(pool, Float64, 7)
        v .= 3.0
        @test length(v) == 7
        @test all(v .== 3.0)
        @test length(pool.float64.vectors) == 1
        rewind!(pool)
    end

    @testset "Bit (BitVector) and fallback (others) types trim too" begin
        pool = AdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Bit, 10_000)       # BitTypedPool path
        acquire!(pool, UInt8, 5_000)      # fallback (others) path
        rewind!(pool)

        @test length(pool.bits.vectors) == 1
        @test pool.others[UInt8].n_active == 0
        @test length(pool.others[UInt8].vectors) == 1

        s = trim!(pool)

        @test length(pool.bits.vectors) == 0
        @test length(pool.others[UInt8].vectors) == 0
        @test s.slots_released == 2
        @test s.estimated_bytes_released > 0
    end

    @testset "per-type trim! never creates a pool for an unused type" begin
        pool = AdaptiveArrayPool{0}()

        # Unused fallback type (UInt8 is not a fixed slot, never acquired):
        # trim! must NOT register a pool in `others` while trying to reclaim.
        @test !haskey(pool.others, UInt8)
        n_before = length(pool.others)
        s = trim!(pool, UInt8)
        @test s == (;
            slots_released = 0, wrappers_released = 0,
            estimated_bytes_released = 0, gc_triggered = false,
        )
        @test length(pool.others) == n_before
        @test !haskey(pool.others, UInt8)          # still not created

        # force_gc is still honored on the no-op path.
        @test trim!(pool, UInt8; force_gc = true).gc_triggered == true
        @test !haskey(pool.others, UInt8)

        # Fixed-slot type (always present) is a safe no-op too — no `others` growth.
        @test trim!(pool, Float64).slots_released == 0
        @test length(pool.others) == n_before

        # An existing fallback type is still trimmed normally.
        checkpoint!(pool)
        acquire!(pool, UInt8, 100)
        rewind!(pool)
        @test haskey(pool.others, UInt8)
        @test trim!(pool, UInt8).slots_released == 1
    end

end
