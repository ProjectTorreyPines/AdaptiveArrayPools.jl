# CUDA trim! Tests
# Parity with CPU/Metal test_trim.jl: drop inactive retained slots, keep active.
# Uses Float32 (universal GPU type).

@testset "CUDA trim!" begin

    @testset "keeps active slots, drops deeper inactive (CUDA, S=0)" begin
        pool = CuAdaptiveArrayPool{0}()

        checkpoint!(pool)               # depth 2
        acquire!(pool, Float32, 100)    # slot 1 (stays active)
        acquire!(pool, Float32, 50)     # slot 2 (stays active)

        checkpoint!(pool)               # depth 3
        acquire!(pool, Float32, 25)     # slot 3
        acquire!(pool, Float32, 10)     # slot 4
        rewind!(pool)                   # back to depth 2: n_active → 2

        @test pool.float32.n_active == 2
        @test length(pool.float32.vectors) == 4

        s = trim!(pool)

        @test pool.float32.n_active == 2
        @test length(pool.float32.vectors) == 2
        @test s.slots_released == 2
        @test s.gc_triggered == false
    end

    @testset "active slots untouched (CUDA, identity + data)" begin
        pool = CuAdaptiveArrayPool{0}()

        checkpoint!(pool)               # depth 2 — active
        a = acquire!(pool, Float32, 8); fill!(a, 7.0f0)
        b = acquire!(pool, Float32, 4); fill!(b, 100.0f0)

        checkpoint!(pool)               # depth 3 — become inactive on rewind
        acquire!(pool, Float32, 1000)
        acquire!(pool, Float32, 2000)
        rewind!(pool)

        v1 = pool.float32.vectors[1]
        v2 = pool.float32.vectors[2]

        s = trim!(pool)

        @test s.slots_released == 2
        @test pool.float32.n_active == 2
        @test length(pool.float32.vectors) == 2
        @test pool.float32.vectors[1] === v1   # same GPU buffers, untouched
        @test pool.float32.vectors[2] === v2
        @test all(==(7.0f0), Array(a))         # data intact (device → host round-trip)
        @test all(==(100.0f0), Array(b))
    end

    @testset "byte estimate uses GPU size (sizeof), not summarysize" begin
        pool = CuAdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Float32, 10_000)   # 40_000 GPU bytes
        rewind!(pool)

        s = trim!(pool)
        @test s.slots_released == 1
        # summarysize(CuArray) reports only the CPU-side handle; maxsize gives the
        # real GPU bytes. This asserts the CUDA byte override is used.
        @test s.estimated_bytes_released >= 40_000
    end

    @testset "byte estimate is capacity-aware in safety mode (CUDA, S=1)" begin
        pool = CuAdaptiveArrayPool{1}()   # S=1: safety invalidation on

        checkpoint!(pool)
        acquire!(pool, Float32, 10_000)   # 40_000 device bytes
        rewind!(pool)                     # S=1: released slot dims → 0, maxsize preserved

        @test length(pool.float32.vectors[1]) == 0   # logical length 0 (sizeof would be 0)...
        s = trim!(pool)
        @test s.slots_released == 1
        @test s.estimated_bytes_released >= 40_000    # ...but maxsize (device capacity) is counted
    end

    @testset "wrapper truncation + per-type + force_gc (CUDA)" begin
        pool = CuAdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Float32, 4, 5)   # 2-D → arr_wrappers[2][1]
        acquire!(pool, Int32, 100)
        rewind!(pool)

        @test pool.float32.arr_wrappers[2] !== nothing
        @test length(pool.float32.arr_wrappers[2]) == 1

        s = trim!(pool, Float32)        # per-type: only Float32
        @test length(pool.float32.vectors) == 0
        @test length(pool.int32.vectors) == 1     # untouched
        @test s.slots_released == 1
        for w in pool.float32.arr_wrappers
            w === nothing && continue
            @test length(w) <= pool.float32.n_active
        end

        @test trim!(pool; force_gc = true).gc_triggered == true
    end

    @testset "self-heal: acquire! after trim! (CUDA)" begin
        pool = CuAdaptiveArrayPool{0}()

        checkpoint!(pool)
        acquire!(pool, Float32, 1000)
        rewind!(pool)
        trim!(pool)
        @test length(pool.float32.vectors) == 0

        checkpoint!(pool)
        v = acquire!(pool, Float32, 7)
        @test length(v) == 7
        @test length(pool.float32.vectors) == 1
        rewind!(pool)
    end

end
