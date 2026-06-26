# CUDA rand! / randn! tests
# =========================
# The live CUDA pool inherits the core _rand_impl!/_randn_impl! (which call
# Random.rand!/randn! on the acquired CuArray — CUDA.jl runs these on-device).
# Only the typed/default forms are supported on GPU; the collection/range form is
# CPU-only (no GPU arbitrary-collection sampler) and must raise a clear error.
#
# Mirrors test/metal/test_random.jl. Run on a CUDA-capable machine to verify parity.

using Random

@testset "CUDA rand!/randn!" begin
    pool = get_task_local_cuda_pool()
    checkpoint!(pool)

    @testset "rand! default type is Float32" begin
        v = rand!(pool, 100)
        @test v isa CuArray{Float32}
        @test length(v) == 100
        va = Array(v)
        @test all(x -> 0.0f0 <= x < 1.0f0, va)

        m = rand!(pool, 8, 8)
        @test m isa CuArray{Float32, 2}
        @test size(m) == (8, 8)
    end

    @testset "rand! explicit type" begin
        v32 = rand!(pool, Float32, 50)
        @test v32 isa CuArray{Float32}
        @test all(x -> 0.0f0 <= x < 1.0f0, Array(v32))

        v16 = rand!(pool, Float16, 50)
        @test v16 isa CuArray{Float16}

        vi = rand!(pool, Int32, 50)
        @test vi isa CuArray{Int32}
    end

    @testset "rand! tuple form" begin
        m = rand!(pool, (4, 5))
        @test m isa CuArray{Float32, 2}
        @test size(m) == (4, 5)

        m32 = rand!(pool, Float32, (3, 4))
        @test m32 isa CuArray{Float32, 2}
        @test size(m32) == (3, 4)
    end

    @testset "randn! default + explicit" begin
        g = randn!(pool, 100)
        @test g isa CuArray{Float32}
        @test all(isfinite, Array(g))

        g32 = randn!(pool, Float32, 4, 5)
        @test g32 isa CuArray{Float32, 2}
        @test size(g32) == (4, 5)
        @test all(isfinite, Array(g32))

        gt = randn!(pool, (6, 7))
        @test size(gt) == (6, 7)
    end

    @testset "returns CuArray (not a view)" begin
        v = rand!(pool, Float32, 10)
        @test v isa CuArray
        @test !(v isa SubArray)
    end

    @testset "collection/range form is rejected (CPU-only)" begin
        @test_throws ArgumentError rand!(pool, 1:6, 10)
        @test_throws ArgumentError rand!(pool, 1:6, (5, 5))
        @test_throws ArgumentError rand!(pool, Int32(1):Int32(6), 8)
    end

    rewind!(pool)

    @testset "DisabledPool{:cuda} fallbacks (GPU-native)" begin
        DC = ext.DISABLED_CUDA

        v = rand!(DC, Float32, 16)
        @test v isa CuArray{Float32}
        @test all(x -> 0.0f0 <= x < 1.0f0, Array(v))

        v2 = rand!(DC, 5, 5)                 # default Float32
        @test v2 isa CuArray{Float32, 2}
        @test size(v2) == (5, 5)

        vt = rand!(DC, Float32, (3, 4))      # NTuple form
        @test vt isa CuArray{Float32, 2}
        @test size(vt) == (3, 4)

        g = randn!(DC, Float32, 16)
        @test g isa CuArray{Float32}
        @test all(isfinite, Array(g))

        gt = randn!(DC, (2, 3))              # default Float32 NTuple
        @test size(gt) == (2, 3)

        # collection rejected on the disabled pool too
        @test_throws ArgumentError rand!(DC, 1:6, 10)
    end
end
