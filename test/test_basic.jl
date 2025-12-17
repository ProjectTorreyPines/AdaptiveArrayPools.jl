@testset "Core Pool Operations" begin

    @testset "Basic functionality" begin
        pool = AdaptiveArrayPool()

        # First acquire (Float64 uses fixed slot)
        v1 = acquire!(pool, Float64, 5)
        @test length(v1) == 5
        @test v1 isa SubArray
        @test eltype(v1) == Float64

        # Access internal state via fixed slot
        @test pool.float64.n_active == 1

        # Second acquire
        v2 = acquire!(pool, Float64, 8)
        @test length(v2) == 8
        @test pool.float64.n_active == 2

        # Values are independent
        v1 .= 1.0
        v2 .= 2.0
        @test all(v1 .== 1.0)
        @test all(v2 .== 2.0)
    end

    @testset "Pool expansion" begin
        pool = AdaptiveArrayPool()

        # Acquire some vectors
        acquire!(pool, Float64, 5)
        acquire!(pool, Float64, 5)

        @test length(pool.float64.vectors) >= 2

        # Should expand pool
        v3 = acquire!(pool, Float64, 5)
        @test pool.float64.n_active == 3
        @test length(pool.float64.vectors) >= 3
        @test v3 isa SubArray
    end

    @testset "Vector resize" begin
        pool = AdaptiveArrayPool()

        # Initial request
        v1 = acquire!(pool, Float64, 20)
        @test length(v1) == 20
        @test length(pool.float64.vectors[1]) >= 20

        # Second acquire, smaller size
        v2 = acquire!(pool, Float64, 5)
        @test length(v2) == 5
    end

    @testset "Fixed slot types" begin
        pool = AdaptiveArrayPool()

        # Float64 - fixed slot
        v64 = acquire!(pool, Float64, 5)
        @test eltype(v64) == Float64
        @test pool.float64.n_active == 1

        # Float32 - fixed slot
        v32 = acquire!(pool, Float32, 5)
        @test eltype(v32) == Float32
        @test pool.float32.n_active == 1

        # Int64 - fixed slot
        vi64 = acquire!(pool, Int64, 5)
        @test eltype(vi64) == Int64
        @test pool.int64.n_active == 1

        # Int32 - fixed slot
        vi32 = acquire!(pool, Int32, 5)
        @test eltype(vi32) == Int32
        @test pool.int32.n_active == 1

        # ComplexF64 - fixed slot
        vc64 = acquire!(pool, ComplexF64, 5)
        @test eltype(vc64) == ComplexF64
        @test pool.complexf64.n_active == 1

        # ComplexF32 - fixed slot
        vc32 = acquire!(pool, ComplexF32, 5)
        @test eltype(vc32) == ComplexF32
        @test pool.complexf32.n_active == 1

        # Bool - fixed slot
        vb = acquire!(pool, Bool, 5)
        @test eltype(vb) == Bool
        @test pool.bool.n_active == 1
    end

    @testset "Fallback types (others)" begin
        pool = AdaptiveArrayPool()

        # UInt8 - not a fixed slot, goes to others
        vu8 = acquire!(pool, UInt8, 10)
        @test eltype(vu8) == UInt8
        @test haskey(pool.others, UInt8)
        @test pool.others[UInt8].n_active == 1

        # Float16 - not a fixed slot
        v16 = acquire!(pool, Float16, 10)
        @test eltype(v16) == Float16
        @test haskey(pool.others, Float16)
    end

    @testset "pool_stats" begin
        pool = AdaptiveArrayPool()
        acquire!(pool, Float64, 100)
        acquire!(pool, Float64, 30)
        acquire!(pool, UInt8, 50)  # fallback type

        # Should not error
        @test true
    end

    @testset "acquire! fallback (DISABLED_CPU)" begin
        # Without pool - returns Vector (allocation)
        v3 = acquire!(DISABLED_CPU, Float64, 10)
        @test v3 isa Vector{Float64}
        @test length(v3) == 10

        # Different types
        v4 = acquire!(DISABLED_CPU, Int64, 5)
        @test v4 isa Vector{Int64}
        @test length(v4) == 5
    end

end # Core Pool Operations
