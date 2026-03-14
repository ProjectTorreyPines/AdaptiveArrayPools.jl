# Tests for DisabledPool{:metal} dispatch methods

using AdaptiveArrayPools: DisabledPool, DISABLED_CPU, pooling_enabled, default_eltype

@testset "DisabledPool{:metal}" begin
    DISABLED_METAL = ext.DISABLED_METAL

    @testset "DISABLED_METAL singleton" begin
        @test DISABLED_METAL isa DisabledPool{:metal}
        @test !pooling_enabled(DISABLED_METAL)
    end

    @testset "default_eltype" begin
        @test default_eltype(DISABLED_METAL) === Float32
    end

    @testset "zeros!" begin
        v1 = zeros!(DISABLED_METAL, Float32, 10)
        @test v1 isa MtlVector{Float32}
        @test length(v1) == 10
        @test all(v1 .== 0.0f0)

        v2 = zeros!(DISABLED_METAL, Float32, 5, 5)
        @test v2 isa MtlArray{Float32, 2}
        @test size(v2) == (5, 5)
        @test all(v2 .== 0.0f0)

        # Without type (default Float32)
        v3 = zeros!(DISABLED_METAL, 8)
        @test v3 isa MtlVector{Float32}
        @test length(v3) == 8

        v4 = zeros!(DISABLED_METAL, 3, 4)
        @test v4 isa MtlArray{Float32, 2}
        @test size(v4) == (3, 4)

        # Tuple dims
        v5 = zeros!(DISABLED_METAL, Float32, (2, 3, 4))
        @test v5 isa MtlArray{Float32, 3}
        @test size(v5) == (2, 3, 4)

        v6 = zeros!(DISABLED_METAL, (5, 6))
        @test v6 isa MtlArray{Float32, 2}
        @test size(v6) == (5, 6)
    end

    @testset "ones!" begin
        v1 = ones!(DISABLED_METAL, Float32, 10)
        @test v1 isa MtlVector{Float32}
        @test length(v1) == 10
        @test all(v1 .== 1.0f0)

        v2 = ones!(DISABLED_METAL, Float32, 5, 5)
        @test v2 isa MtlArray{Float32, 2}
        @test size(v2) == (5, 5)
        @test all(v2 .== 1.0f0)

        # Without type (default Float32)
        v3 = ones!(DISABLED_METAL, 8)
        @test v3 isa MtlVector{Float32}
        @test all(v3 .== 1.0f0)

        v4 = ones!(DISABLED_METAL, 3, 4)
        @test v4 isa MtlArray{Float32, 2}
        @test size(v4) == (3, 4)

        # Tuple dims
        v5 = ones!(DISABLED_METAL, Float32, (2, 3))
        @test v5 isa MtlArray{Float32, 2}
        @test size(v5) == (2, 3)

        v6 = ones!(DISABLED_METAL, (4, 5))
        @test v6 isa MtlArray{Float32, 2}
        @test size(v6) == (4, 5)
    end

    @testset "similar! with MtlArray input" begin
        template = Metal.zeros(Float32, 10)

        v1 = similar!(DISABLED_METAL, template)
        @test v1 isa MtlVector{Float32}
        @test length(v1) == 10

        v2 = similar!(DISABLED_METAL, template, Int32)
        @test v2 isa MtlVector{Int32}
        @test length(v2) == 10

        v3 = similar!(DISABLED_METAL, template, 5, 5)
        @test v3 isa MtlArray{Float32, 2}
        @test size(v3) == (5, 5)

        v4 = similar!(DISABLED_METAL, template, Int32, 3, 4)
        @test v4 isa MtlArray{Int32, 2}
        @test size(v4) == (3, 4)
    end

    @testset "similar! with AbstractArray input (CPU->GPU)" begin
        cpu_template = zeros(Float32, 8)

        v1 = similar!(DISABLED_METAL, cpu_template)
        @test v1 isa MtlArray{Float32}
        @test length(v1) == 8

        v2 = similar!(DISABLED_METAL, cpu_template, Int32)
        @test v2 isa MtlArray{Int32}
        @test length(v2) == 8

        v3 = similar!(DISABLED_METAL, cpu_template, 4, 4)
        @test v3 isa MtlArray{Float32, 2}
        @test size(v3) == (4, 4)

        v4 = similar!(DISABLED_METAL, cpu_template, Int32, 2, 3)
        @test v4 isa MtlArray{Int32, 2}
        @test size(v4) == (2, 3)
    end

    @testset "reshape!" begin
        a = acquire!(DISABLED_METAL, Float32, 12)
        r1 = reshape!(DISABLED_METAL, a, 3, 4)
        @test r1 isa MtlArray{Float32, 2}
        @test size(r1) == (3, 4)

        r2 = reshape!(DISABLED_METAL, a, (4, 3))
        @test r2 isa MtlArray{Float32, 2}
        @test size(r2) == (4, 3)
    end

    @testset "Sub-function passing" begin
        function _metal_helper(pool, n)
            return zeros!(pool, Float32, n)
        end

        function _metal_helper_typed(pool::AbstractArrayPool, n)
            return acquire!(pool, Float32, n)
        end

        function _metal_outer(pool, n)
            return _metal_inner(pool, n)
        end
        function _metal_inner(pool, n)
            return ones!(pool, Float32, n)
        end

        v1 = _metal_helper(DISABLED_METAL, 5)
        @test v1 isa MtlVector{Float32}
        @test all(v1 .== 0.0f0)

        v2 = _metal_helper_typed(DISABLED_METAL, 5)
        @test v2 isa MtlVector{Float32}

        v3 = _metal_outer(DISABLED_METAL, 3)
        @test v3 isa MtlVector{Float32}
        @test all(v3 .== 1.0f0)
    end

    @testset "State management no-ops" begin
        @test checkpoint!(DISABLED_METAL) === nothing
        @test rewind!(DISABLED_METAL) === nothing
        @test reset!(DISABLED_METAL) === nothing
        @test empty!(DISABLED_METAL) === nothing
    end

    @testset "acquire!" begin
        # Type + single dim
        v1 = acquire!(DISABLED_METAL, Float32, 10)
        @test v1 isa MtlVector{Float32}
        @test length(v1) == 10

        # Type + vararg dims
        v2 = acquire!(DISABLED_METAL, Int32, 5, 5)
        @test v2 isa MtlArray{Int32, 2}
        @test size(v2) == (5, 5)

        # Type + tuple dims
        v3 = acquire!(DISABLED_METAL, Float32, (3, 4, 5))
        @test v3 isa MtlArray{Float32, 3}
        @test size(v3) == (3, 4, 5)

        # MtlArray template
        template = Metal.zeros(Float32, 8)
        v4 = acquire!(DISABLED_METAL, template)
        @test v4 isa MtlVector{Float32}
        @test length(v4) == 8

        # AbstractArray template (CPU->GPU)
        cpu_template = zeros(Float32, 6)
        v5 = acquire!(DISABLED_METAL, cpu_template)
        @test v5 isa MtlArray{Float32}
        @test length(v5) == 6
    end

end
