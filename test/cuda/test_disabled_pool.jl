# Tests for DisabledPool{:cuda} dispatch methods
# These ensure correct CuArray allocation when pooling is disabled with :cuda backend

using AdaptiveArrayPools: DisabledPool, DISABLED_CPU, pooling_enabled, default_eltype

@testset "DisabledPool{:cuda}" begin
    # Get DISABLED_CUDA from extension
    DISABLED_CUDA = ext.DISABLED_CUDA

    @testset "DISABLED_CUDA singleton" begin
        @test DISABLED_CUDA isa DisabledPool{:cuda}
        @test !pooling_enabled(DISABLED_CUDA)
    end

    @testset "default_eltype" begin
        @test default_eltype(DISABLED_CUDA) === Float32
    end

    @testset "zeros!" begin
        # With type
        v1 = zeros!(DISABLED_CUDA, Float32, 10)
        @test v1 isa CuVector{Float32}
        @test length(v1) == 10
        @test all(v1 .== 0.0f0)

        v2 = zeros!(DISABLED_CUDA, Float64, 5, 5)
        @test v2 isa CuArray{Float64,2}
        @test size(v2) == (5, 5)
        @test all(v2 .== 0.0)

        # Without type (default Float32)
        v3 = zeros!(DISABLED_CUDA, 8)
        @test v3 isa CuVector{Float32}
        @test length(v3) == 8

        v4 = zeros!(DISABLED_CUDA, 3, 4)
        @test v4 isa CuArray{Float32,2}
        @test size(v4) == (3, 4)

        # Tuple dims
        v5 = zeros!(DISABLED_CUDA, Float32, (2, 3, 4))
        @test v5 isa CuArray{Float32,3}
        @test size(v5) == (2, 3, 4)

        v6 = zeros!(DISABLED_CUDA, (5, 6))
        @test v6 isa CuArray{Float32,2}
        @test size(v6) == (5, 6)
    end

    @testset "ones!" begin
        # With type
        v1 = ones!(DISABLED_CUDA, Float32, 10)
        @test v1 isa CuVector{Float32}
        @test length(v1) == 10
        @test all(v1 .== 1.0f0)

        v2 = ones!(DISABLED_CUDA, Float64, 5, 5)
        @test v2 isa CuArray{Float64,2}
        @test size(v2) == (5, 5)
        @test all(v2 .== 1.0)

        # Without type (default Float32)
        v3 = ones!(DISABLED_CUDA, 8)
        @test v3 isa CuVector{Float32}
        @test all(v3 .== 1.0f0)

        v4 = ones!(DISABLED_CUDA, 3, 4)
        @test v4 isa CuArray{Float32,2}
        @test size(v4) == (3, 4)

        # Tuple dims
        v5 = ones!(DISABLED_CUDA, Float32, (2, 3))
        @test v5 isa CuArray{Float32,2}
        @test size(v5) == (2, 3)

        v6 = ones!(DISABLED_CUDA, (4, 5))
        @test v6 isa CuArray{Float32,2}
        @test size(v6) == (4, 5)
    end

    @testset "similar! with CuArray input" begin
        template = CUDA.zeros(Float32, 10)

        v1 = similar!(DISABLED_CUDA, template)
        @test v1 isa CuVector{Float32}
        @test length(v1) == 10

        v2 = similar!(DISABLED_CUDA, template, Float64)
        @test v2 isa CuVector{Float64}
        @test length(v2) == 10

        v3 = similar!(DISABLED_CUDA, template, 5, 5)
        @test v3 isa CuArray{Float32,2}
        @test size(v3) == (5, 5)

        v4 = similar!(DISABLED_CUDA, template, Float64, 3, 4)
        @test v4 isa CuArray{Float64,2}
        @test size(v4) == (3, 4)
    end

    @testset "similar! with AbstractArray input (CPU->GPU)" begin
        cpu_template = zeros(Float64, 8)

        v1 = similar!(DISABLED_CUDA, cpu_template)
        @test v1 isa CuVector{Float64}
        @test length(v1) == 8

        v2 = similar!(DISABLED_CUDA, cpu_template, Float32)
        @test v2 isa CuVector{Float32}
        @test length(v2) == 8

        v3 = similar!(DISABLED_CUDA, cpu_template, 4, 4)
        @test v3 isa CuArray{Float64,2}
        @test size(v3) == (4, 4)

        v4 = similar!(DISABLED_CUDA, cpu_template, Int32, 2, 3)
        @test v4 isa CuArray{Int32,2}
        @test size(v4) == (2, 3)
    end

    @testset "unsafe_zeros!" begin
        v1 = unsafe_zeros!(DISABLED_CUDA, Float32, 10)
        @test v1 isa CuVector{Float32}
        @test all(v1 .== 0.0f0)

        v2 = unsafe_zeros!(DISABLED_CUDA, Float64, 5, 5)
        @test v2 isa CuArray{Float64,2}
        @test size(v2) == (5, 5)

        # Without type
        v3 = unsafe_zeros!(DISABLED_CUDA, 8)
        @test v3 isa CuVector{Float32}

        # Tuple dims
        v4 = unsafe_zeros!(DISABLED_CUDA, Float32, (3, 4))
        @test v4 isa CuArray{Float32,2}
        @test size(v4) == (3, 4)

        v5 = unsafe_zeros!(DISABLED_CUDA, (2, 3))
        @test v5 isa CuArray{Float32,2}
    end

    @testset "unsafe_ones!" begin
        v1 = unsafe_ones!(DISABLED_CUDA, Float32, 10)
        @test v1 isa CuVector{Float32}
        @test all(v1 .== 1.0f0)

        v2 = unsafe_ones!(DISABLED_CUDA, Float64, 5, 5)
        @test v2 isa CuArray{Float64,2}
        @test size(v2) == (5, 5)

        # Without type
        v3 = unsafe_ones!(DISABLED_CUDA, 8)
        @test v3 isa CuVector{Float32}

        # Tuple dims
        v4 = unsafe_ones!(DISABLED_CUDA, Float32, (3, 4))
        @test v4 isa CuArray{Float32,2}

        v5 = unsafe_ones!(DISABLED_CUDA, (2, 3))
        @test v5 isa CuArray{Float32,2}
    end

    @testset "unsafe_similar! with CuArray input" begin
        template = CUDA.zeros(Float32, 10)

        v1 = unsafe_similar!(DISABLED_CUDA, template)
        @test v1 isa CuVector{Float32}

        v2 = unsafe_similar!(DISABLED_CUDA, template, Float64)
        @test v2 isa CuVector{Float64}

        v3 = unsafe_similar!(DISABLED_CUDA, template, 5, 5)
        @test v3 isa CuArray{Float32,2}

        v4 = unsafe_similar!(DISABLED_CUDA, template, Float64, 3, 4)
        @test v4 isa CuArray{Float64,2}
    end

    @testset "unsafe_similar! with AbstractArray input (CPU->GPU)" begin
        cpu_template = zeros(Float64, 8)

        v1 = unsafe_similar!(DISABLED_CUDA, cpu_template)
        @test v1 isa CuVector{Float64}

        v2 = unsafe_similar!(DISABLED_CUDA, cpu_template, Float32)
        @test v2 isa CuVector{Float32}

        v3 = unsafe_similar!(DISABLED_CUDA, cpu_template, 4, 4)
        @test v3 isa CuArray{Float64,2}

        v4 = unsafe_similar!(DISABLED_CUDA, cpu_template, Int32, 2, 3)
        @test v4 isa CuArray{Int32,2}
    end

    @testset "acquire!" begin
        # Type + single dim
        v1 = acquire!(DISABLED_CUDA, Float32, 10)
        @test v1 isa CuVector{Float32}
        @test length(v1) == 10

        # Type + vararg dims
        v2 = acquire!(DISABLED_CUDA, Float64, 5, 5)
        @test v2 isa CuArray{Float64,2}
        @test size(v2) == (5, 5)

        # Type + tuple dims
        v3 = acquire!(DISABLED_CUDA, Float32, (3, 4, 5))
        @test v3 isa CuArray{Float32,3}
        @test size(v3) == (3, 4, 5)

        # CuArray template
        template = CUDA.zeros(Float32, 8)
        v4 = acquire!(DISABLED_CUDA, template)
        @test v4 isa CuVector{Float32}
        @test length(v4) == 8

        # AbstractArray template (CPU->GPU)
        cpu_template = zeros(Float64, 6)
        v5 = acquire!(DISABLED_CUDA, cpu_template)
        @test v5 isa CuVector{Float64}
        @test length(v5) == 6
    end

    @testset "unsafe_acquire!" begin
        # Type + single dim
        v1 = unsafe_acquire!(DISABLED_CUDA, Float32, 10)
        @test v1 isa CuVector{Float32}
        @test length(v1) == 10

        # Type + vararg dims
        v2 = unsafe_acquire!(DISABLED_CUDA, Float64, 5, 5)
        @test v2 isa CuArray{Float64,2}
        @test size(v2) == (5, 5)

        # Type + tuple dims
        v3 = unsafe_acquire!(DISABLED_CUDA, Float32, (3, 4, 5))
        @test v3 isa CuArray{Float32,3}
        @test size(v3) == (3, 4, 5)

        # CuArray template
        template = CUDA.zeros(Float32, 8)
        v4 = unsafe_acquire!(DISABLED_CUDA, template)
        @test v4 isa CuVector{Float32}
        @test length(v4) == 8

        # AbstractArray template (CPU->GPU)
        cpu_template = zeros(Float64, 6)
        v5 = unsafe_acquire!(DISABLED_CUDA, cpu_template)
        @test v5 isa CuVector{Float64}
        @test length(v5) == 6
    end
end
