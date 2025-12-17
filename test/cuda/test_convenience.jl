@testset "CUDA Convenience Functions" begin
    pool = get_task_local_cuda_pool()
    checkpoint!(pool)

    @testset "zeros! default type is Float32" begin
        v = zeros!(pool, 10)
        @test v isa CuArray{Float32}
        @test length(v) == 10
        @test all(v .== 0.0f0)

        m = zeros!(pool, 3, 4)
        @test m isa CuArray{Float32,2}
        @test size(m) == (3, 4)
        @test all(m .== 0.0f0)

        # Tuple form
        dims = (2, 3)
        t = zeros!(pool, dims)
        @test t isa CuArray{Float32,2}
        @test size(t) == dims
    end

    @testset "zeros! explicit type" begin
        v64 = zeros!(pool, Float64, 10)
        @test v64 isa CuArray{Float64}
        @test all(v64 .== 0.0)

        v16 = zeros!(pool, Float16, 5)
        @test v16 isa CuArray{Float16}

        vi = zeros!(pool, Int32, 8)
        @test vi isa CuArray{Int32}
        @test all(vi .== 0)
    end

    @testset "ones! default type is Float32" begin
        v = ones!(pool, 10)
        @test v isa CuArray{Float32}
        @test length(v) == 10
        @test all(v .== 1.0f0)

        m = ones!(pool, 3, 4)
        @test m isa CuArray{Float32,2}
        @test size(m) == (3, 4)
        @test all(m .== 1.0f0)

        # Tuple form
        dims = (2, 3)
        t = ones!(pool, dims)
        @test t isa CuArray{Float32,2}
        @test size(t) == dims
    end

    @testset "ones! explicit type" begin
        v64 = ones!(pool, Float64, 10)
        @test v64 isa CuArray{Float64}
        @test all(v64 .== 1.0)

        vi = ones!(pool, Int32, 8)
        @test vi isa CuArray{Int32}
        @test all(vi .== 1)
    end

    @testset "similar!" begin
        # Float32 template
        template32 = CUDA.rand(Float32, 5, 5)
        v = similar!(pool, template32)
        @test v isa CuArray{Float32,2}
        @test size(v) == (5, 5)

        # Float64 template
        template64 = CUDA.rand(Float64, 3, 4)
        v64 = similar!(pool, template64)
        @test v64 isa CuArray{Float64,2}
        @test size(v64) == (3, 4)

        # Different type
        v_int = similar!(pool, template32, Int32)
        @test v_int isa CuArray{Int32,2}
        @test size(v_int) == (5, 5)

        # Different dims
        v_dims = similar!(pool, template32, 10)
        @test v_dims isa CuArray{Float32,1}
        @test length(v_dims) == 10

        # Different type and dims
        v_both = similar!(pool, template32, Float64, 2, 3)
        @test v_both isa CuArray{Float64,2}
        @test size(v_both) == (2, 3)
    end

    @testset "unsafe_zeros! default type is Float32" begin
        v = unsafe_zeros!(pool, 10)
        @test v isa CuArray{Float32,1}
        @test !(v isa SubArray)  # Raw array, not view
        @test length(v) == 10
        @test all(v .== 0.0f0)

        m = unsafe_zeros!(pool, 3, 4)
        @test m isa CuArray{Float32,2}
        @test !(m isa SubArray)
        @test size(m) == (3, 4)
        @test all(m .== 0.0f0)

        # Tuple form
        dims = (2, 3)
        t = unsafe_zeros!(pool, dims)
        @test t isa CuArray{Float32,2}
        @test size(t) == dims
    end

    @testset "unsafe_zeros! explicit type" begin
        v64 = unsafe_zeros!(pool, Float64, 10)
        @test v64 isa CuArray{Float64}
        @test !(v64 isa SubArray)
        @test all(v64 .== 0.0)
    end

    @testset "unsafe_ones! default type is Float32" begin
        v = unsafe_ones!(pool, 10)
        @test v isa CuArray{Float32,1}
        @test !(v isa SubArray)
        @test length(v) == 10
        @test all(v .== 1.0f0)

        m = unsafe_ones!(pool, 3, 4)
        @test m isa CuArray{Float32,2}
        @test !(m isa SubArray)
        @test size(m) == (3, 4)
        @test all(m .== 1.0f0)

        # Tuple form
        dims = (2, 3)
        t = unsafe_ones!(pool, dims)
        @test t isa CuArray{Float32,2}
        @test size(t) == dims
    end

    @testset "unsafe_ones! explicit type" begin
        v64 = unsafe_ones!(pool, Float64, 10)
        @test v64 isa CuArray{Float64}
        @test !(v64 isa SubArray)
        @test all(v64 .== 1.0)
    end

    @testset "unsafe_similar!" begin
        # Float32 template
        template32 = CUDA.rand(Float32, 5, 5)
        v = unsafe_similar!(pool, template32)
        @test v isa CuArray{Float32,2}
        @test !(v isa SubArray)
        @test size(v) == (5, 5)

        # Float64 template
        template64 = CUDA.rand(Float64, 3, 4)
        v64 = unsafe_similar!(pool, template64)
        @test v64 isa CuArray{Float64,2}
        @test !(v64 isa SubArray)
        @test size(v64) == (3, 4)

        # Different type
        v_int = unsafe_similar!(pool, template32, Int32)
        @test v_int isa CuArray{Int32,2}
        @test !(v_int isa SubArray)
        @test size(v_int) == (5, 5)

        # Different dims
        v_dims = unsafe_similar!(pool, template32, 10)
        @test v_dims isa CuArray{Float32,1}
        @test !(v_dims isa SubArray)
        @test length(v_dims) == 10

        # Different type and dims
        v_both = unsafe_similar!(pool, template32, Float64, 2, 3)
        @test v_both isa CuArray{Float64,2}
        @test !(v_both isa SubArray)
        @test size(v_both) == (2, 3)
    end

    rewind!(pool)
end
