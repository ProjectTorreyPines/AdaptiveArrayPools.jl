@testset "Metal Convenience Functions" begin
    pool = get_task_local_metal_pool()
    checkpoint!(pool)

    @testset "zeros! default type is Float32" begin
        v = zeros!(pool, 10)
        @test v isa MtlArray{Float32}
        @test length(v) == 10
        @test all(v .== 0.0f0)

        m = zeros!(pool, 3, 4)
        @test m isa MtlArray{Float32, 2}
        @test size(m) == (3, 4)
        @test all(m .== 0.0f0)

        # Tuple form
        dims = (2, 3)
        t = zeros!(pool, dims)
        @test t isa MtlArray{Float32, 2}
        @test size(t) == dims
    end

    @testset "zeros! explicit type" begin
        v16 = zeros!(pool, Float16, 5)
        @test v16 isa MtlArray{Float16}

        vi = zeros!(pool, Int32, 8)
        @test vi isa MtlArray{Int32}
        @test all(vi .== 0)
    end

    @testset "ones! default type is Float32" begin
        v = ones!(pool, 10)
        @test v isa MtlArray{Float32}
        @test length(v) == 10
        @test all(v .== 1.0f0)

        m = ones!(pool, 3, 4)
        @test m isa MtlArray{Float32, 2}
        @test size(m) == (3, 4)
        @test all(m .== 1.0f0)

        # Tuple form
        dims = (2, 3)
        t = ones!(pool, dims)
        @test t isa MtlArray{Float32, 2}
        @test size(t) == dims
    end

    @testset "ones! explicit type" begin
        vi = ones!(pool, Int32, 8)
        @test vi isa MtlArray{Int32}
        @test all(vi .== 1)
    end

    @testset "similar!" begin
        # Float32 template
        template32 = MtlArray(rand(Float32, 5, 5))
        v = similar!(pool, template32)
        @test v isa MtlArray{Float32, 2}
        @test size(v) == (5, 5)

        # Different type
        v_int = similar!(pool, template32, Int32)
        @test v_int isa MtlArray{Int32, 2}
        @test size(v_int) == (5, 5)

        # Different dims
        v_dims = similar!(pool, template32, 10)
        @test v_dims isa MtlArray{Float32, 1}
        @test length(v_dims) == 10

        # Different type and dims
        v_both = similar!(pool, template32, Int32, 2, 3)
        @test v_both isa MtlArray{Int32, 2}
        @test size(v_both) == (2, 3)
    end

    @testset "zeros! returns Array (not view)" begin
        v = zeros!(pool, 10)
        @test v isa MtlArray{Float32, 1}
        @test !(v isa SubArray)
        @test all(v .== 0.0f0)

        m = zeros!(pool, 3, 4)
        @test m isa MtlArray{Float32, 2}
        @test !(m isa SubArray)
        @test all(m .== 0.0f0)
    end

    @testset "ones! returns Array (not view)" begin
        v = ones!(pool, 10)
        @test v isa MtlArray{Float32, 1}
        @test !(v isa SubArray)
        @test all(v .== 1.0f0)

        m = ones!(pool, 3, 4)
        @test m isa MtlArray{Float32, 2}
        @test !(m isa SubArray)
        @test all(m .== 1.0f0)
    end

    @testset "similar! returns Array (not view)" begin
        template32 = MtlArray(rand(Float32, 5, 5))
        v = similar!(pool, template32)
        @test v isa MtlArray{Float32, 2}
        @test !(v isa SubArray)
        @test size(v) == (5, 5)

        v_int = similar!(pool, template32, Int32)
        @test v_int isa MtlArray{Int32, 2}
        @test !(v_int isa SubArray)

        v_dims = similar!(pool, template32, 10)
        @test v_dims isa MtlArray{Float32, 1}
        @test !(v_dims isa SubArray)

        v_both = similar!(pool, template32, Int32, 2, 3)
        @test v_both isa MtlArray{Int32, 2}
        @test !(v_both isa SubArray)
    end

    rewind!(pool)
end
