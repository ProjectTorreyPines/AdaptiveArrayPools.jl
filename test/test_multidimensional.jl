@testset "Multi-dimensional acquire!" begin
    pool = AdaptiveArrayPool()

    # 2D matrix
    mat = acquire!(pool, Float64, 10, 10)
    @test size(mat) == (10, 10)
    @test mat isa Base.ReshapedArray

    @test pool.float64.n_active == 1
    mat .= 1.0
    @test sum(mat) == 100.0

    # 3D tensor
    tensor = acquire!(pool, Float64, 5, 5, 5)
    @test size(tensor) == (5, 5, 5)
    @test tensor isa Base.ReshapedArray
    @test pool.float64.n_active == 2
    tensor .= 2.0
    @test sum(tensor) == 250.0

    # Reset and reuse with new pool
    pool2 = AdaptiveArrayPool()
    checkpoint!(pool2)

    mat2 = acquire!(pool2, Float64, 20, 5)
    @test size(mat2) == (20, 5)

    rewind!(pool2)
    mat3 = acquire!(pool2, Float64, 10, 10)
    @test size(mat3) == (10, 10)

    # Without pool (fallback)
    mat_alloc = acquire!(nothing, Float64, 10, 10)
    @test mat_alloc isa Array{Float64,2}
    @test size(mat_alloc) == (10, 10)
end

@testset "Multi-dimensional with @use_pool" begin
    pool = AdaptiveArrayPool()

    @use_pool pool function matrix_computation(n::Int)
        mat = acquire!(pool, Float64, n, n)
        mat .= 1.0
        vec = acquire!(pool, Float64, n)
        vec .= 2.0
        return sum(mat) + sum(vec)
    end

    # Without pool
    result1 = matrix_computation(10)
    @test result1 == 120.0

    # With pool
    result2 = matrix_computation(10; pool)
    @test result2 == 120.0
    @test pool.float64.n_active == 0
end
