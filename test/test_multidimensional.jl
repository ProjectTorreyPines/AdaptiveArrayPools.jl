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

@testset "Tuple-based acquire!" begin
    pool = AdaptiveArrayPool()

    # Tuple from size()
    ref_array = zeros(3, 4)
    dims = size(ref_array)
    mat = acquire!(pool, Float64, dims)
    @test size(mat) == (3, 4)
    @test mat isa Base.ReshapedArray

    # 3D tuple
    dims3d = (2, 3, 4)
    tensor = acquire!(pool, Float64, dims3d)
    @test size(tensor) == (2, 3, 4)

    # Fallback with nothing
    mat_alloc = acquire!(nothing, Float64, dims)
    @test mat_alloc isa Array{Float64, 2}
    @test size(mat_alloc) == (3, 4)
end

@testset "Multi-dimensional with @with_pool" begin
    # Define a computation function that takes pool as argument
    function matrix_computation(n::Int, pool)
        mat = acquire!(pool, Float64, n, n)
        mat .= 1.0
        vec = acquire!(pool, Float64, n)
        vec .= 2.0
        return sum(mat) + sum(vec)
    end

    # Use @with_pool to manage lifecycle
    result1 = @with_pool pool begin
        matrix_computation(10, pool)
    end
    @test result1 == 120.0
    @test get_task_local_pool().float64.n_active == 0

    # Alternative: function uses get_task_local_pool() directly
    function matrix_computation_global(n::Int)
        pool = get_task_local_pool()
        mat = acquire!(pool, Float64, n, n)
        mat .= 1.0
        vec = acquire!(pool, Float64, n)
        vec .= 2.0
        return sum(mat) + sum(vec)
    end

    result2 = @with_pool begin
        matrix_computation_global(10)
    end
    @test result2 == 120.0
    @test get_task_local_pool().float64.n_active == 0

    # With explicit pool (caller manages checkpoint)
    mypool = AdaptiveArrayPool()
    checkpoint!(mypool)
    result3 = matrix_computation(10, mypool)
    rewind!(mypool)
    @test result3 == 120.0
    @test mypool.float64.n_active == 0
end
