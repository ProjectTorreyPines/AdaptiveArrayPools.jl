using AdaptiveArrayPools: checkpoint!, rewind!

@testset "Multi-dimensional acquire!" begin
    pool = AdaptiveArrayPool()

    # 2D matrix
    mat = acquire!(pool, Float64, 10, 10)
    @test size(mat) == (10, 10)
    @test mat isa SubArray{Float64, 2}
    @test parent(mat) isa Matrix{Float64}

    @test pool.float64.n_active == 1
    mat .= 1.0
    @test sum(mat) == 100.0

    # 3D tensor
    tensor = acquire!(pool, Float64, 5, 5, 5)
    @test size(tensor) == (5, 5, 5)
    @test tensor isa SubArray{Float64, 3}
    @test parent(tensor) isa Array{Float64, 3}
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
    @test mat isa SubArray{Float64, 2}
    @test parent(mat) isa Matrix{Float64}

    # 3D tuple
    dims3d = (2, 3, 4)
    tensor = acquire!(pool, Float64, dims3d)
    @test size(tensor) == (2, 3, 4)
    @test tensor isa SubArray{Float64, 3}

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

@testset "unsafe_acquire! API" begin
    pool = AdaptiveArrayPool()

    # 1D returns Vector
    v = unsafe_acquire!(pool, Float64, 100)
    @test v isa Vector{Float64}
    @test length(v) == 100
    @test pool.float64.n_active == 1

    # 2D returns Matrix
    mat = unsafe_acquire!(pool, Float64, 10, 10)
    @test mat isa Matrix{Float64}
    @test size(mat) == (10, 10)
    @test pool.float64.n_active == 2

    # 3D returns 3D Array
    tensor = unsafe_acquire!(pool, Float64, 5, 5, 5)
    @test tensor isa Array{Float64, 3}
    @test size(tensor) == (5, 5, 5)
    @test pool.float64.n_active == 3

    # Tuple support
    dims = (4, 5, 6)
    arr = unsafe_acquire!(pool, Float64, dims)
    @test size(arr) == dims
    @test arr isa Array{Float64, 3}

    # Data access works
    v .= 1.0
    mat .= 2.0
    tensor .= 3.0
    @test sum(v) == 100.0
    @test sum(mat) == 200.0
    @test sum(tensor) == 375.0
end

@testset "unsafe_acquire! fallback (nothing)" begin
    v = unsafe_acquire!(nothing, Float64, 10)
    @test v isa Vector{Float64}
    @test length(v) == 10

    mat = unsafe_acquire!(nothing, Float64, 10, 10)
    @test mat isa Matrix{Float64}
    @test size(mat) == (10, 10)

    # Tuple support
    arr = unsafe_acquire!(nothing, Float64, (3, 4, 5))
    @test arr isa Array{Float64, 3}
    @test size(arr) == (3, 4, 5)
end

@testset "Memory sharing between acquire! and unsafe_acquire!" begin
    pool = AdaptiveArrayPool()
    checkpoint!(pool)

    # Get a matrix via acquire!
    mat = acquire!(pool, Float64, 10, 10)
    mat .= 42.0

    rewind!(pool)
    checkpoint!(pool)

    # Get the same memory via unsafe_acquire! (1D)
    raw = unsafe_acquire!(pool, Float64, 100)
    @test raw[1] == 42.0  # Memory was reused, data persists

    rewind!(pool)
end

@testset "Similar-style acquire! and unsafe_acquire!" begin
    pool = AdaptiveArrayPool()
    checkpoint!(pool)

    # Test with Matrix
    ref_mat = rand(5, 6)
    mat = acquire!(pool, ref_mat)
    @test size(mat) == size(ref_mat)
    @test eltype(mat) == eltype(ref_mat)
    @test mat isa SubArray{Float64, 2}

    # Test with Vector
    ref_vec = rand(10)
    vec = acquire!(pool, ref_vec)
    @test size(vec) == size(ref_vec)
    @test eltype(vec) == eltype(ref_vec)
    @test vec isa SubArray{Float64, 1}

    # Test with 3D Array
    ref_tensor = rand(2, 3, 4)
    tensor = acquire!(pool, ref_tensor)
    @test size(tensor) == size(ref_tensor)
    @test tensor isa SubArray{Float64, 3}

    # Test with different element types
    ref_int = rand(Int32, 4, 5)
    int_mat = acquire!(pool, ref_int)
    @test eltype(int_mat) == Int32
    @test size(int_mat) == (4, 5)

    rewind!(pool)

    # Test unsafe_acquire! similar style
    checkpoint!(pool)

    unsafe_mat = unsafe_acquire!(pool, ref_mat)
    @test size(unsafe_mat) == size(ref_mat)
    @test unsafe_mat isa Matrix{Float64}

    unsafe_vec = unsafe_acquire!(pool, ref_vec)
    @test size(unsafe_vec) == size(ref_vec)
    @test unsafe_vec isa Vector{Float64}

    rewind!(pool)

    # Test nothing fallback
    nothing_mat = acquire!(nothing, ref_mat)
    @test size(nothing_mat) == size(ref_mat)
    @test nothing_mat isa Matrix{Float64}

    nothing_unsafe = unsafe_acquire!(nothing, ref_mat)
    @test size(nothing_unsafe) == size(ref_mat)
    @test nothing_unsafe isa Matrix{Float64}
end

# Function barrier for accurate allocation measurement
function test_nd_zero_alloc()
    pool = AdaptiveArrayPool()

    # Warmup phase (cache miss → allocations expected)
    @with_pool pool begin
        m = acquire!(pool, Float64, 10, 10)
        m .= 1.0
    end
    @with_pool pool begin
        m = acquire!(pool, Float64, 10, 10)
        m .= 1.0
    end

    # Measure (cache hit → should be 0 bytes)
    alloc = @allocated @with_pool pool begin
        m = acquire!(pool, Float64, 10, 10)
        m .= 1.0
        nothing
    end

    return alloc
end

@testset "N-D acquire! zero-allocation (cache hit)" begin
    # First call compiles the function
    test_nd_zero_alloc()
    test_nd_zero_alloc()

    # Measure
    alloc = test_nd_zero_alloc()
    println("  N-D acquire! allocation (cache hit): $alloc bytes")
    @test alloc == 0
end
