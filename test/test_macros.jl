@testset "@with_pool block mode" begin
    pool = AdaptiveArrayPool()

    v_outer = acquire!(pool, Float64, 10)
    @test pool.float64.n_active == 1

    result = @with_pool pool begin
        v1 = acquire!(pool, Float64, 20)
        v2 = acquire!(pool, Float64, 30)
        @test pool.float64.n_active == 3
        sum(v1) + sum(v2)
    end

    @test pool.float64.n_active == 1
    @test result isa Number

    v_outer .= 42.0
    @test all(v_outer .== 42.0)
end

@testset "@with_pool nested" begin
    pool = AdaptiveArrayPool()

    function inner_computation(pool)
        @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 2.0
            sum(v)
        end
    end

    function outer_computation(pool)
        @with_pool pool begin
            v1 = acquire!(pool, Float64, 20)
            v1 .= 1.0
            inner_result = inner_computation(pool)
            @test all(v1 .== 1.0)
            sum(v1) + inner_result
        end
    end

    result = outer_computation(pool)
    @test result == 40.0
    @test pool.float64.n_active == 0
end

@testset "@with_pool function definition mode" begin
    @with_pool pool function test_auto_inject(x::Vector{Float64})
        temp = acquire!(pool, Float64, length(x))
        temp .= x .* 2
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    result1 = test_auto_inject(x)
    @test result1 == 12.0

    mypool = AdaptiveArrayPool()
    result2 = test_auto_inject(x; pool=mypool)
    @test result2 == 12.0
    @test mypool.float64.n_active == 0
end

@testset "@with_pool short-form function" begin
    @with_pool pool test_short(x) = sum(acquire!(pool, Float64, length(x)) .= x)

    x = [1.0, 2.0, 3.0]
    @test test_short(x) == 6.0

    mypool = AdaptiveArrayPool()
    @test test_short(x; pool=mypool) == 6.0
    @test mypool.float64.n_active == 0
end

@testset "@with_pool with existing kwargs" begin
    @with_pool pool function test_existing_kwargs(x; scale=2.0)
        temp = acquire!(pool, Float64, length(x))
        temp .= x .* scale
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    @test test_existing_kwargs(x) == 12.0
    @test test_existing_kwargs(x; scale=3.0) == 18.0

    mypool = AdaptiveArrayPool()
    @test test_existing_kwargs(x; pool=mypool, scale=4.0) == 24.0
    @test mypool.float64.n_active == 0
end

@testset "@with_pool with where clause" begin
    @with_pool pool function test_where(x::Vector{T}) where {T<:Number}
        temp = acquire!(pool, T, length(x))
        temp .= x .+ one(T)
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    @test test_where(x) == 9.0

    mypool = AdaptiveArrayPool()
    @test test_where(x; pool=mypool) == 9.0
    @test mypool.float64.n_active == 0
end

@testset "@with_pool block mode with POOL_DEBUG" begin
    old_debug = POOL_DEBUG[]
    POOL_DEBUG[] = true

    pool = AdaptiveArrayPool()

    # Safe return (scalar) should work
    result = @with_pool pool begin
        v = acquire!(pool, Float64, 10)
        v .= 1.0
        sum(v)  # Safe: returning scalar
    end
    @test result == 10.0

    # Safe return (copy) should work
    result = @with_pool pool begin
        v = acquire!(pool, Float64, 5)
        v .= 2.0
        collect(v)  # Safe: returning copy
    end
    @test result == [2.0, 2.0, 2.0, 2.0, 2.0]

    # Unsafe return should throw
    @test_throws ErrorException @with_pool pool begin
        v = acquire!(pool, Float64, 10)
        v  # Unsafe: returning pool-backed SubArray
    end

    POOL_DEBUG[] = old_debug
end

@testset "@with_pool function mode with POOL_DEBUG" begin
    old_debug = POOL_DEBUG[]
    POOL_DEBUG[] = true

    # Define function that returns scalar (safe)
    @with_pool pool function safe_sum_func(n::Int)
        v = acquire!(pool, Float64, n)
        v .= 1.0
        sum(v)
    end

    mypool = AdaptiveArrayPool()
    @test safe_sum_func(10; pool=mypool) == 10.0
    @test mypool.float64.n_active == 0

    # Define function that returns copy (safe)
    @with_pool pool function safe_copy_func(n::Int)
        v = acquire!(pool, Float64, n)
        v .= 3.0
        collect(v)
    end

    @test safe_copy_func(3; pool=mypool) == [3.0, 3.0, 3.0]
    @test mypool.float64.n_active == 0

    POOL_DEBUG[] = old_debug
end

@testset "@with_pool short-form with where clause" begin
    # Short-form function with where clause (covers line 201-202)
    @with_pool pool test_short_where(x::Vector{T}) where {T<:Number} = sum(acquire!(pool, T, length(x)) .= x)

    x = [1.0, 2.0, 3.0]
    @test test_short_where(x) == 6.0

    mypool = AdaptiveArrayPool()
    @test test_short_where(x; pool=mypool) == 6.0
    @test mypool.float64.n_active == 0
end
