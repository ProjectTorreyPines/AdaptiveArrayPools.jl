@testset "@use_pool block mode" begin
    pool = AdaptiveArrayPool()

    v_outer = acquire!(pool, Float64, 10)
    @test pool.float64.in_use == 1

    result = @use_pool pool begin
        v1 = acquire!(pool, Float64, 20)
        v2 = acquire!(pool, Float64, 30)
        @test pool.float64.in_use == 3
        sum(v1) + sum(v2)
    end

    @test pool.float64.in_use == 1
    @test result isa Number

    v_outer .= 42.0
    @test all(v_outer .== 42.0)
end

@testset "@use_pool nested" begin
    pool = AdaptiveArrayPool()

    function inner_computation(pool)
        @use_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 2.0
            sum(v)
        end
    end

    function outer_computation(pool)
        @use_pool pool begin
            v1 = acquire!(pool, Float64, 20)
            v1 .= 1.0
            inner_result = inner_computation(pool)
            @test all(v1 .== 1.0)
            sum(v1) + inner_result
        end
    end

    result = outer_computation(pool)
    @test result == 40.0
    @test pool.float64.in_use == 0
end

@testset "@use_pool function definition mode" begin
    @use_pool pool function test_auto_inject(x::Vector{Float64})
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
    @test mypool.float64.in_use == 0
end

@testset "@use_pool short-form function" begin
    @use_pool pool test_short(x) = sum(acquire!(pool, Float64, length(x)) .= x)

    x = [1.0, 2.0, 3.0]
    @test test_short(x) == 6.0

    mypool = AdaptiveArrayPool()
    @test test_short(x; pool=mypool) == 6.0
    @test mypool.float64.in_use == 0
end

@testset "@use_pool with existing kwargs" begin
    @use_pool pool function test_existing_kwargs(x; scale=2.0)
        temp = acquire!(pool, Float64, length(x))
        temp .= x .* scale
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    @test test_existing_kwargs(x) == 12.0
    @test test_existing_kwargs(x; scale=3.0) == 18.0

    mypool = AdaptiveArrayPool()
    @test test_existing_kwargs(x; pool=mypool, scale=4.0) == 24.0
    @test mypool.float64.in_use == 0
end

@testset "@use_pool with where clause" begin
    @use_pool pool function test_where(x::Vector{T}) where {T<:Number}
        temp = acquire!(pool, T, length(x))
        temp .= x .+ one(T)
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    @test test_where(x) == 9.0

    mypool = AdaptiveArrayPool()
    @test test_where(x; pool=mypool) == 9.0
    @test mypool.float64.in_use == 0
end
