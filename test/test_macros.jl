# Tests for explicit pool management with checkpoint!/rewind!
import AdaptiveArrayPools: checkpoint!, rewind!

@testset "Explicit pool with checkpoint!/rewind!" begin
    pool = AdaptiveArrayPool()

    v_outer = acquire!(pool, Float64, 10)
    @test pool.float64.n_active == 1

    checkpoint!(pool)
    v1 = acquire!(pool, Float64, 20)
    v2 = acquire!(pool, Float64, 30)
    @test pool.float64.n_active == 3
    result = sum(v1) + sum(v2)
    rewind!(pool)

    @test pool.float64.n_active == 1
    @test result isa Number

    v_outer .= 42.0
    @test all(v_outer .== 42.0)
end

@testset "Nested checkpoint!/rewind!" begin
    pool = AdaptiveArrayPool()

    function inner_computation(pool)
        checkpoint!(pool)
        try
            v = acquire!(pool, Float64, 10)
            v .= 2.0
            sum(v)
        finally
            rewind!(pool)
        end
    end

    function outer_computation(pool)
        checkpoint!(pool)
        try
            v1 = acquire!(pool, Float64, 20)
            v1 .= 1.0
            inner_result = inner_computation(pool)
            @test all(v1 .== 1.0)
            sum(v1) + inner_result
        finally
            rewind!(pool)
        end
    end

    result = outer_computation(pool)
    @test result == 40.0
    @test pool.float64.n_active == 0
end

@testset "@pool_kwarg function definition" begin
    @pool_kwarg pool function test_auto_inject(x::Vector{Float64})
        temp = acquire!(pool, Float64, length(x))
        temp .= x .* 2
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]

    # Without pool (uses normal allocation)
    result1 = test_auto_inject(x)
    @test result1 == 12.0

    # With explicit pool (caller manages checkpoint)
    mypool = AdaptiveArrayPool()
    checkpoint!(mypool)
    result2 = test_auto_inject(x; pool=mypool)
    rewind!(mypool)
    @test result2 == 12.0
    @test mypool.float64.n_active == 0
end

@testset "@pool_kwarg short-form function" begin
    @pool_kwarg pool test_short(x) = sum(acquire!(pool, Float64, length(x)) .= x)

    x = [1.0, 2.0, 3.0]
    @test test_short(x) == 6.0

    mypool = AdaptiveArrayPool()
    checkpoint!(mypool)
    @test test_short(x; pool=mypool) == 6.0
    rewind!(mypool)
    @test mypool.float64.n_active == 0
end

@testset "@pool_kwarg with existing kwargs" begin
    @pool_kwarg pool function test_existing_kwargs(x; scale=2.0)
        temp = acquire!(pool, Float64, length(x))
        temp .= x .* scale
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    @test test_existing_kwargs(x) == 12.0
    @test test_existing_kwargs(x; scale=3.0) == 18.0

    mypool = AdaptiveArrayPool()
    checkpoint!(mypool)
    @test test_existing_kwargs(x; pool=mypool, scale=4.0) == 24.0
    rewind!(mypool)
    @test mypool.float64.n_active == 0
end

@testset "@pool_kwarg with where clause" begin
    @pool_kwarg pool function test_where(x::Vector{T}) where {T<:Number}
        temp = acquire!(pool, T, length(x))
        temp .= x .+ one(T)
        return sum(temp)
    end

    x = [1.0, 2.0, 3.0]
    @test test_where(x) == 9.0

    mypool = AdaptiveArrayPool()
    checkpoint!(mypool)
    @test test_where(x; pool=mypool) == 9.0
    rewind!(mypool)
    @test mypool.float64.n_active == 0
end

@testset "@pool_kwarg short-form with where clause" begin
    @pool_kwarg pool test_short_where(x::Vector{T}) where {T<:Number} = sum(acquire!(pool, T, length(x)) .= x)

    x = [1.0, 2.0, 3.0]
    @test test_short_where(x) == 6.0

    mypool = AdaptiveArrayPool()
    checkpoint!(mypool)
    @test test_short_where(x; pool=mypool) == 6.0
    rewind!(mypool)
    @test mypool.float64.n_active == 0
end

@testset "@pool_kwarg combined with @with_pool" begin
    # Define a layer function with @pool_kwarg
    @pool_kwarg pool function layer(x)
        out = acquire!(pool, Float64, length(x))
        out .= x .* 2
        return out
    end

    # Use it within @with_pool scope
    result = @with_pool p begin
        x = [1.0, 2.0, 3.0]
        y = layer(x; pool=p)  # Uses the checkpointed pool
        sum(y)
    end
    @test result == 12.0
end

@testset "@with_pool nested scopes" begin
    result = @with_pool p1 begin
        v1 = acquire!(p1, Float64, 10)
        v1 .= 1.0

        inner = @with_pool p2 begin
            v2 = acquire!(p2, Float64, 5)
            v2 .= 2.0
            sum(v2)
        end

        # v1 should still be valid
        @test all(v1 .== 1.0)
        sum(v1) + inner
    end
    @test result == 20.0
end
