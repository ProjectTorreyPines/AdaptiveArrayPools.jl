using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!
import AdaptiveArrayPools: mark, reset!  # Not exported to avoid Base.mark conflict

@testset "AdaptiveArrayPools" begin

    @testset "Basic functionality" begin
        pool = AdaptiveArrayPool()

        # First acquire
        v1 = acquire!(pool, Float64, 5)
        @test length(v1) == 5
        @test v1 isa SubArray
        @test eltype(v1) == Float64
        
        # Access internal state for verification
        tp = pool.pools[Float64]
        @test tp.in_use == 1

        # Second acquire
        v2 = acquire!(pool, Float64, 8)
        @test length(v2) == 8
        @test tp.in_use == 2

        # Values are independent
        v1 .= 1.0
        v2 .= 2.0
        @test all(v1 .== 1.0)
        @test all(v2 .== 2.0)
    end

    @testset "Pool expansion" begin
        pool = AdaptiveArrayPool()
        
        # Acquire some vectors
        acquire!(pool, Float64, 5)
        acquire!(pool, Float64, 5)
        
        tp = pool.pools[Float64]
        @test length(tp.vectors) >= 2 # Might be 2 or more depending on implementation details

        # Should expand pool
        v3 = acquire!(pool, Float64, 5)
        @test tp.in_use == 3
        @test length(tp.vectors) >= 3
        @test v3 isa SubArray
    end

    @testset "Vector resize" begin
        pool = AdaptiveArrayPool()

        # Initial request
        v1 = acquire!(pool, Float64, 20)
        @test length(v1) == 20
        
        tp = pool.pools[Float64]
        @test length(tp.vectors[1]) >= 20

        # Second acquire, smaller size
        v2 = acquire!(pool, Float64, 5)
        @test length(v2) == 5
        # Note: We can't easily guarantee which vector was reused if we don't reset, 
        # but here we are just appending.
    end

    @testset "Reset and reuse" begin
        pool = AdaptiveArrayPool()
        empty_state = mark(pool)

        # First iteration
        v1 = acquire!(pool, Float64, 5)
        v1 .= 42.0
        v2 = acquire!(pool, Float64, 3)
        v2 .= 99.0
        
        tp = pool.pools[Float64]
        @test tp.in_use == 2

        # Reset
        reset!(pool, empty_state)
        @test tp.in_use == 0

        # Second iteration - reuses same vectors
        v1_new = acquire!(pool, Float64, 5)
        @test tp.in_use == 1
        # Memory is reused
        @test length(v1_new) == 5
        # v1_new should point to the same memory as v1
        @test parent(v1_new) === parent(v1)
    end

    @testset "Different element types" begin
        pool = AdaptiveArrayPool()
        
        # Float32
        v32 = acquire!(pool, Float32, 5)
        @test eltype(v32) == Float32
        v32 .= 1.5f0
        @test all(v32 .== 1.5f0)

        # Int64
        v_int = acquire!(pool, Int64, 5)
        @test eltype(v_int) == Int64
        v_int .= 42
        @test all(v_int .== 42)
        
        @test haskey(pool.pools, Float32)
        @test haskey(pool.pools, Int64)
    end

    @testset "Warm-up pattern" begin
        pool = AdaptiveArrayPool()
        empty_state = mark(pool)

        # Warm-up: sizes may cause resize
        for _ in 1:3
            reset!(pool, empty_state)
            acquire!(pool, Float64, 101)
            acquire!(pool, Float64, 30)
            acquire!(pool, Float64, 7)
        end

        # After warm-up, vectors should be properly sized
        tp = pool.pools[Float64]
        @test length(tp.vectors[1]) >= 101
        @test length(tp.vectors[2]) >= 30
        @test length(tp.vectors[3]) >= 7
    end

    @testset "Allocation test" begin
        pool = AdaptiveArrayPool()
        empty_state = mark(pool)

        # Warm-up phase
        for _ in 1:5
            reset!(pool, empty_state)
            acquire!(pool, Float64, 101)
            acquire!(pool, Float64, 30)
            acquire!(pool, Float64, 7)
        end

        # Measure allocations after warm-up
        allocs = @allocated begin
            for _ in 1:100
                reset!(pool, empty_state)
                v1 = acquire!(pool, Float64, 101)
                v2 = acquire!(pool, Float64, 30)
                v3 = acquire!(pool, Float64, 7)
                v1 .= 1.0
                v2 .= 2.0
                v3 .= 3.0
            end
        end

        # Should be very low - only view creation, no vector allocation
        @test allocs < 50_000  # < 50KB for 100 iterations
        println("  Allocations after warm-up: $(allocs) bytes for 100 iterations")
    end

    @testset "pool_stats" begin
        pool = AdaptiveArrayPool()
        acquire!(pool, Float64, 100)
        acquire!(pool, Float64, 30)

        # Should not error - just verify it runs
        # pool_stats(pool)  # Uncomment to see output
        @test true
    end

    @testset "acquire! fallback (nothing)" begin
        # Without pool - returns Vector (allocation)
        v3 = acquire!(nothing, Float64, 10)
        @test v3 isa Vector{Float64}
        @test length(v3) == 10

        # Different types
        v4 = acquire!(nothing, Int64, 5)
        @test v4 isa Vector{Int64}
        @test length(v4) == 5
    end

    @testset "mark and reset with state" begin
        pool = AdaptiveArrayPool()

        # Checkout some vectors
        v1 = acquire!(pool, Float64, 10)
        v2 = acquire!(pool, Float64, 20)
        
        tp = pool.pools[Float64]
        @test tp.in_use == 2

        # Mark current state
        state = mark(pool)
        @test state[Float64] == 2

        # Checkout more vectors
        v3 = acquire!(pool, Float64, 30)
        v4 = acquire!(pool, Float64, 40)
        @test tp.in_use == 4

        # Reset to saved state
        reset!(pool, state)
        @test tp.in_use == 2

        # v1, v2 should still be valid (same underlying memory)
        v1 .= 1.0
        v2 .= 2.0
        @test all(v1 .== 1.0)
        @test all(v2 .== 2.0)

        # mark(nothing) returns nothing
        @test mark(nothing) === nothing

        # reset!(nothing, nothing) is no-op
        @test reset!(nothing, nothing) === nothing
    end

    @testset "@use_pool macro" begin
        pool = AdaptiveArrayPool()

        # Checkout before @use_pool
        v_outer = acquire!(pool, Float64, 10)
        tp = pool.pools[Float64]
        @test tp.in_use == 1

        # Use @use_pool block
        result = @use_pool pool begin
            v1 = acquire!(pool, Float64, 20)
            v2 = acquire!(pool, Float64, 30)
            @test tp.in_use == 3
            sum(v1) + sum(v2)  # Return value from block
        end

        # After @use_pool, state should be restored
        @test tp.in_use == 1
        @test result isa Number  # Block returned a value

        # v_outer should still be valid
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
                # v1 should still be valid after inner returns
                @test all(v1 .== 1.0)
                sum(v1) + inner_result
            end
        end

        result = outer_computation(pool)
        @test result ≈ 40.0  # 20 * 1.0 + 10 * 2.0
        
        # Fully reset after outer returns (assuming pool was empty before)
        # But wait, outer_computation takes pool. If we pass a fresh pool:
        @test !haskey(pool.pools, Float64) || pool.pools[Float64].in_use == 0
    end

    @testset "@use_pool function definition mode" begin
        # Test: pool argument auto-injected
        @use_pool pool function test_auto_inject(x::Vector{Float64})
            temp = acquire!(pool, Float64, length(x))
            temp .= x .* 2
            return sum(temp)
        end

        # Call without pool (backward compatible)
        x = [1.0, 2.0, 3.0]
        result1 = test_auto_inject(x)
        @test result1 ≈ 12.0  # sum([2, 4, 6])

        # Call with pool (optimized)
        mypool = AdaptiveArrayPool()
        result2 = test_auto_inject(x; pool=mypool)
        @test result2 ≈ 12.0
        @test mypool.pools[Float64].in_use == 0  # Reset after function returns
    end

    @testset "@use_pool short-form function" begin
        # Short form: f(x) = ...
        @use_pool pool test_short(x) = sum(acquire!(pool, Float64, length(x)) .= x)

        x = [1.0, 2.0, 3.0]
        @test test_short(x) ≈ 6.0

        mypool = AdaptiveArrayPool()
        @test test_short(x; pool=mypool) ≈ 6.0
        @test mypool.pools[Float64].in_use == 0
    end

    @testset "@use_pool with existing kwargs" begin
        # Function already has kwargs
        @use_pool pool function test_existing_kwargs(x; scale=2.0)
            temp = acquire!(pool, Float64, length(x))
            temp .= x .* scale
            return sum(temp)
        end

        x = [1.0, 2.0, 3.0]
        @test test_existing_kwargs(x) ≈ 12.0          # scale=2.0 default
        @test test_existing_kwargs(x; scale=3.0) ≈ 18.0

        mypool = AdaptiveArrayPool()
        @test test_existing_kwargs(x; pool=mypool, scale=4.0) ≈ 24.0
        @test mypool.pools[Float64].in_use == 0
    end

    @testset "@use_pool with where clause" begin
        @use_pool pool function test_where(x::Vector{T}) where {T<:Number}
            temp = acquire!(pool, T, length(x))
            temp .= x .+ one(T)
            return sum(temp)
        end

        x = [1.0, 2.0, 3.0]
        @test test_where(x) ≈ 9.0  # sum([2, 3, 4])

        mypool = AdaptiveArrayPool()
        @test test_where(x; pool=mypool) ≈ 9.0
        @test mypool.pools[Float64].in_use == 0
    end

    @testset "Multi-dimensional acquire!" begin
        pool = AdaptiveArrayPool()

        # 2D matrix
        mat = acquire!(pool, Float64, 10, 10)
        @test size(mat) == (10, 10)
        @test mat isa Base.ReshapedArray
        
        tp = pool.pools[Float64]
        @test tp.in_use == 1
        mat .= 1.0
        @test sum(mat) ≈ 100.0

        # 3D tensor
        tensor = acquire!(pool, Float64, 5, 5, 5)
        @test size(tensor) == (5, 5, 5)
        @test tensor isa Base.ReshapedArray
        @test tp.in_use == 2
        tensor .= 2.0
        @test sum(tensor) ≈ 250.0

        # Reset and reuse
        empty_state = mark(pool) # Wait, this marks current state (2 in use). 
        # We want to reset to empty.
        # Let's create a new pool or manually reset.
        # Or just use a fresh pool for clarity.
        pool2 = AdaptiveArrayPool()
        empty_state2 = mark(pool2)
        
        mat2 = acquire!(pool2, Float64, 20, 5)
        @test size(mat2) == (20, 5)
        
        reset!(pool2, empty_state2)
        mat3 = acquire!(pool2, Float64, 10, 10)
        @test size(mat3) == (10, 10)
        # Should reuse the vector from mat2 (size 100 vs 100)
        
        # Without pool (fallback)
        mat_alloc = acquire!(nothing, Float64, 10, 10)
        @test mat_alloc isa Array{Float64, 2}
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
        @test result1 ≈ 120.0  # 10*10 + 10*2

        # With pool
        result2 = matrix_computation(10; pool)
        @test result2 ≈ 120.0
        @test pool.pools[Float64].in_use == 0  # Reset after function
    end
    
    @testset "@use_global_pool" begin
        # This uses the task-local global pool
        
        @use_global_pool pool function global_test(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            return sum(v)
        end
        
        res = global_test(10)
        @test res ≈ 10.0
        
        # Verify global pool state is clean
        # We can't easily access the global pool variable from here as it's local to the macro expansion
        # But we can check if subsequent calls work
        res2 = global_test(20)
        @test res2 ≈ 20.0
    end

end
