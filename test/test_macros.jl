# Tests for @with_pool and @maybe_with_pool macros
import AdaptiveArrayPools: checkpoint!, rewind!

@testset "Macro System" begin

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

    @testset "@with_pool basic usage" begin
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            sum(v)
        end
        @test result == 10.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@with_pool 1-arg (no pool name)" begin
        # When you don't need the pool variable, use 1-arg form
        function inner_uses_global(n)
            pool = get_task_local_pool()
            v = acquire!(pool, Float64, n)
            v .= 2.0
            sum(v)
        end

        result = @with_pool begin
            inner_uses_global(5)
        end
        @test result == 10.0
        @test get_task_local_pool().float64.n_active == 0
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

    @testset "@with_pool with function passing pool" begin
        function compute_with_pool(x, pool)
            temp = acquire!(pool, Float64, length(x))
            temp .= x .* 2
            sum(temp)
        end

        x = [1.0, 2.0, 3.0]
        result = @with_pool pool begin
            compute_with_pool(x, pool)
        end
        @test result == 12.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@maybe_with_pool enabled" begin
        MAYBE_POOLING[] = true

        result = @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 3.0
            sum(v)
        end
        @test result == 30.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@maybe_with_pool disabled" begin
        MAYBE_POOLING[] = false

        result = @maybe_with_pool pool begin
            @test pool isa DisabledPool{:cpu}
            @test !pooling_enabled(pool)
            v = acquire!(pool, Float64, 10)  # Falls back to normal allocation
            @test v isa Vector{Float64}
            v .= 4.0
            sum(v)
        end
        @test result == 40.0

        # Reset
        MAYBE_POOLING[] = true
    end

    @testset "@maybe_with_pool 1-arg (no pool name)" begin
        MAYBE_POOLING[] = true

        result = @maybe_with_pool begin
            pool = get_task_local_pool()
            v = acquire!(pool, Float64, 5)
            v .= 1.0
            sum(v)
        end
        @test result == 5.0

        MAYBE_POOLING[] = false
        result2 = @maybe_with_pool begin
            # Pool is nothing, so we allocate normally
            v = Vector{Float64}(undef, 5)
            v .= 2.0
            sum(v)
        end
        @test result2 == 10.0

        # Reset
        MAYBE_POOLING[] = true
    end

    @testset "@with_pool function definition" begin
        @with_pool p1 function my_test_func(n)
            v = acquire!(p1, Float64, n)
            v .= 1.0
            sum(v)
        end

        # Check if it works
        res = my_test_func(10)
        @test res == 10.0

        # Check if pool is clean after call
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@with_pool short-form function definitions" begin
        # Simple short-form: f(x) = expr
        @with_pool p1 short_form_simple(n) = begin
            v = acquire!(p1, Float64, n)
            v .= 2.0
            sum(v)
        end

        @test short_form_simple(5) == 10.0
        @test get_task_local_pool().float64.n_active == 0

        # With type annotation: f(x)::T = expr
        @with_pool p2 short_form_typed(n)::Float64 = begin
            v = acquire!(p2, Float64, n)
            v .= 3.0
            sum(v)
        end

        @test short_form_typed(4) == 12.0
        @test get_task_local_pool().float64.n_active == 0

        # With type annotation on arguments
        @with_pool p3 short_form_arg_typed(x::Vector{Float64}) = begin
            v = acquire!(p3, Float64, length(x))
            v .= x .* 2
            sum(v)
        end

        @test short_form_arg_typed([1.0, 2.0, 3.0]) == 12.0
        @test get_task_local_pool().float64.n_active == 0

        # Combined: return type and argument types
        @with_pool p4 short_form_combined(x::Vector{Float64}, y::Vector{Float64})::Float64 = begin
            v1 = acquire!(p4, Float64, length(x))
            v2 = acquire!(p4, Float64, length(y))
            v1 .= x
            v2 .= y
            sum(v1) + sum(v2)
        end

        @test short_form_combined([1.0, 2.0], [3.0, 4.0]) == 10.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@maybe_with_pool short-form function definitions" begin
        MAYBE_POOLING[] = true

        # Simple short-form with @maybe_with_pool
        @maybe_with_pool p1 maybe_short_form(n) = begin
            v = acquire!(p1, Float64, n)
            v .= 5.0
            sum(v)
        end

        @test maybe_short_form(3) == 15.0
        @test get_task_local_pool().float64.n_active == 0

        # Test with pooling disabled
        MAYBE_POOLING[] = false

        @maybe_with_pool p2 maybe_short_disabled(n) = begin
            @test p2 isa DisabledPool{:cpu}
            @test !pooling_enabled(p2)
            v = acquire!(p2, Float64, n)  # Falls back to allocation
            v .= 1.0
            sum(v)
        end

        @test maybe_short_disabled(5) == 5.0

        # Reset
        MAYBE_POOLING[] = true
    end

    # ==============================================================================
    # Direct-rewind path tests (no try-finally)
    # ==============================================================================

    @testset "Direct rewind: explicit return in @with_pool function" begin
        @with_pool pool function early_return_test(flag)
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            if flag
                return sum(v)  # rewind should happen before return
            end
            v .= 2.0
            sum(v)
        end

        @test early_return_test(true) == 10.0
        @test early_return_test(false) == 20.0

        # Pool should be clean after both paths
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: break inside @with_pool block in loop" begin
        result = 0.0
        for i in 1:10
            @with_pool pool begin
                v = acquire!(pool, Float64, 5)
                v .= Float64(i)
                result = sum(v)
                if i == 3
                    break  # rewind should happen before break
                end
            end
        end

        @test result == 15.0  # 3 * 5
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: continue inside @with_pool block in loop" begin
        total = 0.0
        for i in 1:5
            @with_pool pool begin
                v = acquire!(pool, Float64, 3)
                v .= Float64(i)
                if i == 3
                    continue  # rewind should happen before continue
                end
                total += sum(v)
            end
        end

        # sum for i=1,2,4,5 → 3*(1+2+4+5) = 36
        @test total == 36.0
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: nested catch recovery (entry depth guard)" begin
        @with_pool pool function outer_catches()
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            result = try
                @with_pool pool begin
                    w = acquire!(pool, UInt8, 5)
                    error("boom")  # inner scope leaks
                end
            catch
                42
            end
            sum(v) + result
        end

        @test outer_catches() == 52.0  # 10.0 + 42
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "@safe_with_pool preserves try-finally behavior" begin
        reset!(get_task_local_pool())  # ensure clean state
        try
            @safe_with_pool pool begin
                acquire!(pool, Float64, 10)
                error("simulated failure")
            end
        catch
        end

        # try-finally guarantees cleanup even after exception
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "@safe_maybe_with_pool preserves try-finally behavior" begin
        reset!(get_task_local_pool())  # ensure clean state
        try
            @safe_maybe_with_pool pool begin
                acquire!(pool, Float64, 10)
                error("simulated failure")
            end
        catch
        end

        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

end # Macro System
