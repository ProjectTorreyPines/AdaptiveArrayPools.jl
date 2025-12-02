@testset "Task-local Pool" begin

    _test_accumulator = Ref(0.0)

    @testset "@with_pool" begin
        # Define a function that takes pool as argument
        function global_test(n, pool)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            return sum(v)
        end

        res = @with_pool pool begin
            global_test(10, pool)
        end
        @test res == 10.0

        res2 = @with_pool pool begin
            global_test(20, pool)
        end
        @test res2 == 20.0
    end

    @testset "@with_pool block mode" begin
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 15)
            v .= 3.0
            sum(v)
        end
        @test result == 45.0
    end

    @testset "@maybe_with_pool with pooling enabled" begin
        # Ensure pooling is enabled
        old_state = MAYBE_POOLING_ENABLED[]
        MAYBE_POOLING_ENABLED[] = true

        function maybe_test_enabled(n, pool)
            v = acquire!(pool, Float64, n)
            v .= 2.0
            return sum(v)
        end

        res = @maybe_with_pool pool begin
            maybe_test_enabled(10, pool)
        end
        @test res == 20.0

        # Verify pool was used (pool is not nothing)
        result_type = @maybe_with_pool pool begin
            pool !== nothing
        end
        @test result_type == true

        MAYBE_POOLING_ENABLED[] = old_state
    end

    @testset "@maybe_with_pool with pooling disabled" begin
        old_state = MAYBE_POOLING_ENABLED[]
        MAYBE_POOLING_ENABLED[] = false

        function maybe_test_disabled(n, pool)
            v = acquire!(pool, Float64, n)
            v .= 2.0
            return sum(v)
        end

        res = @maybe_with_pool pool begin
            maybe_test_disabled(10, pool)
        end
        @test res == 20.0

        # Verify pool was nothing (fallback allocation used)
        result_type = @maybe_with_pool pool begin
            pool === nothing
        end
        @test result_type == true

        MAYBE_POOLING_ENABLED[] = old_state
    end

    @testset "@maybe_with_pool block mode" begin
        old_state = MAYBE_POOLING_ENABLED[]

        # Enabled
        MAYBE_POOLING_ENABLED[] = true
        result1 = @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 5)
            v .= 4.0
            sum(v)
        end
        @test result1 == 20.0

        # Disabled
        MAYBE_POOLING_ENABLED[] = false
        result2 = @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 5)
            v .= 4.0
            sum(v)
        end
        @test result2 == 20.0

        MAYBE_POOLING_ENABLED[] = old_state
    end

    # Function barrier for accurate allocation measurement
    function test_zero_alloc_maybe_with_pool()
        # 1. Nothing
        a1 = @allocated @maybe_with_pool pool begin
            nothing
        end

        # 2. acquire! only
        a2 = @allocated @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 100)
            nothing
        end

        # 3. acquire! + fill
        a3 = @allocated @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 100)
            v .= 1.0
            nothing
        end

        # 4. @with_pool acquire! + fill
        a4 = @allocated @with_pool pool begin
            v = acquire!(pool, Float64, 100)
            v .= 1.0
            nothing
        end

        return (a1, a2, a3, a4)
    end

    # Accumulator to prevent compiler from optimizing away allocations

    # Test runtime toggle within a single function
    function test_runtime_toggle()
        # With pooling enabled
        MAYBE_POOLING_ENABLED[] = true
        alloc_with = @allocated begin
            r = @maybe_with_pool pool begin
                v = acquire!(pool, Float64, 100)
                v .= 1.0
                sum(v)
            end
            _test_accumulator[] += r
        end

        # With pooling disabled (toggle in same function!)
        MAYBE_POOLING_ENABLED[] = false
        alloc_without = @allocated begin
            r = @maybe_with_pool pool begin
                v = acquire!(pool, Float64, 100)
                v .= 1.0
                sum(v)
            end
            _test_accumulator[] += r
        end

        return (alloc_with, alloc_without)
    end

    @testset "@maybe_with_pool zero-allocation" begin
        old_state = MAYBE_POOLING_ENABLED[]
        MAYBE_POOLING_ENABLED[] = true

        # Warm-up (compile)
        test_zero_alloc_maybe_with_pool()
        test_zero_alloc_maybe_with_pool()

        # Measure
        a1, a2, a3, a4 = test_zero_alloc_maybe_with_pool()

        println("  @maybe_with_pool nothing: $a1 bytes")
        println("  @maybe_with_pool acquire!: $a2 bytes")
        println("  @maybe_with_pool acquire! + fill: $a3 bytes")
        println("  @with_pool acquire! + fill: $a4 bytes")

        @test a1 == 0
        @test a2 == 0
        @test a3 == 0
        @test a4 == 0

        MAYBE_POOLING_ENABLED[] = old_state
    end

    @testset "@maybe_with_pool pooling vs no-pooling" begin
        old_state = MAYBE_POOLING_ENABLED[]

        # Warm-up (compile the function, warm up pool)
        MAYBE_POOLING_ENABLED[] = true
        test_runtime_toggle()
        test_runtime_toggle()

        # Measure - tests runtime toggle within single function
        alloc_with, alloc_without = test_runtime_toggle()

        println("  Allocations with pooling: $alloc_with bytes")
        println("  Allocations without pooling: $alloc_without bytes")

        # Pooling should allocate less than normal allocation
        # Without pooling: Vector{Float64}(undef, 100) = 800+ bytes
        # With pooling: 0 bytes (after warm-up)
        @test alloc_with < alloc_without

        MAYBE_POOLING_ENABLED[] = old_state
    end

    @testset "Pool growth warning at 512 arrays" begin
        # Use a fresh pool to ensure we start from 0
        pool = AdaptiveArrayPool()

        # Acquire 511 arrays without rewind - no warning yet
        for i in 1:511
            acquire!(pool, Float64, 10)
        end
        @test pool.float64.n_active == 511

        # The 512th acquire should trigger a warning
        @test_logs (:warn, r"TypedPool\{Float64\} growing large \(512 arrays.*Missing rewind") begin
            acquire!(pool, Float64, 10)
        end
        @test pool.float64.n_active == 512

        # Clean up
        empty!(pool)
    end

end # Task-local Pool