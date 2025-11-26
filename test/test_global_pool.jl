@testset "@use_global_pool" begin
    @use_global_pool pool function global_test(n)
        v = acquire!(pool, Float64, n)
        v .= 1.0
        return sum(v)
    end

    res = global_test(10)
    @test res == 10.0

    res2 = global_test(20)
    @test res2 == 20.0
end

@testset "@use_global_pool block mode" begin
    result = @use_global_pool pool begin
        v = acquire!(pool, Float64, 15)
        v .= 3.0
        sum(v)
    end
    @test result == 45.0
end

@testset "@maybe_use_global_pool with pooling enabled" begin
    # Ensure pooling is enabled
    old_state = ENABLE_POOLING[]
    ENABLE_POOLING[] = true

    @maybe_use_global_pool pool function maybe_test_enabled(n)
        v = acquire!(pool, Float64, n)
        v .= 2.0
        return sum(v)
    end

    res = maybe_test_enabled(10)
    @test res == 20.0

    # Verify pool was used (pool is not nothing)
    result_type = @maybe_use_global_pool pool begin
        pool !== nothing
    end
    @test result_type == true

    ENABLE_POOLING[] = old_state
end

@testset "@maybe_use_global_pool with pooling disabled" begin
    old_state = ENABLE_POOLING[]
    ENABLE_POOLING[] = false

    @maybe_use_global_pool pool function maybe_test_disabled(n)
        v = acquire!(pool, Float64, n)
        v .= 2.0
        return sum(v)
    end

    res = maybe_test_disabled(10)
    @test res == 20.0

    # Verify pool was nothing (fallback allocation used)
    result_type = @maybe_use_global_pool pool begin
        pool === nothing
    end
    @test result_type == true

    ENABLE_POOLING[] = old_state
end

@testset "@maybe_use_global_pool block mode" begin
    old_state = ENABLE_POOLING[]

    # Enabled
    ENABLE_POOLING[] = true
    result1 = @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 5)
        v .= 4.0
        sum(v)
    end
    @test result1 == 20.0

    # Disabled
    ENABLE_POOLING[] = false
    result2 = @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 5)
        v .= 4.0
        sum(v)
    end
    @test result2 == 20.0

    ENABLE_POOLING[] = old_state
end

@testset "@maybe_use_global_pool allocation comparison" begin
    old_state = ENABLE_POOLING[]

    # With pooling enabled - should have minimal allocations after warm-up
    ENABLE_POOLING[] = true

    Nvec = 100

    # Warm-up
    for _ in 1:3
        @maybe_use_global_pool pool begin
            v = acquire!(pool, Float64, Nvec)
            v .= 1.0
            nothing
        end
    end

    allocs_enabled = @allocated begin
        for _ in 1:100
            @maybe_use_global_pool pool begin
                v = acquire!(pool, Float64, Nvec)
                v .= 1.0
            end
        end
    end

    # With pooling disabled - should allocate every time
    ENABLE_POOLING[] = false

    allocs_disabled = @allocated begin
        for _ in 1:100
            @maybe_use_global_pool pool begin
                v = acquire!(pool, Float64, Nvec)
                v .= 1.0
            end
        end
    end

    # Pooling should use significantly less memory
    @test allocs_enabled < allocs_disabled
    println("  Allocations with pooling: $(allocs_enabled) bytes")
    println("  Allocations without pooling: $(allocs_disabled) bytes")

    ENABLE_POOLING[] = old_state
end
