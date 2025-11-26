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
    old_state = MAYBE_POOLING_ENABLED[]
    MAYBE_POOLING_ENABLED[] = true

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

    MAYBE_POOLING_ENABLED[] = old_state
end

@testset "@maybe_use_global_pool with pooling disabled" begin
    old_state = MAYBE_POOLING_ENABLED[]
    MAYBE_POOLING_ENABLED[] = false

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

    MAYBE_POOLING_ENABLED[] = old_state
end

@testset "@maybe_use_global_pool block mode" begin
    old_state = MAYBE_POOLING_ENABLED[]

    # Enabled
    MAYBE_POOLING_ENABLED[] = true
    result1 = @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 5)
        v .= 4.0
        sum(v)
    end
    @test result1 == 20.0

    # Disabled
    MAYBE_POOLING_ENABLED[] = false
    result2 = @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 5)
        v .= 4.0
        sum(v)
    end
    @test result2 == 20.0

    MAYBE_POOLING_ENABLED[] = old_state
end

# Function barrier for accurate allocation measurement
function test_zero_alloc_maybe_use_global_pool()
    # 1. Nothing
    a1 = @allocated @maybe_use_global_pool pool begin
        nothing
    end

    # 2. acquire! only
    a2 = @allocated @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 100)
        nothing
    end

    # 3. acquire! + fill
    a3 = @allocated @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 100)
        v .= 1.0
        nothing
    end

    # 4. @use_global_pool acquire! + fill
    a4 = @allocated @use_global_pool pool begin
        v = acquire!(pool, Float64, 100)
        v .= 1.0
        nothing
    end

    return (a1, a2, a3, a4)
end

function test_pooling_vs_no_pooling()
    # With pooling
    MAYBE_POOLING_ENABLED[] = true
    alloc_with = @allocated @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 100)
        v .= 1.0
        nothing
    end

    # Without pooling
    MAYBE_POOLING_ENABLED[] = false
    alloc_without = @allocated @maybe_use_global_pool pool begin
        v = acquire!(pool, Float64, 100)
        v .= 1.0
        nothing
    end

    return (alloc_with, alloc_without)
end

@testset "@maybe_use_global_pool zero-allocation" begin
    old_state = MAYBE_POOLING_ENABLED[]
    MAYBE_POOLING_ENABLED[] = true

    # Warm-up (compile)
    test_zero_alloc_maybe_use_global_pool()
    test_zero_alloc_maybe_use_global_pool()

    # Measure
    a1, a2, a3, a4 = test_zero_alloc_maybe_use_global_pool()

    println("  @maybe_use_global_pool nothing: $a1 bytes")
    println("  @maybe_use_global_pool acquire!: $a2 bytes")
    println("  @maybe_use_global_pool acquire! + fill: $a3 bytes")
    println("  @use_global_pool acquire! + fill: $a4 bytes")

    @test a1 == 0
    @test a2 == 0
    @test a3 == 0
    @test a4 == 0

    MAYBE_POOLING_ENABLED[] = old_state
end

@testset "@maybe_use_global_pool pooling vs no-pooling" begin
    old_state = MAYBE_POOLING_ENABLED[]

    # Warm-up
    MAYBE_POOLING_ENABLED[] = true
    test_pooling_vs_no_pooling()
    test_pooling_vs_no_pooling()

    alloc_with, alloc_without = test_pooling_vs_no_pooling()

    println("  Allocations with pooling: $alloc_with bytes")
    println("  Allocations without pooling: $alloc_without bytes")

    @test alloc_with < alloc_without

    MAYBE_POOLING_ENABLED[] = old_state
end
