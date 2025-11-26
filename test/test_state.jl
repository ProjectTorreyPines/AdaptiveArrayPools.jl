@testset "Reset and reuse" begin
    pool = AdaptiveArrayPool()
    mark!(pool)  # v2: no return value

    # First iteration
    v1 = acquire!(pool, Float64, 5)
    v1 .= 42.0
    v2 = acquire!(pool, Float64, 3)
    v2 .= 99.0

    @test pool.float64.in_use == 2

    # Reset
    reset!(pool)  # v2: no state argument
    @test pool.float64.in_use == 0

    # Second iteration - reuses same vectors
    v1_new = acquire!(pool, Float64, 5)
    @test pool.float64.in_use == 1
    @test length(v1_new) == 5
    @test parent(v1_new) === parent(v1)
end

@testset "Warm-up pattern" begin
    pool = AdaptiveArrayPool()
    mark!(pool)

    # Warm-up: sizes may cause resize
    for _ in 1:3
        reset!(pool)
        mark!(pool)
        acquire!(pool, Float64, 101)
        acquire!(pool, Float64, 30)
        acquire!(pool, Float64, 7)
    end
    reset!(pool)

    # After warm-up, vectors should be properly sized
    @test length(pool.float64.vectors[1]) >= 101
    @test length(pool.float64.vectors[2]) >= 30
    @test length(pool.float64.vectors[3]) >= 7
end

@testset "mark and reset v2 API" begin
    pool = AdaptiveArrayPool()

    v1 = acquire!(pool, Float64, 10)
    v2 = acquire!(pool, Float64, 20)
    @test pool.float64.in_use == 2

    mark!(pool)  # Save state: in_use = 2

    v3 = acquire!(pool, Float64, 30)
    v4 = acquire!(pool, Float64, 40)
    @test pool.float64.in_use == 4

    reset!(pool)  # Restore to in_use = 2
    @test pool.float64.in_use == 2

    v1 .= 1.0
    v2 .= 2.0
    @test all(v1 .== 1.0)
    @test all(v2 .== 2.0)

    # nothing compatibility
    @test mark!(nothing) === nothing
    @test reset!(nothing) === nothing
end

@testset "Nested mark/reset" begin
    pool = AdaptiveArrayPool()

    mark!(pool)  # Level 1: in_use = 0
    v1 = acquire!(pool, Float64, 10)
    @test pool.float64.in_use == 1

    mark!(pool)  # Level 2: in_use = 1
    v2 = acquire!(pool, Float64, 20)
    @test pool.float64.in_use == 2

    mark!(pool)  # Level 3: in_use = 2
    v3 = acquire!(pool, Float64, 30)
    @test pool.float64.in_use == 3

    reset!(pool)  # Back to Level 2
    @test pool.float64.in_use == 2

    reset!(pool)  # Back to Level 1
    @test pool.float64.in_use == 1

    reset!(pool)  # Back to Level 0
    @test pool.float64.in_use == 0
end

@testset "Edge case: new type after mark" begin
    pool = AdaptiveArrayPool()

    mark!(pool)  # UInt16 doesn't exist yet

    # Add new type after mark
    v = acquire!(pool, UInt16, 10)
    @test pool.others[UInt16].in_use == 1

    reset!(pool)  # Should set UInt16.in_use = 0 (empty stack case)
    @test pool.others[UInt16].in_use == 0
end

@testset "Allocation test (Zero Alloc)" begin
    pool = AdaptiveArrayPool()

    # Warm-up phase - allocates saved_stack capacity
    for _ in 1:5
        mark!(pool)
        acquire!(pool, Float64, 101)
        acquire!(pool, Float64, 30)
        acquire!(pool, Float64, 7)
        reset!(pool)
    end

    # Measure allocations after warm-up
    allocs = @allocated begin
        for _ in 1:100
            mark!(pool)
            v1 = acquire!(pool, Float64, 101)
            v2 = acquire!(pool, Float64, 30)
            v3 = acquire!(pool, Float64, 7)
            v1 .= 1.0
            v2 .= 2.0
            v3 .= 3.0
            reset!(pool)
        end
    end

    # Should be very low - no IdDict allocation anymore
    # (some overhead from view allocation is expected)
    @test allocs < 100_000
    println("  Allocations after warm-up (v2): $(allocs) bytes for 100 iterations")
end
