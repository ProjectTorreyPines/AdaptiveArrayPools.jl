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

@testset "View cache hit and miss" begin
    pool = AdaptiveArrayPool()

    # First acquire - creates new slot
    mark!(pool)
    v1 = acquire!(pool, Float64, 100)
    @test length(v1) == 100
    reset!(pool)

    # Same size - cache hit (zero alloc after warmup)
    mark!(pool)
    v2 = acquire!(pool, Float64, 100)
    @test length(v2) == 100
    @test parent(v1) === parent(v2)  # Same backing vector
    reset!(pool)

    # Larger size - cache miss, needs resize
    mark!(pool)
    v3 = acquire!(pool, Float64, 200)
    @test length(v3) == 200
    @test length(parent(v3)) >= 200  # Backing vector was resized
    reset!(pool)

    # Smaller size - cache miss, but no resize needed
    mark!(pool)
    v4 = acquire!(pool, Float64, 50)
    @test length(v4) == 50
    @test length(parent(v4)) >= 200  # Backing vector still large
    reset!(pool)
end

@testset "Fallback types mark/reset" begin
    pool = AdaptiveArrayPool()

    # Use a fallback type (not in fixed slots)
    mark!(pool)
    v1 = acquire!(pool, UInt8, 100)
    v2 = acquire!(pool, UInt8, 50)
    @test pool.others[UInt8].in_use == 2
    reset!(pool)
    @test pool.others[UInt8].in_use == 0

    # Nested mark/reset with fallback
    mark!(pool)
    v1 = acquire!(pool, UInt8, 100)
    @test pool.others[UInt8].in_use == 1

    mark!(pool)  # Nested
    v2 = acquire!(pool, UInt8, 50)
    @test pool.others[UInt8].in_use == 2

    reset!(pool)
    @test pool.others[UInt8].in_use == 1

    reset!(pool)
    @test pool.others[UInt8].in_use == 0
end

@testset "Nothing fallback methods" begin
    # acquire! with nothing pool
    v1 = acquire!(nothing, Float64, 100)
    @test v1 isa Vector{Float64}
    @test length(v1) == 100

    # Multi-dimensional acquire! with nothing
    mat = acquire!(nothing, Float64, 10, 20)
    @test mat isa Array{Float64, 2}
    @test size(mat) == (10, 20)

    tensor = acquire!(nothing, Int32, 3, 4, 5)
    @test tensor isa Array{Int32, 3}
    @test size(tensor) == (3, 4, 5)

    # empty! with nothing
    @test empty!(nothing) === nothing
end

@testset "empty! pool clearing" begin
    import AdaptiveArrayPools: empty!

    pool = AdaptiveArrayPool()

    # Add vectors to fixed slots
    mark!(pool)
    v1 = acquire!(pool, Float64, 100)
    v2 = acquire!(pool, Float32, 50)
    v3 = acquire!(pool, Int64, 25)
    v4 = acquire!(pool, Int32, 10)
    v5 = acquire!(pool, ComplexF64, 5)
    v6 = acquire!(pool, Bool, 20)
    reset!(pool)

    # Add fallback type
    mark!(pool)
    v_uint8 = acquire!(pool, UInt8, 200)
    reset!(pool)

    # Verify pool has data
    @test length(pool.float64.vectors) == 1
    @test length(pool.float32.vectors) == 1
    @test length(pool.int64.vectors) == 1
    @test length(pool.int32.vectors) == 1
    @test length(pool.complexf64.vectors) == 1
    @test length(pool.bool.vectors) == 1
    @test haskey(pool.others, UInt8)

    # Clear the pool
    result = empty!(pool)
    @test result === pool  # Returns self

    # Verify all fixed slots are cleared
    @test isempty(pool.float64.vectors)
    @test isempty(pool.float64.views)
    @test isempty(pool.float64.view_lengths)
    @test pool.float64.in_use == 0
    @test isempty(pool.float64.saved_stack)

    @test isempty(pool.float32.vectors)
    @test isempty(pool.int64.vectors)
    @test isempty(pool.int32.vectors)
    @test isempty(pool.complexf64.vectors)
    @test isempty(pool.bool.vectors)

    # Verify fallback types are cleared
    @test isempty(pool.others)

    # Pool should still be usable after empty!
    mark!(pool)
    v_new = acquire!(pool, Float64, 50)
    @test length(v_new) == 50
    @test pool.float64.in_use == 1
    reset!(pool)
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
