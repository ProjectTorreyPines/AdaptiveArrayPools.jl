@testset "Rewind and reuse" begin
    pool = AdaptiveArrayPool()
    checkpoint!(pool)

    # First iteration
    v1 = acquire!(pool, Float64, 5)
    v1 .= 42.0
    v2 = acquire!(pool, Float64, 3)
    v2 .= 99.0

    @test pool.float64.n_active == 2

    # Rewind
    rewind!(pool)
    @test pool.float64.n_active == 0

    # Second iteration - reuses same vectors
    v1_new = acquire!(pool, Float64, 5)
    @test pool.float64.n_active == 1
    @test length(v1_new) == 5
    @test parent(v1_new) === parent(v1)
end

@testset "Warm-up pattern" begin
    pool = AdaptiveArrayPool()
    checkpoint!(pool)

    # Warm-up: sizes may cause resize
    for _ in 1:3
        rewind!(pool)
        checkpoint!(pool)
        acquire!(pool, Float64, 101)
        acquire!(pool, Float64, 30)
        acquire!(pool, Float64, 7)
    end
    rewind!(pool)

    # After warm-up, vectors should be properly sized
    @test length(pool.float64.vectors[1]) >= 101
    @test length(pool.float64.vectors[2]) >= 30
    @test length(pool.float64.vectors[3]) >= 7
end

@testset "checkpoint and rewind API" begin
    pool = AdaptiveArrayPool()

    v1 = acquire!(pool, Float64, 10)
    v2 = acquire!(pool, Float64, 20)
    @test pool.float64.n_active == 2

    checkpoint!(pool)  # Save state: n_active = 2

    v3 = acquire!(pool, Float64, 30)
    v4 = acquire!(pool, Float64, 40)
    @test pool.float64.n_active == 4

    rewind!(pool)  # Restore to n_active = 2
    @test pool.float64.n_active == 2

    v1 .= 1.0
    v2 .= 2.0
    @test all(v1 .== 1.0)
    @test all(v2 .== 2.0)

    # nothing compatibility
    @test checkpoint!(nothing) === nothing
    @test rewind!(nothing) === nothing
end

@testset "Nested checkpoint/rewind" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)  # Level 1: n_active = 0
    v1 = acquire!(pool, Float64, 10)
    @test pool.float64.n_active == 1

    checkpoint!(pool)  # Level 2: n_active = 1
    v2 = acquire!(pool, Float64, 20)
    @test pool.float64.n_active == 2

    checkpoint!(pool)  # Level 3: n_active = 2
    v3 = acquire!(pool, Float64, 30)
    @test pool.float64.n_active == 3

    rewind!(pool)  # Back to Level 2
    @test pool.float64.n_active == 2

    rewind!(pool)  # Back to Level 1
    @test pool.float64.n_active == 1

    rewind!(pool)  # Back to Level 0
    @test pool.float64.n_active == 0
end

@testset "Edge case: new type after checkpoint" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)  # UInt16 doesn't exist yet

    # Add new type after checkpoint
    v = acquire!(pool, UInt16, 10)
    @test pool.others[UInt16].n_active == 1

    rewind!(pool)  # Sets UInt16.n_active = 0 (empty stack case)
    @test pool.others[UInt16].n_active == 0
end

@testset "View cache hit and miss" begin
    pool = AdaptiveArrayPool()

    # First acquire - creates new slot
    checkpoint!(pool)
    v1 = acquire!(pool, Float64, 100)
    @test length(v1) == 100
    rewind!(pool)

    # Same size - cache hit (zero alloc after warmup)
    checkpoint!(pool)
    v2 = acquire!(pool, Float64, 100)
    @test length(v2) == 100
    @test parent(v1) === parent(v2)  # Same backing vector
    rewind!(pool)

    # Larger size - cache miss, needs resize
    checkpoint!(pool)
    v3 = acquire!(pool, Float64, 200)
    @test length(v3) == 200
    @test length(parent(v3)) >= 200  # Backing vector was resized
    rewind!(pool)

    # Smaller size - cache miss, but no resize needed
    checkpoint!(pool)
    v4 = acquire!(pool, Float64, 50)
    @test length(v4) == 50
    @test length(parent(v4)) >= 200  # Backing vector still large
    rewind!(pool)
end

@testset "Fallback types checkpoint/rewind" begin
    pool = AdaptiveArrayPool()

    # Use a fallback type (not in fixed slots)
    checkpoint!(pool)
    v1 = acquire!(pool, UInt8, 100)
    v2 = acquire!(pool, UInt8, 50)
    @test pool.others[UInt8].n_active == 2
    rewind!(pool)
    @test pool.others[UInt8].n_active == 0

    # Nested checkpoint/rewind with fallback
    checkpoint!(pool)
    v1 = acquire!(pool, UInt8, 100)
    @test pool.others[UInt8].n_active == 1

    checkpoint!(pool)  # Nested level 2
    v2 = acquire!(pool, UInt8, 50)
    @test pool.others[UInt8].n_active == 2

    rewind!(pool)
    @test pool.others[UInt8].n_active == 1

    rewind!(pool)
    @test pool.others[UInt8].n_active == 0
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
    checkpoint!(pool)
    v1 = acquire!(pool, Float64, 100)
    v2 = acquire!(pool, Float32, 50)
    v3 = acquire!(pool, Int64, 25)
    v4 = acquire!(pool, Int32, 10)
    v5 = acquire!(pool, ComplexF64, 5)
    v6 = acquire!(pool, Bool, 20)
    rewind!(pool)

    # Add fallback type
    checkpoint!(pool)
    v_uint8 = acquire!(pool, UInt8, 200)
    rewind!(pool)

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
    @test pool.float64.n_active == 0
    @test isempty(pool.float64.saved_stack)

    @test isempty(pool.float32.vectors)
    @test isempty(pool.int64.vectors)
    @test isempty(pool.int32.vectors)
    @test isempty(pool.complexf64.vectors)
    @test isempty(pool.bool.vectors)

    # Verify fallback types are cleared
    @test isempty(pool.others)

    # Pool should still be usable after empty!
    checkpoint!(pool)
    v_new = acquire!(pool, Float64, 50)
    @test length(v_new) == 50
    @test pool.float64.n_active == 1
    rewind!(pool)
end

@testset "Typed checkpoint!/rewind! (generated functions)" begin
    pool = AdaptiveArrayPool()

    # Single type - checkpoint! and rewind!
    checkpoint!(pool, Float64)
    v1 = acquire!(pool, Float64, 10)
    @test pool.float64.n_active == 1
    rewind!(pool, Float64)
    @test pool.float64.n_active == 0

    # Multiple types - checkpoint! and rewind!
    checkpoint!(pool, Float64, Int64)
    v_f64 = acquire!(pool, Float64, 10)
    v_i64 = acquire!(pool, Int64, 5)
    @test pool.float64.n_active == 1
    @test pool.int64.n_active == 1
    rewind!(pool, Float64, Int64)
    @test pool.float64.n_active == 0
    @test pool.int64.n_active == 0

    # Three types
    checkpoint!(pool, Float64, Int64, Float32)
    v1 = acquire!(pool, Float64, 10)
    v2 = acquire!(pool, Int64, 5)
    v3 = acquire!(pool, Float32, 3)
    @test pool.float64.n_active == 1
    @test pool.int64.n_active == 1
    @test pool.float32.n_active == 1
    rewind!(pool, Float64, Int64, Float32)
    @test pool.float64.n_active == 0
    @test pool.int64.n_active == 0
    @test pool.float32.n_active == 0

    # All fixed types
    checkpoint!(pool, Float64, Float32, Int64, Int32, ComplexF64, Bool)
    acquire!(pool, Float64, 10)
    acquire!(pool, Float32, 10)
    acquire!(pool, Int64, 10)
    acquire!(pool, Int32, 10)
    acquire!(pool, ComplexF64, 10)
    acquire!(pool, Bool, 10)
    @test pool.float64.n_active == 1
    @test pool.float32.n_active == 1
    @test pool.int64.n_active == 1
    @test pool.int32.n_active == 1
    @test pool.complexf64.n_active == 1
    @test pool.bool.n_active == 1
    rewind!(pool, Float64, Float32, Int64, Int32, ComplexF64, Bool)
    @test pool.float64.n_active == 0
    @test pool.float32.n_active == 0
    @test pool.int64.n_active == 0
    @test pool.int32.n_active == 0
    @test pool.complexf64.n_active == 0
    @test pool.bool.n_active == 0

    # nothing fallback with types
    @test checkpoint!(nothing, Float64) === nothing
    @test checkpoint!(nothing, Float64, Int64) === nothing
    @test rewind!(nothing, Float64) === nothing
    @test rewind!(nothing, Float64, Int64) === nothing
end

@testset "Direct TypedPool checkpoint!/rewind!" begin
    import AdaptiveArrayPools: get_typed_pool!
    pool = AdaptiveArrayPool()

    # Get TypedPool directly
    tp = get_typed_pool!(pool, Float64)
    @test tp.n_active == 0

    # Direct TypedPool checkpoint and rewind
    checkpoint!(tp)
    v1 = acquire!(pool, Float64, 100)
    @test tp.n_active == 1
    v2 = acquire!(pool, Float64, 200)
    @test tp.n_active == 2
    rewind!(tp)
    @test tp.n_active == 0

    # Nested checkpoint/rewind on TypedPool
    checkpoint!(tp)
    v1 = acquire!(pool, Float64, 10)
    @test tp.n_active == 1

    checkpoint!(tp)
    v2 = acquire!(pool, Float64, 20)
    @test tp.n_active == 2

    checkpoint!(tp)
    v3 = acquire!(pool, Float64, 30)
    @test tp.n_active == 3

    rewind!(tp)
    @test tp.n_active == 2

    rewind!(tp)
    @test tp.n_active == 1

    rewind!(tp)
    @test tp.n_active == 0

    # Verify type-specific checkpoint delegates to TypedPool
    # (This tests the refactored implementation)
    checkpoint!(pool, Float64)
    v = acquire!(pool, Float64, 50)
    @test tp.n_active == 1
    rewind!(pool, Float64)
    @test tp.n_active == 0
end

@testset "Allocation test (Zero Alloc)" begin
    pool = AdaptiveArrayPool()

    # Warm-up phase - allocates saved_stack capacity
    for _ in 1:5
        checkpoint!(pool)
        acquire!(pool, Float64, 101)
        acquire!(pool, Float64, 30)
        acquire!(pool, Float64, 7)
        rewind!(pool)
    end

    # Measure allocations after warm-up
    allocs = @allocated begin
        for _ in 1:100
            checkpoint!(pool)
            v1 = acquire!(pool, Float64, 101)
            v2 = acquire!(pool, Float64, 30)
            v3 = acquire!(pool, Float64, 7)
            v1 .= 1.0
            v2 .= 2.0
            v3 .= 3.0
            rewind!(pool)
        end
    end

    # Should be very low - no IdDict allocation anymore
    # (some overhead from view allocation is expected)
    @test allocs < 100_000
    println("  Allocations after warm-up: $(allocs) bytes for 100 iterations")
end
