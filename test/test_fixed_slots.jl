@testset "Fixed Slot Infrastructure" begin
    using AdaptiveArrayPools: FIXED_SLOT_FIELDS, foreach_fixed_slot, TypedPool

    @testset "FIXED_SLOT_FIELDS Synchronization" begin
        pool = AdaptiveArrayPool()

        # Forward check: all FIXED_SLOT_FIELDS exist in struct as TypedPool
        for field in FIXED_SLOT_FIELDS
            @test hasfield(AdaptiveArrayPool, field)
            @test fieldtype(AdaptiveArrayPool, field) <: TypedPool
            @test getfield(pool, field) isa TypedPool
        end

        # Reverse check: all TypedPool fields in struct are in FIXED_SLOT_FIELDS
        for (name, type) in zip(fieldnames(AdaptiveArrayPool), fieldtypes(AdaptiveArrayPool))
            if type <: TypedPool
                @test name in FIXED_SLOT_FIELDS
            end
        end

        # Count verification
        typedpool_count = count(t -> t <: TypedPool, fieldtypes(AdaptiveArrayPool))
        @test typedpool_count == length(FIXED_SLOT_FIELDS)
    end

    @testset "foreach_fixed_slot Iteration" begin
        pool = AdaptiveArrayPool()

        # Count iterations
        count_ref = Ref(0)
        foreach_fixed_slot(pool) do tp
            count_ref[] += 1
        end
        @test count_ref[] == length(FIXED_SLOT_FIELDS)

        # Verify all pools are visited
        visited = Set{Symbol}()
        foreach_fixed_slot(pool) do tp
            for field in FIXED_SLOT_FIELDS
                if getfield(pool, field) === tp
                    push!(visited, field)
                    break
                end
            end
        end
        @test visited == Set(FIXED_SLOT_FIELDS)
    end

    @testset "foreach_fixed_slot Zero Allocation" begin
        pool = AdaptiveArrayPool()

        # Warmup with identity
        foreach_fixed_slot(identity, pool)

        # Test zero allocation with identity
        allocs = @allocated foreach_fixed_slot(identity, pool)
        @test allocs == 0

        # Note: Testing with closures requires reusing the SAME closure instance
        # because each `do ... end` block creates a new anonymous function type.
        # The real zero-allocation behavior is verified by test_zero_allocation.jl
        # which tests checkpoint!/rewind! (which use foreach_fixed_slot internally).
    end

    @testset "foreach_fixed_slot Type Stability" begin
        pool = AdaptiveArrayPool()

        # Test that foreach_fixed_slot returns nothing (type stable)
        result = @inferred foreach_fixed_slot(identity, pool)
        @test result === nothing
    end

    @testset "Integration: checkpoint!/rewind! Use foreach_fixed_slot" begin
        pool = AdaptiveArrayPool()

        # Acquire some arrays
        v1 = acquire!(pool, Float64, 10)
        v2 = acquire!(pool, Int64, 5)

        # Checkpoint
        checkpoint!(pool)

        # Verify checkpoint worked
        @test pool.float64.n_active == 1
        @test pool.int64.n_active == 1

        # Acquire more
        v3 = acquire!(pool, Float64, 20)
        @test pool.float64.n_active == 2

        # Rewind
        rewind!(pool)

        # Verify rewind worked
        @test pool.float64.n_active == 1
        @test pool.int64.n_active == 1
    end

    @testset "Integration: empty! Uses foreach_fixed_slot" begin
        pool = AdaptiveArrayPool()

        # Acquire arrays of different types
        acquire!(pool, Float64, 10)
        acquire!(pool, Float32, 10)
        acquire!(pool, Int64, 10)
        acquire!(pool, Int32, 10)
        acquire!(pool, ComplexF64, 10)
        acquire!(pool, Bool, 10)

        # Verify all pools have active arrays
        for field in FIXED_SLOT_FIELDS
            tp = getfield(pool, field)
            @test tp.n_active == 1
        end

        # Empty
        empty!(pool)

        # Verify all pools are empty
        for field in FIXED_SLOT_FIELDS
            tp = getfield(pool, field)
            @test tp.n_active == 0
            @test isempty(tp.vectors)
        end
    end
end
