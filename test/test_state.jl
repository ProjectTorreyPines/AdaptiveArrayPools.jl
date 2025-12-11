@testset "State Management" begin

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
        v6 = acquire!(pool, ComplexF32, 5)
        v7 = acquire!(pool, Bool, 20)
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
        @test length(pool.complexf32.vectors) == 1
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
        @test length(pool.float64._checkpoint_n_active) == 1  # Only sentinel remains

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

    @testset "reset! (state-only reset)" begin
        import AdaptiveArrayPools: reset!

        @testset "basic reset! - n_active to zero" begin
            pool = AdaptiveArrayPool()

            # Acquire some arrays
            v1 = acquire!(pool, Float64, 100)
            v2 = acquire!(pool, Float32, 50)
            v3 = acquire!(pool, Int64, 30)
            @test pool.float64.n_active == 1
            @test pool.float32.n_active == 1
            @test pool.int64.n_active == 1

            # Reset
            result = reset!(pool)
            @test result === pool  # Returns self

            # All n_active should be 0
            @test pool.float64.n_active == 0
            @test pool.float32.n_active == 0
            @test pool.int64.n_active == 0
        end

        @testset "reset! preserves vectors" begin
            pool = AdaptiveArrayPool()

            # Acquire arrays
            v1 = acquire!(pool, Float64, 100)
            v2 = acquire!(pool, Float64, 200)
            v3 = acquire!(pool, Float64, 50)
            @test length(pool.float64.vectors) == 3

            # Reset - should preserve vectors
            reset!(pool)
            @test pool.float64.n_active == 0
            @test length(pool.float64.vectors) == 3  # Vectors preserved!
            @test length(pool.float64.vectors[1]) >= 100
            @test length(pool.float64.vectors[2]) >= 200
            @test length(pool.float64.vectors[3]) >= 50
        end

        @testset "reset! restores checkpoint stacks to sentinel" begin
            pool = AdaptiveArrayPool()

            # Nested checkpoints
            checkpoint!(pool)
            acquire!(pool, Float64, 10)
            checkpoint!(pool)
            acquire!(pool, Float64, 20)
            checkpoint!(pool)
            acquire!(pool, Float64, 30)

            @test pool._current_depth == 4
            @test length(pool.float64._checkpoint_n_active) > 1
            @test length(pool.float64._checkpoint_depths) > 1

            # Reset - should restore sentinel state
            reset!(pool)

            @test pool._current_depth == 1
            @test pool._untracked_flags == [false]
            @test pool.float64._checkpoint_n_active == [0]  # Sentinel only
            @test pool.float64._checkpoint_depths == [0]    # Sentinel only
        end

        @testset "reset! with fallback types" begin
            pool = AdaptiveArrayPool()

            # Use fallback type (not in fixed slots)
            v1 = acquire!(pool, UInt8, 100)
            v2 = acquire!(pool, UInt16, 50)
            @test pool.others[UInt8].n_active == 1
            @test pool.others[UInt16].n_active == 1
            @test length(pool.others[UInt8].vectors) == 1

            # Reset
            reset!(pool)

            # n_active reset but vectors preserved
            @test pool.others[UInt8].n_active == 0
            @test pool.others[UInt16].n_active == 0
            @test length(pool.others[UInt8].vectors) == 1  # Preserved!
            @test length(pool.others[UInt16].vectors) == 1
        end

        @testset "reset!(nothing) compatibility" begin
            @test reset!(nothing) === nothing
        end

        @testset "pool usable after reset!" begin
            pool = AdaptiveArrayPool()

            # First use
            v1 = acquire!(pool, Float64, 100)
            v1 .= 42.0
            backing1 = parent(v1)

            # Reset
            reset!(pool)

            # Should be usable and reuse existing vector
            checkpoint!(pool)
            v2 = acquire!(pool, Float64, 100)
            @test parent(v2) === backing1  # Same backing vector reused
            @test pool.float64.n_active == 1
            rewind!(pool)
            @test pool.float64.n_active == 0
        end

        @testset "A/B scenario - unmanaged then reset" begin
            # Simulates: inner function acquires without management,
            # outer function calls reset! to clean up

            pool = AdaptiveArrayPool()

            # Function that acquires without checkpoint/rewind
            function unmanaged_compute!(p)
                v = acquire!(p, Float64, 100)
                v .= 1.0
                # No rewind!
            end

            # Call multiple times - n_active grows
            for _ in 1:10
                unmanaged_compute!(pool)
            end
            @test pool.float64.n_active == 10
            @test length(pool.float64.vectors) == 10

            # Reset - clean slate but vectors preserved
            reset!(pool)
            @test pool.float64.n_active == 0
            @test length(pool.float64.vectors) == 10  # All preserved for reuse

            # Next use reuses existing vectors
            checkpoint!(pool)
            for _ in 1:5
                acquire!(pool, Float64, 100)
            end
            @test pool.float64.n_active == 5
            @test length(pool.float64.vectors) == 10  # No new allocations
            rewind!(pool)
        end

        @testset "reset! vs empty! comparison" begin
            # Verify reset! preserves while empty! clears

            pool1 = AdaptiveArrayPool()
            pool2 = AdaptiveArrayPool()

            # Both acquire same arrays
            for _ in 1:5
                acquire!(pool1, Float64, 100)
                acquire!(pool2, Float64, 100)
            end
            @test length(pool1.float64.vectors) == 5
            @test length(pool2.float64.vectors) == 5

            # reset! preserves
            reset!(pool1)
            @test pool1.float64.n_active == 0
            @test length(pool1.float64.vectors) == 5  # Preserved

            # empty! clears
            empty!(pool2)
            @test pool2.float64.n_active == 0
            @test length(pool2.float64.vectors) == 0  # Cleared
        end

        @testset "TypedPool reset!" begin
            import AdaptiveArrayPools: get_typed_pool!

            pool = AdaptiveArrayPool()
            tp = get_typed_pool!(pool, Float64)

            # Acquire and checkpoint
            checkpoint!(tp)
            acquire!(pool, Float64, 100)
            checkpoint!(tp)
            acquire!(pool, Float64, 200)
            @test tp.n_active == 2
            @test length(tp._checkpoint_n_active) > 1

            # Reset TypedPool directly
            result = reset!(tp)
            @test result === tp
            @test tp.n_active == 0
            @test tp._checkpoint_n_active == [0]
            @test tp._checkpoint_depths == [0]
            @test length(tp.vectors) == 2  # Vectors preserved
        end
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
        checkpoint!(pool, Float64, Float32, Int64, Int32, ComplexF64, ComplexF32, Bool)
        acquire!(pool, Float64, 10)
        acquire!(pool, Float32, 10)
        acquire!(pool, Int64, 10)
        acquire!(pool, Int32, 10)
        acquire!(pool, ComplexF64, 10)
        acquire!(pool, ComplexF32, 10)
        acquire!(pool, Bool, 10)
        @test pool.float64.n_active == 1
        @test pool.float32.n_active == 1
        @test pool.int64.n_active == 1
        @test pool.int32.n_active == 1
        @test pool.complexf64.n_active == 1
        @test pool.complexf32.n_active == 1
        @test pool.bool.n_active == 1
        rewind!(pool, Float64, Float32, Int64, Int32, ComplexF64, ComplexF32, Bool)
        @test pool.float64.n_active == 0
        @test pool.float32.n_active == 0
        @test pool.int64.n_active == 0
        @test pool.int32.n_active == 0
        @test pool.complexf64.n_active == 0
        @test pool.complexf32.n_active == 0
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

        # Warm-up phase - allocates _checkpoint_n_active capacity
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

    @testset "Parent scope protection via full checkpoint" begin
        # Test: Parent scope arrays are protected by automatic full checkpoint
        # when entering @with_pool with _untracked_flags[_current_depth] = true

        # Helper function that acquires Int64 (called from inside @with_pool)
        # Since it's defined outside the macro, acquire! won't be transformed
        function untracked_helper(p)
            acquire!(p, Int64, 5)  # This will mark _untracked_flags = true
        end

        pool = get_task_local_pool()
        empty!(pool)  # Start fresh

        # Acquire Int64 array OUTSIDE @with_pool - marks global _untracked_flags
        v_parent = acquire!(pool, Int64, 10)
        v_parent .= 42  # Initialize
        @test pool.int64.n_active == 1
        @test pool._untracked_flags[1] == true  # Global scope marked

        # Enter @with_pool - should do FULL checkpoint (because _untracked_flags[1] = true)
        # This protects the parent's Int64 arrays
        @with_pool pool begin
            v_float = acquire!(pool, Float64, 100)  # Tracked
            untracked_helper(pool)                   # Untracked Int64 acquire!
            @test pool.int64.n_active == 2          # Parent + helper
        end

        # After @with_pool: parent's array restored, helper's discarded
        @test pool.int64.n_active == 1  # Only parent's array
        @test all(v_parent .== 42)    # Parent array still valid

        empty!(pool)
    end

    @testset "Helper acquires same type as parent - protected" begin
        # Test that parent's arrays are protected when helper acquires same type

        pool = get_task_local_pool()
        empty!(pool)

        # Acquire Int32 outside @with_pool
        v_parent = acquire!(pool, Int32, 7)
        v_parent .= Int32(123)
        @test pool.int32.n_active == 1
        @test pool._untracked_flags[1] == true

        # Helper for Int32
        function int32_helper(p)
            acquire!(p, Int32, 3)
        end

        # Should NOT error - full checkpoint protects parent
        @with_pool pool begin
            acquire!(pool, Float64, 10)
            int32_helper(pool)
            @test pool.int32.n_active == 2  # Parent + helper
        end

        # Parent's Int32 array still valid
        @test pool.int32.n_active == 1
        @test all(v_parent .== Int32(123))

        empty!(pool)
    end

    @testset "Helper acquires new type - silently reset" begin
        # Test: Helper that acquires a completely new type just gets reset to 0
        # (no error, the arrays are simply discarded)

        function new_type_helper(p)
            acquire!(p, UInt16, 5)  # New type not used by macro body
        end

        pool = get_task_local_pool()
        empty!(pool)

        # Should NOT error - new type just gets n_active=0
        @with_pool pool begin
            acquire!(pool, Float64, 100)
            new_type_helper(pool)
            @test pool.others[UInt16].n_active == 1
        end

        # UInt16 was silently reset to 0
        @test pool.others[UInt16].n_active == 0

        empty!(pool)
    end

    @testset "No untracked in parent - typed checkpoint used" begin
        # Edge case: When parent has no untracked acquires,
        # typed checkpoint is still used for performance
        pool = AdaptiveArrayPool()

        # No global untracked acquire
        @test pool._untracked_flags[1] == false

        # Checkpoint/rewind with typed - should work normally
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        rewind!(pool)

        @test pool.int64.n_active == 0
        @test pool.float64.n_active == 0
    end

    @testset "Complex nested case (design doc 5.3) with functions" begin
        # Test scenario from design/untracked_acquire_design.md section 5.3:
        # L1: @with_pool function (Float64)
        #   L2: regular function with get_task_local_pool() (Int64 + Float32 untracked)
        #     L3: @with_pool function (Bool)
        #
        # This tests realistic usage with functions that use @with_pool
        # and functions that don't (simulating untracked behavior)

        pool = get_task_local_pool()
        empty!(pool)

        # Track results across scope boundaries
        l3_results = Ref{NamedTuple}((;))
        l2_results = Ref{NamedTuple}((;))

        # L3: innermost function WITH @with_pool (function syntax)
        @with_pool pool function level3_with_pool()
            v_bool = acquire!(pool, Bool, 10)
            v_bool .= true
            l3_results[] = (
                bool_n_active = pool.bool.n_active,
                depth = pool._current_depth
            )
        end

        # L2: middle function WITHOUT @with_pool (uses get_task_local_pool directly)
        function level2_no_pool()
            pool = get_task_local_pool()

            # These are untracked acquires (no @with_pool wrapping)
            v_i64 = acquire!(pool, Int64, 50)
            v_i64 .= 64

            # Also untracked Float32
            v_f32 = acquire!(pool, Float32, 20)
            v_f32 .= 32.0f0

            # Call L3
            level3_with_pool()

            l2_results[] = (
                int64_n_active = pool.int64.n_active,
                float32_n_active = pool.float32.n_active,
                l3_bool_after = pool.bool.n_active
            )
        end

        # L1: outermost function WITH @with_pool (function syntax)
        @with_pool pool function level1_with_pool()
            v_f64 = acquire!(pool, Float64, 100)
            v_f64 .= 64.0

            # Call L2 (which doesn't use @with_pool)
            level2_no_pool()

            # After L2 returns - check state
            @test pool.float64.n_active == 1  # Float64 still active in L1
            @test all(v_f64 .== 64.0)  # Array still valid
        end

        # Execute the nested calls
        level1_with_pool()

        # Tests for L3 results (inside L3 @with_pool)
        @test l3_results[].bool_n_active == 1
        @test l3_results[].depth >= 3  # At least 3 levels deep

        # Tests for L2 results (after L3 but still in L2)
        @test l2_results[].int64_n_active >= 1  # Int64 still active in L2
        @test l2_results[].float32_n_active >= 1  # Float32 still active in L2
        @test l2_results[].l3_bool_after == 0  # Bool was released by L3 @with_pool

        # After L1 @with_pool exits - everything should be cleaned up
        @test pool.float64.n_active == 0
        @test pool.int64.n_active == 0
        @test pool.float32.n_active == 0
        @test pool.bool.n_active == 0
        @test pool._current_depth == 1

        empty!(pool)
    end

    @testset "Nested @with_pool functions with untracked middle layer" begin
        # Variation: L1 and L3 use @with_pool function syntax
        # L2 is a plain function (untracked) that calls L3
        # This tests the full checkpoint detection based on parent's untracked flag

        pool = get_task_local_pool()
        empty!(pool)

        test_results = Ref{NamedTuple}((;))

        # L3: innermost @with_pool function - acquires Bool and ComplexF64
        @with_pool pool function inner_level()
            v_bool = acquire!(pool, Bool, 10)       # Tracked
            v_cf64 = acquire!(pool, ComplexF64, 5)  # Tracked
            v_bool .= true
            v_cf64 .= 1.0 + 2.0im
            (pool.bool.n_active, pool.complexf64.n_active)
        end

        # L2: plain function (no @with_pool) - untracked acquires
        function middle_layer()
            p = get_task_local_pool()
            # These are untracked acquires!
            v_f32 = acquire!(p, Float32, 20)
            v_i32 = acquire!(p, Int32, 15)
            v_f32 .= 32.0f0
            v_i32 .= 32

            # Call L3 inside
            l3_active = inner_level()

            # After L3 returns, Bool and ComplexF64 should be cleaned
            test_results[] = (
                float32_active = p.float32.n_active,  # Still active (L2)
                int32_active = p.int32.n_active,      # Still active (L2)
                bool_after_l3 = p.bool.n_active,      # Cleaned by L3
                complexf64_after_l3 = p.complexf64.n_active,  # Cleaned by L3
                l3_bool_was = l3_active[1],
                l3_cf64_was = l3_active[2]
            )
        end

        # L1: outermost @with_pool function - acquires Float64 and Int64
        @with_pool pool function outer_level()
            v_f64 = acquire!(pool, Float64, 100)  # Tracked
            v_i64 = acquire!(pool, Int64, 50)     # Tracked
            v_f64 .= 64.0
            v_i64 .= 64

            @test pool.float64.n_active == 1
            @test pool.int64.n_active == 1

            # Call middle layer (which does untracked acquires and calls inner)
            middle_layer()

            # After middle_layer returns:
            # - Float32 and Int32 were untracked in this scope's view
            # - But they should be cleaned by full rewind detection
            @test test_results[].float32_active == 1  # Was active during L2
            @test test_results[].int32_active == 1    # Was active during L2
            @test test_results[].bool_after_l3 == 0   # Cleaned by L3
            @test test_results[].complexf64_after_l3 == 0  # Cleaned by L3

            # L1's arrays still valid
            @test pool.float64.n_active == 1
            @test pool.int64.n_active == 1
            @test all(v_f64 .== 64.0)
            @test all(v_i64 .== 64)
        end

        # Execute
        outer_level()

        # After all exits
        @test pool.float64.n_active == 0
        @test pool.int64.n_active == 0
        @test pool.float32.n_active == 0
        @test pool.int32.n_active == 0
        @test pool.bool.n_active == 0
        @test pool.complexf64.n_active == 0
        @test pool._current_depth == 1
        @test pool._untracked_flags == [false]

        empty!(pool)
    end

    @testset "Complex nested with unsafe_acquire! (L1→L2→L3)" begin
        # Same L1→L2→L3 pattern but using unsafe_acquire! instead of acquire!
        # This tests that unsafe_acquire! also triggers untracked detection correctly

        pool = get_task_local_pool()
        empty!(pool)

        l3_results = Ref{NamedTuple}((;))
        l2_results = Ref{NamedTuple}((;))

        # L3: innermost @with_pool function using unsafe_acquire!
        @with_pool pool function level3_unsafe()
            # unsafe_acquire! returns raw Array, not SubArray
            v_bool = unsafe_acquire!(pool, Bool, 10)
            @test v_bool isa Vector{Bool}
            v_bool .= true
            l3_results[] = (
                bool_n_active = pool.bool.n_active,
                depth = pool._current_depth
            )
        end

        # L2: plain function using unsafe_acquire! (untracked)
        function level2_unsafe_untracked()
            p = get_task_local_pool()
            # These are untracked unsafe_acquire! calls
            v_i64 = unsafe_acquire!(p, Int64, 50)
            @test v_i64 isa Vector{Int64}
            v_i64 .= 64

            v_f32 = unsafe_acquire!(p, Float32, 20)
            @test v_f32 isa Vector{Float32}
            v_f32 .= 32.0f0

            # Call L3
            level3_unsafe()

            l2_results[] = (
                int64_n_active = p.int64.n_active,
                float32_n_active = p.float32.n_active,
                l3_bool_after = p.bool.n_active
            )
        end

        # L1: outermost @with_pool function using unsafe_acquire!
        @with_pool pool function level1_unsafe()
            v_f64 = unsafe_acquire!(pool, Float64, 100)
            @test v_f64 isa Vector{Float64}
            v_f64 .= 64.0

            level2_unsafe_untracked()

            @test pool.float64.n_active == 1
            @test all(v_f64 .== 64.0)
        end

        # Execute
        level1_unsafe()

        # Verify results
        @test l3_results[].bool_n_active == 1
        @test l3_results[].depth >= 3

        @test l2_results[].int64_n_active >= 1
        @test l2_results[].float32_n_active >= 1
        @test l2_results[].l3_bool_after == 0

        # After L1 exits - all cleaned up
        @test pool.float64.n_active == 0
        @test pool.int64.n_active == 0
        @test pool.float32.n_active == 0
        @test pool.bool.n_active == 0
        @test pool._current_depth == 1

        empty!(pool)
    end

    @testset "Complex nested with acquire_array! alias (L1→L2→L3)" begin
        # Same pattern using acquire_array! (alias for unsafe_acquire!)

        pool = get_task_local_pool()
        empty!(pool)

        l3_results = Ref{NamedTuple}((;))

        # L3: @with_pool with acquire_array!
        @with_pool pool function level3_array()
            v_cf64 = acquire_array!(pool, ComplexF64, 5)
            @test v_cf64 isa Vector{ComplexF64}
            v_cf64 .= 1.0 + 2.0im
            l3_results[] = (complexf64_n_active = pool.complexf64.n_active,)
        end

        # L2: plain function with acquire_array! (untracked)
        function level2_array_untracked()
            p = get_task_local_pool()
            v_i32 = acquire_array!(p, Int32, 15)
            @test v_i32 isa Vector{Int32}
            v_i32 .= Int32(32)

            level3_array()
        end

        # L1: @with_pool with acquire_array!
        @with_pool pool function level1_array()
            v_f64 = acquire_array!(pool, Float64, 10, 10)  # 2D matrix
            @test v_f64 isa Matrix{Float64}
            v_f64 .= 1.0

            level2_array_untracked()

            @test pool.float64.n_active == 1
            @test sum(v_f64) == 100.0
        end

        level1_array()

        @test l3_results[].complexf64_n_active == 1
        @test pool.float64.n_active == 0
        @test pool.int32.n_active == 0
        @test pool.complexf64.n_active == 0
        @test pool._current_depth == 1

        empty!(pool)
    end

    @testset "Mixed acquire!/unsafe_acquire!/acquire_array! in nested scopes" begin
        # Test mixing all acquire variants in a single nested scenario

        pool = get_task_local_pool()
        empty!(pool)

        results = Ref{NamedTuple}((;))

        # L3: @with_pool mixing acquire! and unsafe_acquire!
        @with_pool pool function level3_mixed()
            v_bool = acquire!(pool, Bool, 10)           # SubArray
            v_i32 = unsafe_acquire!(pool, Int32, 5)      # Vector
            v_cf64 = acquire_array!(pool, ComplexF64, 3) # Vector (alias)

            @test v_bool isa SubArray
            @test v_i32 isa Vector{Int32}
            @test v_cf64 isa Vector{ComplexF64}

            v_bool .= true
            v_i32 .= Int32(32)
            v_cf64 .= 3.0 + 4.0im

            (pool.bool.n_active, pool.int32.n_active, pool.complexf64.n_active)
        end

        # L2: plain function with mixed untracked calls
        function level2_mixed_untracked()
            p = get_task_local_pool()

            # Mix of acquire variants (all untracked)
            v_i64 = acquire!(p, Int64, 20)           # SubArray (untracked)
            v_f32 = unsafe_acquire!(p, Float32, 15)  # Vector (untracked)

            @test v_i64 isa SubArray
            @test v_f32 isa Vector{Float32}

            l3_active = level3_mixed()

            results[] = (
                int64_n_active = p.int64.n_active,
                float32_n_active = p.float32.n_active,
                l3_bool = l3_active[1],
                l3_int32 = l3_active[2],
                l3_complexf64 = l3_active[3],
                bool_after = p.bool.n_active,
                int32_after = p.int32.n_active,
                complexf64_after = p.complexf64.n_active
            )
        end

        # L1: @with_pool with mixed acquire calls
        @with_pool pool function level1_mixed()
            v_f64_view = acquire!(pool, Float64, 100)        # SubArray
            v_f64_array = unsafe_acquire!(pool, Float64, 50) # Vector (same type!)

            @test v_f64_view isa SubArray
            @test v_f64_array isa Vector{Float64}

            v_f64_view .= 1.0
            v_f64_array .= 2.0

            @test pool.float64.n_active == 2  # Both from same TypedPool

            level2_mixed_untracked()

            # L1's Float64 arrays still valid
            @test pool.float64.n_active == 2
            @test all(v_f64_view .== 1.0)
            @test all(v_f64_array .== 2.0)
        end

        level1_mixed()

        # Verify L3 had active arrays
        @test results[].l3_bool == 1
        @test results[].l3_int32 == 1
        @test results[].l3_complexf64 == 1

        # Verify L3 cleaned up after exit
        @test results[].bool_after == 0
        @test results[].int32_after == 0
        @test results[].complexf64_after == 0

        # L2's untracked were cleaned by L1's full rewind
        @test results[].int64_n_active >= 1
        @test results[].float32_n_active >= 1

        # After everything - all clean
        @test pool.float64.n_active == 0
        @test pool.int64.n_active == 0
        @test pool.float32.n_active == 0
        @test pool.int32.n_active == 0
        @test pool.bool.n_active == 0
        @test pool.complexf64.n_active == 0
        @test pool._current_depth == 1

        empty!(pool)
    end

    @testset "acquire_view! alias in nested scopes" begin
        # Test acquire_view! (alias for acquire!) in nested scenario

        pool = get_task_local_pool()
        empty!(pool)

        # L2: untracked acquire_view! call
        function level2_view_untracked()
            p = get_task_local_pool()
            v = acquire_view!(p, Int64, 10)
            @test v isa SubArray{Int64}
            v .= 42
        end

        # L1: @with_pool with acquire_view!
        @with_pool pool function level1_view()
            v_f64 = acquire_view!(pool, Float64, 5, 5)  # 2D ReshapedArray
            @test v_f64 isa Base.ReshapedArray{Float64, 2}
            v_f64 .= 1.0

            level2_view_untracked()

            @test pool.float64.n_active == 1
            @test sum(v_f64) == 25.0
        end

        level1_view()

        @test pool.float64.n_active == 0
        @test pool.int64.n_active == 0
        @test pool._current_depth == 1

        empty!(pool)
    end

    @testset "N-D unsafe_acquire! in nested scopes" begin
        # Test multi-dimensional unsafe_acquire! in complex nested scenario

        pool = get_task_local_pool()
        empty!(pool)

        results = Ref{NamedTuple}((;))

        # L3: @with_pool with 3D unsafe_acquire!
        @with_pool pool function level3_nd()
            v_3d = unsafe_acquire!(pool, Float32, 2, 3, 4)
            @test v_3d isa Array{Float32, 3}
            @test size(v_3d) == (2, 3, 4)
            v_3d .= 3.0f0
            results[] = (float32_n_active = pool.float32.n_active,)
        end

        # L2: untracked 2D unsafe_acquire!
        function level2_nd_untracked()
            p = get_task_local_pool()
            v_2d = unsafe_acquire!(p, Int64, 5, 5)
            @test v_2d isa Matrix{Int64}
            v_2d .= 2

            level3_nd()
        end

        # L1: @with_pool with tuple-style dimensions
        @with_pool pool function level1_nd()
            v_tuple = unsafe_acquire!(pool, Float64, (3, 4, 5))
            @test v_tuple isa Array{Float64, 3}
            @test size(v_tuple) == (3, 4, 5)
            v_tuple .= 1.0

            level2_nd_untracked()

            @test pool.float64.n_active == 1
            @test sum(v_tuple) == 60.0
        end

        level1_nd()

        @test results[].float32_n_active == 1
        @test pool.float64.n_active == 0
        @test pool.int64.n_active == 0
        @test pool.float32.n_active == 0
        @test pool._current_depth == 1

        empty!(pool)
    end

end # State Management