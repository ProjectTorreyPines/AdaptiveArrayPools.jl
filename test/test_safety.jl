import AdaptiveArrayPools: _invalidate_released_slots!

@testset "POOL_SAFETY_LV Guard-Level Invalidation" begin

    # ==============================================================================
    # Default values
    # ==============================================================================

    @testset "Default configuration" begin
        @test STATIC_POOL_CHECKS == true
        @test POOL_SAFETY_LV[] == 1
    end

    # ==============================================================================
    # Level 1: acquire! SubArray invalidation
    # ==============================================================================

    @testset "acquire! SubArray invalidated on rewind" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 42.0  # write to confirm it's valid before rewind
        rewind!(pool)

        # Backing vector resized to 0 -> SubArray parent is length 0
        @test length(parent(v)) == 0

        # Accessing stale SubArray should throw BoundsError
        @test_throws BoundsError v[1]

        POOL_SAFETY_LV[] = old_safety
    end

    @testset "acquire! N-D ReshapedArray invalidated on rewind" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        mat = acquire!(pool, Float64, 5, 5)
        mat .= 1.0
        rewind!(pool)

        # Parent chain: ReshapedArray -> SubArray -> Vector (now length 0)
        @test length(parent(parent(mat))) == 0
        @test_throws BoundsError mat[1, 1]

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Level 1: unsafe_acquire! Array wrapper invalidation
    # ==============================================================================

    @testset "unsafe_acquire! Array wrapper invalidated on rewind" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        arr = unsafe_acquire!(pool, Float64, 10)
        arr .= 99.0
        @test size(arr) == (10,)
        rewind!(pool)

        # Wrapper size set to (0,) via setfield!
        @test size(arr) == (0,)
        @test_throws BoundsError arr[1]

        POOL_SAFETY_LV[] = old_safety
    end

    @testset "unsafe_acquire! N-D Array wrapper invalidated on rewind" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        mat = unsafe_acquire!(pool, Float64, 4, 3)
        mat .= 1.0
        @test size(mat) == (4, 3)
        rewind!(pool)

        @test size(mat) == (0, 0)
        @test_throws BoundsError mat[1, 1]

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Level 1: BitArray invalidation
    # ==============================================================================

    @testset "acquire! BitVector invalidated on rewind" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        bv = acquire!(pool, Bit, 100)
        bv .= true
        rewind!(pool)

        # BitVector backing resized to 0
        @test length(pool.bits.vectors[1]) == 0
        # Accessing stale BitVector - len was set to 0 via setfield!
        @test length(bv) == 0

        POOL_SAFETY_LV[] = old_safety
    end

    @testset "acquire! BitMatrix invalidated on rewind" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        ba = acquire!(pool, Bit, 8, 8)
        ba .= true
        @test size(ba) == (8, 8)
        rewind!(pool)

        # BitArray dims set to (0, 0), len set to 0
        @test size(ba) == (0, 0)
        @test length(ba) == 0

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Level 0: No invalidation
    # ==============================================================================

    @testset "POOL_SAFETY_LV=0 bypasses invalidation" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 0

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 7.0
        rewind!(pool)

        # With safety off, backing vector still has length >= 10
        @test length(parent(v)) >= 10
        # Stale access works (this is the unsafe behavior we're protecting against)
        @test v[1] == 7.0

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Re-acquire after invalidation (zero-alloc round-trip)
    # ==============================================================================

    @testset "Re-acquire after invalidation restores vectors" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()

        # First cycle: populate pool
        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 100)
        v1 .= 1.0
        rewind!(pool)

        # Vectors invalidated
        @test length(pool.float64.vectors[1]) == 0

        # Second cycle: re-acquire uses same slot, restores capacity
        checkpoint!(pool)
        v2 = acquire!(pool, Float64, 50)
        v2 .= 2.0
        @test length(parent(v2)) >= 50
        @test v2[1] == 2.0
        # Same backing vector object (capacity preserved through resize round-trip)
        @test parent(v2) === pool.float64.vectors[1]
        rewind!(pool)

        POOL_SAFETY_LV[] = old_safety
    end

    @testset "Re-acquire unsafe_acquire! after invalidation" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()

        # First cycle
        checkpoint!(pool)
        arr = unsafe_acquire!(pool, Float64, 20)
        arr .= 3.0
        rewind!(pool)
        @test size(arr) == (0,)

        # Second cycle: wrapper reused, size restored
        checkpoint!(pool)
        arr2 = unsafe_acquire!(pool, Float64, 15)
        @test size(arr2) == (15,)
        arr2 .= 4.0
        @test arr2[1] == 4.0
        rewind!(pool)

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Nested scopes: inner invalidation doesn't affect outer
    # ==============================================================================

    @testset "Nested checkpoint/rewind: inner invalidated, outer valid" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        v_outer = acquire!(pool, Float64, 10)
        v_outer .= 1.0

        # Inner scope
        checkpoint!(pool)
        v_inner = acquire!(pool, Float64, 20)
        v_inner .= 2.0
        rewind!(pool)

        # Inner is invalidated (slot 2 released)
        @test length(parent(v_inner)) == 0

        # Outer is still valid (slot 1 not released)
        @test length(parent(v_outer)) >= 10
        @test v_outer[1] == 1.0

        rewind!(pool)

        # Now outer is also invalidated
        @test length(parent(v_outer)) == 0

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # reset! invalidation
    # ==============================================================================

    @testset "reset! invalidates all active slots" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 10)
        v2 = acquire!(pool, Float64, 20)
        v1 .= 1.0
        v2 .= 2.0

        reset!(pool.float64)

        @test pool.float64.n_active == 0
        @test length(pool.float64.vectors[1]) == 0
        @test length(pool.float64.vectors[2]) == 0

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Fallback types (pool.others)
    # ==============================================================================

    @testset "Fallback type invalidation" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        v = acquire!(pool, UInt8, 50)
        v .= 0xff
        @test length(parent(v)) >= 50
        rewind!(pool)

        # Fallback type also invalidated
        tp = pool.others[UInt8]
        @test length(tp.vectors[1]) == 0
        @test length(parent(v)) == 0

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # POOL_DEBUG backward compatibility
    # ==============================================================================

    @testset "POOL_DEBUG backward compat with POOL_SAFETY_LV" begin
        old_debug = POOL_DEBUG[]
        old_safety = POOL_SAFETY_LV[]

        # POOL_DEBUG=true still triggers escape detection (regardless of POOL_SAFETY_LV)
        POOL_DEBUG[] = true
        POOL_SAFETY_LV[] = 0
        @test_throws ErrorException @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v  # Unsafe return
        end

        # POOL_SAFETY_LV=2 also triggers escape detection (without POOL_DEBUG)
        POOL_DEBUG[] = false
        POOL_SAFETY_LV[] = 2
        @test_throws ErrorException @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v  # Unsafe return
        end

        # Neither flag -> no escape detection
        POOL_DEBUG[] = false
        POOL_SAFETY_LV[] = 1
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v  # Would be caught at level 2, but not at level 1
        end
        @test result isa SubArray

        POOL_DEBUG[] = old_debug
        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # Multiple types in same scope
    # ==============================================================================

    @testset "Multiple types invalidated together" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        vf = acquire!(pool, Float64, 10)
        vi = acquire!(pool, Int64, 20)
        vb = acquire!(pool, Bit, 30)
        vf .= 1.0
        vi .= 2
        vb .= true
        rewind!(pool)

        @test length(parent(vf)) == 0
        @test length(parent(vi)) == 0
        @test length(vb) == 0

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # @with_pool macro integration
    # ==============================================================================

    @testset "@with_pool invalidates on scope exit" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        pool_ref = Ref{AdaptiveArrayPool}()
        stale_ref = Ref{Any}()

        result = @with_pool pool begin
            pool_ref[] = pool
            v = acquire!(pool, Float64, 10)
            v .= 5.0
            stale_ref[] = v
            sum(v)  # Safe scalar return
        end

        @test result == 50.0
        # After @with_pool exits, the pool's vectors should be invalidated
        v = stale_ref[]
        @test length(parent(v)) == 0
        @test_throws BoundsError v[1]

        POOL_SAFETY_LV[] = old_safety
    end

end # POOL_SAFETY_LV Guard-Level Invalidation
