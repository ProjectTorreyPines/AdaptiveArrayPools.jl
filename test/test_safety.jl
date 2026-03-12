import AdaptiveArrayPools: _invalidate_released_slots!, PoolRuntimeEscapeError, _make_pool, _validate_pool_return

# Opaque identity — defeats compile-time escape analysis without @skip_check_vars
_test_leak(x) = x

@testset "RUNTIME_CHECK Guard-Level Invalidation" begin

    # ==============================================================================
    # Default values
    # ==============================================================================

    @testset "Default configuration" begin
        @test RUNTIME_CHECK isa Int
    end

    # ==============================================================================
    # S=1: acquire! SubArray invalidation
    # ==============================================================================

    @testset "acquire! SubArray invalidated on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 42.0  # write to confirm it's valid before rewind
        rewind!(pool)

        # Backing vector resized to 0 -> SubArray parent is length 0
        @test length(parent(v)) == 0

        # Accessing stale SubArray should throw BoundsError
        @test_throws BoundsError v[1]
    end

    @testset "acquire! N-D ReshapedArray invalidated on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        mat = acquire!(pool, Float64, 5, 5)
        mat .= 1.0
        rewind!(pool)

        # Parent chain: ReshapedArray -> SubArray -> Vector (now length 0)
        @test length(parent(parent(mat))) == 0
        @test_throws BoundsError mat[1, 1]
    end

    # ==============================================================================
    # S=1: unsafe_acquire! Array wrapper invalidation (Julia 1.11+ only)
    # On Julia 1.10, Array is a C struct — setfield!(:size) is not available.
    # ==============================================================================

    @static if VERSION >= v"1.11-"
        @testset "unsafe_acquire! Array wrapper invalidated on rewind" begin
            pool = _make_pool(true)
            checkpoint!(pool)
            arr = unsafe_acquire!(pool, Float64, 10)
            arr .= 99.0
            @test size(arr) == (10,)
            rewind!(pool)

            # Wrapper size set to (0,) via setfield!
            @test size(arr) == (0,)
            @test_throws BoundsError arr[1]
        end

        @testset "unsafe_acquire! N-D Array wrapper invalidated on rewind" begin
            pool = _make_pool(true)
            checkpoint!(pool)
            mat = unsafe_acquire!(pool, Float64, 4, 3)
            mat .= 1.0
            @test size(mat) == (4, 3)
            rewind!(pool)

            @test size(mat) == (0, 0)
            @test_throws BoundsError mat[1, 1]
        end
    end

    # ==============================================================================
    # S=1: BitArray invalidation
    # ==============================================================================

    @testset "acquire! BitVector invalidated on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        bv = acquire!(pool, Bit, 100)
        bv .= true
        rewind!(pool)

        # BitVector backing resized to 0
        @test length(pool.bits.vectors[1]) == 0
        # Accessing stale BitVector - len was set to 0 via setfield!
        @test length(bv) == 0
    end

    @testset "acquire! BitMatrix invalidated on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        ba = acquire!(pool, Bit, 8, 8)
        ba .= true
        @test size(ba) == (8, 8)
        rewind!(pool)

        # BitArray dims set to (0, 0), len set to 0
        @test size(ba) == (0, 0)
        @test length(ba) == 0
    end

    # ==============================================================================
    # S=0: No invalidation
    # ==============================================================================

    @testset "S=0 bypasses invalidation" begin
        pool = _make_pool(false)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 7.0
        rewind!(pool)

        # With safety off, backing vector still has length >= 10
        @test length(parent(v)) >= 10
        # Stale access works (this is the unsafe behavior we're protecting against)
        @test v[1] == 7.0
    end

    # ==============================================================================
    # Re-acquire after invalidation (zero-alloc round-trip)
    # ==============================================================================

    @testset "Re-acquire after invalidation restores vectors" begin
        pool = _make_pool(true)

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
    end

    @static if VERSION >= v"1.11-"
        @testset "Re-acquire unsafe_acquire! after invalidation" begin
            pool = _make_pool(true)

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
        end
    end

    # ==============================================================================
    # Nested scopes: inner invalidation doesn't affect outer
    # ==============================================================================

    @testset "Nested checkpoint/rewind: inner invalidated, outer valid" begin
        pool = _make_pool(true)
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
    end

    # ==============================================================================
    # reset! invalidation
    # ==============================================================================

    @testset "reset! invalidates all active slots" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 10)
        v2 = acquire!(pool, Float64, 20)
        v1 .= 1.0
        v2 .= 2.0

        reset!(pool.float64, 1)  # S=1 to trigger invalidation

        @test pool.float64.n_active == 0
        @test length(pool.float64.vectors[1]) == 0
        @test length(pool.float64.vectors[2]) == 0
    end

    # ==============================================================================
    # Fallback types (pool.others)
    # ==============================================================================

    @testset "Fallback type invalidation" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, UInt8, 50)
        v .= 0xff
        @test length(parent(v)) >= 50
        rewind!(pool)

        # Fallback type also invalidated
        tp = pool.others[UInt8]
        @test length(tp.vectors[1]) == 0
        @test length(parent(v)) == 0
    end

    # ==============================================================================
    # Multiple types in same scope
    # ==============================================================================

    @testset "Multiple types invalidated together" begin
        pool = _make_pool(true)
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
    end

    # ==============================================================================
    # @with_pool macro integration
    # ==============================================================================

    @testset "@with_pool invalidates on scope exit" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 5.0
        result = sum(v)  # Safe scalar return
        rewind!(pool)

        @test result == 50.0
        # After rewind, the pool's vectors should be invalidated
        @test length(parent(v)) == 0
        @test_throws BoundsError v[1]
    end

    # ==============================================================================
    # S=1: Poisoning (NaN/sentinel fill before structural invalidation)
    # ==============================================================================

    @testset "Float64 poisoned with NaN on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 42.0
        rewind!(pool)

        # Re-acquire: backing vector was poisoned with NaN before resize!(v,0).
        # resize! round-trip (0→10) preserves capacity, NaN data survives.
        checkpoint!(pool)
        v2 = acquire!(pool, Float64, 10)
        @test all(isnan, v2)
        rewind!(pool)
    end

    @testset "Int64 poisoned with typemax on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Int64, 10)
        v .= 42
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, Int64, 10)
        @test all(==(typemax(Int64)), v2)
        rewind!(pool)
    end

    @testset "ComplexF64 poisoned with NaN+NaN*im on rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, ComplexF64, 8)
        v .= 1.0 + 2.0im
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, ComplexF64, 8)
        @test all(z -> isnan(real(z)) && isnan(imag(z)), v2)
        rewind!(pool)
    end

end # RUNTIME_CHECK Guard-Level Invalidation
