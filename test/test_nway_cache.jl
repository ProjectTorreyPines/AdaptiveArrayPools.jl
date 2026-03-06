using Test
using AdaptiveArrayPools

@testset "N-way Cache for unsafe_acquire!" begin

    @testset "Type checks" begin
        pool = AdaptiveArrayPool()

        @with_pool pool begin
            # acquire! returns ReshapedArray for N-D
            arr = acquire!(pool, Float64, 10, 10)
            @test arr isa Base.ReshapedArray{Float64, 2}

            # acquire! returns SubArray for 1D
            vec = acquire!(pool, Float64, 100)
            @test vec isa SubArray{Float64, 1}

            # unsafe_acquire! returns Array
            raw = unsafe_acquire!(pool, Float64, 10, 10)
            @test raw isa Array{Float64, 2}
            @test raw isa Matrix{Float64}
        end
    end

    @testset "CACHE_WAYS configuration" begin
        # Verify CACHE_WAYS is exported and accessible
        @test CACHE_WAYS isa Int
        @test 1 <= CACHE_WAYS <= 16  # Valid range

        # Verify set_cache_ways! is exported
        @test isdefined(AdaptiveArrayPools, :set_cache_ways!)
    end

    @testset "set_cache_ways! validation" begin
        # Valid values should return the input value
        @test set_cache_ways!(1) == 1
        @test set_cache_ways!(4) == 4
        @test set_cache_ways!(8) == 8
        @test set_cache_ways!(16) == 16

        # Invalid values should throw ArgumentError
        @test_throws ArgumentError set_cache_ways!(0)
        @test_throws ArgumentError set_cache_ways!(-1)
        @test_throws ArgumentError set_cache_ways!(17)
        @test_throws ArgumentError set_cache_ways!(100)

        # Reset to default after tests
        set_cache_ways!(4)
    end

end

@testset "N-way Zero-Allocation" begin

    @testset "N-D unsafe_acquire!: 4-way alternating is zero-alloc" begin
        pool = AdaptiveArrayPool()

        function test_nd_4way!(p)
            dims_list = ((5, 10), (10, 5), (7, 7), (3, 16))
            for _ in 1:100
                for dims in dims_list
                    @with_pool p begin
                        unsafe_acquire!(p, Float64, dims...)
                    end
                end
            end
        end

        # Warmup
        test_nd_4way!(pool)
        test_nd_4way!(pool)

        # Measure
        allocs = @allocated test_nd_4way!(pool)
        allocs > 0 && @warn "N-D 4-way: $allocs bytes (expected 0)"
        @test allocs == 0
    end

    @testset "N-D acquire!: 5-way is zero-alloc (ReshapedArray)" begin
        # acquire! returns ReshapedArray → no N-way cache needed → always 0 alloc
        pool = AdaptiveArrayPool()

        function test_nd_5way_acquire!(p)
            dims_list = ((5, 10), (10, 5), (7, 7), (3, 16), (4, 12))
            for _ in 1:100
                for dims in dims_list
                    @with_pool p begin
                        acquire!(p, Float64, dims...)  # ReshapedArray
                    end
                end
            end
        end

        # Warmup
        test_nd_5way_acquire!(pool)
        test_nd_5way_acquire!(pool)

        # acquire! uses reshape(1D_view, dims) → 0 alloc regardless of pattern count
        allocs = @allocated test_nd_5way_acquire!(pool)
        allocs > 0 && @warn "N-D acquire! 5-way: $allocs bytes (expected 0)"
        @test allocs == 0
    end

    @testset "N-D unsafe_acquire!: 5-way behavior" begin
        pool = AdaptiveArrayPool()

        function test_nd_5way_unsafe!(p)
            dims_list = ((5, 10), (10, 5), (7, 7), (3, 16), (4, 12))
            for _ in 1:100
                for dims in dims_list
                    @with_pool p begin
                        unsafe_acquire!(p, Float64, dims...)
                    end
                end
            end
        end

        # Warmup
        test_nd_5way_unsafe!(pool)
        test_nd_5way_unsafe!(pool)

        allocs = @allocated test_nd_5way_unsafe!(pool)
        # setfield! reuse: unlimited dim patterns, 0-alloc
        allocs > 0 && @warn "N-D 5-way unsafe: $allocs bytes (expected 0)"
        @test allocs == 0
    end

    @testset "N-D unsafe_acquire!: 10+ patterns per slot is zero-alloc" begin
        # Demonstrates removal of CACHE_WAYS limit via setfield! (Julia 1.11+)
        pool = AdaptiveArrayPool()

        function test_nd_many_patterns!(p)
            dims_list = (
                (2, 50), (5, 20), (10, 10), (20, 5), (50, 2),
                (1, 100), (100, 1), (4, 25), (25, 4), (8, 13),
            )
            for _ in 1:50
                for dims in dims_list
                    @with_pool p begin
                        unsafe_acquire!(p, Float64, dims...)
                    end
                end
            end
        end

        # Warmup
        test_nd_many_patterns!(pool)
        test_nd_many_patterns!(pool)

        allocs = @allocated test_nd_many_patterns!(pool)
        allocs > 0 && @warn "N-D 10+ patterns: $allocs bytes (expected 0)"
        @test allocs == 0
    end

    @testset "Cache invalidation on resize" begin
        pool = AdaptiveArrayPool()

        # Warmup with small array
        @with_pool pool begin
            unsafe_acquire!(pool, Float64, 10, 10)
        end

        # Request larger array (forces resize, invalidates cache)
        @with_pool pool begin
            arr = unsafe_acquire!(pool, Float64, 100, 100)
            @test size(arr) == (100, 100)
        end

        # Warmup again with new size
        @with_pool pool begin
            unsafe_acquire!(pool, Float64, 100, 100)
        end

        # Now should be zero allocation
        allocs = @allocated begin
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 100, 100)
            end
        end
        @test allocs == 0
    end

    @testset "Multiple slots with N-way cache" begin
        pool = AdaptiveArrayPool()

        # Warmup: each slot gets 2 different shapes
        for _ in 1:2
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 5, 5)   # Slot 1
                unsafe_acquire!(pool, Float64, 10, 10) # Slot 2
            end
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 6, 6)   # Slot 1, different dims
                unsafe_acquire!(pool, Float64, 12, 12) # Slot 2, different dims
            end
        end

        # Both slots should have their shapes cached
        allocs = @allocated begin
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 5, 5)
                unsafe_acquire!(pool, Float64, 10, 10)
            end
        end
        @test allocs == 0

        # Alternating dims should also hit cache
        allocs = @allocated begin
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 6, 6)
                unsafe_acquire!(pool, Float64, 12, 12)
            end
        end
        @test allocs == 0
    end

end

# ==============================================================================
# Vector-Based N-D Wrapper Cache Tests (Julia 1.11+)
# ==============================================================================
# These tests verify the Dict→Vector migration for nd_wrappers.

@testset "Vector-based nd_wrappers cache" begin
    using AdaptiveArrayPools: checkpoint!, rewind!

    @testset "nd_wrappers grows correctly for multiple dimensionalities" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # N=1: 1D unsafe_acquire
        v1 = unsafe_acquire!(pool, Float64, 10)
        @test length(pool.float64.nd_wrappers) >= 1

        # N=2: 2D unsafe_acquire — nd_wrappers should grow to index 2
        m1 = unsafe_acquire!(pool, Float64, 3, 4)
        @test length(pool.float64.nd_wrappers) >= 2
        @test pool.float64.nd_wrappers[2] !== nothing  # has a Vector{Any} for N=2

        # N=3: 3D unsafe_acquire — nd_wrappers should grow to index 3
        t1 = unsafe_acquire!(pool, Float64, 2, 3, 4)
        @test length(pool.float64.nd_wrappers) >= 3
        @test pool.float64.nd_wrappers[3] !== nothing  # has a Vector{Any} for N=3

        rewind!(pool)
    end

    @testset "wrapper object identity is preserved on cache hit" begin
        pool = AdaptiveArrayPool()

        # First call: cache miss → creates wrapper
        checkpoint!(pool)
        m1 = unsafe_acquire!(pool, Float64, 3, 4)
        wrapper_id = objectid(m1)
        rewind!(pool)

        # Second call: cache hit → same wrapper object, updated fields
        checkpoint!(pool)
        m2 = unsafe_acquire!(pool, Float64, 5, 6)
        @test objectid(m2) == wrapper_id  # same Array object reused via setfield!
        @test size(m2) == (5, 6)          # dims updated in-place
        rewind!(pool)

        # Third call with same dims: still same wrapper
        checkpoint!(pool)
        m3 = unsafe_acquire!(pool, Float64, 5, 6)
        @test objectid(m3) == wrapper_id
        @test size(m3) == (5, 6)
        rewind!(pool)
    end

    @testset "different N values use independent wrapper slots" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Slot 1 as 2D
        m = unsafe_acquire!(pool, Float64, 3, 4)
        rewind!(pool)

        checkpoint!(pool)
        # Slot 1 as 3D — different N, must create new wrapper
        t = unsafe_acquire!(pool, Float64, 2, 3, 4)
        @test size(t) == (2, 3, 4)

        # Both N=2 and N=3 entries exist
        @test pool.float64.nd_wrappers[2] !== nothing
        @test pool.float64.nd_wrappers[3] !== nothing
        rewind!(pool)
    end

    @testset "nd_wrappers with nothing gaps for skipped N" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Jump directly to N=3 without using N=2
        t = unsafe_acquire!(pool, Float64, 2, 3, 4)
        @test length(pool.float64.nd_wrappers) >= 3

        # N=2 entry should be nothing (never used for N=2)
        @test pool.float64.nd_wrappers[2] === nothing

        rewind!(pool)
    end

    @testset "BitTypedPool nd_wrappers cache" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D BitArray
        bv = acquire!(pool, Bit, 100)
        @test length(pool.bits.nd_wrappers) >= 1

        # 2D BitArray
        ba = acquire!(pool, Bit, 10, 10)
        @test length(pool.bits.nd_wrappers) >= 2
        @test pool.bits.nd_wrappers[2] !== nothing

        rewind!(pool)

        # Verify wrapper reuse for BitArray
        checkpoint!(pool)
        bv2 = acquire!(pool, Bit, 50)
        ba2 = acquire!(pool, Bit, 5, 20)
        @test size(ba2) == (5, 20)
        rewind!(pool)
    end

    @testset "empty! clears nd_wrappers" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        unsafe_acquire!(pool, Float64, 3, 4)
        rewind!(pool)

        @test !isempty(pool.float64.nd_wrappers)
        empty!(pool)
        @test isempty(pool.float64.nd_wrappers)
    end

    @testset "multiple element types have independent nd_wrappers" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        mf = unsafe_acquire!(pool, Float64, 3, 4)
        mi = unsafe_acquire!(pool, Int64, 5, 6)

        @test pool.float64.nd_wrappers[2] !== nothing
        @test pool.int64.nd_wrappers[2] !== nothing

        # They must be separate Vector{Any} instances
        @test pool.float64.nd_wrappers[2] !== pool.int64.nd_wrappers[2]

        rewind!(pool)
    end

    # Function barrier for accurate allocation measurement
    function test_mixed_nd_zero_alloc()
        pool = AdaptiveArrayPool()

        # Warmup: exercise N=1, N=2, N=3 for same slot
        for _ in 1:2
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 100)
            end
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 10, 10)
            end
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 5, 4, 5)
            end
        end

        # Measure: all three should be cache hits
        a1 = @allocated @with_pool pool begin
            unsafe_acquire!(pool, Float64, 50)
        end
        a2 = @allocated @with_pool pool begin
            unsafe_acquire!(pool, Float64, 7, 7)
        end
        a3 = @allocated @with_pool pool begin
            unsafe_acquire!(pool, Float64, 3, 3, 3)
        end
        return (a1, a2, a3)
    end

    @testset "mixed dimensionalities zero-alloc after warmup" begin
        test_mixed_nd_zero_alloc()
        test_mixed_nd_zero_alloc()
        a1, a2, a3 = test_mixed_nd_zero_alloc()
        @test a1 == 0
        @test a2 == 0
        @test a3 == 0
    end
end
