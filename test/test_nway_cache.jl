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
        @test CACHE_WAYS >= 1

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

    @testset "N-D unsafe_acquire!: 5-way causes allocation (cache eviction)" begin
        # unsafe_acquire! uses N-way cache → 5-way exceeds CACHE_WAYS=4
        pool = AdaptiveArrayPool()

        function test_nd_5way_unsafe!(p)
            dims_list = ((5, 10), (10, 5), (7, 7), (3, 16), (4, 12))
            for _ in 1:1
                for dims in dims_list
                    @with_pool p begin
                        unsafe_acquire!(p, Float64, dims...)  # Array with cache
                    end
                end
            end
        end

        # Warmup (fills cache with 4 patterns, 5th evicts one)
        test_nd_5way_unsafe!(pool)
        test_nd_5way_unsafe!(pool)

        # 5-way exceeds 4-way cache → eviction → unsafe_wrap allocation
        allocs = @allocated test_nd_5way_unsafe!(pool)
        @test allocs > 0
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
