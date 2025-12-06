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

    @testset "N-way cache prevents thrashing" begin
        pool = AdaptiveArrayPool()

        # Warmup: populate cache with multiple dimension patterns
        for _ in 1:2
            @with_pool pool begin
                # First slot gets two different dims
                unsafe_acquire!(pool, Float64, 10, 10)
            end
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 20, 20)
            end
        end

        # After warmup, alternating dims should hit cache
        # (both (10,10) and (20,20) should be cached in different ways)
        allocs = @allocated begin
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 10, 10)
            end
        end
        @test allocs == 0

        allocs = @allocated begin
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 20, 20)
            end
        end
        @test allocs == 0
    end

    @testset "Multiple slots with N-way cache" begin
        pool = AdaptiveArrayPool()

        # Warmup with multiple slots and dims
        for _ in 1:2
            @with_pool pool begin
                # Slot 1: two different shapes
                unsafe_acquire!(pool, Float64, 5, 5)
                # Slot 2: two different shapes
                unsafe_acquire!(pool, Float64, 10, 10)
            end
            @with_pool pool begin
                unsafe_acquire!(pool, Float64, 6, 6)
                unsafe_acquire!(pool, Float64, 12, 12)
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
    end

    @testset "Cache invalidation on resize" begin
        pool = AdaptiveArrayPool()

        # Warmup with small array
        @with_pool pool begin
            unsafe_acquire!(pool, Float64, 10, 10)
        end

        # Request larger array (forces resize)
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

    @testset "CACHE_WAYS constant" begin
        # Verify CACHE_WAYS is exported and accessible
        @test AdaptiveArrayPools.CACHE_WAYS == 4
    end
end
