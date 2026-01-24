@testset "BitArray Support" begin

    @testset "BitTypedPool structure" begin
        pool = AdaptiveArrayPool()

        # Verify bits field exists and is correctly typed
        @test pool.bits isa AdaptiveArrayPools.BitTypedPool
        @test pool.bits.n_active == 0
        @test isempty(pool.bits.vectors)
    end

    @testset "acquire_bits! 1D" begin
        pool = AdaptiveArrayPool()

        bv = acquire_bits!(pool, 100)
        @test length(bv) == 100
        @test eltype(bv) == Bool
        @test bv isa SubArray{Bool, 1, BitVector}
        @test pool.bits.n_active == 1

        # Write and read back
        bv .= true
        @test all(bv)
        bv[50] = false
        @test !bv[50]
        @test count(bv) == 99

        # Second acquire
        bv2 = acquire_bits!(pool, 50)
        @test length(bv2) == 50
        @test pool.bits.n_active == 2

        # Independent values
        bv2 .= false
        @test !any(bv2)
        @test count(bv) == 99  # bv unchanged
    end

    @testset "acquire_bits! N-D" begin
        pool = AdaptiveArrayPool()

        # 2D
        ba2 = acquire_bits!(pool, 10, 10)
        @test size(ba2) == (10, 10)
        @test eltype(ba2) == Bool
        @test ba2 isa Base.ReshapedArray
        @test pool.bits.n_active == 1

        # Test indexing
        ba2 .= false
        ba2[1, 1] = true
        ba2[5, 5] = true
        @test count(ba2) == 2
        @test ba2[1, 1]
        @test ba2[5, 5]
        @test !ba2[2, 2]

        # 3D
        ba3 = acquire_bits!(pool, 4, 5, 3)
        @test size(ba3) == (4, 5, 3)
        @test pool.bits.n_active == 2

        # Tuple form
        ba_tuple = acquire_bits!(pool, (3, 4, 2))
        @test size(ba_tuple) == (3, 4, 2)
        @test pool.bits.n_active == 3
    end

    @testset "trues!" begin
        pool = AdaptiveArrayPool()

        # 1D
        t1 = trues!(pool, 100)
        @test length(t1) == 100
        @test all(t1)
        @test pool.bits.n_active == 1

        # 2D
        t2 = trues!(pool, 10, 10)
        @test size(t2) == (10, 10)
        @test all(t2)
        @test count(t2) == 100

        # Tuple form
        t3 = trues!(pool, (5, 5, 4))
        @test size(t3) == (5, 5, 4)
        @test all(t3)
    end

    @testset "falses!" begin
        pool = AdaptiveArrayPool()

        # 1D
        f1 = falses!(pool, 100)
        @test length(f1) == 100
        @test !any(f1)
        @test pool.bits.n_active == 1

        # 2D
        f2 = falses!(pool, 10, 10)
        @test size(f2) == (10, 10)
        @test !any(f2)
        @test count(f2) == 0

        # Tuple form
        f3 = falses!(pool, (5, 5, 4))
        @test size(f3) == (5, 5, 4)
        @test !any(f3)
    end

    @testset "State management" begin
        # Use @with_pool which manages checkpoint/rewind automatically
        @with_pool outer_pool begin
            bv1 = acquire_bits!(outer_pool, 100)
            parent1 = parent(bv1)

            @test outer_pool.bits.n_active == 1

            @with_pool inner_pool begin
                bv2 = acquire_bits!(inner_pool, 200)
                @test inner_pool.bits.n_active == 2
            end
            # After inner scope rewind
            @test outer_pool.bits.n_active == 1

            # bv1 should still be valid (same parent BitVector object)
            bv3 = acquire_bits!(outer_pool, 150)
            @test parent(bv1) === parent1  # Same object identity
        end
        # Pool goes back to task-local state after scope ends
    end

    @testset "checkpoint!/rewind! integration" begin
        pool = AdaptiveArrayPool()

        checkpoint!(pool)
        @test pool.bits.n_active == 0

        bv1 = acquire_bits!(pool, 100)
        t1 = trues!(pool, 50)
        f1 = falses!(pool, 50)
        @test pool.bits.n_active == 3

        rewind!(pool)
        @test pool.bits.n_active == 0
    end

    @testset "reset! and empty!" begin
        pool = AdaptiveArrayPool()

        bv1 = acquire_bits!(pool, 100)
        bv2 = acquire_bits!(pool, 200)
        @test pool.bits.n_active == 2
        @test length(pool.bits.vectors) >= 2

        # reset! preserves vectors
        reset!(pool)
        @test pool.bits.n_active == 0
        @test length(pool.bits.vectors) >= 2  # vectors preserved

        # empty! clears everything
        bv3 = acquire_bits!(pool, 50)
        empty!(pool)
        @test pool.bits.n_active == 0
        @test isempty(pool.bits.vectors)
    end

    @testset "DisabledPool fallback" begin
        # acquire_bits!
        bv = acquire_bits!(DISABLED_CPU, 100)
        @test bv isa BitVector
        @test length(bv) == 100

        # N-D
        ba = acquire_bits!(DISABLED_CPU, 10, 10)
        @test ba isa BitArray{2}
        @test size(ba) == (10, 10)

        # Tuple form
        ba_tuple = acquire_bits!(DISABLED_CPU, (5, 5))
        @test ba_tuple isa BitArray{2}
        @test size(ba_tuple) == (5, 5)

        # trues!
        t = trues!(DISABLED_CPU, 50)
        @test t isa BitVector
        @test all(t)

        t2d = trues!(DISABLED_CPU, 5, 5)
        @test t2d isa BitArray{2}
        @test all(t2d)

        # falses!
        f = falses!(DISABLED_CPU, 50)
        @test f isa BitVector
        @test !any(f)

        f2d = falses!(DISABLED_CPU, 5, 5)
        @test f2d isa BitArray{2}
        @test !any(f2d)
    end

    @testset "Memory efficiency vs Vector{Bool}" begin
        pool = AdaptiveArrayPool()

        # BitVector should use ~8x less memory than Vector{Bool}
        # (1 bit vs 1 byte per element)
        bv = acquire_bits!(pool, 1000)
        vb = acquire!(pool, Bool, 1000)

        bv_parent = parent(bv)
        vb_parent = parent(vb)

        # BitVector stores 64 bits per chunk (UInt64)
        @test sizeof(bv_parent.chunks) < sizeof(vb_parent)
        # Approximate: BitVector ~125 bytes (1000/8), Vector{Bool} ~1000 bytes
        @test sizeof(bv_parent.chunks) <= div(1000, 8) + 8  # allow some overhead
    end

    @testset "@with_pool macro integration" begin
        result = @with_pool pool begin
            bv = acquire_bits!(pool, 100)
            t = trues!(pool, 50)
            f = falses!(pool, 50)

            bv .= true
            sum_bv = count(bv)
            sum_t = count(t)
            sum_f = count(f)

            (sum_bv, sum_t, sum_f)
        end

        @test result == (100, 50, 0)
    end

    @testset "@maybe_with_pool macro integration" begin
        # With pooling enabled (default)
        result1 = @maybe_with_pool pool begin
            bv = acquire_bits!(pool, 100)
            bv .= true
            count(bv)
        end
        @test result1 == 100

        # With pooling disabled
        AdaptiveArrayPools.MAYBE_POOLING_ENABLED[] = false
        try
            result2 = @maybe_with_pool pool begin
                bv = acquire_bits!(pool, 100)
                @test bv isa BitVector  # DisabledPool returns BitVector
                bv .= true
                count(bv)
            end
            @test result2 == 100
        finally
            AdaptiveArrayPools.MAYBE_POOLING_ENABLED[] = true
        end
    end

    @testset "Mixed Bool types" begin
        pool = AdaptiveArrayPool()

        # Vector{Bool} via acquire!
        vb = acquire!(pool, Bool, 100)
        @test vb isa SubArray{Bool, 1, Vector{Bool}}
        @test pool.bool.n_active == 1

        # BitVector via acquire_bits!
        bv = acquire_bits!(pool, 100)
        @test bv isa SubArray{Bool, 1, BitVector}
        @test pool.bits.n_active == 1

        # Both should work independently
        vb .= true
        bv .= false
        @test all(vb)
        @test !any(bv)

        # Separate pools
        @test pool.bool.n_active == 1
        @test pool.bits.n_active == 1
    end

    @testset "Nested scopes" begin
        outer_result = @with_pool outer_pool begin
            outer_bv = acquire_bits!(outer_pool, 100)
            outer_bv .= true

            inner_result = @with_pool inner_pool begin
                inner_bv = acquire_bits!(inner_pool, 50)
                inner_bv .= false
                count(inner_bv)
            end

            # outer_bv should still be valid
            @test all(outer_bv)
            (count(outer_bv), inner_result)
        end

        @test outer_result == (100, 0)
    end

end # BitArray Support
