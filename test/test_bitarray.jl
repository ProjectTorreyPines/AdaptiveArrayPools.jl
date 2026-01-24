@testset "BitArray Support" begin

    @testset "Bit sentinel type" begin
        # Bit is exported and usable
        @test Bit isa DataType
        @test zero(Bit) == false
        @test one(Bit) == true
    end

    @testset "BitTypedPool structure" begin
        pool = AdaptiveArrayPool()

        # Verify bits field exists and is correctly typed
        @test pool.bits isa AdaptiveArrayPools.BitTypedPool
        @test pool.bits.n_active == 0
        @test isempty(pool.bits.vectors)
    end

    @testset "acquire!(pool, Bit, n) - 1D" begin
        pool = AdaptiveArrayPool()

        bv = acquire!(pool, Bit, 100)
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
        bv2 = acquire!(pool, Bit, 50)
        @test length(bv2) == 50
        @test pool.bits.n_active == 2

        # Independent values
        bv2 .= false
        @test !any(bv2)
        @test count(bv) == 99  # bv unchanged
    end

    @testset "acquire!(pool, Bit, dims...) - N-D" begin
        pool = AdaptiveArrayPool()

        # 2D
        ba2 = acquire!(pool, Bit, 10, 10)
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
        ba3 = acquire!(pool, Bit, 4, 5, 3)
        @test size(ba3) == (4, 5, 3)
        @test pool.bits.n_active == 2

        # Tuple form
        ba_tuple = acquire!(pool, Bit, (3, 4, 2))
        @test size(ba_tuple) == (3, 4, 2)
        @test pool.bits.n_active == 3
    end

    @testset "ones!(pool, Bit, dims...) - filled with true" begin
        pool = AdaptiveArrayPool()

        # 1D
        t1 = ones!(pool, Bit, 100)
        @test length(t1) == 100
        @test all(t1)
        @test pool.bits.n_active == 1

        # 2D
        t2 = ones!(pool, Bit, 10, 10)
        @test size(t2) == (10, 10)
        @test all(t2)
        @test count(t2) == 100

        # Tuple form
        t3 = ones!(pool, Bit, (5, 5, 4))
        @test size(t3) == (5, 5, 4)
        @test all(t3)
    end

    @testset "zeros!(pool, Bit, dims...) - filled with false" begin
        pool = AdaptiveArrayPool()

        # 1D
        f1 = zeros!(pool, Bit, 100)
        @test length(f1) == 100
        @test !any(f1)
        @test pool.bits.n_active == 1

        # 2D
        f2 = zeros!(pool, Bit, 10, 10)
        @test size(f2) == (10, 10)
        @test !any(f2)
        @test count(f2) == 0

        # Tuple form
        f3 = zeros!(pool, Bit, (5, 5, 4))
        @test size(f3) == (5, 5, 4)
        @test !any(f3)
    end

    @testset "State management" begin
        # Use @with_pool which manages checkpoint/rewind automatically
        @with_pool outer_pool begin
            bv1 = acquire!(outer_pool, Bit, 100)
            parent1 = parent(bv1)

            @test outer_pool.bits.n_active == 1

            @with_pool inner_pool begin
                bv2 = acquire!(inner_pool, Bit, 200)
                @test inner_pool.bits.n_active == 2
            end
            # After inner scope rewind
            @test outer_pool.bits.n_active == 1

            # bv1 should still be valid (same parent BitVector object)
            bv3 = acquire!(outer_pool, Bit, 150)
            @test parent(bv1) === parent1  # Same object identity
        end
        # Pool goes back to task-local state after scope ends
    end

    @testset "checkpoint!/rewind! integration" begin
        pool = AdaptiveArrayPool()

        checkpoint!(pool)
        @test pool.bits.n_active == 0

        bv1 = acquire!(pool, Bit, 100)
        t1 = ones!(pool, Bit, 50)
        f1 = zeros!(pool, Bit, 50)
        @test pool.bits.n_active == 3

        rewind!(pool)
        @test pool.bits.n_active == 0
    end

    @testset "reset! and empty!" begin
        pool = AdaptiveArrayPool()

        bv1 = acquire!(pool, Bit, 100)
        bv2 = acquire!(pool, Bit, 200)
        @test pool.bits.n_active == 2
        @test length(pool.bits.vectors) >= 2

        # reset! preserves vectors
        reset!(pool)
        @test pool.bits.n_active == 0
        @test length(pool.bits.vectors) >= 2  # vectors preserved

        # empty! clears everything
        bv3 = acquire!(pool, Bit, 50)
        empty!(pool)
        @test pool.bits.n_active == 0
        @test isempty(pool.bits.vectors)
    end

    @testset "DisabledPool fallback" begin
        # acquire! with Bit
        bv = acquire!(DISABLED_CPU, Bit, 100)
        @test bv isa BitVector
        @test length(bv) == 100

        # N-D
        ba = acquire!(DISABLED_CPU, Bit, 10, 10)
        @test ba isa BitArray{2}
        @test size(ba) == (10, 10)

        # Tuple form
        ba_tuple = acquire!(DISABLED_CPU, Bit, (5, 5))
        @test ba_tuple isa BitArray{2}
        @test size(ba_tuple) == (5, 5)

        # ones! with Bit (like trues)
        t = ones!(DISABLED_CPU, Bit, 50)
        @test t isa BitVector
        @test all(t)

        t2d = ones!(DISABLED_CPU, Bit, 5, 5)
        @test t2d isa BitArray{2}
        @test all(t2d)

        # zeros! with Bit (like falses)
        f = zeros!(DISABLED_CPU, Bit, 50)
        @test f isa BitVector
        @test !any(f)

        f2d = zeros!(DISABLED_CPU, Bit, 5, 5)
        @test f2d isa BitArray{2}
        @test !any(f2d)
    end

    @testset "Memory efficiency vs Vector{Bool}" begin
        pool = AdaptiveArrayPool()

        # BitVector should use ~8x less memory than Vector{Bool}
        # (1 bit vs 1 byte per element)
        bv = acquire!(pool, Bit, 1000)
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
            bv = acquire!(pool, Bit, 100)
            t = ones!(pool, Bit, 50)
            f = zeros!(pool, Bit, 50)

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
            bv = acquire!(pool, Bit, 100)
            bv .= true
            count(bv)
        end
        @test result1 == 100

        # With pooling disabled
        AdaptiveArrayPools.MAYBE_POOLING_ENABLED[] = false
        try
            result2 = @maybe_with_pool pool begin
                bv = acquire!(pool, Bit, 100)
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

        # Vector{Bool} via acquire! with Bool
        vb = acquire!(pool, Bool, 100)
        @test vb isa SubArray{Bool, 1, Vector{Bool}}
        @test pool.bool.n_active == 1

        # BitVector via acquire! with Bit
        bv = acquire!(pool, Bit, 100)
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
            outer_bv = acquire!(outer_pool, Bit, 100)
            outer_bv .= true

            inner_result = @with_pool inner_pool begin
                inner_bv = acquire!(inner_pool, Bit, 50)
                inner_bv .= false
                count(inner_bv)
            end

            # outer_bv should still be valid
            @test all(outer_bv)
            (count(outer_bv), inner_result)
        end

        @test outer_result == (100, 0)
    end

    @testset "unsafe_acquire! not supported" begin
        pool = AdaptiveArrayPool()

        # unsafe_acquire! with Bit should throw a clear error
        @test_throws ArgumentError unsafe_acquire!(pool, Bit, 100)
        @test_throws ArgumentError unsafe_acquire!(pool, Bit, 10, 10)

        # Verify the error message is helpful
        try
            unsafe_acquire!(pool, Bit, 100)
        catch e
            @test e isa ArgumentError
            @test occursin("unsafe_acquire!", e.msg)
            @test occursin("Bit", e.msg)
            @test occursin("acquire!", e.msg)  # Suggests alternative
        end
    end

    @testset "API consistency" begin
        # Verify the API is consistent across types
        pool = AdaptiveArrayPool()

        # All these should work with the same pattern
        v_f64 = acquire!(pool, Float64, 10)
        v_i32 = acquire!(pool, Int32, 10)
        v_bool = acquire!(pool, Bool, 10)
        v_bit = acquire!(pool, Bit, 10)

        @test eltype(v_f64) == Float64
        @test eltype(v_i32) == Int32
        @test eltype(v_bool) == Bool
        @test eltype(v_bit) == Bool

        # zeros!/ones! work consistently
        z_f64 = zeros!(pool, Float64, 10)
        z_bit = zeros!(pool, Bit, 10)
        o_f64 = ones!(pool, Float64, 10)
        o_bit = ones!(pool, Bit, 10)

        @test all(z_f64 .== 0.0)
        @test !any(z_bit)
        @test all(o_f64 .== 1.0)
        @test all(o_bit)
    end

end # BitArray Support
