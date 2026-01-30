@testset "BitArray Support" begin

    @testset "Bit sentinel type" begin
        # Bit is exported and usable
        @test Bit isa DataType

        # zero(Bit) and one(Bit) are defined for fill operations
        # These are used by zeros!(pool, Bit, ...) and ones!(pool, Bit, ...)
        @test zero(Bit) === false
        @test one(Bit) === true

        # Verify these work with fill!
        bv = BitVector(undef, 10)
        fill!(bv, zero(Bit))
        @test !any(bv)
        fill!(bv, one(Bit))
        @test all(bv)
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

    @testset "trues!(pool, dims...) - convenience for BitArray filled with true" begin
        pool = AdaptiveArrayPool()

        # 1D
        t1 = trues!(pool, 100)
        @test length(t1) == 100
        @test all(t1)
        @test eltype(t1) == Bool
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

        # Equivalent to ones!(pool, Bit, ...)
        t4 = trues!(pool, 50)
        t5 = ones!(pool, Bit, 50)
        @test all(t4 .== t5)
    end

    @testset "falses!(pool, dims...) - convenience for BitArray filled with false" begin
        pool = AdaptiveArrayPool()

        # 1D
        f1 = falses!(pool, 100)
        @test length(f1) == 100
        @test !any(f1)
        @test eltype(f1) == Bool
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

        # Equivalent to zeros!(pool, Bit, ...)
        f4 = falses!(pool, 50)
        f5 = zeros!(pool, Bit, 50)
        @test all(f4 .== f5)
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

        # ones! with Bit - Tuple form (covers convenience.jl:484)
        t_tuple = ones!(DISABLED_CPU, Bit, (4, 4))
        @test t_tuple isa BitArray{2}
        @test all(t_tuple)

        # zeros! with Bit (like falses)
        f = zeros!(DISABLED_CPU, Bit, 50)
        @test f isa BitVector
        @test !any(f)

        f2d = zeros!(DISABLED_CPU, Bit, 5, 5)
        @test f2d isa BitArray{2}
        @test !any(f2d)

        # zeros! with Bit - Tuple form (covers convenience.jl:482)
        f_tuple = zeros!(DISABLED_CPU, Bit, (4, 4))
        @test f_tuple isa BitArray{2}
        @test !any(f_tuple)

        # trues! (convenience for BitVector filled with true)
        t_trues = trues!(DISABLED_CPU, 50)
        @test t_trues isa BitVector
        @test all(t_trues)

        t_trues_2d = trues!(DISABLED_CPU, 5, 5)
        @test t_trues_2d isa BitArray{2}
        @test all(t_trues_2d)

        t_trues_tuple = trues!(DISABLED_CPU, (4, 4))
        @test t_trues_tuple isa BitArray{2}
        @test all(t_trues_tuple)

        # falses! (convenience for BitVector filled with false)
        f_falses = falses!(DISABLED_CPU, 50)
        @test f_falses isa BitVector
        @test !any(f_falses)

        f_falses_2d = falses!(DISABLED_CPU, 5, 5)
        @test f_falses_2d isa BitArray{2}
        @test !any(f_falses_2d)

        f_falses_tuple = falses!(DISABLED_CPU, (4, 4))
        @test f_falses_tuple isa BitArray{2}
        @test !any(f_falses_tuple)
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

        # Test trues!/falses! within @with_pool
        result2 = @with_pool pool begin
            t = trues!(pool, 100)
            f = falses!(pool, 50)

            @test all(t)
            @test !any(f)

            (count(t), count(f))
        end

        @test result2 == (100, 0)
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

            # Test trues!/falses! with pooling disabled
            result3 = @maybe_with_pool pool begin
                t = trues!(pool, 50)
                f = falses!(pool, 30)
                @test t isa BitVector  # Falls back to Julia's trues()
                @test f isa BitVector  # Falls back to Julia's falses()
                @test all(t)
                @test !any(f)
                (count(t), count(f))
            end
            @test result3 == (50, 0)

            # Test ones!/zeros! with Bit type, pooling disabled
            result4 = @maybe_with_pool pool begin
                o = ones!(pool, Bit, 40)
                z = zeros!(pool, Bit, 20)
                @test o isa BitVector  # Falls back to Julia's trues()
                @test z isa BitVector  # Falls back to Julia's falses()
                @test all(o)
                @test !any(z)
                (count(o), count(z))
            end
            @test result4 == (40, 0)
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

    @testset "unsafe_acquire! returns BitVector with shared chunks" begin
        pool = AdaptiveArrayPool()

        # unsafe_acquire! with Bit returns a real BitVector (not SubArray)
        bv = unsafe_acquire!(pool, Bit, 100)
        @test bv isa BitVector
        @test length(bv) == 100

        # N-D returns BitArray (reshape of BitVector becomes BitArray in Julia)
        ba = unsafe_acquire!(pool, Bit, 10, 10)
        @test ba isa BitMatrix  # reshape(BitVector, dims) → BitArray
        @test size(ba) == (10, 10)

        # Tuple form
        ba_tuple = unsafe_acquire!(pool, Bit, (10, 10))
        @test ba_tuple isa BitMatrix
        @test size(ba_tuple) == (10, 10)

        # Verify chunks sharing (key feature!)
        @with_pool pool2 begin
            bv2 = unsafe_acquire!(pool2, Bit, 100)
            pool_bv = pool2.bits.vectors[1]
            @test bv2.chunks === pool_bv.chunks  # Same chunks object!

            # Verify data is shared
            bv2[1] = true
            @test pool_bv[1] == true
            bv2[1] = false
            @test pool_bv[1] == false
        end
    end

    @testset "unsafe_acquire! SIMD performance" begin
        # Verify that unsafe_acquire! preserves SIMD-optimized operations
        pool = AdaptiveArrayPool()

        @with_pool pool begin
            n = 10000

            # Setup: fill with known pattern
            bv_unsafe = unsafe_acquire!(pool, Bit, n)
            fill!(bv_unsafe, true)

            # count() should work correctly
            @test count(bv_unsafe) == n

            # Verify it's using the fast path (type check)
            @test bv_unsafe isa BitVector

            # Compare with acquire! (SubArray)
            bv_view = acquire!(pool, Bit, n)
            fill!(bv_view, true)
            @test count(bv_view) == n
            @test bv_view isa SubArray
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

    @testset "NTuple form coverage" begin
        pool = AdaptiveArrayPool()

        # Test NTuple forms for trues!/falses! (covers _trues_impl! and _falses_impl! NTuple overloads)
        t_tuple = trues!(pool, (5, 5))
        @test size(t_tuple) == (5, 5)
        @test all(t_tuple)

        f_tuple = falses!(pool, (5, 5))
        @test size(f_tuple) == (5, 5)
        @test !any(f_tuple)

        # Test NTuple forms for zeros!/ones! with Bit type
        # (covers _zeros_impl! and _ones_impl! with Bit NTuple overloads)
        z_bit_tuple = zeros!(pool, Bit, (4, 4))
        @test size(z_bit_tuple) == (4, 4)
        @test !any(z_bit_tuple)

        o_bit_tuple = ones!(pool, Bit, (4, 4))
        @test size(o_bit_tuple) == (4, 4)
        @test all(o_bit_tuple)
    end

    @testset "Generic DisabledPool fallback for unknown backend" begin
        # Test that trues!/falses! throw BackendNotLoadedError for unknown backends
        unknown_pool = DisabledPool{:unknown_backend}()

        @test_throws AdaptiveArrayPools.BackendNotLoadedError trues!(unknown_pool, 10)
        @test_throws AdaptiveArrayPools.BackendNotLoadedError falses!(unknown_pool, 10)

        # Verify error message
        try
            trues!(unknown_pool, 10)
        catch e
            @test e isa AdaptiveArrayPools.BackendNotLoadedError
            @test e.backend == :unknown_backend
        end
    end

    @testset "_impl! delegators for DisabledPool" begin
        # Test _trues_impl! and _falses_impl! for DisabledPool (macro transformation path)
        # These are called when @maybe_with_pool transforms trues!/falses! calls

        # Vararg form
        t = AdaptiveArrayPools._trues_impl!(DISABLED_CPU, 10)
        @test t isa BitVector
        @test all(t)

        f = AdaptiveArrayPools._falses_impl!(DISABLED_CPU, 10)
        @test f isa BitVector
        @test !any(f)

        # NTuple form
        t_tuple = AdaptiveArrayPools._trues_impl!(DISABLED_CPU, (5, 5))
        @test t_tuple isa BitArray{2}
        @test all(t_tuple)

        f_tuple = AdaptiveArrayPools._falses_impl!(DISABLED_CPU, (5, 5))
        @test f_tuple isa BitArray{2}
        @test !any(f_tuple)
    end

    @testset "_impl! with Bit type NTuple for AbstractArrayPool" begin
        # Test _zeros_impl! and _ones_impl! with Bit type NTuple form
        # These are internal functions called by macro transformation
        pool = AdaptiveArrayPool()

        # Direct calls to _impl! functions with Bit type and NTuple
        z = AdaptiveArrayPools._zeros_impl!(pool, Bit, (3, 3))
        @test size(z) == (3, 3)
        @test !any(z)

        o = AdaptiveArrayPools._ones_impl!(pool, Bit, (3, 3))
        @test size(o) == (3, 3)
        @test all(o)
    end

end # BitArray Support
