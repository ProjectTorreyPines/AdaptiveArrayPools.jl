import AdaptiveArrayPools: _validate_pool_return, _check_bitchunks_overlap

@testset "POOL_DEBUG and Safety Validation" begin

    # ==============================================================================
    # POOL_DEBUG flag toggle
    # ==============================================================================

    @testset "POOL_DEBUG flag" begin
        old_debug = POOL_DEBUG[]

        # Default is false
        POOL_DEBUG[] = false

        # When debug is off, no validation happens even if SubArray is returned
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v  # Returning SubArray - would be unsafe in real code
        end
        @test result isa SubArray  # No error when debug is off

        POOL_DEBUG[] = old_debug
    end

    @testset "POOL_DEBUG with safety violation" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # Should throw error when returning SubArray with debug on
        @test_throws ErrorException @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v  # Unsafe: returning pool-backed SubArray
        end

        # Safe returns should work fine
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            sum(v)  # Safe: returning scalar
        end
        @test result == 10.0

        # Returning a copy is also safe
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 5)
            v .= 2.0
            collect(v)  # Safe: returning a copy
        end
        @test result == [2.0, 2.0, 2.0, 2.0, 2.0]

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # _validate_pool_return — direct tests
    # ==============================================================================

    @testset "_validate_pool_return" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Non-SubArray values pass validation
        _validate_pool_return(42, pool)
        _validate_pool_return([1, 2, 3], pool)
        _validate_pool_return("hello", pool)
        _validate_pool_return(nothing, pool)

        # SubArray not from pool passes validation
        external_vec = [1.0, 2.0, 3.0]
        external_view = view(external_vec, 1:2)
        _validate_pool_return(external_view, pool)

        # SubArray from pool fails validation (fixed slot)
        pool_view = acquire!(pool, Float64, 10)
        @test_throws ErrorException _validate_pool_return(pool_view, pool)

        rewind!(pool)

        # Test with fallback type (others)
        checkpoint!(pool)
        pool_view_uint8 = acquire!(pool, UInt8, 10)
        @test_throws ErrorException _validate_pool_return(pool_view_uint8, pool)
        rewind!(pool)

        # DisabledPool always passes
        _validate_pool_return(pool_view, DISABLED_CPU)
        _validate_pool_return(42, DISABLED_CPU)
    end

    @testset "_validate_pool_return with all fixed slots" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Test each fixed slot type
        v_f64 = acquire!(pool, Float64, 5)
        v_f32 = acquire!(pool, Float32, 5)
        v_i64 = acquire!(pool, Int64, 5)
        v_i32 = acquire!(pool, Int32, 5)
        v_c64 = acquire!(pool, ComplexF64, 5)
        v_c32 = acquire!(pool, ComplexF32, 5)
        v_bool = acquire!(pool, Bool, 5)

        @test_throws ErrorException _validate_pool_return(v_f64, pool)
        @test_throws ErrorException _validate_pool_return(v_f32, pool)
        @test_throws ErrorException _validate_pool_return(v_i64, pool)
        @test_throws ErrorException _validate_pool_return(v_i32, pool)
        @test_throws ErrorException _validate_pool_return(v_c64, pool)
        @test_throws ErrorException _validate_pool_return(v_c32, pool)
        @test_throws ErrorException _validate_pool_return(v_bool, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with N-D arrays" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # N-D ReshapedArray from pool should fail validation (pointer overlap check)
        mat = acquire!(pool, Float64, 10, 10)
        @test mat isa Base.ReshapedArray{Float64, 2}
        @test_throws ErrorException _validate_pool_return(mat, pool)

        # 3D ReshapedArray should also fail
        tensor = acquire!(pool, Float64, 5, 5, 5)
        @test tensor isa Base.ReshapedArray{Float64, 3}
        @test_throws ErrorException _validate_pool_return(tensor, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with unsafe_acquire!" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Raw Vector from unsafe_acquire! should fail validation
        v = unsafe_acquire!(pool, Float64, 100)
        @test v isa Vector{Float64}
        @test_throws ErrorException _validate_pool_return(v, pool)

        # Raw Matrix from unsafe_acquire! should fail validation
        mat = unsafe_acquire!(pool, Float64, 10, 10)
        @test mat isa Matrix{Float64}
        @test_throws ErrorException _validate_pool_return(mat, pool)

        # Raw 3D Array from unsafe_acquire! should fail validation
        tensor = unsafe_acquire!(pool, Float64, 5, 5, 5)
        @test tensor isa Array{Float64, 3}
        @test_throws ErrorException _validate_pool_return(tensor, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with view(unsafe_acquire!)" begin
        # Bug fix test: view() wrapped around unsafe_acquire! result
        # The parent Vector/Array is created by unsafe_wrap, not the pool's internal vector
        # This requires pointer overlap check, not identity check
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D: view(unsafe_acquire!(...), :) should fail validation
        v = unsafe_acquire!(pool, Float64, 100)
        v_view = view(v, :)
        @test v_view isa SubArray
        @test parent(v_view) === v  # Parent is unsafe_wrap'd Vector, not pool's internal vector
        @test_throws ErrorException _validate_pool_return(v_view, pool)

        # Partial view should also fail
        v_partial = view(v, 1:50)
        @test_throws ErrorException _validate_pool_return(v_partial, pool)

        # 2D: view(unsafe_acquire!(...), :, :) should fail validation
        mat = unsafe_acquire!(pool, Float64, 10, 10)
        mat_view = view(mat, :, :)
        @test mat_view isa SubArray
        @test_throws ErrorException _validate_pool_return(mat_view, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return external arrays pass" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Acquire some memory to populate the pool
        _ = acquire!(pool, Float64, 100)

        # External N-D arrays should pass validation
        external_mat = zeros(Float64, 10, 10)
        external_view = view(external_mat, :, :)
        _validate_pool_return(external_view, pool)
        _validate_pool_return(external_mat, pool)

        # External 3D array should pass
        external_tensor = zeros(Float64, 5, 5, 5)
        _validate_pool_return(external_tensor, pool)

        rewind!(pool)
    end

    @testset "POOL_DEBUG with N-D arrays" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # N-D ReshapedArray should throw error when returned
        @test_throws ErrorException @with_pool pool begin
            mat = acquire!(pool, Float64, 10, 10)
            mat  # Unsafe: returning pool-backed N-D ReshapedArray
        end

        # Raw Array from unsafe_acquire! should throw error when returned
        @test_throws ErrorException @with_pool pool begin
            mat = unsafe_acquire!(pool, Float64, 10, 10)
            mat  # Unsafe: returning raw Array backed by pool
        end

        # Safe returns should work fine
        result = @with_pool pool begin
            mat = acquire!(pool, Float64, 10, 10)
            mat .= 1.0
            sum(mat)  # Safe: returning scalar
        end
        @test result == 100.0

        # Returning a copy is also safe
        result = @with_pool pool begin
            mat = acquire!(pool, Float64, 3, 3)
            mat .= 2.0
            collect(mat)  # Safe: returning a copy
        end
        @test result == fill(2.0, 3, 3)

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # BitArray overlap detection (_check_bitchunks_overlap)
    # ==============================================================================

    @testset "_check_bitchunks_overlap - direct BitArray validation" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D BitVector from pool - should detect overlap
        bv = acquire!(pool, Bit, 100)
        @test bv isa BitVector
        @test_throws ErrorException _check_bitchunks_overlap(bv, pool)

        # N-D BitArray from pool - should detect overlap (shares chunks with pool)
        ba = acquire!(pool, Bit, 10, 10)
        @test ba isa BitMatrix
        @test_throws ErrorException _check_bitchunks_overlap(ba, pool)

        # 3D BitArray from pool
        ba3 = acquire!(pool, Bit, 4, 5, 3)
        @test ba3 isa BitArray{3}
        @test_throws ErrorException _check_bitchunks_overlap(ba3, pool)

        rewind!(pool)
    end

    @testset "_check_bitchunks_overlap - external BitArray passes" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Populate pool with some BitVectors
        _ = acquire!(pool, Bit, 100)
        _ = acquire!(pool, Bit, 200)

        # External BitVector (not from pool) should pass validation
        external_bv = BitVector(undef, 50)
        _check_bitchunks_overlap(external_bv, pool)  # Should not throw

        # External BitMatrix should pass
        external_ba = BitArray(undef, 10, 10)
        _check_bitchunks_overlap(external_ba, pool)  # Should not throw

        # External 3D BitArray should pass
        external_ba3 = BitArray(undef, 5, 5, 5)
        _check_bitchunks_overlap(external_ba3, pool)  # Should not throw

        rewind!(pool)
    end

    @testset "_validate_pool_return with BitArray (via _check_bitchunks_overlap)" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Direct BitVector from pool fails validation
        bv = acquire!(pool, Bit, 100)
        @test_throws ErrorException _validate_pool_return(bv, pool)

        # Direct BitMatrix from pool fails validation
        ba = acquire!(pool, Bit, 10, 10)
        @test_throws ErrorException _validate_pool_return(ba, pool)

        # External BitArray passes validation
        external_bv = BitVector(undef, 50)
        _validate_pool_return(external_bv, pool)  # Should not throw

        rewind!(pool)
    end

    @testset "_validate_pool_return with SubArray{BitArray} parent" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Create a view of a pool BitVector
        bv = acquire!(pool, Bit, 100)
        bv_view = view(bv, 1:50)
        @test bv_view isa SubArray
        @test parent(bv_view) isa BitVector
        @test_throws ErrorException _validate_pool_return(bv_view, pool)

        # View of external BitVector should pass
        external_bv = BitVector(undef, 100)
        external_view = view(external_bv, 1:50)
        _validate_pool_return(external_view, pool)  # Should not throw

        rewind!(pool)
    end

    @testset "POOL_DEBUG with BitArray" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # BitVector from pool should throw error when returned with debug on
        @test_throws ErrorException @with_pool pool begin
            bv = acquire!(pool, Bit, 100)
            bv  # Unsafe: returning pool-backed BitVector
        end

        # BitMatrix from pool should throw error when returned
        @test_throws ErrorException @with_pool pool begin
            ba = acquire!(pool, Bit, 10, 10)
            ba  # Unsafe: returning pool-backed BitMatrix
        end

        # Safe returns should work fine
        result = @with_pool pool begin
            bv = acquire!(pool, Bit, 100)
            bv .= true
            count(bv)  # Safe: returning scalar
        end
        @test result == 100

        # Returning a copy is also safe
        result = @with_pool pool begin
            bv = acquire!(pool, Bit, 5)
            bv .= true
            copy(bv)  # Safe: returning a copy
        end
        @test result == trues(5)

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # POOL_DEBUG with function definition forms
    # ==============================================================================

    @testset "POOL_DEBUG with @with_pool function definition" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # Unsafe: function implicitly returns pool-backed SubArray
        @with_pool pool function _test_debug_func_unsafe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            v  # Implicit return of pool-backed SubArray
        end
        @test_throws ErrorException _test_debug_func_unsafe(10)

        # Safe: function returns scalar
        @with_pool pool function _test_debug_func_safe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            sum(v)
        end
        @test _test_debug_func_safe(10) == 10.0

        # Safe: function returns a copy
        @with_pool pool function _test_debug_func_copy(n)
            v = acquire!(pool, Float64, n)
            v .= 2.0
            collect(v)
        end
        @test _test_debug_func_copy(5) == fill(2.0, 5)

        # Unsafe: N-D ReshapedArray from function
        @with_pool pool function _test_debug_func_nd(m, n)
            mat = acquire!(pool, Float64, m, n)
            mat .= 1.0
            mat
        end
        @test_throws ErrorException _test_debug_func_nd(3, 4)

        # Unsafe: BitVector from function
        @with_pool pool function _test_debug_func_bit(n)
            bv = acquire!(pool, Bit, n)
            bv .= true
            bv
        end
        @test_throws ErrorException _test_debug_func_bit(100)

        POOL_DEBUG[] = old_debug
    end

    @testset "POOL_DEBUG with @maybe_with_pool function definition" begin
        old_debug = POOL_DEBUG[]
        old_maybe = MAYBE_POOLING[]
        POOL_DEBUG[] = true
        MAYBE_POOLING[] = true

        # Unsafe: function returns pool-backed array
        @maybe_with_pool pool function _test_maybe_debug_unsafe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            v
        end
        @test_throws ErrorException _test_maybe_debug_unsafe(10)

        # Safe: function returns scalar
        @maybe_with_pool pool function _test_maybe_debug_safe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            sum(v)
        end
        @test _test_maybe_debug_safe(10) == 10.0

        # When pooling disabled, no validation needed (DisabledPool returns fresh arrays)
        MAYBE_POOLING[] = false
        @maybe_with_pool pool function _test_maybe_debug_disabled(n)
            v = zeros!(pool, n)
            v
        end
        result = _test_maybe_debug_disabled(5)
        @test result == zeros(5)

        POOL_DEBUG[] = old_debug
        MAYBE_POOLING[] = old_maybe
    end

    @testset "POOL_DEBUG with @with_pool :cpu function definition" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # Unsafe: backend function returns pool-backed array
        @with_pool :cpu pool function _test_backend_debug_unsafe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            v
        end
        @test_throws ErrorException _test_backend_debug_unsafe(10)

        # Safe: returns scalar
        @with_pool :cpu pool function _test_backend_debug_safe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            sum(v)
        end
        @test _test_backend_debug_safe(10) == 10.0

        POOL_DEBUG[] = old_debug
    end

end # POOL_DEBUG and Safety Validation
