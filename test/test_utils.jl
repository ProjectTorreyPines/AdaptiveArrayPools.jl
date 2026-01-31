import AdaptiveArrayPools: _validate_pool_return

# Helper macro to capture stdout (must be defined before use)
macro capture_out(expr)
    quote
        local old_stdout = stdout
        local rd, wr = redirect_stdout()
        try
            $(esc(expr))
            redirect_stdout(old_stdout)
            close(wr)
            read(rd, String)
        catch e
            redirect_stdout(old_stdout)
            close(wr)
            rethrow(e)
        end
    end
end

@testset "Utilities and Debugging" begin

    # ==============================================================================
    # Tests for utils.jl: POOL_DEBUG, _validate_pool_return, pool_stats
    # ==============================================================================


    @testset "pool_stats" begin
        pool = AdaptiveArrayPool()

        # Empty pool stats
        output = @capture_out pool_stats(pool)
        @test occursin("AdaptiveArrayPool", output)
        @test occursin("empty", output)

        # Add some vectors to fixed slots
        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 100)
        v2 = acquire!(pool, Float32, 50)
        v3 = acquire!(pool, Int64, 25)

        output = @capture_out pool_stats(pool)
        @test occursin("Float64 (fixed)", output)
        @test occursin("Float32 (fixed)", output)
        @test occursin("Int64 (fixed)", output)
        @test occursin("slots: 1", output)
        @test occursin("active: 1", output)

        rewind!(pool)

        # Test with fallback types (others)
        checkpoint!(pool)
        v_uint8 = acquire!(pool, UInt8, 200)

        output = @capture_out pool_stats(pool)
        @test occursin("UInt8 (fallback)", output)
        @test occursin("elements: 200", output)

        rewind!(pool)
    end

    @testset "pool_stats with backend symbol" begin
        # pool_stats(:cpu) should work
        output = @capture_out pool_stats(:cpu)
        @test occursin("AdaptiveArrayPool", output)

        # pool_stats(:cuda) should throw MethodError (extension not loaded)
        @test_throws MethodError pool_stats(:cuda)

        # pool_stats() without args should work (shows all pools)
        pool = get_task_local_pool()
        checkpoint!(pool)
        acquire!(pool, Float64, 100)

        output = @capture_out pool_stats()
        @test occursin("AdaptiveArrayPool", output)
        @test occursin("Float64", output)

        rewind!(pool)
    end

    @testset "pool_stats output format" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Use acquire! to populate pool
        v = acquire!(pool, Float64, 100)

        output = @capture_out pool_stats(pool)

        # Check format
        @test occursin("slots:", output)
        @test occursin("elements:", output)

        rewind!(pool)
    end

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

    @testset "Base.show for TypedPool & BitTypedPool" begin
        import AdaptiveArrayPools: TypedPool, BitTypedPool 

        # Empty TypedPool - compact show
        tp_empty = TypedPool{Float64}()
        output = sprint(show, tp_empty)
        @test output == "TypedPool{Float64}(empty)"

        # Non-empty TypedPool - compact show
        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Float64, 50)

        acquire!(pool, Bit, 10)

        output = sprint(show, pool.float64)
        @test occursin("TypedPool{Float64}", output)
        @test occursin("slots=2", output)
        @test occursin("active=2", output)
        @test occursin("elements=150", output)

        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool.float64)
        @test occursin("TypedPool{Float64}", output)
        @test occursin("slots:", output)
        @test occursin("active:", output)

        # BitTypedPool - compact show
        output = sprint(show, pool.bits)
        @test output == "BitTypedPool(slots=1, active=1, bits=10)"
        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool.bits)
        @test occursin("BitTypedPool", output)
        @test occursin("slots:", output)
        @test occursin("active:", output)
        @test occursin("bits:", output)

        rewind!(pool)
    end

    @testset "Base.show for AdaptiveArrayPool" begin
        # Empty pool - compact show
        pool_empty = AdaptiveArrayPool()
        output = sprint(show, pool_empty)
        @test occursin("AdaptiveArrayPool", output)
        @test occursin("types=0", output)
        @test occursin("slots=0", output)
        @test occursin("active=0", output)

        # Non-empty pool - compact show
        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Int64, 50)
        acquire!(pool, UInt8, 25)  # fallback type

        output = sprint(show, pool)
        @test occursin("AdaptiveArrayPool", output)
        @test occursin("types=3", output)
        @test occursin("slots=3", output)
        @test occursin("active=3", output)

        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool)
        @test occursin("AdaptiveArrayPool", output)
        @test occursin("Float64 (fixed)", output)
        @test occursin("Int64 (fixed)", output)
        @test occursin("UInt8 (fallback)", output)

        rewind!(pool)
    end

    @testset "pool_stats for empty TypedPool" begin
        import AdaptiveArrayPools: TypedPool

        tp = TypedPool{Float64}()
        output = @capture_out pool_stats(tp)
        @test occursin("Float64", output)
        @test occursin("empty", output)
    end

    @testset "pool_stats for BitTypedPool" begin
        import AdaptiveArrayPools: BitTypedPool

        # Empty BitTypedPool
        btp = BitTypedPool()
        output = @capture_out pool_stats(btp)
        @test occursin("Bit", output)
        @test occursin("empty", output)

        # BitTypedPool with content (via AdaptiveArrayPool)
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Acquire some BitVectors
        bv1 = acquire!(pool, Bit, 100)
        bv2 = acquire!(pool, Bit, 200)

        output = @capture_out pool_stats(pool)
        @test occursin("Bit (fixed)", output)
        @test occursin("slots: 2", output)
        @test occursin("active: 2", output)
        @test occursin("bits:", output)  # BitTypedPool uses "bits" label, not "elements"
        @test occursin("300", output)     # Total bits: 100 + 200

        rewind!(pool)

        # Test direct BitTypedPool stats
        btp2 = BitTypedPool()
        # Manually add vectors for testing
        push!(btp2.vectors, BitVector(undef, 64))
        btp2.n_active = 1

        output = @capture_out pool_stats(btp2)
        @test occursin("Bit", output)
        @test occursin("slots: 1", output)
        @test occursin("bits: 64", output)
    end

    @testset "direct call of internal helpers" begin
        import AdaptiveArrayPools: _default_type_name, _vector_bytes, _count_label, TypedPool, BitTypedPool
        @test _default_type_name(TypedPool{Float64}()) == "Float64"
        @test _default_type_name(BitTypedPool()) == "Bit"
        @test _vector_bytes([1, 2, 3]) == Base.summarysize([1, 2, 3])
        @test _vector_bytes(BitVector(undef, 100)) == sizeof(BitVector(undef, 100).chunks)
        @test _count_label(TypedPool{Int}()) == "elements"
        @test _count_label(BitTypedPool()) == "bits"
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
    # Tests for _check_bitchunks_overlap (BitArray safety validation)
    # ==============================================================================

    @testset "_check_bitchunks_overlap - direct BitArray validation" begin
        import AdaptiveArrayPools: _check_bitchunks_overlap

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
        import AdaptiveArrayPools: _check_bitchunks_overlap

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

end # Utilities and Debugging