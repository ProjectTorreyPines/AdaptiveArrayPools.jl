# Helper macro to capture stdout (must be defined before use)
macro capture_out(expr)
    return quote
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

@testset "Statistics and Display" begin

    # ==============================================================================
    # Tests for utils.jl: pool_stats, Base.show
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

end # Statistics and Display
