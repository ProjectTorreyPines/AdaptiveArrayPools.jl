# ==============================================================================
# Tests for utils.jl: POOL_DEBUG, _validate_pool_return, pool_stats
# ==============================================================================

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

@testset "pool_stats" begin
    pool = AdaptiveArrayPool()

    # Empty pool stats
    output = @capture_out pool_stats(pool)
    @test occursin("AdaptiveArrayPool Statistics:", output)
    # No fixed slots should be printed for empty pool
    @test !occursin("Float64", output)

    # Add some vectors to fixed slots
    checkpoint!(pool)
    v1 = acquire!(pool, Float64, 100)
    v2 = acquire!(pool, Float32, 50)
    v3 = acquire!(pool, Int64, 25)

    output = @capture_out pool_stats(pool)
    @test occursin("Float64 (fixed slot):", output)
    @test occursin("Float32 (fixed slot):", output)
    @test occursin("Int64 (fixed slot):", output)
    @test occursin("Vectors: 1", output)
    @test occursin("Active:  1", output)

    rewind!(pool)

    # Test with fallback types (others)
    checkpoint!(pool)
    v_uint8 = acquire!(pool, UInt8, 200)

    output = @capture_out pool_stats(pool)
    @test occursin("UInt8 (fallback):", output)
    @test occursin("Total elements: 200", output)

    rewind!(pool)
end

@testset "POOL_DEBUG flag" begin
    old_debug = POOL_DEBUG[]

    # Default is false
    POOL_DEBUG[] = false
    pool = AdaptiveArrayPool()

    # When debug is off, no validation happens even if SubArray is returned
    result = @use_pool pool begin
        v = acquire!(pool, Float64, 10)
        v  # Returning SubArray - would be unsafe in real code
    end
    @test result isa SubArray  # No error when debug is off

    POOL_DEBUG[] = old_debug
end

@testset "POOL_DEBUG with safety violation" begin
    old_debug = POOL_DEBUG[]
    POOL_DEBUG[] = true

    pool = AdaptiveArrayPool()

    # Should throw error when returning SubArray with debug on
    @test_throws ErrorException @use_pool pool begin
        v = acquire!(pool, Float64, 10)
        v  # Unsafe: returning pool-backed SubArray
    end

    # Safe returns should work fine
    result = @use_pool pool begin
        v = acquire!(pool, Float64, 10)
        v .= 1.0
        sum(v)  # Safe: returning scalar
    end
    @test result == 10.0

    # Returning a copy is also safe
    result = @use_pool pool begin
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

    # Nothing pool always passes
    _validate_pool_return(pool_view, nothing)
    _validate_pool_return(42, nothing)
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
    v_bool = acquire!(pool, Bool, 5)

    @test_throws ErrorException _validate_pool_return(v_f64, pool)
    @test_throws ErrorException _validate_pool_return(v_f32, pool)
    @test_throws ErrorException _validate_pool_return(v_i64, pool)
    @test_throws ErrorException _validate_pool_return(v_i32, pool)
    @test_throws ErrorException _validate_pool_return(v_c64, pool)
    @test_throws ErrorException _validate_pool_return(v_bool, pool)

    rewind!(pool)
end
