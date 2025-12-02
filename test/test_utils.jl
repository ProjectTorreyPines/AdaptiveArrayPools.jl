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
    @test occursin("arrays: 1", output)
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

@testset "_format_bytes" begin
    import AdaptiveArrayPools: _format_bytes

    # Bytes (< 1024)
    @test _format_bytes(0) == "0 bytes"
    @test _format_bytes(100) == "100 bytes"
    @test _format_bytes(1023) == "1023 bytes"

    # KiB (1024 <= bytes < 1024^2)
    @test _format_bytes(1024) == "1.000 KiB"
    @test _format_bytes(2048) == "2.000 KiB"
    @test _format_bytes(1536) == "1.500 KiB"  # 1.5 KiB

    # MiB (1024^2 <= bytes < 1024^3)
    @test _format_bytes(1024^2) == "1.000 MiB"
    @test _format_bytes(2 * 1024^2) == "2.000 MiB"
    @test _format_bytes(Int(1.5 * 1024^2)) == "1.500 MiB"

    # GiB (bytes >= 1024^3)
    @test _format_bytes(1024^3) == "1.000 GiB"
    @test _format_bytes(2 * 1024^3) == "2.000 GiB"
end

@testset "Base.show for TypedPool" begin
    import AdaptiveArrayPools: TypedPool

    # Empty TypedPool - compact show
    tp_empty = TypedPool{Float64}()
    output = sprint(show, tp_empty)
    @test output == "TypedPool{Float64}(empty)"

    # Non-empty TypedPool - compact show
    pool = AdaptiveArrayPool()
    checkpoint!(pool)
    acquire!(pool, Float64, 100)
    acquire!(pool, Float64, 50)

    output = sprint(show, pool.float64)
    @test occursin("TypedPool{Float64}", output)
    @test occursin("vectors=2", output)
    @test occursin("active=2", output)
    @test occursin("elements=150", output)

    # Multi-line show (MIME"text/plain")
    output = sprint(show, MIME("text/plain"), pool.float64)
    @test occursin("TypedPool{Float64}", output)
    @test occursin("arrays:", output)
    @test occursin("active:", output)

    rewind!(pool)
end

@testset "Base.show for AdaptiveArrayPool" begin
    # Empty pool - compact show
    pool_empty = AdaptiveArrayPool()
    output = sprint(show, pool_empty)
    @test occursin("AdaptiveArrayPool", output)
    @test occursin("types=0", output)
    @test occursin("vectors=0", output)
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
    @test occursin("vectors=3", output)
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

@testset "_validate_pool_return with N-D arrays" begin
    pool = AdaptiveArrayPool()
    checkpoint!(pool)

    # N-D SubArray from pool should fail validation (pointer overlap check)
    mat = acquire!(pool, Float64, 10, 10)
    @test mat isa SubArray{Float64, 2}
    @test_throws ErrorException _validate_pool_return(mat, pool)

    # 3D SubArray should also fail
    tensor = acquire!(pool, Float64, 5, 5, 5)
    @test tensor isa SubArray{Float64, 3}
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

    # N-D SubArray should throw error when returned
    @test_throws ErrorException @with_pool pool begin
        mat = acquire!(pool, Float64, 10, 10)
        mat  # Unsafe: returning pool-backed N-D SubArray
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
