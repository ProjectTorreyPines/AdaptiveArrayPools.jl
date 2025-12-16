# CUDA Display Tests
# Tests for pool_stats and Base.show methods for CuTypedPool and CuAdaptiveArrayPool

# Helper macro to capture stdout
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

@testset "CUDA Display" begin

    @testset "pool_stats for CuAdaptiveArrayPool" begin
        pool = get_task_local_cuda_pool()
        empty!(pool)

        # Empty pool stats
        output = @capture_out pool_stats(pool)
        @test occursin("CuAdaptiveArrayPool", output)
        @test occursin("device", output)
        @test occursin("empty", output)

        # Add some arrays
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Float32, 50)
        acquire!(pool, Int32, 25)

        output = @capture_out pool_stats(pool)
        @test occursin("Float64 (fixed)", output)
        @test occursin("Float32 (fixed)", output)
        @test occursin("Int32 (fixed)", output)
        @test occursin("GPU", output)
        @test occursin("slots:", output)
        @test occursin("active:", output)

        rewind!(pool)
    end

    @testset "pool_stats(:cuda) dispatch" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, Float64, 100)

        output = @capture_out pool_stats(:cuda)
        @test occursin("CuAdaptiveArrayPool", output)
        @test occursin("Float64", output)

        rewind!(pool)
    end

    @testset "pool_stats output format" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, Float64, 100)

        output = @capture_out pool_stats(pool)

        # Check format elements
        @test occursin("slots:", output)
        @test occursin("elements:", output)
        @test occursin("bytes", output)  # Size formatting

        rewind!(pool)
    end

    @testset "pool_stats for CuTypedPool" begin
        pool = get_task_local_cuda_pool()
        empty!(pool)

        # Empty CuTypedPool
        output = @capture_out pool_stats(pool.float64)
        @test occursin("Float64", output)
        @test occursin("empty", output)

        # Non-empty CuTypedPool
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Float64, 200)

        output = @capture_out pool_stats(pool.float64)
        @test occursin("Float64", output)
        @test occursin("GPU", output)
        @test occursin("slots:", output)
        @test occursin("elements:", output)

        rewind!(pool)
    end

    @testset "pool_stats with fallback types" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, UInt8, 200)  # Fallback type

        output = @capture_out pool_stats(pool)
        @test occursin("UInt8 (fallback)", output)
        @test occursin("elements: 200", output)

        rewind!(pool)
    end

    @testset "Base.show for CuTypedPool" begin
        pool = get_task_local_cuda_pool()
        empty!(pool)

        # Empty CuTypedPool - compact show
        output = sprint(show, pool.float64)
        @test output == "CuTypedPool{Float64}(empty)"

        # Non-empty CuTypedPool - compact show
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Float64, 50)

        output = sprint(show, pool.float64)
        @test occursin("CuTypedPool{Float64}", output)
        @test occursin("slots=2", output)
        @test occursin("active=2", output)
        @test occursin("elements=150", output)

        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool.float64)
        @test occursin("CuTypedPool{Float64}", output)
        @test occursin("slots:", output)
        @test occursin("GPU", output)

        rewind!(pool)
    end

    @testset "Base.show for CuAdaptiveArrayPool" begin
        pool = get_task_local_cuda_pool()
        empty!(pool)

        # Empty pool - compact show
        output = sprint(show, pool)
        @test occursin("CuAdaptiveArrayPool", output)
        @test occursin("device=", output)
        @test occursin("types=0", output)
        @test occursin("slots=0", output)

        # Non-empty pool - compact show
        checkpoint!(pool)
        acquire!(pool, Float64, 100)
        acquire!(pool, Int32, 50)
        acquire!(pool, UInt8, 25)  # fallback

        output = sprint(show, pool)
        @test occursin("CuAdaptiveArrayPool", output)
        @test occursin("types=3", output)
        @test occursin("slots=3", output)
        @test occursin("active=3", output)

        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool)
        @test occursin("CuAdaptiveArrayPool", output)
        @test occursin("Float64 (fixed)", output)
        @test occursin("Int32 (fixed)", output)
        @test occursin("UInt8 (fallback)", output)

        rewind!(pool)
    end

    @testset "pool_stats returns nothing" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # pool_stats should return nothing
        result = pool_stats(pool; io=devnull)
        @test result === nothing

        result = pool_stats(:cuda; io=devnull)
        @test result === nothing
    end

    @testset "Float16 display (GPU ML type)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, Float16, 100)

        output = @capture_out pool_stats(pool)
        @test occursin("Float16 (fixed)", output)
        @test occursin("GPU", output)

        rewind!(pool)
    end

end
