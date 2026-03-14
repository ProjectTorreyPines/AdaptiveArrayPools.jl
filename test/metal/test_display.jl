# Metal Display Tests
# Tests for pool_stats and Base.show methods for MetalTypedPool and MetalAdaptiveArrayPool

# Helper macro to capture stdout
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

@testset "Metal Display" begin

    @testset "pool_stats for MetalAdaptiveArrayPool" begin
        pool = get_task_local_metal_pool()
        empty!(pool)

        # Empty pool stats
        output = @capture_out pool_stats(pool)
        @test occursin("MetalAdaptiveArrayPool", output)
        @test occursin("device", output)
        @test occursin("empty", output)

        # Add some arrays
        checkpoint!(pool)
        acquire!(pool, Float32, 100)
        acquire!(pool, Int32, 50)
        acquire!(pool, Float16, 25)

        output = @capture_out pool_stats(pool)
        @test occursin("Float32 (fixed)", output)
        @test occursin("Int32 (fixed)", output)
        @test occursin("Float16 (fixed)", output)
        @test occursin("Metal", output)
        @test occursin("slots:", output)
        @test occursin("active:", output)

        rewind!(pool)
    end

    @testset "pool_stats(:metal) dispatch" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, Float32, 100)

        output = @capture_out pool_stats(:metal)
        @test occursin("MetalAdaptiveArrayPool", output)
        @test occursin("Float32", output)

        rewind!(pool)
    end

    @testset "pool_stats output format" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, Float32, 100)

        output = @capture_out pool_stats(pool)

        @test occursin("slots:", output)
        @test occursin("elements:", output)
        @test occursin("bytes", output)

        rewind!(pool)
    end

    @testset "pool_stats for MetalTypedPool" begin
        pool = get_task_local_metal_pool()
        empty!(pool)

        # Empty MetalTypedPool
        output = @capture_out pool_stats(pool.float32)
        @test occursin("Float32", output)
        @test occursin("empty", output)

        # Non-empty MetalTypedPool
        checkpoint!(pool)
        acquire!(pool, Float32, 100)
        acquire!(pool, Float32, 200)

        output = @capture_out pool_stats(pool.float32)
        @test occursin("Float32", output)
        @test occursin("Metal", output)
        @test occursin("slots:", output)
        @test occursin("elements:", output)

        rewind!(pool)
    end

    @testset "pool_stats with fallback types" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, UInt8, 200)

        output = @capture_out pool_stats(pool)
        @test occursin("UInt8 (fallback)", output)
        @test occursin("elements: 200", output)

        rewind!(pool)
    end

    @testset "Base.show for MetalTypedPool" begin
        pool = get_task_local_metal_pool()
        empty!(pool)

        # Empty - compact show
        output = sprint(show, pool.float32)
        @test occursin("MetalTypedPool", output)
        @test occursin("empty", output)

        # Non-empty - compact show
        checkpoint!(pool)
        acquire!(pool, Float32, 100)
        acquire!(pool, Float32, 50)

        output = sprint(show, pool.float32)
        @test occursin("MetalTypedPool", output)
        @test occursin("slots=2", output)
        @test occursin("active=2", output)
        @test occursin("elements=150", output)

        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool.float32)
        @test occursin("MetalTypedPool", output)
        @test occursin("slots:", output)
        @test occursin("Metal", output)

        rewind!(pool)
    end

    @testset "Base.show for MetalAdaptiveArrayPool" begin
        pool = get_task_local_metal_pool()
        empty!(pool)

        # Empty pool - compact show
        output = sprint(show, pool)
        @test occursin("MetalAdaptiveArrayPool", output)
        @test occursin("types=0", output)
        @test occursin("slots=0", output)

        # Non-empty pool - compact show
        checkpoint!(pool)
        acquire!(pool, Float32, 100)
        acquire!(pool, Int32, 50)
        acquire!(pool, UInt8, 25)  # fallback

        output = sprint(show, pool)
        @test occursin("MetalAdaptiveArrayPool", output)
        @test occursin("types=3", output)
        @test occursin("slots=3", output)
        @test occursin("active=3", output)

        # Multi-line show (MIME"text/plain")
        output = sprint(show, MIME("text/plain"), pool)
        @test occursin("MetalAdaptiveArrayPool", output)
        @test occursin("Float32 (fixed)", output)
        @test occursin("Int32 (fixed)", output)
        @test occursin("UInt8 (fallback)", output)

        rewind!(pool)
    end

    @testset "pool_stats returns nothing" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        result = pool_stats(pool; io = devnull)
        @test result === nothing

        result = pool_stats(:metal; io = devnull)
        @test result === nothing
    end

    @testset "Float16 display (GPU ML type)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        checkpoint!(pool)
        acquire!(pool, Float16, 100)

        output = @capture_out pool_stats(pool)
        @test occursin("Float16 (fixed)", output)
        @test occursin("Metal", output)

        rewind!(pool)
    end

end
