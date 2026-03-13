# CUDA Allocation Tests
# Verifies zero-allocation pooling behavior and GPU memory reuse

@testset "GPU Allocation" begin

    @testset "Memory reuse (same size)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # First acquire - populates pool
        @with_pool :cuda p begin
            v = acquire!(p, Float32, 100)
            v .= 1.0f0
        end

        # Second acquire (same size) - should reuse
        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                v = acquire!(p, Float32, 100)
                v .= 2.0f0
            end
        end

        # GPU allocation should be 0 (memory reused)
        @test alloc == 0
    end

    @testset "Memory reuse (multiple arrays)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup with 3 arrays
        @with_pool :cuda p begin
            acquire!(p, Float32, 100)
            acquire!(p, Float32, 200)
            acquire!(p, Float32, 300)
        end

        # Second pass should reuse all
        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                v1 = acquire!(p, Float32, 100)
                v2 = acquire!(p, Float32, 200)
                v3 = acquire!(p, Float32, 300)
                v1 .= 1.0f0; v2 .= 2.0f0; v3 .= 3.0f0
            end
        end

        @test alloc == 0
    end

    @testset "Memory reuse (N-D arrays)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup with 2D array
        @with_pool :cuda p begin
            A = acquire!(p, Float64, 10, 10)
            A .= 1.0
        end

        # Reuse check
        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 10, 10)
                A .= 2.0
            end
        end

        @test alloc == 0
    end

    @testset "Memory reuse (3D arrays)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup with 3D array
        @with_pool :cuda p begin
            T = acquire!(p, Float32, 5, 5, 4)
            T .= 1.0f0
        end

        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                T = acquire!(p, Float32, 5, 5, 4)
                T .= 2.0f0
            end
        end

        @test alloc == 0
    end

    @testset "Pointer reuse verification" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        ptr1 = Ref{UInt}(0)
        ptr2 = Ref{UInt}(0)

        @with_pool :cuda p begin
            v = acquire!(p, Float32, 1000)
            ptr1[] = UInt(pointer(v))
        end

        @with_pool :cuda p begin
            v = acquire!(p, Float32, 1000)
            ptr2[] = UInt(pointer(v))
        end

        # Same GPU memory address should be reused
        @test ptr1[] == ptr2[]
        @test ptr1[] != 0
    end

    @testset "acquire! N-D memory reuse" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup
        @with_pool :cuda p begin
            A = acquire!(p, Float64, 10, 10)
            A .= 1.0
        end

        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 10, 10)
                A .= 2.0
            end
        end

        @test alloc == 0
    end

    @testset "Comparison: pooled vs direct allocation" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)
        N = 1000
        ITERS = 10

        # Warmup pool
        @with_pool :cuda p begin
            acquire!(p, Float32, N)
        end

        # Measure pooled allocation
        GC.gc(); CUDA.reclaim()
        pooled_alloc = CUDA.@allocated begin
            for _ in 1:ITERS
                @with_pool :cuda p begin
                    v = acquire!(p, Float32, N)
                    v .= 1.0f0
                end
            end
        end

        # Measure direct allocation (no pool)
        GC.gc(); CUDA.reclaim()
        direct_alloc = CUDA.@allocated begin
            for _ in 1:ITERS
                v = CUDA.zeros(Float32, N)
                v .= 1.0f0
            end
        end

        # Pooled should allocate significantly less
        @test pooled_alloc < direct_alloc
    end

end

@testset "_resize_to_fit! GPU memory preservation" begin
    _resize_to_fit! = ext._resize_to_fit!

    @testset "Shrink preserves GPU pointer" begin
        v = CUDA.zeros(Float32, 1000)
        ptr = UInt(pointer(v))
        _resize_to_fit!(v, 100)
        @test length(v) == 100
        @test UInt(pointer(v)) == ptr
    end

    @testset "Grow-back within capacity: no realloc" begin
        v = CUDA.zeros(Float32, 1000)
        ptr = UInt(pointer(v))
        # Shrink first
        _resize_to_fit!(v, 100)
        @test length(v) == 100
        @test UInt(pointer(v)) == ptr
        # Grow back to original size — maxsize preserved, so no GPU realloc
        _resize_to_fit!(v, 1000)
        @test length(v) == 1000
        @test UInt(pointer(v)) == ptr
    end

    @testset "Shrink to 0, grow back preserves pointer" begin
        v = CUDA.zeros(Float32, 500)
        ptr = UInt(pointer(v))
        _resize_to_fit!(v, 0)
        @test length(v) == 0
        # GPU memory still allocated (not freed)
        # Grow back from 0 — within capacity, so no GPU realloc
        _resize_to_fit!(v, 500)
        @test length(v) == 500
        @test UInt(pointer(v)) == ptr
    end

    @testset "Grow within capacity after invalidation: no realloc" begin
        # This is the key test: after safety invalidation (dims→0),
        # re-acquire within original capacity should NOT trigger GPU realloc.
        # (CUDA.jl v5.9.x resize! would always reallocate; _resize_to_fit! avoids this)
        v = CUDA.zeros(Float32, 1000)
        ptr = UInt(pointer(v))
        # Simulate safety invalidation
        _resize_to_fit!(v, 0)
        @test length(v) == 0
        # Re-acquire at smaller size (still within original capacity)
        _resize_to_fit!(v, 200)
        @test length(v) == 200
        @test UInt(pointer(v)) == ptr  # Same GPU buffer
    end

    @testset "No-op when n == length" begin
        v = CUDA.zeros(Float32, 200)
        ptr = UInt(pointer(v))
        _resize_to_fit!(v, 200)
        @test length(v) == 200
        @test UInt(pointer(v)) == ptr
    end

    @testset "Grow beyond capacity delegates to resize!" begin
        v = CUDA.zeros(Float32, 100)
        _resize_to_fit!(v, 10_000)
        @test length(v) == 10_000
        # Pointer may change (new allocation) — just verify length is correct
    end
end

@testset "CPU Allocation (CuArray wrapper)" begin

    @testset "acquire! N-D has low CPU allocation (cache hit)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Wrap in function for proper JIT warmup (pool binding let-block
        # needs function boundary to avoid @allocated counting JIT artifacts)
        function _test_cuda_nd_alloc!()
            @with_pool :cuda p begin
                acquire!(p, Float64, 10, 10)
            end
        end

        # Warmup (JIT + cache)
        _test_cuda_nd_alloc!()
        _test_cuda_nd_alloc!()

        # Cache hit should have minimal CPU allocation
        cpu_alloc = @allocated _test_cuda_nd_alloc!()
        @test cpu_alloc < 100  # Allow some overhead
    end

    @testset "acquire! N-D cache hit returns cached wrapper" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function _test_cuda_nd_cached_alloc!()
            @with_pool :cuda p begin
                acquire!(p, Float64, 10, 10)
            end
        end

        # Warmup (JIT + cache)
        _test_cuda_nd_cached_alloc!()
        _test_cuda_nd_cached_alloc!()

        # Cache hit should have minimal CPU allocation
        cpu_alloc = @allocated _test_cuda_nd_cached_alloc!()
        @test cpu_alloc < 100  # Allow some overhead
    end

    @testset "acquire! 1D has low CPU allocation" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function _test_cuda_1d_alloc!()
            @with_pool :cuda p begin
                acquire!(p, Float64, 100)
            end
        end

        # Warmup (JIT + cache)
        _test_cuda_1d_alloc!()
        _test_cuda_1d_alloc!()

        # 1D acquire! uses view path, should be efficient
        cpu_alloc = @allocated _test_cuda_1d_alloc!()
        @test cpu_alloc < 200
    end

end

@testset "Mixed Type Allocation" begin

    @testset "Multiple types maintain separate pools" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup all types
        @with_pool :cuda p begin
            acquire!(p, Float32, 100)
            acquire!(p, Float64, 100)
            acquire!(p, Int32, 100)
        end

        # Reuse all types
        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                v32 = acquire!(p, Float32, 100)
                v64 = acquire!(p, Float64, 100)
                vi32 = acquire!(p, Int32, 100)
                v32 .= 1.0f0; v64 .= 2.0; vi32 .= 3
            end
        end

        @test alloc == 0
    end

    @testset "Float16 support (GPU ML type)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup
        @with_pool :cuda p begin
            v = acquire!(p, Float16, 100)
            v .= Float16(1.0)
        end

        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                v = acquire!(p, Float16, 100)
                v .= Float16(2.0)
            end
        end

        @test alloc == 0
    end

end
