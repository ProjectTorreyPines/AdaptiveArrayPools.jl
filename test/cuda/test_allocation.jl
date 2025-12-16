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
                v1 .= 1f0; v2 .= 2f0; v3 .= 3f0
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

    @testset "unsafe_acquire! memory reuse" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup
        @with_pool :cuda p begin
            A = unsafe_acquire!(p, Float64, 10, 10)
            A .= 1.0
        end

        alloc = CUDA.@allocated begin
            @with_pool :cuda p begin
                A = unsafe_acquire!(p, Float64, 10, 10)
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

@testset "CPU Allocation (CuArray wrapper)" begin

    @testset "acquire! N-D has low CPU allocation (cache hit)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup (populates cache)
        @with_pool :cuda p begin
            acquire!(p, Float64, 10, 10)
        end
        @with_pool :cuda p begin
            acquire!(p, Float64, 10, 10)
        end

        # Measure CPU allocation
        cpu_alloc = @allocated begin
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 10, 10)
            end
        end

        # Cache hit should have minimal CPU allocation
        @test cpu_alloc < 100  # Allow some overhead
    end

    @testset "unsafe_acquire! cache hit returns cached wrapper" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup (populates cache)
        @with_pool :cuda p begin
            unsafe_acquire!(p, Float64, 10, 10)
        end
        @with_pool :cuda p begin
            unsafe_acquire!(p, Float64, 10, 10)
        end

        # After warmup, cache hit should be low/zero allocation
        cpu_alloc = @allocated begin
            @with_pool :cuda p begin
                A = unsafe_acquire!(p, Float64, 10, 10)
            end
        end

        # Cache hit should have minimal CPU allocation
        @test cpu_alloc < 100  # Allow some overhead
    end

    @testset "acquire! 1D has low CPU allocation" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup
        @with_pool :cuda p begin
            acquire!(p, Float64, 100)
        end
        @with_pool :cuda p begin
            acquire!(p, Float64, 100)
        end

        cpu_alloc = @allocated begin
            @with_pool :cuda p begin
                v = acquire!(p, Float64, 100)
            end
        end

        # 1D acquire! uses view path, should be efficient
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
                v32 .= 1f0; v64 .= 2.0; vi32 .= 3
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
