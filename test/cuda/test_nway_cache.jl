# CUDA N-way Cache Tests
# Verifies N-way cache behavior for CuArray wrapper reuse
# Key: 4-way cache means 4 dimension patterns = zero-alloc, 5+ = allocation

@testset "N-way Cache Types" begin

    @testset "acquire! returns CuArray" begin
        @with_pool :cuda pool begin
            # acquire! N-D returns CuArray
            arr = acquire!(pool, Float64, 10, 10)
            @test arr isa CuArray{Float64, 2}

            # acquire! 1D returns CuArray view
            vec = acquire!(pool, Float64, 100)
            @test vec isa CuArray{Float64, 1}
        end
    end

    @testset "unsafe_acquire! returns CuArray" begin
        @with_pool :cuda pool begin
            # unsafe_acquire! N-D returns CuArray
            arr = unsafe_acquire!(pool, Float64, 10, 10)
            @test arr isa CuArray{Float64, 2}

            # unsafe_acquire! 1D returns CuArray
            vec = unsafe_acquire!(pool, Float64, 100)
            @test vec isa CuArray{Float64, 1}
        end
    end

    @testset "CACHE_WAYS configuration" begin
        # Verify CACHE_WAYS is accessible
        @test AdaptiveArrayPools.CACHE_WAYS isa Int
        @test 1 <= AdaptiveArrayPools.CACHE_WAYS <= 16
    end

end

@testset "N-way Cache Behavior" begin

    # Key principles:
    # 1. GPU allocation should ALWAYS be 0 (memory reused from pool)
    # 2. CPU allocation: cache hit (4-way) = 0, cache miss (5-way) = >0

    # =========================================================================
    # GPU Allocation Tests (with fill! to actually use the arrays)
    # =========================================================================

    @testset "GPU: 4-way zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_4way_gpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                end
            end
        end

        # Warmup
        test_4way_gpu()
        test_4way_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_4way_gpu()
        @test gpu_alloc == 0
    end

    @testset "GPU: 5-way zero-alloc (even with cache miss)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25), (2, 50))

        function test_5way_gpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                end
            end
        end

        # Warmup
        test_5way_gpu()
        test_5way_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_5way_gpu()
        @test gpu_alloc == 0
    end

    # =========================================================================
    # CPU Allocation Tests (no fill! to avoid CUDA kernel overhead)
    # =========================================================================

    @testset "CPU: 4-way zero-alloc (cache hit)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_4way_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_4way_cpu()
        test_4way_cpu()
        GC.gc()

        cpu_alloc = @allocated test_4way_cpu()
        @test cpu_alloc == 0  # 4 patterns fit in 4-way cache
    end

    @testset "CPU: 5-way causes allocation (cache miss)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25), (2, 50))

        function test_5way_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_5way_cpu()
        test_5way_cpu()
        GC.gc()

        cpu_alloc = @allocated test_5way_cpu()
        @test cpu_alloc > 0  # 5 patterns exceed 4-way cache
    end

    # =========================================================================
    # unsafe_acquire! Tests
    # =========================================================================

    @testset "unsafe_acquire! GPU: 4-way zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((8, 8), (4, 16), (16, 4), (2, 32))

        function test_unsafe_4way_gpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = unsafe_acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                end
            end
        end

        # Warmup
        test_unsafe_4way_gpu()
        test_unsafe_4way_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_unsafe_4way_gpu()
        @test gpu_alloc == 0
    end

    @testset "unsafe_acquire! CPU: 4-way zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((8, 8), (4, 16), (16, 4), (2, 32))

        function test_unsafe_4way_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = unsafe_acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_unsafe_4way_cpu()
        test_unsafe_4way_cpu()
        GC.gc()

        cpu_alloc = @allocated test_unsafe_4way_cpu()
        @test cpu_alloc == 0
    end

    @testset "unsafe_acquire! CPU: 5-way causes allocation" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((8, 8), (4, 16), (16, 4), (2, 32), (32, 2))

        function test_unsafe_5way_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = unsafe_acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_unsafe_5way_cpu()
        test_unsafe_5way_cpu()
        GC.gc()

        cpu_alloc = @allocated test_unsafe_5way_cpu()
        @test cpu_alloc > 0
    end

end

@testset "N-way Cache: Loop Patterns" begin

    @testset "100 iterations: GPU always zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_loop_4way()
            for _ in 1:100
                for dims in dims_list
                    @with_pool :cuda p begin
                        A = acquire!(p, Float64, dims...)
                        fill!(A, 1.0)
                    end
                end
            end
        end

        # Warmup
        test_loop_4way()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_loop_4way()
        @test gpu_alloc == 0  # GPU memory always reused
    end

    @testset "100 iterations with 5 patterns: GPU still zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25), (2, 50))

        function test_loop_5way()
            for _ in 1:100
                for dims in dims_list
                    @with_pool :cuda p begin
                        A = acquire!(p, Float64, dims...)
                        fill!(A, 1.0)
                    end
                end
            end
        end

        # Warmup
        test_loop_5way()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_loop_5way()
        @test gpu_alloc == 0  # GPU memory reused even with cache thrashing
    end

end

@testset "N-way Cache: Multiple Slots" begin

    @testset "Multiple arrays per iteration: GPU zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_multi_slot()
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 10, 10)  # Slot 1
                B = acquire!(p, Float64, 20, 20)  # Slot 2
                C = acquire!(p, Float64, 30, 30)  # Slot 3
                fill!(A, 1.0)
                fill!(B, 2.0)
                fill!(C, 3.0)
            end
        end

        # Warmup
        test_multi_slot()
        test_multi_slot()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_multi_slot()
        @test gpu_alloc == 0
    end

    @testset "Each slot with varying patterns: GPU zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Use same dims for both slots, just vary across iterations
        # This tests GPU memory reuse, not cache behavior
        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_multi_slot_varying()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = acquire!(p, Float64, dims...)
                    B = acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                    fill!(B, 2.0)
                end
            end
        end

        # Warmup
        test_multi_slot_varying()
        test_multi_slot_varying()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_multi_slot_varying()
        @test gpu_alloc == 0
    end

end

@testset "N-way Cache: Resize Behavior" begin

    @testset "Resize: GPU zero-alloc maintained" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Warmup with small array
        @with_pool :cuda p begin
            A = acquire!(p, Float64, 10, 10)
            fill!(A, 1.0)
        end
        @with_pool :cuda p begin
            A = acquire!(p, Float64, 10, 10)
            fill!(A, 1.0)
        end
        GC.gc(); CUDA.reclaim()

        # Small array - GPU should be zero
        gpu_small = CUDA.@allocated begin
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 10, 10)
                fill!(A, 1.0)
            end
        end
        @test gpu_small == 0

        # Request larger array (forces resize)
        @with_pool :cuda p begin
            A = acquire!(p, Float64, 100, 100)
            @test size(A) == (100, 100)
            fill!(A, 2.0)
        end

        # Re-warmup with new size
        @with_pool :cuda p begin
            A = acquire!(p, Float64, 100, 100)
            fill!(A, 2.0)
        end
        GC.gc(); CUDA.reclaim()

        # After re-warmup, GPU should still be zero
        gpu_large = CUDA.@allocated begin
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 100, 100)
                fill!(A, 3.0)
            end
        end
        @test gpu_large == 0
    end

end
