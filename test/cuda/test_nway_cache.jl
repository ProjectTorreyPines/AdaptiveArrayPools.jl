# CUDA arr_wrappers Tests
# Verifies setfield!-based CuArray wrapper reuse for zero-allocation acquire.
# Key: arr_wrappers[N][slot] caches one CuArray{T,N} per (dimensionality N, slot).
# Same N = setfield!(:dims) → zero-alloc for unlimited dimension patterns.
# Different N = separate wrapper per N.

@testset "CuArray Wrapper Types" begin

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

end

@testset "arr_wrappers: Unlimited Same-N Patterns" begin

    # With arr_wrappers, same-N dimension patterns use setfield!(:dims).
    # Unlike the old 4-way cache, there is NO eviction — unlimited patterns per N.

    # =========================================================================
    # GPU Allocation Tests (with fill! to actually use the arrays)
    # =========================================================================

    @testset "GPU: 4 patterns zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_4pat_gpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                end
            end
        end

        # Warmup
        test_4pat_gpu()
        test_4pat_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_4pat_gpu()
        @test gpu_alloc == 0
    end

    @testset "GPU: 5+ patterns zero-alloc (no eviction with arr_wrappers)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25), (2, 50))

        function test_5pat_gpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                end
            end
        end

        # Warmup
        test_5pat_gpu()
        test_5pat_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_5pat_gpu()
        @test gpu_alloc == 0
    end

    # =========================================================================
    # CPU Allocation Tests (no fill! to avoid CUDA kernel overhead)
    # =========================================================================

    @testset "CPU: 4 patterns zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_4pat_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_4pat_cpu()
        test_4pat_cpu()
        GC.gc()

        cpu_alloc = @allocated test_4pat_cpu()
        @test cpu_alloc == 0
    end

    @testset "CPU: 5+ patterns zero-alloc (arr_wrappers: same-N uses setfield!)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # All 2D — same N=2 → single wrapper per slot, setfield!(:dims) only
        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25), (2, 50))

        function test_5pat_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_5pat_cpu()
        test_5pat_cpu()
        GC.gc()

        cpu_alloc = @allocated test_5pat_cpu()
        @test cpu_alloc == 0  # No eviction — arr_wrappers support unlimited same-N patterns
    end

    # =========================================================================
    # unsafe_acquire! Tests
    # =========================================================================

    @testset "unsafe_acquire! GPU: 4 patterns zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((8, 8), (4, 16), (16, 4), (2, 32))

        function test_unsafe_4pat_gpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    A = unsafe_acquire!(p, Float64, dims...)
                    fill!(A, 1.0)
                end
            end
        end

        # Warmup
        test_unsafe_4pat_gpu()
        test_unsafe_4pat_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_unsafe_4pat_gpu()
        @test gpu_alloc == 0
    end

    @testset "unsafe_acquire! CPU: 4 patterns zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((8, 8), (4, 16), (16, 4), (2, 32))

        function test_unsafe_4pat_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = unsafe_acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_unsafe_4pat_cpu()
        test_unsafe_4pat_cpu()
        GC.gc()

        cpu_alloc = @allocated test_unsafe_4pat_cpu()
        @test cpu_alloc == 0
    end

    @testset "unsafe_acquire! CPU: 5+ patterns zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((8, 8), (4, 16), (16, 4), (2, 32), (32, 2))

        function test_unsafe_5pat_cpu()
            for dims in dims_list
                @with_pool :cuda p begin
                    _ = unsafe_acquire!(p, Float64, dims...)
                end
            end
        end

        # Warmup
        test_unsafe_5pat_cpu()
        test_unsafe_5pat_cpu()
        GC.gc()

        cpu_alloc = @allocated test_unsafe_5pat_cpu()
        @test cpu_alloc == 0  # arr_wrappers: unlimited same-N patterns
    end

end

@testset "arr_wrappers: Mixed-N Patterns (1D + 2D + 3D)" begin

    # arr_wrappers[N][slot] caches a separate wrapper per dimensionality N.
    # Same slot, different N → each N gets its own wrapper (first use = cache miss).
    # After warmup of all (slot, N) combos → zero-alloc for any mix.

    # =========================================================================
    # GPU Allocation Tests
    # =========================================================================

    @testset "GPU: 1D + 2D + 3D mixed acquire zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Each @with_pool scope resets slot counter → slot 1 gets 1D/2D/3D wrappers
        function test_mixed_n_gpu()
            @with_pool :cuda p begin
                v = acquire!(p, Float64, 100)      # 1D — arr_wrappers[1][1]
                fill!(v, 1.0)
            end
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 10, 10)    # 2D — arr_wrappers[2][1]
                fill!(A, 2.0)
            end
            @with_pool :cuda p begin
                T = acquire!(p, Float64, 5, 5, 4)   # 3D — arr_wrappers[3][1]
                fill!(T, 3.0)
            end
        end

        # Warmup (populates all 3 wrappers per N)
        test_mixed_n_gpu()
        test_mixed_n_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_mixed_n_gpu()
        @test gpu_alloc == 0
    end

    @testset "GPU: mixed-N with varying dims zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Vary dims within each N across iterations
        function test_mixed_n_varying_gpu()
            for (d1, d2, d3) in ((100, (10, 10), (5, 5, 4)),
                                  (200, (5, 20),  (4, 5, 10)),
                                  (50,  (20, 5),  (2, 5, 10)))
                @with_pool :cuda p begin
                    v = acquire!(p, Float64, d1)
                    fill!(v, 1.0)
                end
                @with_pool :cuda p begin
                    A = acquire!(p, Float64, d2...)
                    fill!(A, 2.0)
                end
                @with_pool :cuda p begin
                    T = acquire!(p, Float64, d3...)
                    fill!(T, 3.0)
                end
            end
        end

        # Warmup
        test_mixed_n_varying_gpu()
        test_mixed_n_varying_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_mixed_n_varying_gpu()
        @test gpu_alloc == 0
    end

    @testset "GPU: multi-slot mixed-N zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Multiple acquires within one scope → different slots, different N
        function test_multi_slot_mixed_n_gpu()
            @with_pool :cuda p begin
                v = acquire!(p, Float64, 100)        # Slot 1, 1D
                A = acquire!(p, Float64, 10, 10)     # Slot 2, 2D
                T = acquire!(p, Float64, 5, 5, 4)    # Slot 3, 3D
                fill!(v, 1.0)
                fill!(A, 2.0)
                fill!(T, 3.0)
            end
        end

        # Warmup
        test_multi_slot_mixed_n_gpu()
        test_multi_slot_mixed_n_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_multi_slot_mixed_n_gpu()
        @test gpu_alloc == 0
    end

    # =========================================================================
    # CPU Allocation Tests (no fill! — wrapper creation overhead only)
    # =========================================================================

    @testset "CPU: 1D + 2D + 3D mixed acquire zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_mixed_n_cpu()
            @with_pool :cuda p begin
                _ = acquire!(p, Float64, 100)        # 1D
            end
            @with_pool :cuda p begin
                _ = acquire!(p, Float64, 10, 10)     # 2D
            end
            @with_pool :cuda p begin
                _ = acquire!(p, Float64, 5, 5, 4)    # 3D
            end
        end

        # Warmup
        test_mixed_n_cpu()
        test_mixed_n_cpu()
        GC.gc()

        cpu_alloc = @allocated test_mixed_n_cpu()
        @test cpu_alloc == 0
    end

    @testset "CPU: mixed-N with varying dims zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_mixed_n_varying_cpu()
            for (d1, d2, d3) in ((100, (10, 10), (5, 5, 4)),
                                  (200, (5, 20),  (4, 5, 10)),
                                  (50,  (20, 5),  (2, 5, 10)))
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, d1)
                end
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, d2...)
                end
                @with_pool :cuda p begin
                    _ = acquire!(p, Float64, d3...)
                end
            end
        end

        # Warmup
        test_mixed_n_varying_cpu()
        test_mixed_n_varying_cpu()
        GC.gc()

        cpu_alloc = @allocated test_mixed_n_varying_cpu()
        @test cpu_alloc == 0
    end

    @testset "CPU: multi-slot mixed-N zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_multi_slot_mixed_n_cpu()
            @with_pool :cuda p begin
                _ = acquire!(p, Float64, 100)        # Slot 1, 1D
                _ = acquire!(p, Float64, 10, 10)     # Slot 2, 2D
                _ = acquire!(p, Float64, 5, 5, 4)    # Slot 3, 3D
            end
        end

        # Warmup
        test_multi_slot_mixed_n_cpu()
        test_multi_slot_mixed_n_cpu()
        GC.gc()

        cpu_alloc = @allocated test_multi_slot_mixed_n_cpu()
        @test cpu_alloc == 0
    end

    # =========================================================================
    # unsafe_acquire! Mixed-N Tests
    # =========================================================================

    @testset "unsafe_acquire! GPU: mixed-N zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_unsafe_mixed_n_gpu()
            @with_pool :cuda p begin
                v = unsafe_acquire!(p, Float64, 100)
                A = unsafe_acquire!(p, Float64, 10, 10)
                T = unsafe_acquire!(p, Float64, 5, 5, 4)
                fill!(v, 1.0)
                fill!(A, 2.0)
                fill!(T, 3.0)
            end
        end

        # Warmup
        test_unsafe_mixed_n_gpu()
        test_unsafe_mixed_n_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_unsafe_mixed_n_gpu()
        @test gpu_alloc == 0
    end

    @testset "unsafe_acquire! CPU: mixed-N zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_unsafe_mixed_n_cpu()
            @with_pool :cuda p begin
                _ = unsafe_acquire!(p, Float64, 100)
                _ = unsafe_acquire!(p, Float64, 10, 10)
                _ = unsafe_acquire!(p, Float64, 5, 5, 4)
            end
        end

        # Warmup
        test_unsafe_mixed_n_cpu()
        test_unsafe_mixed_n_cpu()
        GC.gc()

        cpu_alloc = @allocated test_unsafe_mixed_n_cpu()
        @test cpu_alloc == 0
    end

end

@testset "arr_wrappers: Loop Patterns" begin

    @testset "100 iterations: GPU always zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25))

        function test_loop_4pat()
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
        test_loop_4pat()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_loop_4pat()
        @test gpu_alloc == 0  # GPU memory always reused
    end

    @testset "100 iterations with 5+ patterns: GPU still zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        dims_list = ((10, 10), (5, 20), (20, 5), (4, 25), (2, 50))

        function test_loop_5pat()
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
        test_loop_5pat()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_loop_5pat()
        @test gpu_alloc == 0  # GPU memory reused — no cache eviction
    end

end

@testset "arr_wrappers: Multiple Slots" begin

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

@testset "arr_wrappers: reshape! Zero-Alloc" begin

    # _reshape_impl! for CuArray uses arr_wrappers cache for cross-dim reshape,
    # and in-place setfield!(:dims) for same-dim reshape (no pool interaction).

    @testset "GPU: cross-dim reshape zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_reshape_cross_dim_gpu()
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 12)
                CUDA.fill!(A, 1.0)
                # 1D → 2D (cross-dim: claims slot, uses arr_wrappers[2])
                B = reshape!(p, A, 3, 4)
                CUDA.fill!(B, 2.0)
            end
        end

        # Warmup
        test_reshape_cross_dim_gpu()
        test_reshape_cross_dim_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_reshape_cross_dim_gpu()
        @test gpu_alloc == 0
    end

    @testset "CPU: cross-dim reshape zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_reshape_cross_dim_cpu()
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 12)
                B = reshape!(p, A, 3, 4)
            end
        end

        test_reshape_cross_dim_cpu()
        test_reshape_cross_dim_cpu()
        GC.gc()

        cpu_alloc = @allocated test_reshape_cross_dim_cpu()
        @test cpu_alloc == 0
    end

    @testset "GPU: same-dim reshape zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_reshape_same_dim_gpu()
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 3, 4)
                CUDA.fill!(A, 1.0)
                # 2D → 2D (same-dim: in-place setfield!, no pool interaction)
                B = reshape!(p, A, 4, 3)
                CUDA.fill!(B, 2.0)
            end
        end

        test_reshape_same_dim_gpu()
        test_reshape_same_dim_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_reshape_same_dim_gpu()
        @test gpu_alloc == 0
    end

    @testset "GPU: mixed reshape sequence zero-alloc" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        function test_reshape_mixed_gpu()
            @with_pool :cuda p begin
                A = acquire!(p, Float64, 24)
                CUDA.fill!(A, 1.0)
                B = reshape!(p, A, 4, 6)       # 1D → 2D
                C = reshape!(p, A, 2, 3, 4)    # 1D → 3D
                CUDA.fill!(B, 2.0)
                CUDA.fill!(C, 3.0)
            end
        end

        test_reshape_mixed_gpu()
        test_reshape_mixed_gpu()
        GC.gc(); CUDA.reclaim()

        gpu_alloc = CUDA.@allocated test_reshape_mixed_gpu()
        @test gpu_alloc == 0
    end

    @testset "Correctness: data sharing through reshape" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        @with_pool :cuda p begin
            A = acquire!(p, Float64, 12)
            CUDA.fill!(A, 1.0)
            B = reshape!(p, A, 3, 4)
            @test size(B) == (3, 4)
            @test B isa CuArray{Float64, 2}
            # Data identity: B shares GPU memory with A
            @test length(B) == length(A)
        end
    end

    @testset "DimensionMismatch" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        @with_pool :cuda p begin
            A = acquire!(p, Float64, 12)
            @test_throws DimensionMismatch reshape!(p, A, 5, 5)
        end
    end

end

@testset "arr_wrappers: Resize Behavior" begin

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
