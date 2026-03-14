# Metal Allocation Tests
# Verifies zero-allocation pooling behavior and GPU memory reuse

@testset "GPU Allocation" begin

    @testset "Memory reuse (same size)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # First acquire - populates pool
        @with_pool :metal p begin
            v = acquire!(p, Float32, 100)
            v .= 1.0f0
        end

        # Second acquire (same size) - should reuse GPU memory
        alloc = Metal.@allocated begin
            @with_pool :metal p begin
                v = acquire!(p, Float32, 100)
                v .= 2.0f0
            end
        end

        # GPU allocation should be minimal on cache hit (kernel launch overhead only)
        @test alloc < 200
    end

    @testset "Memory reuse (multiple arrays)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # Warmup with 3 arrays
        @with_pool :metal p begin
            acquire!(p, Float32, 100)
            acquire!(p, Float32, 200)
            acquire!(p, Float32, 300)
        end

        # Second pass should reuse all GPU memory
        alloc = Metal.@allocated begin
            @with_pool :metal p begin
                v1 = acquire!(p, Float32, 100)
                v2 = acquire!(p, Float32, 200)
                v3 = acquire!(p, Float32, 300)
                v1 .= 1.0f0; v2 .= 2.0f0; v3 .= 3.0f0
            end
        end

        # 3 kernel launches × ~56 bytes each ≈ ~168 bytes overhead
        @test alloc < 200
    end

    @testset "Memory reuse (N-D arrays)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # Warmup with 2D array
        @with_pool :metal p begin
            A = acquire!(p, Float32, 10, 10)
            A .= 1.0f0
        end

        # Reuse check — GPU allocation only
        alloc = Metal.@allocated begin
            @with_pool :metal p begin
                A = acquire!(p, Float32, 10, 10)
                A .= 2.0f0
            end
        end

        @test alloc < 200
    end

    @testset "Memory reuse (3D arrays)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # Warmup with 3D array
        @with_pool :metal p begin
            T = acquire!(p, Float32, 5, 5, 4)
            T .= 1.0f0
        end

        alloc = Metal.@allocated begin
            @with_pool :metal p begin
                T = acquire!(p, Float32, 5, 5, 4)
                T .= 2.0f0
            end
        end

        @test alloc < 200
    end

    @testset "Pointer reuse verification" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        ptr1 = Ref{UInt}(0)
        ptr2 = Ref{UInt}(0)

        @with_pool :metal p begin
            v = acquire!(p, Float32, 1000)
            ptr1[] = UInt(pointer(v).offset)
        end

        @with_pool :metal p begin
            v = acquire!(p, Float32, 1000)
            ptr2[] = UInt(pointer(v).offset)
        end

        # Same GPU memory offset should be reused
        @test ptr1[] == ptr2[]
    end

end

@testset "_resize_to_fit! Metal memory preservation" begin
    _resize_to_fit! = ext._resize_to_fit!

    @testset "Shrink preserves Metal pointer" begin
        v = MtlArray(zeros(Float32, 1000))
        ptr = UInt(pointer(v).offset)
        _resize_to_fit!(v, 100)
        @test length(v) == 100
        @test UInt(pointer(v).offset) == ptr
    end

    @testset "Grow-back within capacity: no realloc" begin
        v = MtlArray(zeros(Float32, 1000))
        ptr = UInt(pointer(v).offset)
        _resize_to_fit!(v, 100)
        @test length(v) == 100
        @test UInt(pointer(v).offset) == ptr
        _resize_to_fit!(v, 1000)
        @test length(v) == 1000
        @test UInt(pointer(v).offset) == ptr
    end

    @testset "Shrink to 0, grow back preserves pointer" begin
        v = MtlArray(zeros(Float32, 500))
        ptr = UInt(pointer(v).offset)
        _resize_to_fit!(v, 0)
        @test length(v) == 0
        _resize_to_fit!(v, 500)
        @test length(v) == 500
        @test UInt(pointer(v).offset) == ptr
    end

    @testset "Grow within capacity after invalidation: no realloc" begin
        v = MtlArray(zeros(Float32, 1000))
        ptr = UInt(pointer(v).offset)
        _resize_to_fit!(v, 0)
        @test length(v) == 0
        _resize_to_fit!(v, 200)
        @test length(v) == 200
        @test UInt(pointer(v).offset) == ptr
    end

    @testset "No-op when n == length" begin
        v = MtlArray(zeros(Float32, 200))
        ptr = UInt(pointer(v).offset)
        _resize_to_fit!(v, 200)
        @test length(v) == 200
        @test UInt(pointer(v).offset) == ptr
    end

    @testset "Grow beyond capacity delegates to resize!" begin
        v = MtlArray(zeros(Float32, 100))
        _resize_to_fit!(v, 10_000)
        @test length(v) == 10_000
    end
end

@testset "CPU Allocation (MtlArray wrapper)" begin

    @testset "acquire! N-D has low CPU allocation (cache hit)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        function _test_metal_nd_alloc!()
            @with_pool :metal p begin
                acquire!(p, Float32, 10, 10)
            end
        end

        # Warmup (JIT + cache)
        _test_metal_nd_alloc!()
        _test_metal_nd_alloc!()

        cpu_alloc = @allocated _test_metal_nd_alloc!()
        @test cpu_alloc < 100
    end

    @testset "acquire! 1D has low CPU allocation" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        function _test_metal_1d_alloc!()
            @with_pool :metal p begin
                acquire!(p, Float32, 100)
            end
        end

        # Warmup (JIT + cache)
        _test_metal_1d_alloc!()
        _test_metal_1d_alloc!()

        cpu_alloc = @allocated _test_metal_1d_alloc!()
        @test cpu_alloc < 200
    end

end

@testset "Mixed Type Allocation" begin

    @testset "Multiple types maintain separate pools" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # Warmup all types
        @with_pool :metal p begin
            acquire!(p, Float32, 100)
            acquire!(p, Int32, 100)
            acquire!(p, Float16, 100)
        end

        # Reuse all types — check GPU allocation only
        alloc = Metal.@allocated begin
            @with_pool :metal p begin
                v32 = acquire!(p, Float32, 100)
                vi32 = acquire!(p, Int32, 100)
                v16 = acquire!(p, Float16, 100)
                v32 .= 1.0f0; vi32 .= 3; v16 .= Float16(4.0)
            end
        end

        # 3 kernel launches overhead
        @test alloc < 200
    end

    @testset "Float16 support (GPU ML type)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # Warmup
        @with_pool :metal p begin
            v = acquire!(p, Float16, 100)
            v .= Float16(1.0)
        end

        alloc = Metal.@allocated begin
            @with_pool :metal p begin
                v = acquire!(p, Float16, 100)
                v .= Float16(2.0)
            end
        end

        @test alloc < 200
    end

end
