# Metal Extension Core Tests
# Tests for MetalTypedPool, MetalAdaptiveArrayPool, state management, and macros

@testset "Extension Types" begin
    @testset "MetalTypedPool structure" begin
        tp_fields = fieldnames(MetalTypedPool)
        @test :vectors in tp_fields
        @test :n_active in tp_fields
        @test :arr_wrappers in tp_fields
        @test :_checkpoint_n_active in tp_fields
        @test :_checkpoint_depths in tp_fields
    end

    @testset "MetalAdaptiveArrayPool structure" begin
        pool_fields = fieldnames(MetalAdaptiveArrayPool)
        @test :float16 in pool_fields
        @test :device_key in pool_fields
        @test :others in pool_fields
        # Metal does NOT have float64/complexf64
        @test !(:float64 in pool_fields)
        @test !(:complexf64 in pool_fields)
    end

    @testset "Type hierarchy" begin
        @test MetalTypedPool <: AbstractTypedPool
        @test MetalAdaptiveArrayPool <: AbstractArrayPool
    end

    @testset "Instance creation" begin
        tp = MetalTypedPool{Float32, Metal.PrivateStorage}()
        @test tp.n_active == 0
        @test length(tp.vectors) == 0

        pool = MetalAdaptiveArrayPool()
        @test pool.device_key == Metal.device()
        @test pool._current_depth == 1
    end

    @testset "METAL_FIXED_SLOT_FIELDS" begin
        @test :float16 in METAL_FIXED_SLOT_FIELDS
        @test first(METAL_FIXED_SLOT_FIELDS) == :float32
        @test length(METAL_FIXED_SLOT_FIELDS) == 6
        # No Float64/ComplexF64
        @test !(:float64 in METAL_FIXED_SLOT_FIELDS)
        @test !(:complexf64 in METAL_FIXED_SLOT_FIELDS)
    end
end

@testset "Dispatch Methods" begin
    @testset "allocate_vector" begin
        tp = MetalTypedPool{Float32, Metal.PrivateStorage}()
        vec = AdaptiveArrayPools.allocate_vector(tp, 100)
        @test vec isa MtlVector{Float32}
        @test length(vec) == 100
    end

    @testset "get_typed_pool! fixed slots" begin
        pool = MetalAdaptiveArrayPool()
        test_types = [Float32, Float16, Int32, Int64, ComplexF32, Bool]
        for T in test_types
            tp = get_typed_pool!(pool, T)
            @test tp isa MetalTypedPool{T}
        end
    end

    @testset "get_typed_pool! rejects Float64/ComplexF64" begin
        pool = MetalAdaptiveArrayPool()
        @test_throws ArgumentError get_typed_pool!(pool, Float64)
        @test_throws ArgumentError get_typed_pool!(pool, ComplexF64)
    end

    @testset "get_typed_pool! fallback (rare types)" begin
        pool = MetalAdaptiveArrayPool()
        tp = get_typed_pool!(pool, UInt8)
        @test tp isa MetalTypedPool{UInt8}
        @test haskey(pool.others, UInt8)
    end

    @testset "get_view!" begin
        tp = MetalTypedPool{Float32, Metal.PrivateStorage}()
        @test tp.n_active == 0

        v1 = get_view!(tp, 100)
        @test v1 isa MtlArray
        @test length(v1) == 100
        @test tp.n_active == 1

        v2 = get_view!(tp, 200)
        @test v2 isa MtlArray
        @test length(v2) == 200
        @test tp.n_active == 2
    end

    @testset "Checkpoint auto-init for dynamic types" begin
        pool = MetalAdaptiveArrayPool()
        checkpoint!(pool)  # Properly enter depth 2

        tp = get_typed_pool!(pool, UInt16)
        @test tp._checkpoint_n_active == [0, 0]
        @test tp._checkpoint_depths == [0, 2]
    end
end

@testset "State Management" begin
    @testset "Basic checkpoint/rewind" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @test pool._current_depth == 1
        @test pool.float32.n_active == 0

        checkpoint!(pool)
        @test pool._current_depth == 2

        get_view!(pool.float32, 100)
        get_view!(pool.float32, 200)
        @test pool.float32.n_active == 2

        rewind!(pool)
        @test pool._current_depth == 1
        @test pool.float32.n_active == 0
        @test length(pool.float32.vectors) >= 2  # Memory preserved
    end

    @testset "Nested checkpoint/rewind" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # Outer
        checkpoint!(pool)
        @test pool._current_depth == 2
        get_view!(pool.float32, 50)
        @test pool.float32.n_active == 1

        # Inner
        checkpoint!(pool)
        @test pool._current_depth == 3
        get_view!(pool.float32, 100)
        get_view!(pool.float32, 150)
        @test pool.float32.n_active == 3

        # Inner rewind
        rewind!(pool)
        @test pool._current_depth == 2
        @test pool.float32.n_active == 1

        # Outer rewind
        rewind!(pool)
        @test pool._current_depth == 1
        @test pool.float32.n_active == 0
    end

    @testset "reset!" begin
        pool = get_task_local_metal_pool()
        get_view!(pool.float32, 100)
        get_view!(pool.int32, 200)
        vectors_count = length(pool.float32.vectors)

        reset!(pool)
        @test pool.float32.n_active == 0
        @test pool.int32.n_active == 0
        @test pool._current_depth == 1
        @test length(pool.float32.vectors) == vectors_count  # Memory preserved
    end

    @testset "empty!" begin
        pool = get_task_local_metal_pool()
        get_view!(pool.float32, 100)
        @test length(pool.float32.vectors) >= 1

        empty!(pool)
        @test pool.float32.n_active == 0
        @test length(pool.float32.vectors) == 0  # Memory cleared
    end

    @testset "foreach_fixed_slot" begin
        pool = get_task_local_metal_pool()
        slot_count = Ref(0)
        foreach_fixed_slot(pool) do tp
            slot_count[] += 1
        end
        @test slot_count[] == 6
    end

    @testset "Type-specific checkpoint/rewind" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        checkpoint!(pool, Float32)
        get_view!(pool.float32, 100)
        get_view!(pool.int32, 200)
        @test pool.float32.n_active == 1
        @test pool.int32.n_active == 1

        rewind!(pool, Float32)
        @test pool.float32.n_active == 0
    end

    @testset "Multi-type checkpoint/rewind" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        checkpoint!(pool, Float32, Int32)
        @test pool._current_depth == 2

        get_view!(pool.float32, 100)
        get_view!(pool.int32, 200)
        @test pool.float32.n_active == 1
        @test pool.int32.n_active == 1

        rewind!(pool, Float32, Int32)
        @test pool._current_depth == 1
        @test pool.float32.n_active == 0
        @test pool.int32.n_active == 0
    end

    @testset "Type-specific reset" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        get_view!(pool.float32, 100)
        get_view!(pool.int32, 200)
        @test pool.float32.n_active == 1
        @test pool.int32.n_active == 1

        reset!(pool, Float32)
        @test pool.float32.n_active == 0
        @test pool.int32.n_active == 1  # Not affected
    end

    @testset "Rewind at depth=1 (edge case)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @test pool._current_depth == 1
        get_view!(pool.float32, 100)
        @test pool.float32.n_active == 1

        rewind!(pool)
        @test pool._current_depth == 1
        @test pool.float32.n_active == 0
    end

    @testset "Type-specific rewind at depth=1" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @test pool._current_depth == 1
        get_view!(pool.float32, 100)
        @test pool.float32.n_active == 1

        rewind!(pool, Float32)
        @test pool.float32.n_active == 0
    end

    @testset "Multi-type rewind at depth=1" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @test pool._current_depth == 1
        get_view!(pool.float32, 100)
        get_view!(pool.int32, 200)

        rewind!(pool, Float32, Int32)
        @test pool.float32.n_active == 0
        @test pool.int32.n_active == 0
    end

    @testset "State operations with rare types (pool.others)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        tp_uint8 = get_typed_pool!(pool, UInt8)
        @test haskey(pool.others, UInt8)

        checkpoint!(pool)
        get_view!(tp_uint8, 50)
        @test tp_uint8.n_active == 1

        rewind!(pool)
        @test tp_uint8.n_active == 0

        get_view!(tp_uint8, 100)
        @test tp_uint8.n_active == 1
        reset!(pool)
        @test tp_uint8.n_active == 0

        get_view!(tp_uint8, 100)
        @test length(tp_uint8.vectors) >= 1
        empty!(pool)
        @test tp_uint8.n_active == 0
        @test length(tp_uint8.vectors) == 0
    end
end

@testset "Macro Integration" begin
    @testset "@with_pool :metal basic" begin
        result = @with_pool :metal pool begin
            @test pool isa MetalAdaptiveArrayPool
            v = acquire!(pool, Float32, 100)
            v .= 1.0f0
            sum(v)
        end
        @test result == 100.0f0
        @test get_task_local_metal_pool().float32.n_active == 0
    end

    @testset "@with_pool :metal without pool name" begin
        result = @with_pool :metal begin
            pool = get_task_local_metal_pool()
            v = acquire!(pool, Float32, 50)
            v .= 2.0f0
            sum(v)
        end
        @test result == 100.0f0
    end

    @testset "Nested CPU/Metal pools" begin
        result = @with_pool cpu_pool begin
            cpu_v = acquire!(cpu_pool, Float64, 10)
            cpu_v .= 1.0

            gpu_result = @with_pool :metal gpu_pool begin
                gpu_v = acquire!(gpu_pool, Float32, 10)
                gpu_v .= 2.0f0
                sum(gpu_v)
            end

            sum(cpu_v) + gpu_result
        end
        @test result == 30.0
    end

    @testset "Rewind on normal exit" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @with_pool :metal p begin
            acquire!(p, Float32, 100)
            acquire!(p, Float32, 200)
            @test p.float32.n_active == 2
        end

        @test pool.float32.n_active == 0
    end

    @testset "Rewind on error" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        try
            @safe_with_pool :metal p begin
                acquire!(p, Float32, 100)
                @test p.float32.n_active == 1
                error("Intentional error")
            end
        catch e
            @test e isa ErrorException
        end

        @test pool.float32.n_active == 0
    end

    @testset "Multi-dimensional acquire" begin
        result = @with_pool :metal pool begin
            A = acquire!(pool, Float32, 10, 10)
            @test size(A) == (10, 10)
            A .= 1.0f0
            sum(A)
        end
        @test result == 100.0f0
    end

    @testset "acquire! returns MtlArray" begin
        result = @with_pool :metal pool begin
            A = acquire!(pool, Float32, 100)
            @test A isa MtlArray{Float32, 1}
            A .= 2.0f0
            sum(A)
        end
        @test result == 200.0f0
    end

    @testset "Direct rewind: explicit return" begin
        @with_pool :metal pool function metal_early_return(flag)
            v = acquire!(pool, Float32, 10)
            v .= 1.0f0
            if flag
                return sum(v)
            end
            v .= 2.0f0
            sum(v)
        end

        @test metal_early_return(true) == 10.0f0
        @test metal_early_return(false) == 20.0f0
        @test get_task_local_metal_pool()._current_depth == 1
    end

    @testset "Direct rewind: break/continue in loop" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        total = 0.0f0
        for i in 1:5
            @with_pool :metal p begin
                v = acquire!(p, Float32, 3)
                v .= Float32(i)
                if i == 3
                    continue
                end
                total += sum(v)
            end
        end
        @test total == 3.0f0 * (1 + 2 + 4 + 5)
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: nested catch recovery (entry depth guard)" begin
        reset!(get_task_local_metal_pool())

        @with_pool :metal pool function metal_outer_catches()
            v = acquire!(pool, Float32, 10)
            v .= 1.0f0
            result = try
                @with_pool :metal pool begin
                    acquire!(pool, Int32, 5)
                    error("boom")
                end
            catch
                42
            end
            sum(v) + result
        end

        @test metal_outer_catches() == 52.0f0
        @test get_task_local_metal_pool()._current_depth == 1
    end

    @testset "Uncaught exception corrupts Metal pool (documented)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        try
            @with_pool :metal p begin
                acquire!(p, Float32, 10)
                error("uncaught!")
            end
        catch
        end

        @test pool._current_depth > 1  # corrupted — expected
        reset!(pool)
        @test pool._current_depth == 1
    end
end

@testset "Acquire API" begin
    @testset "acquire! with MetalAdaptiveArrayPool" begin
        pool = MetalAdaptiveArrayPool()
        v = acquire!(pool, Float32, 100)
        @test v isa MtlArray
        @test length(v) == 100
    end

    @testset "acquire! multi-dim" begin
        pool = MetalAdaptiveArrayPool()
        A = acquire!(pool, Float32, 10, 10)
        @test size(A) == (10, 10)
    end

    @testset "acquire! tuple dims" begin
        pool = MetalAdaptiveArrayPool()
        dims = (5, 5, 5)
        A = acquire!(pool, Float32, dims)
        @test size(A) == dims
    end

    @testset "acquire! similar-style" begin
        pool = MetalAdaptiveArrayPool()
        original = MtlArray(rand(Float32, 10, 10))
        A = acquire!(pool, original)
        @test size(A) == size(original)
        @test eltype(A) == eltype(original)
    end

    @testset "acquire! all dimensionalities" begin
        pool = MetalAdaptiveArrayPool()

        v = acquire!(pool, Float32, 100)
        @test v isa MtlArray{Float32, 1}

        A = acquire!(pool, Int32, 10, 10)
        @test A isa MtlArray{Int32, 2}

        B = acquire!(pool, Int32, (5, 5))
        @test B isa MtlArray{Int32, 2}
    end

    @testset "acquire! rejects Float64" begin
        pool = MetalAdaptiveArrayPool()
        @test_throws ArgumentError acquire!(pool, Float64, 10)
    end
end
