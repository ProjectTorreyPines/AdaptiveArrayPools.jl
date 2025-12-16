# CUDA Extension Core Tests
# Tests for CuTypedPool, CuAdaptiveArrayPool, state management, and macros

@testset "Extension Types" begin
    @testset "CuTypedPool structure" begin
        tp_fields = fieldnames(CuTypedPool)
        @test :vectors in tp_fields
        @test :n_active in tp_fields
        # N-way cache fields
        @test :views in tp_fields
        @test :view_dims in tp_fields
        @test :next_way in tp_fields  # Round-robin counter
        # State management
        @test :_checkpoint_n_active in tp_fields
        @test :_checkpoint_depths in tp_fields
    end

    @testset "CuAdaptiveArrayPool structure" begin
        pool_fields = fieldnames(CuAdaptiveArrayPool)
        @test :float16 in pool_fields  # GPU ML support
        @test :device_id in pool_fields  # Multi-GPU safety
        @test :others in pool_fields
    end

    @testset "Type hierarchy" begin
        @test CuTypedPool <: AbstractTypedPool
        @test CuAdaptiveArrayPool <: AbstractArrayPool
    end

    @testset "Instance creation" begin
        tp = CuTypedPool{Float32}()
        @test tp.n_active == 0
        @test length(tp.vectors) == 0

        pool = CuAdaptiveArrayPool()
        @test pool.device_id == CUDA.deviceid(CUDA.device())
        @test pool._current_depth == 1
    end

    @testset "GPU_FIXED_SLOT_FIELDS" begin
        @test :float16 in GPU_FIXED_SLOT_FIELDS
        @test first(GPU_FIXED_SLOT_FIELDS) == :float32
        @test length(GPU_FIXED_SLOT_FIELDS) == 8
    end
end

@testset "Dispatch Methods" begin
    @testset "allocate_vector" begin
        tp = CuTypedPool{Float32}()
        vec = AdaptiveArrayPools.allocate_vector(tp, 100)
        @test vec isa CuVector{Float32}
        @test length(vec) == 100
    end

    @testset "wrap_array" begin
        tp = CuTypedPool{Float32}()
        vec = CUDA.zeros(Float32, 50)
        flat_view = view(vec, 1:50)
        wrapped = AdaptiveArrayPools.wrap_array(tp, flat_view, (10, 5))
        @test wrapped isa CuArray{Float32,2}
        @test size(wrapped) == (10, 5)
    end

    @testset "get_typed_pool! fixed slots" begin
        pool = CuAdaptiveArrayPool()
        test_types = [Float32, Float64, Float16, Int32, Int64, ComplexF32, ComplexF64, Bool]
        for T in test_types
            tp = get_typed_pool!(pool, T)
            @test tp isa CuTypedPool{T}
        end
    end

    @testset "get_typed_pool! fallback (rare types)" begin
        pool = CuAdaptiveArrayPool()
        tp = get_typed_pool!(pool, UInt8)
        @test tp isa CuTypedPool{UInt8}
        @test haskey(pool.others, UInt8)
    end

    @testset "get_view!" begin
        tp = CuTypedPool{Float32}()
        @test tp.n_active == 0

        v1 = get_view!(tp, 100)
        @test v1 isa CuArray
        @test length(v1) == 100
        @test tp.n_active == 1

        v2 = get_view!(tp, 200)
        @test v2 isa CuArray
        @test length(v2) == 200
        @test tp.n_active == 2
    end

    @testset "Checkpoint auto-init for dynamic types" begin
        pool = CuAdaptiveArrayPool()
        pool._current_depth = 2  # Simulate inside @with_pool scope

        tp = get_typed_pool!(pool, UInt16)
        @test tp._checkpoint_n_active == [0, 0]
        @test tp._checkpoint_depths == [0, 2]
    end
end

@testset "Task-Local Pool" begin
    @testset "get_task_local_cuda_pool" begin
        pool1 = get_task_local_cuda_pool()
        @test pool1 isa CuAdaptiveArrayPool
        @test pool1.device_id == CUDA.deviceid(CUDA.device())

        pool2 = get_task_local_cuda_pool()
        @test pool1 === pool2  # Same pool on second call
    end

    @testset "get_task_local_cuda_pools" begin
        pools_dict = get_task_local_cuda_pools()
        @test pools_dict isa Dict{Int, CuAdaptiveArrayPool}
        pool = get_task_local_cuda_pool()
        @test haskey(pools_dict, pool.device_id)
    end

    @testset "get_task_local_cuda_pools before pool creation" begin
        # Test in a fresh task where no pool exists yet
        result = fetch(Threads.@spawn begin
            # Call get_task_local_cuda_pools() FIRST (before get_task_local_cuda_pool)
            pools = get_task_local_cuda_pools()
            @test pools isa Dict{Int, CuAdaptiveArrayPool}
            @test isempty(pools)  # No pools created yet
            true
        end)
        @test result == true
    end

    @testset "Multi-device safety (single device verification)" begin
        # 1. Verify device_id is captured correctly at pool creation
        pool = get_task_local_cuda_pool()
        current_dev_id = CUDA.deviceid(CUDA.device())
        @test pool.device_id == current_dev_id

        # 2. Verify Dict key matches pool's device_id
        pools = get_task_local_cuda_pools()
        @test haskey(pools, current_dev_id)
        @test pools[current_dev_id] === pool
        @test pools[current_dev_id].device_id == current_dev_id

        # 3. Verify different device IDs get different pool entries
        # (Simulate multi-device by manually adding fake entries)
        fake_dev_id = 999
        @test !haskey(pools, fake_dev_id)

        fake_pool = CuAdaptiveArrayPool()
        pools[fake_dev_id] = fake_pool

        # Real device pool unchanged
        @test pools[current_dev_id] === pool
        # Fake device has its own pool
        @test pools[fake_dev_id] === fake_pool
        @test pools[fake_dev_id] !== pools[current_dev_id]

        # Cleanup fake entry
        delete!(pools, fake_dev_id)
        @test !haskey(pools, fake_dev_id)

        # 4. get_task_local_cuda_pool() still returns same pool (not affected by fake)
        @test get_task_local_cuda_pool() === pool
    end
end

@testset "State Management" begin
    @testset "Basic checkpoint/rewind" begin
        pool = get_task_local_cuda_pool()
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
        pool = get_task_local_cuda_pool()
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
        pool = get_task_local_cuda_pool()
        get_view!(pool.float32, 100)
        get_view!(pool.float64, 200)
        vectors_count = length(pool.float32.vectors)

        reset!(pool)
        @test pool.float32.n_active == 0
        @test pool.float64.n_active == 0
        @test pool._current_depth == 1
        @test length(pool.float32.vectors) == vectors_count  # Memory preserved
    end

    @testset "empty!" begin
        pool = get_task_local_cuda_pool()
        get_view!(pool.float32, 100)
        @test length(pool.float32.vectors) >= 1

        empty!(pool)
        @test pool.float32.n_active == 0
        @test length(pool.float32.vectors) == 0  # Memory cleared
    end

    @testset "foreach_fixed_slot" begin
        pool = get_task_local_cuda_pool()
        slot_count = Ref(0)
        foreach_fixed_slot(pool) do tp
            slot_count[] += 1
        end
        @test slot_count[] == 8
    end

    @testset "Type-specific checkpoint/rewind" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        checkpoint!(pool, Float32)
        get_view!(pool.float32, 100)
        get_view!(pool.float64, 200)
        @test pool.float32.n_active == 1
        @test pool.float64.n_active == 1

        rewind!(pool, Float32)
        @test pool.float32.n_active == 0
    end

    @testset "Multi-type checkpoint/rewind" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Multi-type checkpoint
        checkpoint!(pool, Float32, Float64)
        @test pool._current_depth == 2

        get_view!(pool.float32, 100)
        get_view!(pool.float64, 200)
        @test pool.float32.n_active == 1
        @test pool.float64.n_active == 1

        # Multi-type rewind
        rewind!(pool, Float32, Float64)
        @test pool._current_depth == 1
        @test pool.float32.n_active == 0
        @test pool.float64.n_active == 0
    end

    @testset "Type-specific reset" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        get_view!(pool.float32, 100)
        get_view!(pool.float64, 200)
        @test pool.float32.n_active == 1
        @test pool.float64.n_active == 1

        reset!(pool, Float32)
        @test pool.float32.n_active == 0
        @test pool.float64.n_active == 1  # Not affected
    end

    @testset "Rewind at depth=1 (edge case)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        @test pool._current_depth == 1
        get_view!(pool.float32, 100)
        @test pool.float32.n_active == 1

        # Rewind at depth=1 should delegate to reset!
        rewind!(pool)
        @test pool._current_depth == 1
        @test pool.float32.n_active == 0
    end

    @testset "Type-specific rewind at depth=1" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        @test pool._current_depth == 1
        get_view!(pool.float32, 100)
        @test pool.float32.n_active == 1

        # Type-specific rewind at depth=1 should reset that type
        rewind!(pool, Float32)
        @test pool.float32.n_active == 0
    end

    @testset "Multi-type rewind at depth=1" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        @test pool._current_depth == 1
        get_view!(pool.float32, 100)
        get_view!(pool.float64, 200)

        # Multi-type rewind at depth=1 should reset those types
        rewind!(pool, Float32, Float64)
        @test pool.float32.n_active == 0
        @test pool.float64.n_active == 0
    end

    @testset "State operations with rare types (pool.others)" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Use a rare type that goes into pool.others
        tp_uint8 = get_typed_pool!(pool, UInt8)
        @test haskey(pool.others, UInt8)

        # checkpoint! with rare type in others
        checkpoint!(pool)
        get_view!(tp_uint8, 50)
        @test tp_uint8.n_active == 1

        # rewind! should also rewind rare types
        rewind!(pool)
        @test tp_uint8.n_active == 0

        # reset! with rare type
        get_view!(tp_uint8, 100)
        @test tp_uint8.n_active == 1
        reset!(pool)
        @test tp_uint8.n_active == 0

        # empty! with rare type
        get_view!(tp_uint8, 100)
        @test length(tp_uint8.vectors) >= 1
        empty!(pool)
        @test tp_uint8.n_active == 0
        @test length(tp_uint8.vectors) == 0
    end
end

@testset "Macro Integration" begin
    @testset "@with_pool :cuda basic" begin
        result = @with_pool :cuda pool begin
            @test pool isa CuAdaptiveArrayPool
            v = acquire!(pool, Float32, 100)
            v .= 1.0f0
            sum(v)
        end
        @test result == 100.0f0
        @test get_task_local_cuda_pool().float32.n_active == 0
    end

    @testset "@with_pool :cuda without pool name" begin
        result = @with_pool :cuda begin
            pool = get_task_local_cuda_pool()
            v = acquire!(pool, Float64, 50)
            v .= 2.0
            sum(v)
        end
        @test result == 100.0
    end

    @testset "Nested CPU/GPU pools" begin
        result = @with_pool cpu_pool begin
            cpu_v = acquire!(cpu_pool, Float64, 10)
            cpu_v .= 1.0

            gpu_result = @with_pool :cuda gpu_pool begin
                gpu_v = acquire!(gpu_pool, Float32, 10)
                gpu_v .= 2.0f0
                sum(gpu_v)
            end

            sum(cpu_v) + gpu_result
        end
        @test result == 30.0
    end

    @testset "Rewind on normal exit" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        @with_pool :cuda p begin
            acquire!(p, Float32, 100)
            acquire!(p, Float32, 200)
            @test p.float32.n_active == 2
        end

        @test pool.float32.n_active == 0
    end

    @testset "Rewind on error" begin
        pool = get_task_local_cuda_pool()
        reset!(pool)

        try
            @with_pool :cuda p begin
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
        result = @with_pool :cuda pool begin
            A = acquire!(pool, Float32, 10, 10)
            @test size(A) == (10, 10)
            A .= 1.0f0
            sum(A)
        end
        @test result == 100.0f0
    end

    @testset "unsafe_acquire!" begin
        result = @with_pool :cuda pool begin
            A = unsafe_acquire!(pool, Float32, 100)
            @test A isa CuArray{Float32,1}
            A .= 2.0f0
            sum(A)
        end
        @test result == 200.0f0
    end
end

@testset "Acquire API" begin
    @testset "acquire! with CuAdaptiveArrayPool" begin
        pool = CuAdaptiveArrayPool()
        v = acquire!(pool, Float32, 100)
        @test v isa CuArray
        @test length(v) == 100
    end

    @testset "acquire! multi-dim" begin
        pool = CuAdaptiveArrayPool()
        A = acquire!(pool, Float32, 10, 10)
        @test size(A) == (10, 10)
    end

    @testset "acquire! tuple dims" begin
        pool = CuAdaptiveArrayPool()
        dims = (5, 5, 5)
        A = acquire!(pool, Float64, dims)
        @test size(A) == dims
    end

    @testset "acquire! similar-style" begin
        pool = CuAdaptiveArrayPool()
        original = CUDA.rand(Float32, 10, 10)
        A = acquire!(pool, original)
        @test size(A) == size(original)
        @test eltype(A) == eltype(original)
    end

    @testset "unsafe_acquire! variants" begin
        pool = CuAdaptiveArrayPool()

        v = unsafe_acquire!(pool, Float32, 100)
        @test v isa CuArray{Float32,1}

        A = unsafe_acquire!(pool, Float64, 10, 10)
        @test A isa CuArray{Float64,2}

        B = unsafe_acquire!(pool, Int32, (5, 5))
        @test B isa CuArray{Int32,2}
    end
end
