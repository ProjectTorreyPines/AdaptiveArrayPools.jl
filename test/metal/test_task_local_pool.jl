# Metal Task-Local Pool Tests

@testset "Metal Task-Local Pool" begin

    @testset "get_task_local_metal_pool" begin
        pool1 = get_task_local_metal_pool()
        @test pool1 isa MetalAdaptiveArrayPool
        @test pool1.device_key == Metal.device()

        pool2 = get_task_local_metal_pool()
        @test pool1 === pool2  # Same pool on second call
    end

    @testset "get_task_local_metal_pools" begin
        pools_dict = get_task_local_metal_pools()
        @test pools_dict isa Dict{UInt64, MetalAdaptiveArrayPool}
        pool = get_task_local_metal_pool()
        dev_key = objectid(Metal.device())
        @test haskey(pools_dict, dev_key)
    end

    @testset "get_task_local_metal_pools before pool creation" begin
        result = fetch(
            Threads.@spawn begin
                pools = get_task_local_metal_pools()
                @test pools isa Dict{UInt64, MetalAdaptiveArrayPool}
                @test isempty(pools)
                true
            end
        )
        @test result == true
    end

    @testset "Device key verification" begin
        pool = get_task_local_metal_pool()
        current_dev = Metal.device()
        @test pool.device_key == current_dev

        pools = get_task_local_metal_pools()
        dev_key = objectid(current_dev)
        @test haskey(pools, dev_key)
        @test pools[dev_key] === pool

        @test get_task_local_metal_pool() === pool
    end

end
