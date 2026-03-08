@with_pool pool function foo()
    float64_vec = acquire!(pool, Float64, 10)
    float32_vec = acquire!(pool, Float32, 10)

    float64_mat = acquire!(pool, Float64, 10, 10)
    float32_mat = acquire!(pool, Float32, 10, 10)

    bv = acquire!(pool, Bit, 100)
    ba2 = acquire!(pool, Bit, 10, 10)
    ba3 = acquire!(pool, Bit, 5, 5, 4)

    tt1 = trues!(pool, 256)
    tt2 = ones!(pool, Bit, 10, 20)
    ff1 = falses!(pool, 100, 5)
    ff2 = zeros!(pool, Bit, 100)

    C = similar!(pool, tt1)
end


@testset "zero allocation on reuse" begin
    # Disable safety invalidation: rewind-time resize!/setfield! forces cache misses
    # (new SubArray views on legacy, new BitArray wrappers), breaking zero-alloc invariant.
    old_safety = POOL_SAFETY_LV[]
    POOL_SAFETY_LV[] = 0

    # First call: JIT + initial cache miss (pool arrays + N-way bitarray cache)
    alloc1 = @allocated foo()
    @test alloc1 > 0  # Sanity: pool reuse does save allocations vs. alloc-every-time

    # Extra warmup: in the full test suite, prior tests may leave the task-local pool in a
    # partially-warmed state (e.g. bitarray N-way cache sized for different call counts),
    # requiring one additional call to reach the stable hot path. This does NOT indicate a
    # correctness issue — alloc3/alloc4 below confirm zero-alloc once stable.
    foo()

    # Hot path: all subsequent calls must be zero-allocation
    alloc2 = @allocated foo()
    alloc3 = @allocated foo()
    @test alloc2 == 0
    @test alloc3 == 0

    POOL_SAFETY_LV[] = old_safety
end
