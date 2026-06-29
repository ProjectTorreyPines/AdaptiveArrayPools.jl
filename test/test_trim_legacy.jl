using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: trim!, get_task_local_pool

# Legacy path (Julia < 1.12): trim! is a defined, exported NO-OP that returns a
# zero summary and warns once. This keeps the public API portable across the full
# supported Julia range (dependents can import/call trim! on any version).

@testset "trim! (legacy no-op, Julia < 1.12)" begin
    pool = get_task_local_pool()

    # First call warns once and returns a zero summary.
    s = @test_logs (:warn,) match_mode = :any trim!(pool)
    @test s.slots_released == 0
    @test s.wrappers_released == 0
    @test s.estimated_bytes_released == 0
    @test s.gc_triggered == false

    # Same zero-summary shape for the other entry points.
    @test trim!(pool, Float64).slots_released == 0
    @test trim!().slots_released == 0

    # Legacy no-op ignores force_gc (no reclamation to do).
    @test trim!(pool; force_gc = true).gc_triggered == false

    # DisabledPool: zero-summary no-op on the legacy path too.
    zd = trim!(DISABLED_CPU)
    @test zd.slots_released == 0
    @test zd.wrappers_released == 0
    @test zd.estimated_bytes_released == 0
    @test zd.gc_triggered == false
    @test trim!(DISABLED_CPU, Float64).slots_released == 0
end
