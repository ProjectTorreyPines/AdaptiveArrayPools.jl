using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: compact!, get_task_local_pool

# Legacy path (Julia < 1.12): compact! is a defined, exported NO-OP that returns a
# zero summary and warns once. This keeps the public API portable across the full
# supported Julia range (dependents can import/call compact! on any version).
# Mirrors test_trim_legacy.jl.

@testset "compact! (legacy no-op, Julia < 1.12)" begin
    pool = get_task_local_pool()

    # First call warns once and returns a zero summary.
    s = @test_logs (:warn,) match_mode = :any compact!(pool)
    @test s.slots_compacted == 0
    @test s.bytes_reclaimed == 0
    @test s.gc_triggered == false

    # Same zero-summary shape for the other entry points (single-type, varargs, no-arg).
    @test compact!(pool, Float64).slots_compacted == 0
    @test compact!(pool, Float64, Int64).slots_compacted == 0
    @test compact!().slots_compacted == 0

    # Legacy no-op ignores the active / force_gc kwargs (no reclamation to do).
    @test compact!(pool; active = true).slots_compacted == 0
    @test compact!(pool; active = false).slots_compacted == 0
    @test compact!(pool; force_gc = true).gc_triggered == false

    # DisabledPool: zero-summary no-op on the legacy path too.
    zd = compact!(DISABLED_CPU)
    @test zd.slots_compacted == 0
    @test zd.bytes_reclaimed == 0
    @test zd.gc_triggered == false
    @test compact!(DISABLED_CPU, Float64).slots_compacted == 0
    @test compact!(DISABLED_CPU, Float64, Int64).slots_compacted == 0
end
