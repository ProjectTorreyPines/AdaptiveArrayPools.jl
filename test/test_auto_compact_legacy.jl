using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_task_local_pool

# Legacy path (Julia < 1.12): the auto-compact API is defined but a NO-OP (the legacy
# pool architecture has no capacity compaction). `const AUTO_COMPACT = false` makes the
# shared @with_pool scope-exit hook DCE away, and the public enable/disable/enabled API
# stays callable across the supported Julia range. Mirrors test_compact_legacy.jl.

@testset "auto-compact (legacy no-op, Julia < 1.12)" begin
    @test AdaptiveArrayPools.AUTO_COMPACT == false
    @test auto_compact_enabled() == false

    # enable! warns once and stays a no-op; disable!/enabled are safe and idempotent.
    @test (@test_logs (:warn,) match_mode = :any enable_auto_compact!()) === nothing
    @test auto_compact_enabled() == false
    @test enable_auto_compact!(interval = 5.0, factor = 20, active = false) === nothing
    @test disable_auto_compact!() === nothing
    @test auto_compact_enabled() == false

    # The internal hooks the shared code references are no-ops too.
    @test AdaptiveArrayPools.register_auto_compact!(get_task_local_pool()) === nothing
    @test AdaptiveArrayPools._maybe_auto_compact!(get_task_local_pool()) === nothing

    # @with_pool still works on legacy (the hook DCEs to nothing; no auto-compaction).
    r = @with_pool pool begin
        x = acquire!(pool, Float64, 4)
        x .= 1.0
        sum(x)
    end
    @test r == 4.0
end
