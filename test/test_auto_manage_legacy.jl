using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_task_local_pool

# Legacy path (Julia < 1.12): the auto-manage API is defined but a NO-OP (the legacy
# pool architecture has no capacity compaction). `const AUTO_MANAGE = false` makes the
# shared @with_pool scope-entry hook DCE away, and the public enable/disable/enabled API
# stays callable across the supported Julia range. Mirrors test_compact_legacy.jl.

@testset "auto-manage (legacy no-op, Julia < 1.12)" begin
    @test AdaptiveArrayPools.AUTO_MANAGE == false
    @test auto_manage_enabled() == false

    # enable! warns once and stays a no-op; disable!/enabled are safe and idempotent.
    @test (@test_logs (:warn,) match_mode = :any enable_auto_manage!()) === nothing
    @test auto_manage_enabled() == false
    # Accepts the full 1.12+ keyword set (incl. `trim_interval`) so the public API is
    # call-compatible across all supported Julia — passing it must not MethodError.
    @test enable_auto_manage!(compact_interval = 5.0, trim_interval = 60.0, compact_bloat_factor = 20) === nothing
    @test disable_auto_manage!() === nothing
    @test auto_manage_enabled() == false

    # The internal hooks the shared code references are no-ops too.
    @test AdaptiveArrayPools.register_auto_manage!(get_task_local_pool()) === nothing
    @test AdaptiveArrayPools._maybe_auto_manage!(get_task_local_pool()) === nothing

    # @with_pool still works on legacy (the hook DCEs to nothing; no auto-management).
    r = @with_pool pool begin
        x = acquire!(pool, Float64, 4)
        x .= 1.0
        sum(x)
    end
    @test r == 4.0
end
