# Tests for the touched-others stack: per-scope selective fallback checkpoint/rewind.
# See docs/plans/DESIGN_fallback_touch_tracking.md.

using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!, checkpoint!, rewind!,
    _lazy_checkpoint!, _lazy_rewind!, _typed_lazy_checkpoint!, _typed_lazy_rewind!,
    _tracked_mask_for_types, _can_use_typed_path

# Distinct isbits fallback types (not fixed slots)
struct TOFooA
    x::Float64
end
struct TOFooB
    x::Float64
end
struct TOFooC
    x::Float64
end

@testset "touched-others: fields & lifecycle" begin
    pool = AdaptiveArrayPool()
    @test pool._touched_others == Any[]
    @test pool._touched_others_checkpoints == [0]

    # reset! clears transient scope state, keeps registry
    acquire!(pool, TOFooA, 4)
    reset!(pool)
    @test isempty(pool._touched_others)
    @test pool._touched_others_checkpoints == [0]
    @test haskey(pool.others, TOFooA)          # registry kept

    # empty! clears everything
    acquire!(pool, TOFooA, 4)
    empty!(pool)
    @test isempty(pool._touched_others)
    @test pool._touched_others_checkpoints == [0]
    @test !haskey(pool.others, TOFooA)
end
