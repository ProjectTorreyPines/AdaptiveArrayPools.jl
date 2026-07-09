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

@testset "touched-others: checkpoint/rewind plumbing balance" begin
    pool = AdaptiveArrayPool()

    # lazy pair
    _lazy_checkpoint!(pool)
    @test pool._touched_others_checkpoints == [0, 0]
    _lazy_rewind!(pool)
    @test pool._touched_others_checkpoints == [0]

    # typed single pair (fixed-slot type)
    checkpoint!(pool, Float64)
    @test length(pool._touched_others_checkpoints) == 2
    rewind!(pool, Float64)
    @test pool._touched_others_checkpoints == [0]

    # typed multi pair
    checkpoint!(pool, Float64, Int64)
    rewind!(pool, Float64, Int64)
    @test pool._touched_others_checkpoints == [0]

    # full pair
    checkpoint!(pool)
    rewind!(pool)
    @test pool._touched_others_checkpoints == [0]

    # typed-lazy pair
    _typed_lazy_checkpoint!(pool, Float64)
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test pool._touched_others_checkpoints == [0]

    # nesting
    _lazy_checkpoint!(pool)
    checkpoint!(pool, Float64)
    @test length(pool._touched_others_checkpoints) == 3
    rewind!(pool, Float64)
    _lazy_rewind!(pool)
    @test pool._touched_others_checkpoints == [0]
end
