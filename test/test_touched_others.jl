# Tests for the touched-others stack: per-scope selective fallback checkpoint/rewind.
# Invariant: a fallback pool has a checkpoint entry at depth d ⟺ it is in the depth-d
# segment of pool._touched_others — except under full checkpoint!(pool), whose eager
# sweep pairs with full rewind!(pool)'s sweep (segment stays empty, truncate-only).
# See the docstrings of _touch_fallback_pool!/_drain_touched_others! in src/state.jl.

using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!, checkpoint!, rewind!,
    _lazy_checkpoint!, _lazy_rewind!, _typed_lazy_checkpoint!, _typed_lazy_rewind!,
    _tracked_mask_for_types, _can_use_typed_path, get_task_local_pool, @with_pool

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

@testset "touched-others: no eager checkpoint on lazy entry (pollution regression)" begin
    pool = AdaptiveArrayPool()
    # Register three fallback types at global scope, then reset counters
    acquire!(pool, TOFooA, 4); acquire!(pool, TOFooB, 4); acquire!(pool, TOFooC, 4)
    reset!(pool)
    tpA = get_typed_pool!(pool, TOFooA)
    tpB = get_typed_pool!(pool, TOFooB)
    tpC = get_typed_pool!(pool, TOFooC)

    _lazy_checkpoint!(pool)
    # THE regression assertion: unrelated registered fallbacks are NOT touched
    @test tpB._checkpoint_depths == [0]
    @test tpC._checkpoint_depths == [0]

    v = acquire!(pool, TOFooA, 8)
    @test tpA._checkpoint_depths[end] == 2       # first-touch checkpoint at depth 2
    @test length(pool._touched_others) == 1
    @test pool._touched_others[end] === tpA
    @test tpA.n_active == 1

    # Re-acquire same type: no duplicate stack entry
    acquire!(pool, TOFooA, 8)
    @test length(pool._touched_others) == 1

    _lazy_rewind!(pool)
    @test tpA.n_active == 0
    @test isempty(pool._touched_others)
    @test tpB._checkpoint_depths == [0]          # still never visited
    @test tpC._checkpoint_depths == [0]
end

@testset "touched-others: nested scopes, same fallback type at two depths" begin
    pool = AdaptiveArrayPool()
    tpA = get_typed_pool!(pool, TOFooA)

    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, TOFooA, 4)
    @test tpA.n_active == 1

    _lazy_checkpoint!(pool)                       # depth 3
    acquire!(pool, TOFooA, 4)
    acquire!(pool, TOFooA, 4)
    @test tpA.n_active == 3
    @test length(pool._touched_others) == 2       # one entry per depth

    _lazy_rewind!(pool)                           # exit depth 3
    @test tpA.n_active == 1

    _lazy_rewind!(pool)                           # exit depth 2
    @test tpA.n_active == 0
end

@testset "touched-others: nested scope NOT touching outer's fallback" begin
    pool = AdaptiveArrayPool()
    tpA = get_typed_pool!(pool, TOFooA)

    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, TOFooA, 4)
    _lazy_checkpoint!(pool)                       # depth 3: does not touch TOFooA
    zeros!(pool, 16)                              # Float64 fixed-slot work only
    @test length(pool._touched_others) == 1       # no new fallback entry
    _lazy_rewind!(pool)
    @test tpA.n_active == 1                       # outer's array untouched
    _lazy_rewind!(pool)
    @test tpA.n_active == 0
end

@testset "touched-others: typed scope with helper touching a fallback (typed-lazy)" begin
    pool = AdaptiveArrayPool()
    tpB = get_typed_pool!(pool, TOFooB)

    _typed_lazy_checkpoint!(pool, Float64)
    zeros!(pool, 8)                               # tracked fixed-slot work
    acquire!(pool, TOFooB, 4)                     # untracked helper-style fallback touch
    @test pool._touched_has_others[end] == true
    @test pool._touched_others[end] === tpB
    @test !_can_use_typed_path(pool, _tracked_mask_for_types(Float64))
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test tpB.n_active == 0
end

@testset "touched-others: tracked fallback type via typed checkpoint!" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool, TOFooA)                     # fallback T tracked by macro
    tpA = get_typed_pool!(pool, TOFooA)
    @test pool._touched_others[end] === tpA       # pushed at checkpoint
    acquire!(pool, TOFooA, 4)                     # public-API acquire: no double push
    @test count(tp -> tp === tpA, pool._touched_others) == 1
    # macro exit path for has_others=true is _typed_lazy_rewind!
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(TOFooA))
    @test tpA.n_active == 0
    @test pool._touched_others_checkpoints == [0]
end

@testset "touched-others: new type registered mid-scope" begin
    pool = AdaptiveArrayPool()
    _lazy_checkpoint!(pool)
    acquire!(pool, TOFooC, 4)                     # first-ever registration, in-scope
    tpC = get_typed_pool!(pool, TOFooC)
    @test pool._touched_others[end] === tpC
    @test count(tp -> tp === tpC, pool._touched_others) == 1
    _lazy_rewind!(pool)
    @test tpC.n_active == 0
end

@testset "touched-others: full checkpoint!/rewind! pairing unchanged" begin
    pool = AdaptiveArrayPool()
    acquire!(pool, TOFooA, 4); reset!(pool)
    tpA = get_typed_pool!(pool, TOFooA)

    checkpoint!(pool)                             # eager: checkpoints ALL others
    @test tpA._checkpoint_depths[end] == 2
    acquire!(pool, TOFooA, 4)
    @test isempty(pool._touched_others)           # guard saw existing depth-2 entry
    rewind!(pool)
    @test tpA.n_active == 0
    @test pool._touched_others_checkpoints == [0]
end

@testset "touched-others: similar! records fallback touch" begin
    pool = AdaptiveArrayPool()
    src = [TOFooA(1.0), TOFooA(2.0)]
    _lazy_checkpoint!(pool)
    similar!(pool, src)
    tpA = get_typed_pool!(pool, TOFooA)
    @test pool._touched_others[end] === tpA
    _lazy_rewind!(pool)
    @test tpA.n_active == 0
end

@testset "touched-others: @with_pool integration + exception-leak recovery" begin
    # The task-local pool is a process-wide singleton shared with every other test
    # file. `@with_pool`'s fast (non-try/finally) path does not guarantee cleanup
    # for exceptions that escape ITS OWN scope (only nested leaks within one
    # invocation are caught by the entry-depth guard — see macros.jl docstring),
    # so earlier test files that deliberately throw across a `@with_pool` boundary
    # (e.g. stack-trace tests in test_macro_expansion.jl) can leave `_current_depth`
    # elevated. Start from a known-clean baseline so the absolute-depth assertions
    # below are independent of test file execution order.
    empty!(get_task_local_pool())

    # Integration through the real macro (task-local pool)
    f_leaf(n) = @with_pool p begin
        q = acquire!(p, TOFooA, n)
        length(q)
    end
    @test f_leaf(8) == 8
    tl = get_task_local_pool()
    @test get_typed_pool!(tl, TOFooA).n_active == 0
    @test isempty(tl._touched_others)

    # Inner scope throws, outer catches: outer exit must clean up leaked state
    function f_outer()
        @with_pool p begin
            acquire!(p, TOFooB, 4)
            try
                @with_pool p2 begin
                    acquire!(p2, TOFooC, 4)
                    error("boom")
                end
            catch
            end
            1
        end
    end
    @test f_outer() == 1
    @test tl._current_depth == 1
    @test get_typed_pool!(tl, TOFooB).n_active == 0
    @test get_typed_pool!(tl, TOFooC).n_active == 0
    @test isempty(tl._touched_others)
    @test tl._touched_others_checkpoints == [0]
    empty!(tl)   # leave the task-local pool clean for other test files
end

# Function barriers for @allocated measurements
function _to_lazy_roundtrip(pool, n)
    _lazy_checkpoint!(pool)
    v = acquire!(pool, TOFooA, n)
    s = length(v)
    _lazy_rewind!(pool)
    return s
end

function _to_macro_roundtrip(n)
    return @with_pool p begin
        v = acquire!(p, TOFooA, n)
        length(v)
    end
end

@testset "touched-others: zero allocation after warmup" begin
    pool = AdaptiveArrayPool()
    # Pollute the registry to also confirm cost independence at the alloc level
    acquire!(pool, TOFooB, 4); acquire!(pool, TOFooC, 4); reset!(pool)

    _to_lazy_roundtrip(pool, 32); _to_lazy_roundtrip(pool, 32)   # warmup
    @test @allocated(_to_lazy_roundtrip(pool, 32)) == 0

    _to_macro_roundtrip(32); _to_macro_roundtrip(32)             # warmup
    @test @allocated(_to_macro_roundtrip(32)) == 0
    empty!(get_task_local_pool())
end
