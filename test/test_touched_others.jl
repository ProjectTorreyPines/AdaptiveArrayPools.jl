# Tests for the touched-others stack: per-scope selective fallback checkpoint/rewind.
# Invariant: a fallback pool has an entry tagged with depth d in the depth-tagged
# stack (states/depths, kept in lockstep) ⟺ it was first touched at depth d — except
# under full checkpoint!(pool), whose eager sweep pairs with full rewind!(pool)'s
# sweep (stack stays empty, truncate-only).
# See the docstrings of the touch/drain helpers in src/state.jl.

using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!, checkpoint!, rewind!,
    _lazy_checkpoint!, _lazy_rewind!, _typed_lazy_checkpoint!, _typed_lazy_rewind!,
    _tracked_mask_for_types, _can_use_typed_path, get_task_local_pool, @with_pool

# Depth-tagged stack shape: states/depths always in lockstep; pools populated
# only in runtime-check builds.
function _to_stack_shape_ok(pool)
    ok = length(pool._touched_others_states) == length(pool._touched_others_depths)
    expected_pools = AdaptiveArrayPools.RUNTIME_CHECK >= 1 ?
        length(pool._touched_others_states) : 0
    return ok && length(pool._touched_others_pools) == expected_pools
end
_to_stack_len(pool) = length(pool._touched_others_states)

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
    @test isempty(pool._touched_others_states) && isempty(pool._touched_others_depths) && isempty(pool._touched_others_pools)
    @test isempty(pool._touched_others_depths)

    # reset! clears transient scope state, keeps registry
    acquire!(pool, TOFooA, 4)
    reset!(pool)
    @test _to_stack_len(pool) == 0
    @test isempty(pool._touched_others_depths)
    @test haskey(pool.others, TOFooA)          # registry kept

    # empty! clears everything
    acquire!(pool, TOFooA, 4)
    empty!(pool)
    @test _to_stack_len(pool) == 0
    @test isempty(pool._touched_others_depths)
    @test !haskey(pool.others, TOFooA)
end

@testset "touched-others: checkpoint/rewind plumbing balance" begin
    pool = AdaptiveArrayPool()

    # lazy pair
    _lazy_checkpoint!(pool)
    @test isempty(pool._touched_others_depths)
    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_depths)

    # typed single pair (fixed-slot type)
    checkpoint!(pool, Float64)
    @test isempty(pool._touched_others_depths)
    rewind!(pool, Float64)
    @test isempty(pool._touched_others_depths)

    # typed multi pair
    checkpoint!(pool, Float64, Int64)
    rewind!(pool, Float64, Int64)
    @test isempty(pool._touched_others_depths)

    # full pair
    checkpoint!(pool)
    rewind!(pool)
    @test isempty(pool._touched_others_depths)

    # typed-lazy pair
    _typed_lazy_checkpoint!(pool, Float64)
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test isempty(pool._touched_others_depths)

    # nesting
    _lazy_checkpoint!(pool)
    checkpoint!(pool, Float64)
    @test isempty(pool._touched_others_depths)
    rewind!(pool, Float64)
    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_depths)
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
    @test _to_stack_len(pool) == 1 && _to_stack_shape_ok(pool)
    @test pool._touched_others_states[end] === tpA.state
    @test tpA.n_active == 1

    # Re-acquire same type: no duplicate stack entry
    acquire!(pool, TOFooA, 8)
    @test _to_stack_len(pool) == 1 && _to_stack_shape_ok(pool)

    _lazy_rewind!(pool)
    @test tpA.n_active == 0
    @test _to_stack_len(pool) == 0
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
    @test _to_stack_len(pool) == 2 && _to_stack_shape_ok(pool)       # one entry per depth

    _lazy_rewind!(pool)                           # exit depth 3
    @test tpA.n_active == 1

    _lazy_rewind!(pool)                           # exit depth 2
    @test tpA.n_active == 0
end

@testset "touched-others: depth tags are exact and monotone" begin
    pool = AdaptiveArrayPool()
    tpA = get_typed_pool!(pool, TOFooA)

    _lazy_checkpoint!(pool)                       # depth 2
    @test isempty(pool._touched_others_depths)    # entry pushes nothing
    acquire!(pool, TOFooA, 4)
    @test pool._touched_others_depths == [2]
    @test pool._touched_others_states[end] === tpA.state

    _lazy_checkpoint!(pool)                       # depth 3
    acquire!(pool, TOFooB, 4)
    acquire!(pool, TOFooA, 4)                     # same type, new depth → new entry
    @test pool._touched_others_depths == [2, 3, 3]
    @test issorted(pool._touched_others_depths)   # monotone invariant
    @test _to_stack_shape_ok(pool)

    _lazy_rewind!(pool)                           # drains ONLY the ==3 entries
    @test pool._touched_others_depths == [2]
    @test tpA.n_active == 1
    @test get_typed_pool!(pool, TOFooB).n_active == 0

    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_depths)
    @test tpA.n_active == 0
end

@testset "touched-others: nested scope NOT touching outer's fallback" begin
    pool = AdaptiveArrayPool()
    tpA = get_typed_pool!(pool, TOFooA)

    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, TOFooA, 4)
    _lazy_checkpoint!(pool)                       # depth 3: does not touch TOFooA
    zeros!(pool, 16)                              # Float64 fixed-slot work only
    @test _to_stack_len(pool) == 1 && _to_stack_shape_ok(pool)       # no new fallback entry
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
    @test pool._touched_others_states[end] === tpB.state
    @test !_can_use_typed_path(pool, _tracked_mask_for_types(Float64))
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test tpB.n_active == 0
end

@testset "touched-others: tracked fallback type via typed checkpoint!" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool, TOFooA)                     # fallback T tracked by macro
    tpA = get_typed_pool!(pool, TOFooA)
    @test pool._touched_others_states[end] === tpA.state       # pushed at checkpoint
    acquire!(pool, TOFooA, 4)                     # public-API acquire: no double push
    @test count(st -> st === tpA.state, pool._touched_others_states) == 1
    # macro exit path for has_others=true is _typed_lazy_rewind!
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(TOFooA))
    @test tpA.n_active == 0
    @test isempty(pool._touched_others_depths)
end

@testset "touched-others: new type registered mid-scope" begin
    pool = AdaptiveArrayPool()
    _lazy_checkpoint!(pool)
    acquire!(pool, TOFooC, 4)                     # first-ever registration, in-scope
    tpC = get_typed_pool!(pool, TOFooC)
    @test pool._touched_others_states[end] === tpC.state
    @test count(st -> st === tpC.state, pool._touched_others_states) == 1
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
    @test _to_stack_len(pool) == 0           # guard saw existing depth-2 entry
    rewind!(pool)
    @test tpA.n_active == 0
    @test isempty(pool._touched_others_depths)
end

@testset "touched-others: similar! records fallback touch" begin
    pool = AdaptiveArrayPool()
    src = [TOFooA(1.0), TOFooA(2.0)]
    _lazy_checkpoint!(pool)
    similar!(pool, src)
    tpA = get_typed_pool!(pool, TOFooA)
    @test pool._touched_others_states[end] === tpA.state
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
    @test isempty(tl._touched_others_states)

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
    @test isempty(tl._touched_others_states)
    @test isempty(tl._touched_others_depths)
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

@testset "fallback lookup memo: identity and invalidation" begin
    pool = AdaptiveArrayPool()
    tp1 = get_typed_pool!(pool, TOFooA)
    @test get_typed_pool!(pool, TOFooA) === tp1     # repeat lookup: same pool
    @test get_typed_pool!(pool, TOFooB) !== tp1     # different type: different pool
    @test get_typed_pool!(pool, TOFooA) === tp1     # alternating types stay correct

    reset!(pool)                                     # keeps registry → memo may stay
    @test get_typed_pool!(pool, TOFooA) === tp1

    empty!(pool)                                     # kills registry → memo MUST die
    tp2 = get_typed_pool!(pool, TOFooA)
    @test tp2 !== tp1                                # stale-memo regression guard
    @test tp2 === pool.others[TOFooA]

    # end-to-end: acquire after empty! must use the fresh pool
    v = acquire!(pool, TOFooA, 4)
    @test tp2.n_active == 1 && tp1.n_active == 0
    reset!(pool)
end
