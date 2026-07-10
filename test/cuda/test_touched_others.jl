# CUDA port of test/test_touched_others.jl — depth-tagged touched-others stack
# and fallback lookup memo. Invariant: a fallback pool has an entry tagged with
# depth d in the depth-tagged stack (states/depths, kept in lockstep) ⟺ it was
# first touched at depth d — except under full checkpoint!(pool), whose eager
# sweep pairs with full rewind!(pool)'s sweep (stack stays empty, truncate-only).
#
# Fallback exercise types: UInt16, UInt8, Int8, UInt32, Int16,
# UInt64, Int128, UInt128 — CUDA would accept Int128/UInt128 as CuArray element
# types, but two isbits structs are substituted anyway (same as Metal) to keep
# the two GPU test files diffable against each other.
# Float16 is a fixed struct field with `_fixed_slot_bit == 0` on CUDA too (bit-7
# reassignment, same as Metal), so it must NEVER be routed through the
# touched-others stack (see the dedicated "Float16 bit-7" testsets below).

using AdaptiveArrayPools: _lazy_checkpoint!, _lazy_rewind!,
    _typed_lazy_checkpoint!, _typed_lazy_rewind!,
    _tracked_mask_for_types, _can_use_typed_path

# Named distinctly from the CPU test suite's TOFooA/B/C (both files load into
# the same top-level module during a full-suite run) to avoid a
# struct-redefinition clash.
struct CudaFallbackStructA
    x::Float32
end
struct CudaFallbackStructB
    x::Float32
end

@testset "touched-others: fields & lifecycle" begin
    pool = CuAdaptiveArrayPool()
    @test isempty(pool._touched_others_states) && isempty(pool._touched_others_depths) && isempty(pool._touched_others_pools)

    # reset! clears transient scope state, keeps registry
    acquire!(pool, UInt16, 4)
    reset!(pool)
    @test length(pool._touched_others_states) == 0
    @test isempty(pool._touched_others_depths)
    @test haskey(pool.others, UInt16)          # registry kept

    # empty! clears everything
    acquire!(pool, UInt16, 4)
    empty!(pool)
    @test length(pool._touched_others_states) == 0
    @test isempty(pool._touched_others_depths)
    @test !haskey(pool.others, UInt16)
end

@testset "touched-others: checkpoint/rewind plumbing balance" begin
    pool = CuAdaptiveArrayPool()

    # lazy pair
    _lazy_checkpoint!(pool)
    @test isempty(pool._touched_others_depths)
    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_depths)

    # typed single pair (fixed-slot type)
    checkpoint!(pool, Float32)
    @test isempty(pool._touched_others_depths)
    rewind!(pool, Float32)
    @test isempty(pool._touched_others_depths)

    # typed multi pair
    checkpoint!(pool, Float32, Int32)
    rewind!(pool, Float32, Int32)
    @test isempty(pool._touched_others_depths)

    # full pair
    checkpoint!(pool)
    rewind!(pool)
    @test isempty(pool._touched_others_depths)

    # typed-lazy pair
    _typed_lazy_checkpoint!(pool, Float32)
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float32))
    @test isempty(pool._touched_others_depths)

    # nesting
    _lazy_checkpoint!(pool)
    checkpoint!(pool, Float32)
    @test isempty(pool._touched_others_depths)
    rewind!(pool, Float32)
    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_depths)
end

@testset "touched-others: no eager checkpoint on lazy entry (pollution regression)" begin
    pool = CuAdaptiveArrayPool()
    # Register all 8 fallback exercise types at global scope, then reset counters.
    # (Int128/UInt128 substituted with distinct isbits structs — kept for
    # diffability against the Metal test file, which must substitute them.)
    fallback_types = (UInt16, UInt8, Int8, UInt32, Int16, UInt64, CudaFallbackStructA, CudaFallbackStructB)
    for T in fallback_types
        acquire!(pool, T, 4)
    end
    reset!(pool)
    tps = Dict(T => get_typed_pool!(pool, T) for T in fallback_types)

    _lazy_checkpoint!(pool)
    # THE regression assertion: unrelated registered fallbacks are NOT touched
    for T in fallback_types[2:end]
        @test tps[T]._checkpoint_depths == [0]
    end

    acquire!(pool, UInt16, 8)
    @test tps[UInt16]._checkpoint_depths[end] == 2       # first-touch checkpoint at depth 2
    @test length(pool._touched_others_depths) == 1
    @test pool._touched_others_states[end] === tps[UInt16].state
    @test tps[UInt16].n_active == 1

    # Re-acquire same type: no duplicate stack entry
    acquire!(pool, UInt16, 8)
    @test length(pool._touched_others_depths) == 1

    _lazy_rewind!(pool)
    @test tps[UInt16].n_active == 0
    @test isempty(pool._touched_others_depths)
    for T in fallback_types[2:end]
        @test tps[T]._checkpoint_depths == [0]           # still never visited
    end
end

@testset "touched-others: nested scopes, same fallback type at two depths" begin
    pool = CuAdaptiveArrayPool()
    tpA = get_typed_pool!(pool, UInt16)

    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, UInt16, 4)
    @test tpA.n_active == 1

    _lazy_checkpoint!(pool)                       # depth 3
    acquire!(pool, UInt16, 4)
    acquire!(pool, UInt16, 4)
    @test tpA.n_active == 3
    @test length(pool._touched_others_depths) == 2   # one entry per depth

    _lazy_rewind!(pool)                           # exit depth 3
    @test tpA.n_active == 1

    _lazy_rewind!(pool)                           # exit depth 2
    @test tpA.n_active == 0
end

@testset "touched-others: depth tags are exact and monotone" begin
    pool = CuAdaptiveArrayPool()
    tpA = get_typed_pool!(pool, UInt16)

    _lazy_checkpoint!(pool)                       # depth 2
    @test isempty(pool._touched_others_depths)    # entry pushes nothing
    acquire!(pool, UInt16, 4)
    @test pool._touched_others_depths == [2]
    @test pool._touched_others_states[end] === tpA.state

    _lazy_checkpoint!(pool)                       # depth 3
    acquire!(pool, UInt8, 4)                      # different fallback type, new depth
    @test pool._touched_others_depths == [2, 3]
    @test issorted(pool._touched_others_depths)   # monotone invariant

    _lazy_rewind!(pool)                           # drains ONLY the ==3 entries
    @test pool._touched_others_depths == [2]
    @test tpA.n_active == 1
    @test get_typed_pool!(pool, UInt8).n_active == 0

    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_depths)
    @test tpA.n_active == 0
end

@testset "touched-others: nested scope NOT touching outer's fallback" begin
    pool = CuAdaptiveArrayPool()
    tpA = get_typed_pool!(pool, UInt16)

    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, UInt16, 4)
    _lazy_checkpoint!(pool)                       # depth 3: does not touch UInt16
    acquire!(pool, Float32, 16)                   # Float32 fixed-slot work only
    @test length(pool._touched_others_depths) == 1   # no new fallback entry
    _lazy_rewind!(pool)
    @test tpA.n_active == 1                       # outer's array untouched
    _lazy_rewind!(pool)
    @test tpA.n_active == 0
end

@testset "touched-others: typed scope with helper touching a fallback (typed-lazy)" begin
    pool = CuAdaptiveArrayPool()
    tpB = get_typed_pool!(pool, UInt8)

    _typed_lazy_checkpoint!(pool, Float32)
    acquire!(pool, Float32, 8)                    # tracked fixed-slot work
    acquire!(pool, UInt8, 4)                      # untracked helper-style fallback touch
    @test pool._touched_has_others[end] == true
    @test pool._touched_others_states[end] === tpB.state
    @test !_can_use_typed_path(pool, _tracked_mask_for_types(Float32))
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float32))
    @test tpB.n_active == 0
end

@testset "touched-others: tracked fallback type via typed checkpoint!" begin
    pool = CuAdaptiveArrayPool()

    checkpoint!(pool, UInt16)                     # fallback T tracked by macro
    tpA = get_typed_pool!(pool, UInt16)
    @test pool._touched_others_states[end] === tpA.state       # pushed at checkpoint
    acquire!(pool, UInt16, 4)                     # public-API acquire: no double push
    @test count(st -> st === tpA.state, pool._touched_others_states) == 1
    # macro exit path for has_others=true is _typed_lazy_rewind!
    _typed_lazy_rewind!(pool, _tracked_mask_for_types(UInt16))
    @test tpA.n_active == 0
    @test isempty(pool._touched_others_depths)
end

@testset "touched-others: new type registered mid-scope" begin
    pool = CuAdaptiveArrayPool()
    _lazy_checkpoint!(pool)
    acquire!(pool, Int8, 4)                       # first-ever registration, in-scope
    tpC = get_typed_pool!(pool, Int8)
    @test pool._touched_others_states[end] === tpC.state
    @test count(st -> st === tpC.state, pool._touched_others_states) == 1
    _lazy_rewind!(pool)
    @test tpC.n_active == 0
end

@testset "touched-others: full checkpoint!/rewind! pairing unchanged" begin
    pool = CuAdaptiveArrayPool()
    acquire!(pool, UInt16, 4)
    reset!(pool)
    tpA = get_typed_pool!(pool, UInt16)

    checkpoint!(pool)                             # eager: checkpoints ALL others
    @test tpA._checkpoint_depths[end] == 2
    acquire!(pool, UInt16, 4)
    @test length(pool._touched_others_depths) == 0   # guard saw existing depth-2 entry
    rewind!(pool)
    @test tpA.n_active == 0
    @test isempty(pool._touched_others_depths)
end

@testset "touched-others: similar! records fallback touch" begin
    pool = CuAdaptiveArrayPool()
    src = CuArray(UInt16[1, 2])
    _lazy_checkpoint!(pool)
    similar!(pool, src)
    tpA = get_typed_pool!(pool, UInt16)
    @test pool._touched_others_states[end] === tpA.state
    _lazy_rewind!(pool)
    @test tpA.n_active == 0
end

@testset "touched-others: @with_pool integration + exception-leak recovery" begin
    # The task-local CUDA pool is a process-wide-per-device singleton shared with
    # every other test file. Start from a known-clean baseline so the absolute-depth
    # assertions below are independent of test file execution order.
    empty!(get_task_local_cuda_pool())

    # Integration through the real macro (task-local pool)
    f_leaf(n) = @with_pool :cuda p begin
        q = acquire!(p, UInt16, n)
        length(q)
    end
    @test f_leaf(8) == 8
    tl = get_task_local_cuda_pool()
    @test get_typed_pool!(tl, UInt16).n_active == 0
    @test isempty(tl._touched_others_states)

    # Inner scope throws, outer catches: outer exit must clean up leaked state
    function f_outer()
        @with_pool :cuda p begin
            acquire!(p, UInt8, 4)
            try
                @with_pool :cuda p2 begin
                    acquire!(p2, Int8, 4)
                    error("boom")
                end
            catch
            end
            1
        end
    end
    @test f_outer() == 1
    @test tl._current_depth == 1
    @test get_typed_pool!(tl, UInt8).n_active == 0
    @test get_typed_pool!(tl, Int8).n_active == 0
    @test isempty(tl._touched_others_states)
    @test isempty(tl._touched_others_depths)
    empty!(tl)   # leave the task-local pool clean for other test files
end

# ==============================================================================
# S=1: runtime-check-gated pools vector + invalidation
# ==============================================================================
# Every touched-others testset above drives CuAdaptiveArrayPool() at the
# default RUNTIME_CHECK level (S=0, no preference flip), so none of them exercise
# the `_runtime_check(pool)` branch in `_drain_touched_others!`/`_touch_fallback_pool!`
# that pushes/pops `_touched_others_pools` and invalidates a released fallback
# slot. Construct an S=1 pool directly to close that gap.

@testset "touched-others: S=1 lazy-drain lockstep (pools vector populated, invalidation engaged)" begin
    pool = ext._make_cuda_pool(1)
    _lazy_checkpoint!(pool)                       # depth 2
    v = acquire!(pool, UInt16, 8)                 # genuine fallback acquire
    tp = get_typed_pool!(pool, UInt16)
    @test length(pool._touched_others_pools) == 1   # S=1-only: pools vector populated
    @test pool._touched_others_depths == [2]
    @test pool._touched_others_pools[end] === tp

    _lazy_rewind!(pool)
    @test isempty(pool._touched_others_states)
    @test isempty(pool._touched_others_depths)
    @test isempty(pool._touched_others_pools)
    @test tp.n_active == 0

    # S=1 invalidation actually engaged: the previously-acquired wrapper's dims
    # were zeroed (same assertion pattern as "arr_wrappers invalidated on rewind"
    # in test_cuda_safety.jl).
    @test all(==(0), size(v))
end

@testset "fallback lookup memo: fields and lifecycle" begin
    pool = CuAdaptiveArrayPool()
    tp = get_typed_pool!(pool, UInt16)             # slow-path lookup
    @test pool._lookup_memo_type === UInt16
    @test pool._lookup_memo_tp === tp

    empty!(pool)
    @test pool._lookup_memo_type === nothing
    @test pool._lookup_memo_tp === nothing

    tp2 = get_typed_pool!(pool, UInt16)            # re-register
    @test pool._lookup_memo_type === UInt16
    @test pool._lookup_memo_tp === tp2

    reset!(pool)                                    # keeps registry AND memo
    @test pool._lookup_memo_type === UInt16
    @test pool._lookup_memo_tp === tp2
end

@testset "fallback lookup memo: identity and invalidation" begin
    pool = CuAdaptiveArrayPool()
    tp1 = get_typed_pool!(pool, UInt16)
    @test get_typed_pool!(pool, UInt16) === tp1     # repeat lookup: same pool
    @test get_typed_pool!(pool, UInt8) !== tp1      # different type: different pool
    @test get_typed_pool!(pool, UInt16) === tp1     # alternating types stay correct

    reset!(pool)                                     # keeps registry → memo may stay
    @test get_typed_pool!(pool, UInt16) === tp1

    empty!(pool)                                     # kills registry → memo MUST die
    tp2 = get_typed_pool!(pool, UInt16)
    @test tp2 !== tp1                                # stale-memo regression guard
    @test tp2 === pool.others[UInt16]

    # end-to-end: acquire after empty! must use the fresh pool
    acquire!(pool, UInt16, 4)
    @test tp2.n_active == 1 && tp1.n_active == 0
    reset!(pool)
end

# ==============================================================================
# Float16 bit-7 non-interaction (GPU-only divergence — CPU has no such field)
# ==============================================================================
# Float16 is a FIXED struct field (pool.float16) but `_fixed_slot_bit(Float16) ==
# 0` (Float16 is absent from the base module's generic fixed-slot bit table).
# Routing MUST key off `_cuda_is_fallback_type(T) = !(T <: _CUDA_FIXED_TYPES)`,
# never `_fixed_slot_bit(T) == 0` — Float16 goes through the direct
# `_checkpoint_typed_pool!`/`_rewind_typed_pool!` path like other fixed slots and
# must NEVER get a touched-others stack entry.

@testset "Float16 bit-7: lazy scope acquiring Float16 pushes no stack entry" begin
    pool = CuAdaptiveArrayPool()
    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, Float16, 4)
    @test isempty(pool._touched_others_depths)
    @test pool.float16.n_active == 1
    _lazy_rewind!(pool)
    @test pool.float16.n_active == 0
    @test isempty(pool._touched_others_depths)
end

@testset "Float16 bit-7: typed inner scope alongside live parent Float16 arrays" begin
    empty!(get_task_local_cuda_pool())
    pool = get_task_local_cuda_pool()
    depth_before = pool._current_depth

    _lazy_checkpoint!(pool)                       # depth 2 (parent, lazy)
    v_outer = acquire!(pool, Float16, 4)
    v_outer .= Float16(3)
    @test pool.float16.n_active == 1

    # acquire!(pool2, Float16, 8) below is a static-type call, so the macro
    # statically resolves Float16 and drives the TYPED path
    # (_typed_lazy_checkpoint!/_typed_lazy_rewind!) rather than the untyped
    # _lazy_checkpoint!/_lazy_rewind! path — no separate typed-macro syntax
    # exists or is needed to opt in.
    result = @with_pool :cuda pool2 begin
        w = acquire!(pool2, Float16, 8)
        @test isempty(pool._touched_others_depths)    # Float16 never stack-managed, even in typed scope
        w .= Float16(9)
        sum(w)
    end
    @test result == Float16(9) * 8
    @test isempty(pool._touched_others_depths)    # Float16 never stack-managed
    @test pool.float16.n_active == 1              # parent's slot survives the inner exit
    @test all(Array(v_outer) .== Float16(3))      # parent's array contents survive

    _lazy_rewind!(pool)
    @test pool.float16.n_active == 0
    @test pool._current_depth == depth_before     # depth balance: no leak across the testset
    empty!(get_task_local_cuda_pool())
end

@testset "Float16 bit-7: mixed scope Float16 + fallback pushes exactly one entry" begin
    pool = CuAdaptiveArrayPool()
    _lazy_checkpoint!(pool)                       # depth 2
    acquire!(pool, Float16, 4)
    acquire!(pool, UInt16, 4)
    @test length(pool._touched_others_depths) == 1
    @test pool._touched_others_depths == [2]
    tpA = get_typed_pool!(pool, UInt16)
    @test pool._touched_others_states[end] === tpA.state
    _lazy_rewind!(pool)
    @test pool.float16.n_active == 0
    @test tpA.n_active == 0
    @test isempty(pool._touched_others_depths)
end
