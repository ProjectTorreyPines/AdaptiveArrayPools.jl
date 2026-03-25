import AdaptiveArrayPools: _validate_pool_return, _check_bitchunks_overlap,
    PoolRuntimeEscapeError, _make_pool,
    _lazy_checkpoint!, _lazy_rewind!,
    checkpoint!, rewind!

# ==============================================================================
# Scope-Aware Validation: outer-scope arrays must NOT trigger escape errors
# in inner scopes.
#
# Bug: _check_pointer_overlap iterates ALL tp.vectors, not just the ones
# acquired in the current scope. Arrays acquired in an outer scope should be
# freely usable (and returnable) in inner scopes without false-positive errors.
# ==============================================================================

@testset "Scope-depth-aware validation" begin

    # ------------------------------------------------------------------
    # Scenario 1: acquire at depth 1 → validate inside depth 2
    #   Array acquired before @with_pool, then used inside @with_pool.
    #   The inner scope's _validate_pool_return should NOT flag it.
    # ------------------------------------------------------------------
    @testset "outer acquire, inner validate — should NOT throw (Array)" begin
        pool = _make_pool(true)

        # Depth 1: acquire an array
        checkpoint!(pool)
        v = acquire!(pool, Float64, 100)
        fill!(v, 1.0)

        # Depth 2: inner scope
        checkpoint!(pool)
        w = acquire!(pool, Float64, 50)   # inner-scope acquire

        # v was acquired in depth 1 — returning it from depth 2 is safe
        _validate_pool_return(v, pool)   # ← Should NOT throw

        rewind!(pool)   # rewind depth 2 (releases w, keeps v)
        @test sum(v) == 100.0   # v is still valid
        rewind!(pool)   # rewind depth 1
    end

    @testset "outer acquire, inner validate — should NOT throw (BitArray)" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        bv = acquire!(pool, Bit, 100)
        fill!(bv, true)

        checkpoint!(pool)
        _ = acquire!(pool, Bit, 50)   # inner-scope acquire

        # bv was acquired in depth 1 — should NOT be flagged
        _validate_pool_return(bv, pool)   # ← Should NOT throw

        rewind!(pool)
        @test count(bv) == 100
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 2: nested scopes — acquire at depth 2, validate at depth 3
    # ------------------------------------------------------------------
    @testset "nested scope — outer array returned from inner scope" begin
        pool = _make_pool(true)

        # Depth 1
        checkpoint!(pool)

        # Depth 2: acquire
        checkpoint!(pool)
        v = acquire!(pool, Float64, 20)
        fill!(v, 2.0)

        # Depth 3: inner scope uses v
        checkpoint!(pool)
        _ = acquire!(pool, Float64, 10)   # inner-scope acquire

        # v belongs to depth 2, we're at depth 3 → safe
        _validate_pool_return(v, pool)   # ← Should NOT throw

        rewind!(pool)   # exit depth 3
        @test sum(v) == 40.0

        rewind!(pool)   # exit depth 2
        rewind!(pool)   # exit depth 1
    end

    # ------------------------------------------------------------------
    # Scenario 3: in-place function return pattern
    #   acquire → pass to in-place function → function returns the array
    #   → validate on that return value inside the same outer scope
    # ------------------------------------------------------------------
    @testset "in-place function return — should NOT throw" begin
        inplace_fill!(arr) = (fill!(arr, 3.14); arr)   # returns the same array

        pool = _make_pool(true)

        checkpoint!(pool)
        v = acquire!(pool, Float64, 50)

        # Inner scope: call in-place function, result is v
        checkpoint!(pool)
        result = inplace_fill!(v)

        # result === v, acquired in outer scope → safe
        _validate_pool_return(result, pool)   # ← Should NOT throw

        rewind!(pool)
        @test result[1] ≈ 3.14
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 4: inner-scope acquire SHOULD still throw
    #   Only arrays acquired in the CURRENT scope should be flagged.
    # ------------------------------------------------------------------
    @testset "inner-scope acquire — SHOULD still throw" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        _ = acquire!(pool, Float64, 100)   # outer acquire (safe)

        checkpoint!(pool)
        w = acquire!(pool, Float64, 50)    # inner acquire (should be caught)

        @test_throws PoolRuntimeEscapeError _validate_pool_return(w, pool)

        rewind!(pool)
        rewind!(pool)
    end

    @testset "inner-scope acquire BitArray — SHOULD still throw" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        _ = acquire!(pool, Bit, 100)   # outer

        checkpoint!(pool)
        bw = acquire!(pool, Bit, 50)   # inner (should be caught)

        @test_throws PoolRuntimeEscapeError _validate_pool_return(bw, pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 5: lazy checkpoint/rewind path (same bug, different entry)
    # ------------------------------------------------------------------
    @testset "lazy path — outer acquire, inner validate" begin
        pool = _make_pool(true)

        _lazy_checkpoint!(pool)
        v = acquire!(pool, Float64, 100)
        fill!(v, 5.0)

        _lazy_checkpoint!(pool)
        _ = acquire!(pool, Float64, 30)

        # v acquired at outer depth → safe
        _validate_pool_return(v, pool)   # ← Should NOT throw

        _lazy_rewind!(pool)
        @test sum(v) == 500.0
        _lazy_rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 6: container wrapping outer-scope array
    #   Tuple/NamedTuple containing an outer-scope array should pass.
    # ------------------------------------------------------------------
    @testset "outer array inside container — should NOT throw" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        fill!(v, 1.0)

        checkpoint!(pool)
        _ = acquire!(pool, Float64, 5)

        # v inside tuple — v belongs to outer scope → safe
        _validate_pool_return((sum(v), v), pool)   # ← Should NOT throw for v

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 7: others (non-fixed-slot) type from outer scope
    # ------------------------------------------------------------------
    @testset "outer acquire non-fixed-slot type — should NOT throw" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        v = acquire!(pool, UInt8, 100)
        fill!(v, 0x42)

        checkpoint!(pool)
        _ = acquire!(pool, UInt8, 50)   # inner

        _validate_pool_return(v, pool)   # ← Should NOT throw

        rewind!(pool)
        @test v[1] == 0x42
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 8: mixed — outer fixed-slot + inner different type
    #   The outer Float64 should pass even when inner Int64 exists.
    # ------------------------------------------------------------------
    @testset "outer Float64, inner Int64 — outer should NOT throw" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        v_f64 = acquire!(pool, Float64, 20)
        fill!(v_f64, 9.0)

        checkpoint!(pool)
        _ = acquire!(pool, Int64, 10)   # different type in inner scope

        _validate_pool_return(v_f64, pool)   # ← Should NOT throw

        rewind!(pool)
        @test sum(v_f64) == 180.0
        rewind!(pool)
    end

    # ==================================================================
    # Complex scenarios: partial escape from mixed inner/outer arrays
    # (compile-time can't catch these; runtime check must)
    # ==================================================================

    # ------------------------------------------------------------------
    # Scenario 9: same-type mix — outer + inner acquire same type,
    #   return tuple where outer is safe but inner escapes
    # ------------------------------------------------------------------
    @testset "same-type mix: outer safe, inner escapes via tuple" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        v_outer = acquire!(pool, Float64, 20)
        fill!(v_outer, 1.0)

        checkpoint!(pool)
        v_inner = acquire!(pool, Float64, 10)
        fill!(v_inner, 2.0)

        # Returning only the inner array should throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_inner, pool)

        # Returning tuple with inner array should also throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(
            (sum(v_outer), v_inner), pool
        )

        # Returning only the outer array should NOT throw
        _validate_pool_return(v_outer, pool)

        # Returning tuple with only outer array and scalars is safe
        _validate_pool_return((v_outer, sum(v_inner)), pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 10: in-place function modifies inner array, returns it
    #   through opaque function — runtime must still catch it
    # ------------------------------------------------------------------
    @testset "opaque in-place on inner array — should throw" begin
        opaque_fill!(arr) = (fill!(arr, 99.0); arr)

        pool = _make_pool(true)

        checkpoint!(pool)
        v_outer = acquire!(pool, Float64, 20)

        checkpoint!(pool)
        v_inner = acquire!(pool, Float64, 10)

        # In-place on inner → returns inner → escape
        leaked = opaque_fill!(v_inner)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(leaked, pool)

        # In-place on outer → returns outer → safe (outer scope)
        safe_result = opaque_fill!(v_outer)
        _validate_pool_return(safe_result, pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 11: 3-level nesting — acquire at each level,
    #   validate at deepest with mixed results
    # ------------------------------------------------------------------
    @testset "3-level nesting: only deepest acquire should throw" begin
        pool = _make_pool(true)

        # Depth 1
        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 100)
        fill!(v1, 1.0)

        # Depth 2
        checkpoint!(pool)
        v2 = acquire!(pool, Float64, 50)
        fill!(v2, 2.0)

        # Depth 3
        checkpoint!(pool)
        v3 = acquire!(pool, Float64, 25)
        fill!(v3, 3.0)

        # At depth 3: v1 and v2 are from outer scopes → safe
        _validate_pool_return(v1, pool)
        _validate_pool_return(v2, pool)

        # v3 is from current (depth 3) scope → escape!
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v3, pool)

        # Tuple mixing all three — v3 causes the throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return((v1, v2, v3), pool)

        # Tuple with only v1 and v2 — safe
        _validate_pool_return((v1, v2), pool)

        rewind!(pool)   # exit depth 3

        # At depth 2: v1 is outer → safe, v2 is current → escape
        _validate_pool_return(v1, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v2, pool)

        rewind!(pool)   # exit depth 2

        # At depth 1: v1 is current → escape
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v1, pool)

        rewind!(pool)   # exit depth 1
    end

    # ------------------------------------------------------------------
    # Scenario 12: NamedTuple return with partial escape —
    #   common pattern: (result=scalar, buffer=pool_array)
    # ------------------------------------------------------------------
    @testset "NamedTuple partial escape: inner buffer leaks" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        outer_buf = acquire!(pool, Float64, 100)

        checkpoint!(pool)
        inner_buf = acquire!(pool, Float64, 50)
        fill!(inner_buf, 1.0)

        # Realistic return pattern: scalar result + leaked buffer
        @test_throws PoolRuntimeEscapeError _validate_pool_return(
            (result = sum(inner_buf), buffer = inner_buf), pool
        )

        # Safe pattern: scalar result + outer buffer (not escaping)
        _validate_pool_return(
            (result = sum(inner_buf), buffer = outer_buf), pool
        )

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 13: view of inner array escapes — SubArray detection
    # ------------------------------------------------------------------
    @testset "view of inner array — should throw" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        v_outer = acquire!(pool, Float64, 100)

        checkpoint!(pool)
        v_inner = acquire!(pool, Float64, 50)

        # View of inner array → escape
        inner_view = view(v_inner, 1:25)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(inner_view, pool)

        # View of outer array → safe (outer scope)
        outer_view = view(v_outer, 1:50)
        _validate_pool_return(outer_view, pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 14: BitArray mixed — outer Bit safe, inner Bit escapes
    # ------------------------------------------------------------------
    @testset "BitArray mixed scope — inner escapes, outer safe" begin
        pool = _make_pool(true)

        checkpoint!(pool)
        bv_outer = acquire!(pool, Bit, 200)
        fill!(bv_outer, true)

        checkpoint!(pool)
        bv_inner = acquire!(pool, Bit, 100)

        # Inner BitArray should throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(bv_inner, pool)

        # Outer BitArray should pass
        _validate_pool_return(bv_outer, pool)

        # View of inner → throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(
            view(bv_inner, 1:50), pool
        )

        # View of outer → safe
        _validate_pool_return(view(bv_outer, 1:100), pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Scenario 15: lazy path + mixed types — Float64 outer, Int64 inner
    #   with partial escape through Dict
    # ------------------------------------------------------------------
    @testset "lazy path mixed types — Dict partial escape" begin
        pool = _make_pool(true)

        _lazy_checkpoint!(pool)
        v_f64 = acquire!(pool, Float64, 30)
        fill!(v_f64, 1.0)

        _lazy_checkpoint!(pool)
        v_i64 = acquire!(pool, Int64, 20)
        fill!(v_i64, 2)

        # Dict with inner array → throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(
            Dict(:data => v_i64), pool
        )

        # Dict with only outer array → safe
        _validate_pool_return(Dict(:data => v_f64), pool)

        # Mixed Dict — inner array present → throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(
            Dict(:outer => v_f64, :inner => v_i64), pool
        )

        _lazy_rewind!(pool)
        _lazy_rewind!(pool)
    end

end
