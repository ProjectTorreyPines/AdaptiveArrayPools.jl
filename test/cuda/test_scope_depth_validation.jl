import AdaptiveArrayPools: PoolRuntimeEscapeError, _validate_pool_return,
    _lazy_checkpoint!, _lazy_rewind!

const _make_cuda_pool_scope = ext._make_cuda_pool

# ==============================================================================
# CUDA Scope-Aware Validation: mirrors CPU test_scope_depth_validation.jl
# for CuAdaptiveArrayPool overlap checks (_check_tp_cuda_overlap).
# ==============================================================================

@testset "CUDA Scope-depth-aware validation" begin

    # ------------------------------------------------------------------
    # Outer acquire, inner validate — should NOT throw
    # ------------------------------------------------------------------
    @testset "outer acquire, inner validate — should NOT throw (CuArray)" begin
        pool = _make_cuda_pool_scope(true)

        checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        CUDA.fill!(v, 1.0f0)

        checkpoint!(pool)
        _ = acquire!(pool, Float32, 50)   # inner-scope acquire

        # v was acquired in depth 1 — returning it from depth 2 is safe
        _validate_pool_return(v, pool)   # ← Should NOT throw

        rewind!(pool)
        @test CUDA.sum(v) ≈ 100.0f0
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Inner-scope acquire SHOULD still throw
    # ------------------------------------------------------------------
    @testset "inner-scope acquire — SHOULD still throw" begin
        pool = _make_cuda_pool_scope(true)

        checkpoint!(pool)
        _ = acquire!(pool, Float32, 100)   # outer acquire (safe)

        checkpoint!(pool)
        w = acquire!(pool, Float32, 50)    # inner acquire (should be caught)

        @test_throws PoolRuntimeEscapeError _validate_pool_return(w, pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # 3-level nesting: only deepest acquire should throw
    # ------------------------------------------------------------------
    @testset "3-level nesting: only deepest acquire should throw" begin
        pool = _make_cuda_pool_scope(true)

        # Depth 1
        checkpoint!(pool)
        v1 = acquire!(pool, Float32, 100)

        # Depth 2
        checkpoint!(pool)
        v2 = acquire!(pool, Float32, 50)

        # Depth 3
        checkpoint!(pool)
        v3 = acquire!(pool, Float32, 25)

        # At depth 3: v1 and v2 are from outer scopes → safe
        _validate_pool_return(v1, pool)
        _validate_pool_return(v2, pool)

        # v3 is from current (depth 3) scope → escape!
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v3, pool)

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
    # Mixed types: outer Float32, inner Int32
    # ------------------------------------------------------------------
    @testset "outer Float32, inner Int32 — outer should NOT throw" begin
        pool = _make_cuda_pool_scope(true)

        checkpoint!(pool)
        v_f32 = acquire!(pool, Float32, 20)

        checkpoint!(pool)
        _ = acquire!(pool, Int32, 10)   # different type in inner scope

        _validate_pool_return(v_f32, pool)   # ← Should NOT throw

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Lazy path — outer acquire, inner validate
    # ------------------------------------------------------------------
    @testset "lazy path — outer acquire, inner validate" begin
        pool = _make_cuda_pool_scope(true)

        _lazy_checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        CUDA.fill!(v, 5.0f0)

        _lazy_checkpoint!(pool)
        _ = acquire!(pool, Float32, 30)

        # v acquired at outer depth → safe
        _validate_pool_return(v, pool)   # ← Should NOT throw

        _lazy_rewind!(pool)
        @test CUDA.sum(v) ≈ 500.0f0
        _lazy_rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Container wrapping outer-scope array — should NOT throw
    # ------------------------------------------------------------------
    @testset "outer array inside tuple — should NOT throw" begin
        pool = _make_cuda_pool_scope(true)

        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 1.0f0)

        checkpoint!(pool)
        _ = acquire!(pool, Float32, 5)

        # v inside tuple — v belongs to outer scope → safe
        _validate_pool_return((CUDA.sum(v), v), pool)

        rewind!(pool)
        rewind!(pool)
    end

    # ------------------------------------------------------------------
    # Leaked inner scope — outer array must still be caught
    # ------------------------------------------------------------------
    @testset "leaked inner scope — outer array must still be caught" begin
        pool = _make_cuda_pool_scope(true)

        entry_depth = pool._current_depth
        checkpoint!(pool)

        v = acquire!(pool, Float32, 100)

        # Inner scope leaks (no rewind)
        checkpoint!(pool)
        _ = acquire!(pool, Float32, 50)

        # Leaked scope cleanup (simulates macro reorder fix)
        while pool._current_depth > entry_depth + 1
            rewind!(pool)
        end

        # v acquired at this scope → escape → must throw
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v, pool)

        rewind!(pool)
    end

end
