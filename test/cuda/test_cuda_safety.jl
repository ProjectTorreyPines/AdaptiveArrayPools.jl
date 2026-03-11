import AdaptiveArrayPools: PoolRuntimeEscapeError, PoolEscapeError, _safety_level, POOL_DEBUG

const _make_cuda_pool = ext._make_cuda_pool

# Opaque identity — defeats compile-time escape analysis
_cuda_test_leak(x) = x

@testset "CUDA Safety Dispatch (CuAdaptiveArrayPool{S})" begin

    # ==============================================================================
    # Type parameterization basics
    # ==============================================================================

    @testset "CuAdaptiveArrayPool{S} construction and _safety_level" begin
        p0 = _make_cuda_pool(0)
        p1 = _make_cuda_pool(1)
        p2 = _make_cuda_pool(2)
        p3 = _make_cuda_pool(3)

        @test p0 isa CuAdaptiveArrayPool{0}
        @test p1 isa CuAdaptiveArrayPool{1}
        @test p2 isa CuAdaptiveArrayPool{2}
        @test p3 isa CuAdaptiveArrayPool{3}

        @test _safety_level(p0) == 0
        @test _safety_level(p1) == 1
        @test _safety_level(p2) == 2
        @test _safety_level(p3) == 3

        # Borrow fields exist at all levels (required by macro-injected field access)
        @test hasfield(typeof(p0), :_pending_callsite)
        @test hasfield(typeof(p0), :_pending_return_site)
        @test hasfield(typeof(p0), :_borrow_log)
    end

    # ==============================================================================
    # Level 0: No poisoning, no validation
    # ==============================================================================

    @testset "Level 0: no poisoning on rewind" begin
        pool = _make_cuda_pool(0)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 42.0f0)
        rewind!(pool)

        # With safety off, backing vector still has valid data
        @test length(pool.float32.vectors[1]) >= 10
        # Data should still be there (no poisoning)
        checkpoint!(pool)
        v2 = acquire!(pool, Float32, 10)
        @test all(x -> x == 42.0f0, Array(v2))
        rewind!(pool)
    end

    # ==============================================================================
    # Level 1: Poisoning + structural invalidation (length → 0)
    # ==============================================================================
    # CUDA Level 1 now: poison fill → _resize_without_shrink!(vec, 0)
    # Backing vector length becomes 0 (GPU memory preserved via maxsize).
    # Poison data persists in GPU memory and is visible on re-acquire (grow-back).

    @testset "Level 1: released vectors have length 0 after rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        CUDA.fill!(v, 42.0f0)
        rewind!(pool)

        # Structural invalidation: length → 0 (matches CPU behavior)
        @test length(pool.float32.vectors[1]) == 0
    end

    @testset "Level 1: Float32 poisoned with NaN on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 42.0f0)
        rewind!(pool)

        # Backing vector length is 0 after invalidation
        @test length(pool.float32.vectors[1]) == 0

        # Re-acquire: grow-back reuses same GPU memory → poison data visible
        checkpoint!(pool)
        v2 = acquire!(pool, Float32, 10)
        @test all(isnan, Array(v2))
        rewind!(pool)
    end

    @testset "Level 1: Int32 poisoned with typemax on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Int32, 8)
        CUDA.fill!(v, Int32(42))
        rewind!(pool)

        # Verify via re-acquire (backing vector length is 0 after invalidation)
        checkpoint!(pool)
        v2 = acquire!(pool, Int32, 8)
        @test all(==(typemax(Int32)), Array(v2))
        rewind!(pool)
    end

    @testset "Level 1: ComplexF32 poisoned with NaN on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, ComplexF32, 8)
        CUDA.fill!(v, ComplexF32(1.0f0 + 2.0f0im))
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, ComplexF32, 8)
        @test all(z -> isnan(real(z)) && isnan(imag(z)), Array(v2))
        rewind!(pool)
    end

    @testset "Level 1: Bool poisoned with true on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Bool, 16)
        CUDA.fill!(v, false)
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, Bool, 16)
        @test all(==(true), Array(v2))
        rewind!(pool)
    end

    @testset "Level 1: Float16 poisoned with NaN on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float16, 10)
        CUDA.fill!(v, Float16(42.0))
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, Float16, 10)
        @test all(isnan, Array(v2))
        rewind!(pool)
    end

    @testset "Level 1: N-way cache invalidated on poisoned rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 1.0f0)
        rewind!(pool)

        # Cached views should be cleared (nothing) after poisoning
        base = 0 * ext.CACHE_WAYS
        for k in 1:ext.CACHE_WAYS
            @test pool.float32.views[base + k] === nothing
        end
    end

    @testset "Level 1: no escape detection" begin
        # Level 1 should NOT throw on escape (that's Level 2+)
        pool = _make_cuda_pool(1)
        result = begin
            checkpoint!(pool)
            v = acquire!(pool, Float32, 10)
            rewind!(pool)
            v  # "escaping" — should not throw at Level 1
        end
        @test result isa CuArray
    end

    # ==============================================================================
    # Level 0: Verify no poisoning
    # ==============================================================================

    @testset "Level 0: no poisoning (verify data survives rewind)" begin
        pool = _make_cuda_pool(0)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 42.0f0)
        rewind!(pool)

        # Data should NOT be poisoned at Level 0
        cpu_data = Array(pool.float32.vectors[1])
        @test all(x -> x == 42.0f0, cpu_data[1:10])
    end

    # ==============================================================================
    # Level 2: Escape detection
    # ==============================================================================

    @testset "Level 2: escape detection catches CuArray leak" begin
        pool = _make_cuda_pool(2)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                # Simulate what _validate_pool_return does
                AdaptiveArrayPools._validate_pool_return(_cuda_test_leak(v), pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "Level 2: safe scalar return does not throw" begin
        pool = _make_cuda_pool(2)
        checkpoint!(pool)
        try
            v = acquire!(pool, Float32, 10)
            CUDA.fill!(v, 3.0f0)
            result = sum(Array(v))  # scalar — safe
            AdaptiveArrayPools._validate_pool_return(result, pool)
            @test result == 30.0f0
        finally
            rewind!(pool)
        end
    end

    @testset "Level 2: escape detection with Tuple containing CuArray" begin
        pool = _make_cuda_pool(2)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                val = (42, _cuda_test_leak(v))
                AdaptiveArrayPools._validate_pool_return(val, pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "Level 2: escape detection with Dict containing CuArray" begin
        pool = _make_cuda_pool(2)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                val = Dict(:data => _cuda_test_leak(v))
                AdaptiveArrayPools._validate_pool_return(val, pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "Level 0 and 1: no escape detection" begin
        for lv in (0, 1)
            pool = _make_cuda_pool(lv)
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                # Should NOT throw — escape detection requires Level 2+
                AdaptiveArrayPools._validate_pool_return(_cuda_test_leak(v), pool)
            finally
                rewind!(pool)
            end
        end
    end

    # ==============================================================================
    # Level 3: Borrow tracking
    # ==============================================================================

    @testset "Level 3: borrow fields functional" begin
        pool = _make_cuda_pool(3)
        @test pool._pending_callsite == ""
        @test pool._pending_return_site == ""
        @test pool._borrow_log === nothing  # lazily created
    end

    @testset "Level 3: _set_pending_callsite! works" begin
        pool = _make_cuda_pool(3)
        AdaptiveArrayPools._set_pending_callsite!(pool, "test.jl:42\nacquire!(pool, Float32, 10)")
        @test pool._pending_callsite == "test.jl:42\nacquire!(pool, Float32, 10)"

        # At Level 0, should be no-op
        pool0 = _make_cuda_pool(0)
        AdaptiveArrayPools._set_pending_callsite!(pool0, "should not be set")
        @test pool0._pending_callsite == ""
    end

    @testset "Level 3: _maybe_record_borrow! records callsite" begin
        pool = _make_cuda_pool(3)
        checkpoint!(pool)
        tp = get_typed_pool!(pool, Float32)

        # Set pending callsite, then acquire to increment n_active
        AdaptiveArrayPools._set_pending_callsite!(pool, "test.jl:99\nacquire!(pool, Float32, 5)")
        acquire!(pool, Float32, 5)

        # The borrow log should now have an entry
        @test pool._borrow_log !== nothing
        @test length(pool._borrow_log) >= 1

        rewind!(pool)
    end

    # ==============================================================================
    # set_safety_level! — all-device replacement
    # ==============================================================================

    @testset "set_safety_level! replaces pool with state preservation" begin
        # Get current pool (creates one at default safety level)
        pool = get_task_local_cuda_pool()
        reset!(pool)

        # Populate with some data
        checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        CUDA.fill!(v, 1.0f0)
        rewind!(pool)

        # Change safety level
        set_safety_level!(2)
        new_pool = get_task_local_cuda_pool()

        @test new_pool isa CuAdaptiveArrayPool{2}
        @test _safety_level(new_pool) == 2
        # Cached vectors should be preserved (same object reference)
        @test new_pool.float32.vectors[1] === pool.float32.vectors[1]

        # Restore
        set_safety_level!(0)
        @test get_task_local_cuda_pool() isa CuAdaptiveArrayPool{0}
    end

    @testset "set_safety_level! rejects inside active scope" begin
        pool = get_task_local_cuda_pool()
        checkpoint!(pool)
        try
            @test_throws ArgumentError set_safety_level!(2)
        finally
            rewind!(pool)
        end
    end

    # ==============================================================================
    # Nested scopes: inner poisoned, outer valid
    # ==============================================================================

    @testset "Nested scopes: inner poisoned, outer still valid" begin
        pool = _make_cuda_pool(1)

        checkpoint!(pool)
        v_outer = acquire!(pool, Float32, 10)
        CUDA.fill!(v_outer, 1.0f0)

        # Inner scope
        checkpoint!(pool)
        v_inner = acquire!(pool, Float32, 20)
        CUDA.fill!(v_inner, 2.0f0)
        rewind!(pool)

        # Inner should be invalidated (slot 2: length → 0, poisoned)
        @test length(pool.float32.vectors[2]) == 0
        # Verify poison via re-acquire
        checkpoint!(pool)
        v_inner2 = acquire!(pool, Float32, 20)
        @test all(isnan, Array(v_inner2))
        rewind!(pool)

        # Outer should still be valid (slot 1 not released)
        cpu_outer = Array(v_outer)
        @test all(x -> x == 1.0f0, cpu_outer)

        rewind!(pool)
        # Now outer is also invalidated (length → 0, poisoned)
        @test length(pool.float32.vectors[1]) == 0
        # Verify poison via re-acquire
        checkpoint!(pool)
        v_outer2 = acquire!(pool, Float32, 10)
        @test all(isnan, Array(v_outer2))
        rewind!(pool)
    end

    # ==============================================================================
    # reset! with safety
    # ==============================================================================

    @testset "reset! clears borrow tracking state" begin
        pool = _make_cuda_pool(3)
        pool._pending_callsite = "test"
        pool._pending_return_site = "test"
        pool._borrow_log = IdDict{Any, String}()

        reset!(pool)

        @test pool._pending_callsite == ""
        @test pool._pending_return_site == ""
        @test pool._borrow_log === nothing
    end

    # ==============================================================================
    # Display includes {S} and safety label
    # ==============================================================================

    @testset "show includes {S} and safety label" begin
        pool = _make_cuda_pool(2)
        s = sprint(show, pool)
        @test occursin("{2}", s)
        @test occursin("safety=full", s)

        pool0 = _make_cuda_pool(0)
        s0 = sprint(show, pool0)
        @test occursin("{0}", s0)
        @test occursin("safety=off", s0)
    end

    # ==============================================================================
    # POOL_DEBUG backward compat with CUDA
    # ==============================================================================

    @testset "POOL_DEBUG backward compat triggers CUDA escape detection" begin
        old_debug = POOL_DEBUG[]

        POOL_DEBUG[] = true
        pool = _make_cuda_pool(0)  # Safety off, but POOL_DEBUG overrides
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                AdaptiveArrayPools._validate_pool_return(_cuda_test_leak(v), pool)
            finally
                rewind!(pool)
            end
        end

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # Fallback types (pool.others) poisoning
    # ==============================================================================

    @testset "Fallback type (UInt8) poisoned on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, UInt8, 16)
        CUDA.fill!(v, UInt8(42))
        rewind!(pool)

        # Backing vector length → 0 after invalidation
        tp = pool.others[UInt8]
        @test length(tp.vectors[1]) == 0

        # Verify poison via re-acquire
        checkpoint!(pool)
        v2 = acquire!(pool, UInt8, 16)
        @test all(==(typemax(UInt8)), Array(v2))
        rewind!(pool)
    end

    # ==============================================================================
    # @with_pool :cuda integration with safety
    # ==============================================================================

    @testset "@with_pool :cuda with escape detection" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true  # Use POOL_DEBUG to trigger on any safety level

        @test_throws PoolRuntimeEscapeError @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            _cuda_test_leak(v)
        end

        POOL_DEBUG[] = old_debug
    end

    @testset "@with_pool :cuda safe return" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        result = @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            CUDA.fill!(v, 3.0f0)
            sum(Array(v))  # scalar return — safe
        end
        @test result == 30.0f0

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # Compile-time escape detection (@with_pool :cuda)
    # ==============================================================================

    @testset "Compile-time: direct CuArray escape caught at macro expansion" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            v  # direct escape in tail position
        end
    end

    @testset "Compile-time: safe scalar return passes" begin
        # Should NOT throw at macro expansion time
        ex = @macroexpand @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            sum(Array(v))  # scalar — safe
        end
        @test ex isa Expr
    end

    @testset "Compile-time: zeros!/ones! escape caught" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :cuda pool begin
            v = zeros!(pool, Float32, 10)
            v
        end
    end

    # ==============================================================================
    # @with_pool :cuda at native Level 2 (no POOL_DEBUG hack)
    # ==============================================================================

    @testset "@with_pool :cuda Level 2 escape detection (native S=2)" begin
        set_safety_level!(2)

        @test_throws PoolRuntimeEscapeError @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            _cuda_test_leak(v)
        end

        set_safety_level!(0)
    end

    @testset "@with_pool :cuda Level 2 safe return (native S=2)" begin
        set_safety_level!(2)

        result = @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            CUDA.fill!(v, 5.0f0)
            sum(Array(v))
        end
        @test result == 50.0f0

        set_safety_level!(0)
    end

    @testset "@with_pool :cuda Level 1 no escape detection (native S=1)" begin
        set_safety_level!(1)

        # Level 1 should NOT trigger escape detection
        result = @with_pool :cuda pool begin
            v = acquire!(pool, Float32, 10)
            _cuda_test_leak(v)
        end
        @test result isa CuArray

        set_safety_level!(0)
    end

    # ==============================================================================
    # Level 3 borrow tracking via macro path
    # ==============================================================================

    @testset "@with_pool :cuda Level 3 escape error includes callsite" begin
        set_safety_level!(3)

        err = try
            @with_pool :cuda pool begin
                v = acquire!(pool, Float32, 10)
                _cuda_test_leak(v)
            end
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")  # "file:line" format

        set_safety_level!(0)
    end

    @testset "@with_pool :cuda Level 3 callsite includes expression text" begin
        set_safety_level!(3)

        err = try
            @with_pool :cuda pool begin
                v = zeros!(pool, Float32, 10)
                _cuda_test_leak(v)
            end
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, "\n")
        @test contains(err.callsite, "zeros!(pool, Float32, 10)")

        set_safety_level!(0)
    end

    @testset "LV<3 does not create borrow log on CUDA" begin
        for lv in (0, 1, 2)
            pool = _make_cuda_pool(lv)
            checkpoint!(pool)
            _ = acquire!(pool, Float32, 10)
            @test pool._borrow_log === nothing
            rewind!(pool)
        end
    end

    @testset "LV=3 creates borrow log on CUDA acquire" begin
        pool = _make_cuda_pool(3)
        checkpoint!(pool)
        _ = acquire!(pool, Float32, 10)
        @test pool._borrow_log !== nothing
        @test pool._borrow_log isa IdDict
        rewind!(pool)
    end

    # ==============================================================================
    # Error message content (showerror)
    # ==============================================================================

    @testset "showerror: CuArray escape error message format" begin
        # LV≥2 without callsite → "Tip: set LV=3"
        err = PoolRuntimeEscapeError("CuArray{Float32, 1}", "Float32", nothing, nothing)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "PoolEscapeError")
        @test contains(msg, "CuArray{Float32, 1}")
        @test contains(msg, "Float32")
        @test contains(msg, "POOL_SAFETY_LV ≥ 2")
        @test contains(msg, "Tip:")
        @test contains(msg, "POOL_SAFETY_LV[] = 3")
    end

    @testset "showerror: CuArray with callsite (LV≥3)" begin
        err = PoolRuntimeEscapeError(
            "CuArray{Float32, 1}", "Float32",
            "test_cuda.jl:42\nacquire!(pool, Float32, 10)", nothing
        )
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "acquired at")
        @test contains(msg, "test_cuda.jl:42")
        @test contains(msg, "acquire!(pool, Float32, 10)")
        @test contains(msg, "POOL_SAFETY_LV ≥ 3")
        @test !contains(msg, "Tip:")  # No tip when callsite is present
    end

    # ==============================================================================
    # Function form: @with_pool :cuda pool function ...
    # ==============================================================================

    @testset "Function form: escape detection with explicit return" begin
        set_safety_level!(2)

        @with_pool :cuda pool function _cuda_test_return_escape()
            v = acquire!(pool, Float32, 10)
            return _cuda_test_leak(v)
        end

        @test_throws PoolRuntimeEscapeError _cuda_test_return_escape()

        set_safety_level!(0)
    end

    @testset "Function form: safe scalar return passes" begin
        set_safety_level!(2)

        @with_pool :cuda pool function _cuda_test_safe_return()
            v = acquire!(pool, Float32, 5)
            CUDA.fill!(v, 4.0f0)
            return sum(Array(v))
        end

        @test _cuda_test_safe_return() == 20.0f0

        set_safety_level!(0)
    end

    @testset "Function form: bare return (nothing) passes" begin
        set_safety_level!(2)

        @with_pool :cuda pool function _cuda_test_bare_return()
            _ = acquire!(pool, Float32, 10)
            return
        end

        @test _cuda_test_bare_return() === nothing

        set_safety_level!(0)
    end

end  # CUDA Safety Dispatch
