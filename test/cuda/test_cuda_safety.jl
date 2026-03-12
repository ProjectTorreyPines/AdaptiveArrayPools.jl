import AdaptiveArrayPools: PoolRuntimeEscapeError, PoolEscapeError, _runtime_check,
    _validate_pool_return, _lazy_checkpoint!, _lazy_rewind!

const _make_cuda_pool = ext._make_cuda_pool

# Opaque identity — defeats compile-time escape analysis
_cuda_test_leak(x) = x

@testset "CUDA Safety (CuAdaptiveArrayPool{S}, Binary S=0/1)" begin

    # ==============================================================================
    # Type parameterization basics
    # ==============================================================================

    @testset "CuAdaptiveArrayPool{S} construction and _runtime_check" begin
        p0 = _make_cuda_pool(0)
        p1 = _make_cuda_pool(1)

        @test p0 isa CuAdaptiveArrayPool{0}
        @test p1 isa CuAdaptiveArrayPool{1}

        @test _runtime_check(p0) == false
        @test _runtime_check(p1) == true

        # Borrow fields exist at all levels (required by macro-injected field access)
        @test hasfield(typeof(p0), :_pending_callsite)
        @test hasfield(typeof(p0), :_pending_return_site)
        @test hasfield(typeof(p0), :_borrow_log)
    end

    # ==============================================================================
    # S=0: No poisoning, no validation
    # ==============================================================================

    @testset "S=0: no poisoning on rewind" begin
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

    @testset "S=0: no poisoning (verify data survives rewind)" begin
        pool = _make_cuda_pool(0)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 42.0f0)
        rewind!(pool)

        # Data should NOT be poisoned at S=0
        cpu_data = Array(pool.float32.vectors[1])
        @test all(x -> x == 42.0f0, cpu_data[1:10])
    end

    @testset "S=0: no escape detection" begin
        pool = _make_cuda_pool(0)
        checkpoint!(pool)
        try
            v = acquire!(pool, Float32, 10)
            # Should NOT throw — escape detection requires S=1
            _validate_pool_return(_cuda_test_leak(v), pool)
        finally
            rewind!(pool)
        end
    end

    # ==============================================================================
    # S=1: Poisoning + structural invalidation + escape detection + borrow tracking
    # ==============================================================================
    # CUDA S=1: poison fill → _resize_to_fit!(vec, 0) + arr_wrappers invalidation
    # Backing vector length becomes 0 (GPU memory preserved via maxsize).
    # Poison data persists in GPU memory and is visible on re-acquire (grow-back).

    @testset "S=1: released vectors have length 0 after rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        CUDA.fill!(v, 42.0f0)
        rewind!(pool)

        # Structural invalidation: length → 0 (matches CPU behavior)
        @test length(pool.float32.vectors[1]) == 0
    end

    @testset "S=1: Float32 poisoned with NaN on rewind" begin
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

    @testset "S=1: Int32 poisoned with typemax on rewind" begin
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

    @testset "S=1: ComplexF32 poisoned with NaN on rewind" begin
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

    @testset "S=1: Bool poisoned with true on rewind" begin
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

    @testset "S=1: Float16 poisoned with NaN on rewind" begin
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

    @testset "S=1: arr_wrappers invalidated on rewind" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 1.0f0)
        rewind!(pool)

        # arr_wrappers for released slots should have zero-dims after invalidation
        tp = pool.float32
        for N_idx in 1:length(tp.arr_wrappers)
            wrappers_for_N = tp.arr_wrappers[N_idx]
            wrappers_for_N === nothing && continue
            for wrapper in wrappers_for_N
                wrapper === nothing && continue
                @test all(==(0), size(wrapper))
            end
        end
    end

    # ==============================================================================
    # S=1: Escape detection
    # ==============================================================================

    @testset "S=1: escape detection catches CuArray leak" begin
        pool = _make_cuda_pool(1)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                _validate_pool_return(_cuda_test_leak(v), pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "S=1: safe scalar return does not throw" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        try
            v = acquire!(pool, Float32, 10)
            CUDA.fill!(v, 3.0f0)
            result = sum(Array(v))  # scalar — safe
            _validate_pool_return(result, pool)
            @test result == 30.0f0
        finally
            rewind!(pool)
        end
    end

    @testset "S=1: escape detection with Tuple containing CuArray" begin
        pool = _make_cuda_pool(1)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                val = (42, _cuda_test_leak(v))
                _validate_pool_return(val, pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "S=1: escape detection with Dict containing CuArray" begin
        pool = _make_cuda_pool(1)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                val = Dict(:data => _cuda_test_leak(v))
                _validate_pool_return(val, pool)
            finally
                rewind!(pool)
            end
        end
    end

    # ==============================================================================
    # S=1: Borrow tracking
    # ==============================================================================

    @testset "S=1: borrow fields functional" begin
        pool = _make_cuda_pool(1)
        @test pool._pending_callsite == ""
        @test pool._pending_return_site == ""
        @test pool._borrow_log === nothing  # lazily created
    end

    @testset "S=1: _set_pending_callsite! works" begin
        pool = _make_cuda_pool(1)
        AdaptiveArrayPools._set_pending_callsite!(pool, "test.jl:42\nacquire!(pool, Float32, 10)")
        @test pool._pending_callsite == "test.jl:42\nacquire!(pool, Float32, 10)"

        # At S=0, should be no-op
        pool0 = _make_cuda_pool(0)
        AdaptiveArrayPools._set_pending_callsite!(pool0, "should not be set")
        @test pool0._pending_callsite == ""
    end

    @testset "S=1: _maybe_record_borrow! records callsite" begin
        pool = _make_cuda_pool(1)
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

    @testset "S=0: does not create borrow log on CUDA" begin
        pool = _make_cuda_pool(0)
        checkpoint!(pool)
        _ = acquire!(pool, Float32, 10)
        @test pool._borrow_log === nothing
        rewind!(pool)
    end

    @testset "S=1: creates borrow log on CUDA acquire" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        _ = acquire!(pool, Float32, 10)
        @test pool._borrow_log !== nothing
        @test pool._borrow_log isa IdDict
        rewind!(pool)
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
        pool = _make_cuda_pool(1)
        pool._pending_callsite = "test"
        pool._pending_return_site = "test"
        pool._borrow_log = IdDict{Any, String}()

        reset!(pool)

        @test pool._pending_callsite == ""
        @test pool._pending_return_site == ""
        @test pool._borrow_log === nothing
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
    # Display includes {S} and check label
    # ==============================================================================

    @testset "show includes {S} and check label" begin
        pool1 = _make_cuda_pool(1)
        s1 = sprint(show, pool1)
        @test occursin("{1}", s1)
        @test occursin("check=on", s1)

        pool0 = _make_cuda_pool(0)
        s0 = sprint(show, pool0)
        @test occursin("{0}", s0)
        @test occursin("check=off", s0)
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
    # S=1 escape detection via direct checkpoint/validate/rewind
    # (replaces old set_safety_level! + @with_pool tests)
    # ==============================================================================

    @testset "Pool{1} escape detection via direct validate" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        err = try
            v = acquire!(pool, Float32, 10)
            _validate_pool_return(_cuda_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
    end

    @testset "Pool{1} safe scalar via direct validate" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        CUDA.fill!(v, 5.0f0)
        result = sum(Array(v))
        _validate_pool_return(result, pool)
        rewind!(pool)
        @test result == 50.0f0
    end

    # ==============================================================================
    # S=1 borrow tracking: callsite in escape error
    # ==============================================================================

    @testset "Pool{1} escape error includes callsite when set" begin
        pool = _make_cuda_pool(1)
        checkpoint!(pool)

        # Manually set callsite (normally macro-injected)
        pool._pending_callsite = "test_cuda.jl:42\nacquire!(pool, Float32, 10)"
        v = acquire!(pool, Float32, 10)

        err = try
            _validate_pool_return(_cuda_test_leak(v), pool)
            nothing
        catch e
            e
        end
        rewind!(pool)

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, "test_cuda.jl:42")
        @test contains(err.callsite, "acquire!(pool, Float32, 10)")
    end

    # ==============================================================================
    # Error message content (showerror)
    # ==============================================================================

    @testset "showerror: CuArray escape error message format" begin
        err = PoolRuntimeEscapeError("CuArray{Float32, 1}", "Float32", nothing, nothing)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "PoolEscapeError")
        @test contains(msg, "CuArray{Float32, 1}")
        @test contains(msg, "Float32")
        @test contains(msg, "RUNTIME_CHECK")
    end

    @testset "showerror: CuArray with callsite" begin
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
    end

    # ==============================================================================
    # Function form: @with_pool :cuda pool function ...
    # (Compile-time only — no runtime escape detection test via @with_pool,
    #  because RUNTIME_CHECK is a compile-time const and @with_pool type-asserts
    #  to Pool{RUNTIME_CHECK}. Use direct _make_pool(1) + validate for runtime tests.)
    # ==============================================================================

    @testset "Function form: compile-time escape detection" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :cuda pool function _cuda_test_escape_fn()
            v = acquire!(pool, Float32, 10)
            return v  # direct escape
        end
    end

    @testset "Function form: safe scalar return compiles" begin
        ex = @macroexpand @with_pool :cuda pool function _cuda_test_safe_fn()
            v = acquire!(pool, Float32, 5)
            return sum(Array(v))
        end
        @test ex isa Expr
    end

    @testset "Function form: bare return compiles" begin
        ex = @macroexpand @with_pool :cuda pool function _cuda_test_bare_fn()
            _ = acquire!(pool, Float32, 10)
            return
        end
        @test ex isa Expr
    end

end  # CUDA Safety
