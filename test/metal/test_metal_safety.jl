import AdaptiveArrayPools: PoolRuntimeEscapeError, PoolEscapeError, _runtime_check,
    _validate_pool_return, _lazy_checkpoint!, _lazy_rewind!

const _make_metal_pool = ext._make_metal_pool

# Opaque identity — defeats compile-time escape analysis
_metal_test_leak(x) = x

@testset "Metal Safety (MetalAdaptiveArrayPool{R}, Binary R=0/1)" begin

    # ==============================================================================
    # Type parameterization basics
    # ==============================================================================

    @testset "MetalAdaptiveArrayPool{R} construction and _runtime_check" begin
        p0 = _make_metal_pool(0)
        p1 = _make_metal_pool(1)

        @test p0 isa MetalAdaptiveArrayPool{0}
        @test p1 isa MetalAdaptiveArrayPool{1}

        @test _runtime_check(p0) == false
        @test _runtime_check(p1) == true

        # Borrow fields exist at all levels
        @test hasfield(typeof(p0), :_pending_callsite)
        @test hasfield(typeof(p0), :_pending_return_site)
        @test hasfield(typeof(p0), :_borrow_log)
    end

    # ==============================================================================
    # R=0: No poisoning, no validation
    # ==============================================================================

    @testset "R=0: no poisoning on rewind" begin
        pool = _make_metal_pool(0)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        Metal.fill!(v, 42.0f0)
        rewind!(pool)

        @test length(pool.float32.vectors[1]) >= 10
        checkpoint!(pool)
        v2 = acquire!(pool, Float32, 10)
        @test all(x -> x == 42.0f0, Array(v2))
        rewind!(pool)
    end

    @testset "R=0: no poisoning (verify data survives rewind)" begin
        pool = _make_metal_pool(0)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        Metal.fill!(v, 42.0f0)
        rewind!(pool)

        cpu_data = Array(pool.float32.vectors[1])
        @test all(x -> x == 42.0f0, cpu_data[1:10])
    end

    @testset "R=0: no escape detection" begin
        pool = _make_metal_pool(0)
        checkpoint!(pool)
        try
            v = acquire!(pool, Float32, 10)
            _validate_pool_return(_metal_test_leak(v), pool)
        finally
            rewind!(pool)
        end
    end

    # ==============================================================================
    # R=1: Poisoning + structural invalidation + escape detection + borrow tracking
    # ==============================================================================

    @testset "R=1: released vectors have length 0 after rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        Metal.fill!(v, 42.0f0)
        rewind!(pool)

        @test length(pool.float32.vectors[1]) == 0
    end

    @testset "R=1: Float32 poisoned with NaN on rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        Metal.fill!(v, 42.0f0)
        rewind!(pool)

        @test length(pool.float32.vectors[1]) == 0

        checkpoint!(pool)
        v2 = acquire!(pool, Float32, 10)
        @test all(isnan, Array(v2))
        rewind!(pool)
    end

    @testset "R=1: Int32 poisoned with typemax on rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Int32, 8)
        Metal.fill!(v, Int32(42))
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, Int32, 8)
        @test all(==(typemax(Int32)), Array(v2))
        rewind!(pool)
    end

    @testset "R=1: ComplexF32 poisoned with NaN on rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, ComplexF32, 8)
        Metal.fill!(v, ComplexF32(1.0f0 + 2.0f0im))
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, ComplexF32, 8)
        @test all(z -> isnan(real(z)) && isnan(imag(z)), Array(v2))
        rewind!(pool)
    end

    @testset "R=1: Bool poisoned with true on rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Bool, 16)
        Metal.fill!(v, false)
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, Bool, 16)
        @test all(==(true), Array(v2))
        rewind!(pool)
    end

    @testset "R=1: Float16 poisoned with NaN on rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float16, 10)
        Metal.fill!(v, Float16(42.0))
        rewind!(pool)

        checkpoint!(pool)
        v2 = acquire!(pool, Float16, 10)
        @test all(isnan, Array(v2))
        rewind!(pool)
    end

    @testset "R=1: arr_wrappers invalidated on rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        Metal.fill!(v, 1.0f0)
        rewind!(pool)

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
    # R=1: Escape detection
    # ==============================================================================

    @testset "R=1: escape detection catches MtlArray leak" begin
        pool = _make_metal_pool(1)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                _validate_pool_return(_metal_test_leak(v), pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "R=1: safe scalar return does not throw" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        try
            v = acquire!(pool, Float32, 10)
            Metal.fill!(v, 3.0f0)
            result = sum(Array(v))
            _validate_pool_return(result, pool)
            @test result == 30.0f0
        finally
            rewind!(pool)
        end
    end

    @testset "R=1: escape detection with Tuple containing MtlArray" begin
        pool = _make_metal_pool(1)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                val = (42, _metal_test_leak(v))
                _validate_pool_return(val, pool)
            finally
                rewind!(pool)
            end
        end
    end

    @testset "R=1: escape detection with Dict containing MtlArray" begin
        pool = _make_metal_pool(1)
        @test_throws PoolRuntimeEscapeError begin
            checkpoint!(pool)
            try
                v = acquire!(pool, Float32, 10)
                val = Dict(:data => _metal_test_leak(v))
                _validate_pool_return(val, pool)
            finally
                rewind!(pool)
            end
        end
    end

    # ==============================================================================
    # R=1: Borrow tracking
    # ==============================================================================

    @testset "R=1: borrow fields functional" begin
        pool = _make_metal_pool(1)
        @test pool._pending_callsite == ""
        @test pool._pending_return_site == ""
        @test pool._borrow_log === nothing
    end

    @testset "R=1: _set_pending_callsite! works" begin
        pool = _make_metal_pool(1)
        AdaptiveArrayPools._set_pending_callsite!(pool, "test.jl:42\nacquire!(pool, Float32, 10)")
        @test pool._pending_callsite == "test.jl:42\nacquire!(pool, Float32, 10)"

        pool0 = _make_metal_pool(0)
        AdaptiveArrayPools._set_pending_callsite!(pool0, "should not be set")
        @test pool0._pending_callsite == ""
    end

    @testset "R=1: _maybe_record_borrow! records callsite" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        tp = get_typed_pool!(pool, Float32)

        AdaptiveArrayPools._set_pending_callsite!(pool, "test.jl:99\nacquire!(pool, Float32, 5)")
        acquire!(pool, Float32, 5)

        @test pool._borrow_log !== nothing
        @test length(pool._borrow_log) >= 1

        rewind!(pool)
    end

    @testset "R=0: does not create borrow log on Metal" begin
        pool = _make_metal_pool(0)
        checkpoint!(pool)
        _ = acquire!(pool, Float32, 10)
        @test pool._borrow_log === nothing
        rewind!(pool)
    end

    @testset "R=1: creates borrow log on Metal acquire" begin
        pool = _make_metal_pool(1)
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
        pool = _make_metal_pool(1)

        checkpoint!(pool)
        v_outer = acquire!(pool, Float32, 10)
        Metal.fill!(v_outer, 1.0f0)

        # Inner scope
        checkpoint!(pool)
        v_inner = acquire!(pool, Float32, 20)
        Metal.fill!(v_inner, 2.0f0)
        rewind!(pool)

        # Inner should be invalidated
        @test length(pool.float32.vectors[2]) == 0
        checkpoint!(pool)
        v_inner2 = acquire!(pool, Float32, 20)
        @test all(isnan, Array(v_inner2))
        rewind!(pool)

        # Outer should still be valid
        cpu_outer = Array(v_outer)
        @test all(x -> x == 1.0f0, cpu_outer)

        rewind!(pool)
        @test length(pool.float32.vectors[1]) == 0
        checkpoint!(pool)
        v_outer2 = acquire!(pool, Float32, 10)
        @test all(isnan, Array(v_outer2))
        rewind!(pool)
    end

    # ==============================================================================
    # reset! with safety
    # ==============================================================================

    @testset "reset! clears borrow tracking state" begin
        pool = _make_metal_pool(1)
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
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, UInt8, 16)
        Metal.fill!(v, UInt8(42))
        rewind!(pool)

        tp = pool.others[UInt8]
        @test length(tp.vectors[1]) == 0

        checkpoint!(pool)
        v2 = acquire!(pool, UInt8, 16)
        @test all(==(typemax(UInt8)), Array(v2))
        rewind!(pool)
    end

    # ==============================================================================
    # Display includes {R} and check label
    # ==============================================================================

    @testset "show includes {R} and check label" begin
        pool1 = _make_metal_pool(1)
        s1 = sprint(show, pool1)
        @test occursin("{1", s1)
        @test occursin("check=on", s1)

        pool0 = _make_metal_pool(0)
        s0 = sprint(show, pool0)
        @test occursin("{0", s0)
        @test occursin("check=off", s0)
    end

    # ==============================================================================
    # Compile-time escape detection (@with_pool :metal)
    # ==============================================================================

    @testset "Compile-time: direct MtlArray escape caught at macro expansion" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :metal pool begin
            v = acquire!(pool, Float32, 10)
            v
        end
    end

    @testset "Compile-time: safe scalar return passes" begin
        ex = @macroexpand @with_pool :metal pool begin
            v = acquire!(pool, Float32, 10)
            sum(Array(v))
        end
        @test ex isa Expr
    end

    @testset "Compile-time: zeros!/ones! escape caught" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :metal pool begin
            v = zeros!(pool, Float32, 10)
            v
        end
    end

    # ==============================================================================
    # R=1 escape detection via direct checkpoint/validate/rewind
    # ==============================================================================

    @testset "Pool{1} escape detection via direct validate" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        err = try
            v = acquire!(pool, Float32, 10)
            _validate_pool_return(_metal_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
    end

    @testset "Pool{1} safe scalar via direct validate" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        Metal.fill!(v, 5.0f0)
        result = sum(Array(v))
        _validate_pool_return(result, pool)
        rewind!(pool)
        @test result == 50.0f0
    end

    # ==============================================================================
    # R=1 borrow tracking: callsite in escape error
    # ==============================================================================

    @testset "Pool{1} escape error includes callsite when set" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)

        pool._pending_callsite = "test_metal.jl:42\nacquire!(pool, Float32, 10)"
        v = acquire!(pool, Float32, 10)

        err = try
            _validate_pool_return(_metal_test_leak(v), pool)
            nothing
        catch e
            e
        end
        rewind!(pool)

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, "test_metal.jl:42")
        @test contains(err.callsite, "acquire!(pool, Float32, 10)")
    end

    # ==============================================================================
    # Error message content (showerror)
    # ==============================================================================

    @testset "showerror: MtlArray escape error message format" begin
        err = PoolRuntimeEscapeError("MtlArray{Float32, 1}", "Float32", nothing, nothing)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "PoolEscapeError")
        @test contains(msg, "MtlArray{Float32, 1}")
        @test contains(msg, "Float32")
        @test contains(msg, "RUNTIME_CHECK")
    end

    @testset "showerror: MtlArray with callsite" begin
        err = PoolRuntimeEscapeError(
            "MtlArray{Float32, 1}", "Float32",
            "test_metal.jl:42\nacquire!(pool, Float32, 10)", nothing
        )
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "acquired at")
        @test contains(msg, "test_metal.jl:42")
        @test contains(msg, "acquire!(pool, Float32, 10)")
    end

    # ==============================================================================
    # Function form: @with_pool :metal pool function ...
    # ==============================================================================

    @testset "Function form: compile-time escape detection" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :metal pool function _metal_test_escape_fn()
            v = acquire!(pool, Float32, 10)
            return v
        end
    end

    @testset "Function form: safe scalar return compiles" begin
        ex = @macroexpand @with_pool :metal pool function _metal_test_safe_fn()
            v = acquire!(pool, Float32, 5)
            return sum(Array(v))
        end
        @test ex isa Expr
    end

    @testset "Function form: bare return compiles" begin
        ex = @macroexpand @with_pool :metal pool function _metal_test_bare_fn()
            _ = acquire!(pool, Float32, 10)
            return
        end
        @test ex isa Expr
    end

end  # Metal Safety
