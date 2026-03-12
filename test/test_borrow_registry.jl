import AdaptiveArrayPools: _validate_pool_return, _lookup_borrow_callsite,
    PoolRuntimeEscapeError, Bit, _make_pool, _lazy_checkpoint!, _lazy_rewind!

_test_leak(x) = x

@testset "Borrow Registry (RUNTIME_CHECK)" begin

    # ==============================================================================
    # Basic recording: S=1 direct path → callsite in escape error
    # ==============================================================================

    @testset "Direct path: escape error includes callsite" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            pool._pending_callsite = "test_borrow:1\nv = acquire!(pool, Float64, 10)"
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")  # "file:line" format
    end

    @testset "Direct path: unsafe_acquire! escape includes callsite" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            pool._pending_callsite = "test_borrow:2\nv = unsafe_acquire!(pool, Float64, 10)"
            v = unsafe_acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")
    end

    # ==============================================================================
    # Non-macro path: direct acquire! → generic label
    # ==============================================================================

    @testset "Direct acquire! shows generic callsite label" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        v = acquire!(pool, Float64, 10)
        err = try
            _validate_pool_return(v, pool)
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite == "<direct acquire! call>"

        _lazy_rewind!(pool)
    end

    @testset "Direct unsafe_acquire! shows generic callsite label" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        v = unsafe_acquire!(pool, Float64, 10)
        err = try
            _validate_pool_return(v, pool)
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite == "<direct unsafe_acquire! call>"

        _lazy_rewind!(pool)
    end

    # ==============================================================================
    # Convenience functions via direct path → callsite
    # ==============================================================================

    @testset "Direct path: zeros! escape includes callsite" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            pool._pending_callsite = "test_borrow:3\nv = zeros!(pool, Float64, 10)"
            v = zeros!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")
    end

    @testset "Direct path: callsite includes expression text" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            pool._pending_callsite = "test_borrow:4\nzeros!(pool, Float64, 10)"
            v = zeros!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        # Callsite should contain expression text after \n
        @test contains(err.callsite, "\n")
        @test contains(err.callsite, "zeros!(pool, Float64, 10)")
    end

    @testset "Direct zeros! shows generic callsite label" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        v = zeros!(pool, Float64, 10)
        err = try
            _validate_pool_return(v, pool)
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite == "<direct zeros! call>"

        _lazy_rewind!(pool)
    end

    # ==============================================================================
    # BitArray path → callsite
    # ==============================================================================

    @testset "Direct path: BitArray acquire escape includes callsite" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            pool._pending_callsite = "test_borrow:5\nv = acquire!(pool, Bit, 100)"
            v = acquire!(pool, Bit, 100)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")
    end

    # ==============================================================================
    # S=0: no borrow log overhead
    # ==============================================================================

    @testset "S=0 does not create borrow log" begin
        pool = _make_pool(false)
        _lazy_checkpoint!(pool)
        _ = acquire!(pool, Float64, 10)
        @test pool._borrow_log === nothing
        _lazy_rewind!(pool)
    end

    # ==============================================================================
    # S=1: borrow log IS created
    # ==============================================================================

    @testset "S=1 creates borrow log on acquire" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        _ = acquire!(pool, Float64, 10)
        @test pool._borrow_log !== nothing
        @test pool._borrow_log isa IdDict
        _lazy_rewind!(pool)
    end

    # ==============================================================================
    # reset! clears borrow log
    # ==============================================================================

    @testset "reset! clears borrow log and pending callsite" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        _ = acquire!(pool, Float64, 10)
        @test pool._borrow_log !== nothing

        reset!(pool)
        @test pool._borrow_log === nothing
        @test pool._pending_callsite == ""
    end

    # ==============================================================================
    # Error message format: showerror output
    # ==============================================================================

    @testset "showerror: 'acquired at' shown when callsite present (S=1)" begin
        err = PoolRuntimeEscapeError("SubArray{Float64, 1}", "Float64", "test.jl:42", nothing)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "acquired at")
        @test contains(msg, "test.jl:42")
        @test contains(msg, "RUNTIME_CHECK >= 1")
        @test !contains(msg, "Tip:")
    end

    @testset "showerror: expression text shown when present in callsite" begin
        err = PoolRuntimeEscapeError(
            "SubArray{Float64, 1}", "Float64",
            "test.jl:42\nzeros!(pool, Float64, 10)", nothing
        )
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "acquired at")
        @test contains(msg, "test.jl:42")
        @test contains(msg, "zeros!(pool, Float64, 10)")
    end

    @testset "showerror: short path used for absolute paths" begin
        err = PoolRuntimeEscapeError(
            "SubArray{Float64, 1}", "Float64",
            "$(homedir())/.julia/dev/Foo/src/bar.jl:99\nacquire!(pool, Float64, 5)", nothing
        )
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "acquired at")
        # Should NOT contain the full absolute homedir path
        @test !contains(msg, homedir())
        @test contains(msg, "bar.jl:99")
        @test contains(msg, "acquire!(pool, Float64, 5)")
    end

    @testset "showerror: no callsite still works" begin
        err = PoolRuntimeEscapeError("SubArray{Float64, 1}", "Float64", nothing, nothing)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test !contains(msg, "acquired at")
        @test contains(msg, "RUNTIME_CHECK >= 1")
    end

    # ==============================================================================
    # Multiple types: each gets correct callsite
    # ==============================================================================

    @testset "Multiple types record independent callsites" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        v_f64 = acquire!(pool, Float64, 10)
        v_i32 = acquire!(pool, Int32, 5)

        # Both should have callsite recorded
        tp_f64 = get_typed_pool!(pool, Float64)
        tp_i32 = get_typed_pool!(pool, Int32)

        cs_f64 = _lookup_borrow_callsite(pool, tp_f64.vectors[1])
        cs_i32 = _lookup_borrow_callsite(pool, tp_i32.vectors[1])

        @test cs_f64 !== nothing
        @test cs_i32 !== nothing
        # Both should be generic labels (direct calls)
        @test cs_f64 == "<direct acquire! call>"
        @test cs_i32 == "<direct acquire! call>"

        _lazy_rewind!(pool)
    end

    # ==============================================================================
    # Return statement validation: explicit return in function form
    # ==============================================================================

    @testset "Function form: explicit return triggers escape detection" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
    end

    @testset "Function form: safe return passes validation" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        v = acquire!(pool, Float64, 5)
        v .= 3.0
        result = sum(v)  # scalar — safe
        _validate_pool_return(result, pool)

        _lazy_rewind!(pool)

        @test result == 15.0
    end

    @testset "Function form: bare return (nothing) passes" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        _ = acquire!(pool, Float64, 10)
        _validate_pool_return(nothing, pool)

        _lazy_rewind!(pool)
    end

    @testset "Function form: return with callsite at S=1" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        pool._pending_callsite = "test_borrow:6\nv = acquire!(pool, Float64, 10)"
        err = try
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")
    end

    @testset "Block form: return in enclosing function triggers validation" begin
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)

        err = try
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end

        @test err isa PoolRuntimeEscapeError
    end

end
