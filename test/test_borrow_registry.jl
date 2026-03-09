import AdaptiveArrayPools: _validate_pool_return, _lookup_borrow_callsite,
    PoolRuntimeEscapeError, Bit

@testset "Borrow Registry (POOL_SAFETY_LV=3)" begin

    # ==============================================================================
    # Basic recording: LV=3 macro path → callsite in escape error
    # ==============================================================================

    @testset "Macro path: escape error includes callsite" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        err = try
            @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                @skip_check_vars v
                identity(v)
            end
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")  # "file:line" format

        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Macro path: unsafe_acquire! escape includes callsite" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        err = try
            @with_pool pool begin
                v = unsafe_acquire!(pool, Float64, 10)
                @skip_check_vars v
                identity(v)
            end
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")

        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # Non-macro path: direct acquire! → generic label
    # ==============================================================================

    @testset "Direct acquire! shows generic callsite label" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)
        err = try
            _validate_pool_return(v, pool)
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite == "<direct acquire! call>"

        rewind!(pool)
        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Direct unsafe_acquire! shows generic callsite label" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = unsafe_acquire!(pool, Float64, 10)
        err = try
            _validate_pool_return(v, pool)
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite == "<direct unsafe_acquire! call>"

        rewind!(pool)
        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # Convenience functions via macro → callsite
    # ==============================================================================

    @testset "Macro path: zeros! escape includes callsite" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        err = try
            @with_pool pool begin
                v = zeros!(pool, Float64, 10)
                @skip_check_vars v
                identity(v)
            end
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")

        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Direct zeros! shows generic callsite label" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = zeros!(pool, Float64, 10)
        err = try
            _validate_pool_return(v, pool)
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite == "<direct zeros! call>"

        rewind!(pool)
        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # BitArray path → callsite
    # ==============================================================================

    @testset "Macro path: BitArray acquire escape includes callsite" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        err = try
            @with_pool pool begin
                v = acquire!(pool, Bit, 100)
                @skip_check_vars v
                identity(v)
            end
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")

        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # LV<3: no borrow log overhead
    # ==============================================================================

    @testset "LV<3 does not create borrow log" begin
        for lv in (0, 1, 2)
            old_lv = POOL_SAFETY_LV[]
            POOL_SAFETY_LV[] = lv

            pool = AdaptiveArrayPool()
            checkpoint!(pool)
            _ = acquire!(pool, Float64, 10)
            @test pool._borrow_log === nothing
            rewind!(pool)

            POOL_SAFETY_LV[] = old_lv
        end
    end

    # ==============================================================================
    # LV=3: borrow log IS created
    # ==============================================================================

    @testset "LV=3 creates borrow log on acquire" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        _ = acquire!(pool, Float64, 10)
        @test pool._borrow_log !== nothing
        @test pool._borrow_log isa IdDict
        rewind!(pool)

        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # reset! clears borrow log
    # ==============================================================================

    @testset "reset! clears borrow log and pending callsite" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        _ = acquire!(pool, Float64, 10)
        @test pool._borrow_log !== nothing

        reset!(pool)
        @test pool._borrow_log === nothing
        @test pool._pending_callsite == ""

        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # Error message format: showerror output
    # ==============================================================================

    @testset "showerror: 'acquired at' shown when callsite present (LV≥3)" begin
        err = PoolRuntimeEscapeError("SubArray{Float64, 1}", "Float64", "test.jl:42")
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test contains(msg, "acquired at")
        @test contains(msg, "test.jl:42")
        @test contains(msg, "POOL_SAFETY_LV ≥ 3")
        @test !contains(msg, "Tip:")
    end

    @testset "showerror: 'Tip: set LV=3' shown when no callsite (LV=2)" begin
        err = PoolRuntimeEscapeError("SubArray{Float64, 1}", "Float64", nothing)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))

        @test !contains(msg, "acquired at")
        @test contains(msg, "POOL_SAFETY_LV ≥ 2")
        @test contains(msg, "Tip:")
        @test contains(msg, "POOL_SAFETY_LV[] = 3")
    end

    # ==============================================================================
    # Multiple types: each gets correct callsite
    # ==============================================================================

    @testset "Multiple types record independent callsites" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        pool = AdaptiveArrayPool()
        checkpoint!(pool)

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

        rewind!(pool)
        POOL_SAFETY_LV[] = old_lv
    end

    # ==============================================================================
    # Return statement validation: explicit return in function form
    # ==============================================================================

    @testset "Function form: explicit return triggers escape detection" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 2

        # Function with explicit return of pool-backed array should throw
        @with_pool pool function _test_return_escape()
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            return identity(v)
        end

        @test_throws PoolRuntimeEscapeError _test_return_escape()

        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Function form: safe return passes validation" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 2

        @with_pool pool function _test_safe_return()
            v = acquire!(pool, Float64, 5)
            v .= 3.0
            return sum(v)  # scalar — safe
        end

        @test _test_safe_return() == 15.0

        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Function form: bare return (nothing) passes" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 2

        @with_pool pool function _test_bare_return()
            _ = acquire!(pool, Float64, 10)
            return
        end

        @test _test_bare_return() === nothing

        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Function form: return with callsite at LV=3" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 3

        @with_pool pool function _test_return_callsite()
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            return identity(v)
        end

        err = try
            _test_return_callsite()
            nothing
        catch e
            e
        end

        @test err isa PoolRuntimeEscapeError
        @test err.callsite !== nothing
        @test contains(err.callsite, ":")

        POOL_SAFETY_LV[] = old_lv
    end

    @testset "Block form: return in enclosing function triggers validation" begin
        old_lv = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 2

        function _test_block_return_escape()
            @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                @skip_check_vars v
                return identity(v)
            end
        end

        @test_throws PoolRuntimeEscapeError _test_block_return_escape()

        POOL_SAFETY_LV[] = old_lv
    end

end
