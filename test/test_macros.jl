# Tests for @with_pool and @maybe_with_pool macros
import AdaptiveArrayPools: checkpoint!, rewind!

@testset "Macro System" begin

    @testset "Explicit pool with checkpoint!/rewind!" begin
        pool = AdaptiveArrayPool()

        v_outer = acquire!(pool, Float64, 10)
        @test pool.float64.n_active == 1

        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 20)
        v2 = acquire!(pool, Float64, 30)
        @test pool.float64.n_active == 3
        result = sum(v1) + sum(v2)
        rewind!(pool)

        @test pool.float64.n_active == 1
        @test result isa Number

        v_outer .= 42.0
        @test all(v_outer .== 42.0)
    end

    @testset "Nested checkpoint!/rewind!" begin
        pool = AdaptiveArrayPool()

        function inner_computation(pool)
            checkpoint!(pool)
            try
                v = acquire!(pool, Float64, 10)
                v .= 2.0
                sum(v)
            finally
                rewind!(pool)
            end
        end

        function outer_computation(pool)
            checkpoint!(pool)
            try
                v1 = acquire!(pool, Float64, 20)
                v1 .= 1.0
                inner_result = inner_computation(pool)
                @test all(v1 .== 1.0)
                sum(v1) + inner_result
            finally
                rewind!(pool)
            end
        end

        result = outer_computation(pool)
        @test result == 40.0
        @test pool.float64.n_active == 0
    end

    @testset "@with_pool basic usage" begin
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            sum(v)
        end
        @test result == 10.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@with_pool 1-arg (no pool name)" begin
        # When you don't need the pool variable, use 1-arg form
        function inner_uses_global(n)
            pool = get_task_local_pool()
            v = acquire!(pool, Float64, n)
            v .= 2.0
            sum(v)
        end

        result = @with_pool begin
            inner_uses_global(5)
        end
        @test result == 10.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@with_pool nested scopes" begin
        result = @with_pool p1 begin
            v1 = acquire!(p1, Float64, 10)
            v1 .= 1.0

            inner = @with_pool p2 begin
                v2 = acquire!(p2, Float64, 5)
                v2 .= 2.0
                sum(v2)
            end

            # v1 should still be valid
            @test all(v1 .== 1.0)
            sum(v1) + inner
        end
        @test result == 20.0
    end

    @testset "@with_pool with function passing pool" begin
        function compute_with_pool(x, pool)
            temp = acquire!(pool, Float64, length(x))
            temp .= x .* 2
            sum(temp)
        end

        x = [1.0, 2.0, 3.0]
        result = @with_pool pool begin
            compute_with_pool(x, pool)
        end
        @test result == 12.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@maybe_with_pool enabled" begin
        MAYBE_POOLING[] = true

        result = @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 3.0
            sum(v)
        end
        @test result == 30.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@maybe_with_pool disabled" begin
        MAYBE_POOLING[] = false

        result = @maybe_with_pool pool begin
            @test pool isa DisabledPool{:cpu}
            @test !pooling_enabled(pool)
            v = acquire!(pool, Float64, 10)  # Falls back to normal allocation
            @test v isa Vector{Float64}
            v .= 4.0
            sum(v)
        end
        @test result == 40.0

        # Reset
        MAYBE_POOLING[] = true
    end

    @testset "@maybe_with_pool 1-arg (no pool name)" begin
        MAYBE_POOLING[] = true

        result = @maybe_with_pool begin
            pool = get_task_local_pool()
            v = acquire!(pool, Float64, 5)
            v .= 1.0
            sum(v)
        end
        @test result == 5.0

        MAYBE_POOLING[] = false
        result2 = @maybe_with_pool begin
            # Pool is nothing, so we allocate normally
            v = Vector{Float64}(undef, 5)
            v .= 2.0
            sum(v)
        end
        @test result2 == 10.0

        # Reset
        MAYBE_POOLING[] = true
    end

    @testset "@with_pool function definition" begin
        @with_pool p1 function my_test_func(n)
            v = acquire!(p1, Float64, n)
            v .= 1.0
            sum(v)
        end

        # Check if it works
        res = my_test_func(10)
        @test res == 10.0

        # Check if pool is clean after call
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@with_pool short-form function definitions" begin
        # Simple short-form: f(x) = expr
        @with_pool p1 short_form_simple(n) = begin
            v = acquire!(p1, Float64, n)
            v .= 2.0
            sum(v)
        end

        @test short_form_simple(5) == 10.0
        @test get_task_local_pool().float64.n_active == 0

        # With type annotation: f(x)::T = expr
        @with_pool p2 short_form_typed(n)::Float64 = begin
            v = acquire!(p2, Float64, n)
            v .= 3.0
            sum(v)
        end

        @test short_form_typed(4) == 12.0
        @test get_task_local_pool().float64.n_active == 0

        # With type annotation on arguments
        @with_pool p3 short_form_arg_typed(x::Vector{Float64}) = begin
            v = acquire!(p3, Float64, length(x))
            v .= x .* 2
            sum(v)
        end

        @test short_form_arg_typed([1.0, 2.0, 3.0]) == 12.0
        @test get_task_local_pool().float64.n_active == 0

        # Combined: return type and argument types
        @with_pool p4 short_form_combined(x::Vector{Float64}, y::Vector{Float64})::Float64 = begin
            v1 = acquire!(p4, Float64, length(x))
            v2 = acquire!(p4, Float64, length(y))
            v1 .= x
            v2 .= y
            sum(v1) + sum(v2)
        end

        @test short_form_combined([1.0, 2.0], [3.0, 4.0]) == 10.0
        @test get_task_local_pool().float64.n_active == 0
    end

    @testset "@maybe_with_pool short-form function definitions" begin
        MAYBE_POOLING[] = true

        # Simple short-form with @maybe_with_pool
        @maybe_with_pool p1 maybe_short_form(n) = begin
            v = acquire!(p1, Float64, n)
            v .= 5.0
            sum(v)
        end

        @test maybe_short_form(3) == 15.0
        @test get_task_local_pool().float64.n_active == 0

        # Test with pooling disabled
        MAYBE_POOLING[] = false

        @maybe_with_pool p2 maybe_short_disabled(n) = begin
            @test p2 isa DisabledPool{:cpu}
            @test !pooling_enabled(p2)
            v = acquire!(p2, Float64, n)  # Falls back to allocation
            v .= 1.0
            sum(v)
        end

        @test maybe_short_disabled(5) == 5.0

        # Reset
        MAYBE_POOLING[] = true
    end

    # ==============================================================================
    # Direct-rewind path tests (no try-finally)
    # ==============================================================================

    @testset "Direct rewind: explicit return in @with_pool function" begin
        @with_pool pool function early_return_test(flag)
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            if flag
                return sum(v)  # rewind should happen before return
            end
            v .= 2.0
            sum(v)
        end

        @test early_return_test(true) == 10.0
        @test early_return_test(false) == 20.0

        # Pool should be clean after both paths
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: break inside @with_pool block in loop" begin
        result = 0.0
        for i in 1:10
            @with_pool pool begin
                v = acquire!(pool, Float64, 5)
                v .= Float64(i)
                result = sum(v)
                if i == 3
                    break  # rewind should happen before break
                end
            end
        end

        @test result == 15.0  # 3 * 5
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: continue inside @with_pool block in loop" begin
        total = 0.0
        for i in 1:5
            @with_pool pool begin
                v = acquire!(pool, Float64, 3)
                v .= Float64(i)
                if i == 3
                    continue  # rewind should happen before continue
                end
                total += sum(v)
            end
        end

        # sum for i=1,2,4,5 → 3*(1+2+4+5) = 36
        @test total == 36.0
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "Direct rewind: nested catch recovery (entry depth guard)" begin
        @with_pool pool function outer_catches()
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            result = try
                @with_pool pool begin
                    w = acquire!(pool, UInt8, 5)
                    error("boom")  # inner scope leaks
                end
            catch
                42
            end
            sum(v) + result
        end

        @test outer_catches() == 52.0  # 10.0 + 42
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "@safe_with_pool preserves try-finally behavior" begin
        reset!(get_task_local_pool())  # ensure clean state
        try
            @safe_with_pool pool begin
                acquire!(pool, Float64, 10)
                error("simulated failure")
            end
        catch
        end

        # try-finally guarantees cleanup even after exception
        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    @testset "@safe_maybe_with_pool preserves try-finally behavior" begin
        reset!(get_task_local_pool())  # ensure clean state
        try
            @safe_maybe_with_pool pool begin
                acquire!(pool, Float64, 10)
                error("simulated failure")
            end
        catch
        end

        pool = get_task_local_pool()
        @test pool._current_depth == 1
    end

    # ==============================================================================
    # @goto safety checks
    # ==============================================================================

    @testset "@goto safety in @with_pool" begin
        # Internal @goto/@label: allowed (both are inside the pool body)
        @testset "Internal @goto is allowed" begin
            result = @with_pool pool begin
                x = acquire!(pool, Float64, 10)
                x .= 1.0
                s = sum(x)
                if s < 100
                    @goto done
                end
                s *= 2
                @label done
                s
            end
            @test result == 10.0
            pool = get_task_local_pool()
            @test pool._current_depth == 1
        end

        # External @goto: hard error at macro expansion time
        @testset "External @goto is a hard error" begin
            @test_throws ErrorException @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                @goto outside
            end
        end

        # @safe_with_pool allows any @goto (try-finally protects)
        @testset "@safe_with_pool allows @goto" begin
            expr = @macroexpand @safe_with_pool pool begin
                v = acquire!(pool, Float64, 10)
                @goto outside
            end
            @test expr isa Expr  # no error thrown
        end

        # Multiple internal @goto to different labels
        @testset "Multiple internal @goto targets" begin
            result = @with_pool pool begin
                v = acquire!(pool, Float64, 5)
                v .= 1.0
                x = sum(v)
                if x > 10.0
                    @goto big
                elseif x > 0.0
                    @goto small
                end
                @label big
                x *= 100
                @label small
                x
            end
            @test result == 5.0  # falls through to @label small
            @test get_task_local_pool()._current_depth == 1
        end

        # @goto in function form (not just block)
        @testset "External @goto error in function form" begin
            @test_throws ErrorException @macroexpand @with_pool pool function goto_func()
                v = acquire!(pool, Float64, 10)
                @goto escape
            end
        end

        # @goto inside inner lambda is ignored (separate scope)
        @testset "@goto inside inner function is ignored" begin
            expr = @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                f = () -> @goto somewhere  # inner function — not our scope
                sum(v)
            end
            @test expr isa Expr  # no error — inner lambda @goto is skipped
        end

        # Mix of internal and external @goto: external wins → error
        @testset "Mixed internal+external @goto errors on external" begin
            @test_throws ErrorException @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                @goto internal_label
                @label internal_label
                @goto external_label  # this one has no matching @label
            end
        end

        # Quoted @label must NOT mask real external @goto
        @testset "Quoted @label does not mask external @goto" begin
            @test_throws ErrorException @macroexpand @with_pool pool begin
                q = quote
                    @label escape  # just AST data, not a real label
                end
                @goto escape  # real goto — should be caught as external
            end
        end

        # Quoted @goto should NOT trigger false-positive error
        @testset "Quoted @goto does not trigger false error" begin
            expr = @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                q = quote
                    @goto somewhere  # just AST data, harmless
                end
                sum(v)
            end
            @test expr isa Expr  # no error — quoted @goto is ignored
        end
    end

    # ==============================================================================
    # Exception edge cases (deferred recovery)
    # ==============================================================================

    @testset "Exception edge cases" begin
        # Multi-level nested throw: 2 inner scopes leak, outer catches
        @testset "Multi-level nested leak recovery" begin
            reset!(get_task_local_pool())
            @with_pool pool function multi_level_leak()
                v = acquire!(pool, Float64, 10)
                v .= 1.0
                result = try
                    @with_pool pool begin
                        acquire!(pool, UInt8, 5)
                        @with_pool pool begin
                            acquire!(pool, Int32, 3)
                            error("deep boom")  # 2 inner scopes leak
                        end
                    end
                catch
                    99
                end
                sum(v) + result
            end

            @test multi_level_leak() == 109.0  # 10.0 + 99
            @test get_task_local_pool()._current_depth == 1
        end

        # Multi-type cross-scope throw: inner uses different types than outer
        @testset "Cross-type throw recovery" begin
            reset!(get_task_local_pool())
            @with_pool pool function cross_type_throw()
                v = acquire!(pool, Float64, 10)
                v .= 2.0
                result = try
                    @with_pool pool begin
                        w = acquire!(pool, Int64, 5)  # different type from outer
                        w .= 1
                        error("type mismatch boom")
                    end
                catch
                    0
                end
                sum(v) + result
            end

            @test cross_type_throw() == 20.0  # sum(v)=20 + 0
            pool = get_task_local_pool()
            @test pool._current_depth == 1
            @test pool.float64.n_active == 0
        end

        # Uncaught exception → pool state is corrupted (documented limitation)
        @testset "Uncaught exception corrupts pool (documented)" begin
            reset!(get_task_local_pool())
            try
                @with_pool pool begin
                    acquire!(pool, Float64, 10)
                    error("uncaught!")
                end
            catch
            end
            # Without try-finally, rewind! was never called
            pool = get_task_local_pool()
            @test pool._current_depth > 1  # corrupted — this is expected behavior

            # reset! recovers
            reset!(pool)
            @test pool._current_depth == 1
        end

        # @safe_with_pool handles uncaught exception correctly
        @testset "@safe_with_pool handles uncaught exception" begin
            reset!(get_task_local_pool())
            try
                @safe_with_pool pool begin
                    acquire!(pool, Float64, 10)
                    error("caught by safe!")
                end
            catch
            end
            # try-finally guarantees cleanup
            @test get_task_local_pool()._current_depth == 1
        end
    end

    # ==============================================================================
    # Leaked scope warning (RUNTIME_CHECK-gated)
    # ==============================================================================

    @testset "Leaked scope warning" begin
        import AdaptiveArrayPools: _warn_leaked_scope, _runtime_check

        # 1. Macro expansion includes _warn_leaked_scope call
        @testset "Warning present in macro expansion" begin
            expr = @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                sum(v)
            end
            expr_str = string(expr)
            @test occursin("_warn_leaked_scope", expr_str)
        end

        # 2. Warning is gated by _runtime_check (returns false at RUNTIME_CHECK=0)
        @testset "Warning gated by _runtime_check" begin
            pool_s0 = AdaptiveArrayPool{0}()
            @test _runtime_check(pool_s0) == false   # guard is false → warning never fires

            pool_s1 = AdaptiveArrayPool{1}()
            @test _runtime_check(pool_s1) == true    # guard is true → warning can fire
        end

        # 3. Warning fires on RUNTIME_CHECK=1 pool with simulated leak
        @testset "Warning fires on leaked scope (RUNTIME_CHECK=1)" begin
            pool = AdaptiveArrayPool{1}()
            @test _runtime_check(pool) == true

            # Simulate: checkpoint without matching rewind (leak)
            checkpoint!(pool)         # depth 1→2 (outer scope)
            checkpoint!(pool)         # depth 2→3 (inner scope, will "leak")
            # skip inner rewind — simulates leaked @with_pool

            entry_depth = 1  # outer scope's entry depth
            @test pool._current_depth > entry_depth + 1  # guard condition is true

            # Verify _warn_leaked_scope fires @error log
            @test_logs (:error, r"Leaked @with_pool scope") _warn_leaked_scope(pool, entry_depth)

            # Cleanup
            reset!(pool)
        end

        # 4. Warning does NOT fire when depth is correct
        @testset "No warning on normal depth" begin
            pool = AdaptiveArrayPool{1}()
            checkpoint!(pool)   # depth 1→2
            # No leak — depth is entry_depth + 1

            entry_depth = 1
            @test pool._current_depth == entry_depth + 1  # guard condition is false
            # _warn_leaked_scope would NOT be called (the if guard prevents it)

            rewind!(pool)
            @test pool._current_depth == 1
        end
    end

end # Macro System
