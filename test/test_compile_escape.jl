import AdaptiveArrayPools: _extract_acquired_vars, _get_last_expression,
    _find_direct_exposure, _remove_flat_reassigned!, _check_compile_time_escape,
    _is_definite_escape

@testset "Compile-Time Escape Detection" begin

    # ==============================================================================
    # _extract_acquired_vars: find variables assigned from acquire calls
    # ==============================================================================

    @testset "_extract_acquired_vars" begin
        # Basic acquire!
        vars = _extract_acquired_vars(
            :(v = acquire!(pool, Float64, 10)),
            :pool
        )
        @test :v in vars

        # Multiple acquire functions
        vars = _extract_acquired_vars(
            quote
                v = acquire!(pool, Float64, 10)
                w = zeros!(pool, 5)
                x = ones!(pool, Int64, 3)
                y = similar!(pool, some_array)
                z = unsafe_acquire!(pool, Float64, 20)
                bv = trues!(pool, 100)
                bf = falses!(pool, 50)
                r = reshape!(pool, some_array, 3, 4)
            end,
            :pool
        )
        @test :v in vars
        @test :w in vars
        @test :x in vars
        @test :y in vars
        @test :z in vars
        @test :bv in vars
        @test :bf in vars
        @test :r in vars

        # Non-acquire call → not tracked
        vars = _extract_acquired_vars(
            :(v = sum(data)),
            :pool
        )
        @test isempty(vars)

        # Different pool → not tracked
        vars = _extract_acquired_vars(
            :(v = acquire!(other_pool, Float64, 10)),
            :pool
        )
        @test isempty(vars)

        # Mixed: only acquire calls tracked
        vars = _extract_acquired_vars(
            quote
                v = acquire!(pool, Float64, 10)
                w = sum(v)
                x = zeros!(pool, 5)
            end,
            :pool
        )
        @test :v in vars
        @test :x in vars
        @test !(:w in vars)
    end

    # ==============================================================================
    # _get_last_expression: find the block's return value expression
    # ==============================================================================

    @testset "_get_last_expression" begin
        # Simple block
        @test _get_last_expression(quote
            a = 1
            b = 2
            c
        end) == :c

        # Block ending with LineNumberNode → skip it
        block = Expr(:block, :a, LineNumberNode(1, :test))
        @test _get_last_expression(block) == :a

        # Non-block expression
        @test _get_last_expression(:v) == :v
        @test _get_last_expression(42) == 42

        # Nested block
        @test _get_last_expression(quote
            begin
                x
            end
        end) == :x

        # Empty block
        @test _get_last_expression(Expr(:block)) === nothing
    end

    # ==============================================================================
    # _find_direct_exposure: detect acquired vars in return expression
    # ==============================================================================

    @testset "_find_direct_exposure" begin
        acquired = Set{Symbol}([:v, :w])

        # Bare symbol → detected
        @test :v in _find_direct_exposure(:v, acquired)
        @test :w in _find_direct_exposure(:w, acquired)

        # Non-acquired symbol → not detected
        @test isempty(_find_direct_exposure(:x, acquired))

        # Function call → NOT detected (can't know return type)
        @test isempty(_find_direct_exposure(:(sum(v)), acquired))
        @test isempty(_find_direct_exposure(:(f(v, w)), acquired))

        # Indexing → NOT detected (element access)
        @test isempty(_find_direct_exposure(:(v[1]), acquired))

        # Tuple containing acquired vars → detected
        found = _find_direct_exposure(:(v, w), acquired)
        @test :v in found
        @test :w in found

        # Tuple with mix of acquired and non-acquired
        found = _find_direct_exposure(:(v, 42, x), acquired)
        @test :v in found
        @test !(:x in found)

        # Array literal
        found = _find_direct_exposure(:([v, w]), acquired)
        @test :v in found
        @test :w in found

        # NamedTuple with = syntax: (a=v,)
        found = _find_direct_exposure(
            Expr(:tuple, Expr(:(=), :a, :v)),
            acquired
        )
        @test :v in found

        # NamedTuple with kw syntax: (a=v,)
        found = _find_direct_exposure(
            Expr(:parameters, Expr(:kw, :a, :v)),
            acquired
        )
        @test :v in found

        # return v
        found = _find_direct_exposure(Expr(:return, :v), acquired)
        @test :v in found

        # return (v, w)
        found = _find_direct_exposure(
            Expr(:return, Expr(:tuple, :v, :w)),
            acquired
        )
        @test :v in found
        @test :w in found

        # Scalar literal → not detected
        @test isempty(_find_direct_exposure(42, acquired))
        @test isempty(_find_direct_exposure(3.14, acquired))
    end

    # ==============================================================================
    # _remove_flat_reassigned!: handle v=acquire!() then v=other pattern
    # ==============================================================================

    @testset "_remove_flat_reassigned!" begin
        # Simple reassignment removes from set
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                v = zeros(10)
            end,
            acquired, :pool
        )
        @test isempty(acquired)

        # Reassignment to another acquire keeps it
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                v = zeros!(pool, 10)
            end,
            acquired, :pool
        )
        @test :v in acquired

        # Different variable reassigned → original stays
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                w = zeros(10)
            end,
            acquired, :pool
        )
        @test :v in acquired

        # Non-block expression → no change
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(:(v = zeros(10)), acquired, :pool)
        @test :v in acquired  # only processes :block heads
    end

    # ==============================================================================
    # _check_compile_time_escape: integration tests
    # ==============================================================================

    @testset "_check_compile_time_escape" begin
        src = LineNumberNode(1, :test)

        # Bare variable return → error "Pool escape"
        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            :pool, src
        )

        # Tuple containing acquired var → warning "Possible pool escape"
        @test_warn "Possible pool escape" _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                w = acquire!(pool, Float64, 5)
                (sum(v), w)
            end,
            :pool, src
        )

        # Safe: scalar return → no warning
        @test_nowarn _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                sum(v)
            end,
            :pool, src
        )

        # Safe: collect return → no warning
        @test_nowarn _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                collect(v)
            end,
            :pool, src
        )

        # Safe: literal return → no warning
        @test_nowarn _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v .= 1.0
                42
            end,
            :pool, src
        )

        # Safe: reassigned then returned → no warning
        @test_nowarn _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v .= data
                v = collect(v)
                v
            end,
            :pool, src
        )

        # Safe: no acquire calls → no warning
        @test_nowarn _check_compile_time_escape(
            quote
                x = sum(data)
                x
            end,
            :pool, src
        )

        # zeros!/ones!/similar! also detected
        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = zeros!(pool, 10)
                v
            end,
            :pool, src
        )

        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = ones!(pool, Float32, 10)
                v
            end,
            :pool, src
        )

        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = similar!(pool, some_array)
                v
            end,
            :pool, src
        )

        # unsafe_acquire! also detected
        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = unsafe_acquire!(pool, Float64, 10)
                v
            end,
            :pool, src
        )

        # trues!/falses! also detected
        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                bv = trues!(pool, 100)
                bv
            end,
            :pool, src
        )

        # Different pool name → no warning (not our pool)
        @test_nowarn _check_compile_time_escape(
            quote
                v = acquire!(other_pool, Float64, 10)
                v
            end,
            :pool, src
        )

        # source=nothing also works
        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            :pool, nothing
        )

        # `return v` is also a definite escape (error)
        @test_throws "Pool escape" _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                return v
            end,
            :pool, src
        )

        # `return (v, w)` is a possible escape (warning, not error)
        @test_warn "Possible pool escape" _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                return (v, sum(v))
            end,
            :pool, src
        )
    end

    # ==============================================================================
    # _is_definite_escape: bare symbol or return-of-symbol
    # ==============================================================================

    @testset "_is_definite_escape" begin
        # Bare symbol → definite
        @test _is_definite_escape(:v, :v)
        @test !_is_definite_escape(:w, :v)

        # return v → definite
        @test _is_definite_escape(Expr(:return, :v), :v)
        @test !_is_definite_escape(Expr(:return, :w), :v)

        # return (v, w) → NOT definite (container)
        @test !_is_definite_escape(Expr(:return, Expr(:tuple, :v, :w)), :v)

        # Tuple → NOT definite
        @test !_is_definite_escape(Expr(:tuple, :v, :w), :v)

        # Literal → NOT definite
        @test !_is_definite_escape(42, :v)
    end

    # ==============================================================================
    # Integration: compile-time error via @macroexpand
    # ==============================================================================

    @testset "Compile-time error through macro pipeline" begin
        # Bare variable: macro expansion itself throws
        @test_throws "Pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Safe return: macro expansion succeeds
        expanded = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end
        @test expanded isa Expr

        # Function form: bare return also caught
        @test_throws "Pool escape" @macroexpand @with_pool pool function test_fn(n)
            v = acquire!(pool, Float64, n)
            v
        end
    end

    # ==============================================================================
    # Integration: block form — false positive prevention
    # ==============================================================================

    @testset "Block form: safe patterns (no false positives)" begin
        # Function call on acquired var → safe (can't determine return type)
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            identity(v)
        end

        # Indexing → safe (element access, not array itself)
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v[1]
        end

        # Arithmetic expression → safe
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v) + 1.0
        end

        # Reassigned from collect, then returned → safe
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v = collect(v)
            v
        end

        # Non-acquired variable returned → safe
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = sum(v)
            w
        end

        # nothing return → safe
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            nothing
        end

        # Multiple acquires, safe return
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = zeros!(pool, 5)
            sum(v) + sum(w)
        end

        # No acquire calls at all → safe
        @test_nowarn @macroexpand @with_pool pool begin
            x = [1, 2, 3]
            sum(x)
        end

        # Boolean comparison → safe
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            length(v) > 0
        end

        # NamedTuple: key name matches acquired var, but VALUE is safe
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            (v = collect(v), total = sum(v))
        end

        # NamedTuple: key name coincidentally same, value is unrelated
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= data
            (v = zeros(10), u = sum(v))
        end

        # --- Tricky-but-safe edge cases (true negatives) ---
        # Each pattern looks like it might escape pool memory, but is genuinely safe.
        # The checker correctly does NOT flag these.

        # Safe: v is reassigned to non-pool arrays twice; final v is a fresh collect'd copy
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            v = v .+ 1.0
            v = collect(v)
            v
        end

        # Safe: w is a plain view (not from acquire!), and return is scalar sum
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = view(v, 1:5)
            sum(w)
        end

        # Safe: similar() (no !) allocates a fresh independent array — w is not pool-backed
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = similar(v)
            w .= 1.0
            w
        end

        # Safe: copy() returns an independent deep copy — no pool memory escapes
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            copy(v)
        end

        # Safe: comprehension allocates a fresh Array from element values
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            [v[i]^2 for i in 1:10]
        end

        # Safe: result is a String, not an array
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            "total = $(sum(v))"
        end

        # Safe: ternary returns scalar from both branches
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            length(v) > 5 ? sum(v) : 0.0
        end

        # Safe: pipe evaluates to sum(v) — a scalar, not the array
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            v |> sum
        end

        # Safe: broadcast allocates a fresh result array — neither v nor w escapes
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 10)
            v .+ w
        end

        # Safe: Dict holds only scalars (sum, length) — no pool array reference
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            Dict(:sum => sum(v), :len => length(v))
        end

        # Safe: let block returns scalar s*2 — pool array v stays local
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 2.0
            let s = sum(v)
                s * 2
            end
        end

        # Safe: map() allocates a fresh Array with transformed elements
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            map(v) do x
                x^2
            end
        end
    end

    @testset "Block form: additional escape scenarios" begin
        # zeros! — definite escape
        @test_throws "Pool escape" @macroexpand @with_pool pool begin
            v = zeros!(pool, 10)
            v
        end

        # trues! — definite escape
        @test_throws "Pool escape" @macroexpand @with_pool pool begin
            bv = trues!(pool, 100)
            bv
        end

        # Explicit return — definite escape
        @test_throws "Pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            return v
        end

        # Tuple with acquired var — possible escape (warning)
        @test_warn "Possible pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (v, 42)
        end

        # Array literal — possible escape (warning)
        @test_warn "Possible pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            [v, nothing]
        end

        # return (v, scalar) — possible escape (warning)
        @test_warn "Possible pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            return (v, sum(v))
        end

        # Re-acquire reassignment: v still tracked after v = zeros!(pool, ...)
        @test_throws "Pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v = zeros!(pool, 20)
            v
        end

        # NamedTuple with acquired var as VALUE → escape
        @test_warn "Possible pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (result = v, n = 42)
        end

        # NamedTuple shorthand (v = v) → value IS acquired → escape
        @test_warn "Possible pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (v = v,)
        end
    end

    # ==============================================================================
    # Integration: @with_pool function definition form
    # ==============================================================================

    @testset "Function form: escape detection" begin
        # Definite — bare variable return
        @test_throws "Pool escape" @macroexpand @with_pool pool function fn_esc1(n)
            v = acquire!(pool, Float64, n)
            v
        end

        # Definite — explicit return
        @test_throws "Pool escape" @macroexpand @with_pool pool function fn_esc2(n)
            v = zeros!(pool, n)
            v .= 1.0
            return v
        end

        # Definite — trues!
        @test_throws "Pool escape" @macroexpand @with_pool pool function fn_esc3(n)
            bv = trues!(pool, n)
            bv
        end

        # Possible — tuple return
        @test_warn "Possible pool escape" @macroexpand @with_pool pool function fn_warn1(n)
            v = acquire!(pool, Float64, n)
            (v, sum(v))
        end

        # Possible — return tuple
        @test_warn "Possible pool escape" @macroexpand @with_pool pool function fn_warn2(n)
            v = acquire!(pool, Float64, n)
            return (v, n)
        end
    end

    @testset "Function form: safe patterns (no false positives)" begin
        # Scalar return
        @test_nowarn @macroexpand @with_pool pool function fn_safe1(n)
            v = acquire!(pool, Float64, n)
            sum(v)
        end

        # collect() return
        @test_nowarn @macroexpand @with_pool pool function fn_safe2(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            collect(v)
        end

        # nothing return (side-effect function)
        @test_nowarn @macroexpand @with_pool pool function fn_safe3!(out, n)
            v = acquire!(pool, Float64, n)
            out .= v
            nothing
        end

        # Reassigned then returned
        @test_nowarn @macroexpand @with_pool pool function fn_safe4(n)
            v = acquire!(pool, Float64, n)
            v = collect(v)
            v
        end

        # Function call wrapping acquired var
        @test_nowarn @macroexpand @with_pool pool function fn_safe5(n)
            v = acquire!(pool, Float64, n)
            identity(v)
        end

        # Multiple acquires, safe return
        @test_nowarn @macroexpand @with_pool pool function fn_safe6(m, n)
            v = acquire!(pool, Float64, m)
            w = zeros!(pool, n)
            sum(v) + sum(w)
        end

        # Safe: pipe evaluates to scalar sum(v)
        @test_nowarn @macroexpand @with_pool pool function fn_safe_pipe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            v |> sum
        end

        # Safe: comprehension allocates fresh array from element values
        @test_nowarn @macroexpand @with_pool pool function fn_safe_comp(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            [v[i] for i in 1:n]
        end

        # Safe: ternary returns scalar from both branches
        @test_nowarn @macroexpand @with_pool pool function fn_safe_ternary(n)
            v = acquire!(pool, Float64, n)
            n > 5 ? sum(v) : 0.0
        end

        # Safe: v reassigned to fresh broadcast result — no longer pool-backed
        @test_nowarn @macroexpand @with_pool pool function fn_safe_bcast(n)
            v = acquire!(pool, Float64, n)
            v = v .+ 1.0
            v
        end

        # Safe: copy() returns independent deep copy
        @test_nowarn @macroexpand @with_pool pool function fn_safe_copy(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            copy(v)
        end
    end

    # ==============================================================================
    # Integration: @maybe_with_pool forms
    # ==============================================================================

    @testset "@maybe_with_pool block form" begin
        # Definite escape → error
        @test_throws "Pool escape" @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Possible escape → warning
        @test_warn "Possible pool escape" @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (v, sum(v))
        end

        # Safe → no warning
        @test_nowarn @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end
    end

    @testset "@maybe_with_pool function form" begin
        # Definite escape → error
        @test_throws "Pool escape" @macroexpand @maybe_with_pool pool function mwp_esc(n)
            v = acquire!(pool, Float64, n)
            v
        end

        # Safe → no warning
        @test_nowarn @macroexpand @maybe_with_pool pool function mwp_safe(n)
            v = acquire!(pool, Float64, n)
            sum(v)
        end
    end

    # ==============================================================================
    # Integration: backend forms (@with_pool :cpu, @maybe_with_pool :cpu)
    # ==============================================================================

    @testset "@with_pool :cpu block form" begin
        @test_throws "Pool escape" @macroexpand @with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        @test_nowarn @macroexpand @with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end
    end

    @testset "@with_pool :cpu function form" begin
        @test_throws "Pool escape" @macroexpand @with_pool :cpu pool function cpu_esc(n)
            v = acquire!(pool, Float64, n)
            v
        end

        @test_nowarn @macroexpand @with_pool :cpu pool function cpu_safe(n)
            v = acquire!(pool, Float64, n)
            sum(v)
        end
    end

    @testset "@maybe_with_pool :cpu forms" begin
        # Block — error
        @test_throws "Pool escape" @macroexpand @maybe_with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Block — safe
        @test_nowarn @macroexpand @maybe_with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end

        # Function — error
        @test_throws "Pool escape" @macroexpand @maybe_with_pool :cpu pool function mcpu_esc(n)
            v = acquire!(pool, Float64, n)
            v
        end

        # Function — safe
        @test_nowarn @macroexpand @maybe_with_pool :cpu pool function mcpu_safe(n)
            v = acquire!(pool, Float64, n)
            sum(v)
        end
    end

    # ==============================================================================
    # Integration: nested @with_pool scopes
    # ==============================================================================

    @testset "Nested @with_pool scopes" begin
        # Note: @macroexpand expands only the outermost macro; inner @with_pool
        # inside esc() boundaries are not recursively expanded by macroexpand().
        # Inner scope escape detection runs when code is actually compiled.

        # Both scopes safe → no error
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            inner = @with_pool pool begin
                w = acquire!(pool, Float64, 5)
                sum(w)
            end
            sum(v) + inner
        end

        # Outer scope escape → error from outer macro check
        @test_throws "Pool escape" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @with_pool pool begin
                w = acquire!(pool, Float64, 5)
                w .= 1.0
                nothing
            end
            v  # ← outer definite escape
        end
    end

    # ==============================================================================
    # Error/warning messages: verify variable names and suggestions
    # ==============================================================================

    @testset "Error messages identify escaping variables" begin
        # Error includes specific variable name
        @test_throws r"`v`" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Error includes collect() suggestion
        @test_throws r"collect" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Different variable name in error
        @test_throws r"`data`" @macroexpand @with_pool pool begin
            data = zeros!(pool, 10)
            data
        end

        # Function form: error includes variable name
        @test_throws r"`result`" @macroexpand @with_pool pool function msg_fn(n)
            result = acquire!(pool, Float64, n)
            result
        end
    end

    @testset "Warning messages identify escaping variables in containers" begin
        # Warning identifies specific variable in tuple (only w escapes, not sum(v))
        @test_warn r"`w`" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            (sum(v), w)
        end

        # Multiple variables: both v and w warned about
        @test_warn r"`v`" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            (v, w)
        end
        @test_warn r"`w`" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            (v, w)
        end

        # Array literal: warning identifies variable
        @test_warn r"`v`" @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            [v]
        end

        # return tuple in function: warning includes variable name
        @test_warn r"`v`" @macroexpand @with_pool pool function msg_fn_warn(n)
            v = acquire!(pool, Float64, n)
            return (v, n)
        end
    end

end # Compile-Time Escape Detection
