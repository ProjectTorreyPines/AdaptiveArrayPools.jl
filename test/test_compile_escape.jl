import AdaptiveArrayPools: _extract_acquired_vars, _get_last_expression,
    _find_direct_exposure, _remove_flat_reassigned!, _check_compile_time_escape,
    _collect_all_return_values, _collect_explicit_returns!, _collect_implicit_return_values!,
    _get_last_expression_with_line, _render_return_expr,
    _acquire_call_kind, _classify_escaped_vars

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

        # Destructuring with tuple RHS: only acquire elements tracked
        vars = _extract_acquired_vars(
            :((v, w) = (acquire!(pool, Float64, 10), safe_func())),
            :pool
        )
        @test :v in vars
        @test !(:w in vars)

        # Destructuring: both acquire calls tracked
        vars = _extract_acquired_vars(
            :((v, w) = (acquire!(pool, Float64, 10), zeros!(pool, 5))),
            :pool
        )
        @test :v in vars
        @test :w in vars

        # Destructuring with function call RHS: can't determine → nothing tracked
        vars = _extract_acquired_vars(
            :((v, w) = foo()),
            :pool
        )
        @test isempty(vars)

        # Destructuring: mixed with regular assignment
        vars = _extract_acquired_vars(
            quote
                (a, b) = (acquire!(pool, Float64, 10), safe())
                c = zeros!(pool, 5)
            end,
            :pool
        )
        @test :a in vars
        @test !(:b in vars)
        @test :c in vars
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

        # Destructuring with function call RHS: v removed (reassigned to unknown)
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                (result, v) = process(v)
            end,
            acquired, :pool
        )
        @test isempty(acquired)

        # Destructuring with tuple RHS: element-wise check
        acquired = Set{Symbol}([:v, :w])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                w = acquire!(pool, Float64, 5)
                (v, w) = (safe_func(), acquire!(pool, Float64, 3))
            end,
            acquired, :pool
        )
        @test !(:v in acquired)  # reassigned to safe_func() → removed
        @test :w in acquired     # reassigned to acquire!() → stays

        # Comma destructuring (same AST as tuple): v removed
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                a, v = foo()
            end,
            acquired, :pool
        )
        @test isempty(acquired)

        # Destructuring doesn't affect vars not in acquired
        acquired = Set{Symbol}([:v])
        _remove_flat_reassigned!(
            quote
                v = acquire!(pool, Float64, 10)
                (x, y) = foo()
            end,
            acquired, :pool
        )
        @test :v in acquired  # v not in destructuring → untouched
    end

    # ==============================================================================
    # _check_compile_time_escape: integration tests
    # ==============================================================================

    @testset "_check_compile_time_escape" begin
        src = LineNumberNode(1, :test)

        # Bare variable return → error "Pool escape"
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            :pool, src
        )

        # Tuple containing acquired var → error
        @test_throws PoolEscapeError _check_compile_time_escape(
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
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = zeros!(pool, 10)
                v
            end,
            :pool, src
        )

        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = ones!(pool, Float32, 10)
                v
            end,
            :pool, src
        )

        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = similar!(pool, some_array)
                v
            end,
            :pool, src
        )

        # unsafe_acquire! also detected
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = unsafe_acquire!(pool, Float64, 10)
                v
            end,
            :pool, src
        )

        # trues!/falses! also detected
        @test_throws PoolEscapeError _check_compile_time_escape(
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
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            :pool, nothing
        )

        # `return v` is also a definite escape (error)
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                return v
            end,
            :pool, src
        )

        # `return (v, w)` is an escape (error)
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                return (v, sum(v))
            end,
            :pool, src
        )
    end

    # ==============================================================================
    # _collect_all_return_values: explicit returns + implicit if/else branches
    # ==============================================================================

    @testset "_collect_all_return_values (expr, line) pairs" begin
        # Simple block: implicit return only
        vals = _collect_all_return_values(quote
            x = 1
            x
        end)
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :x, exprs)

        # Explicit return inside if branch
        vals = _collect_all_return_values(quote
            v = acquire!(pool, Float64, 10)
            if cond
                return v
            end
            sum(v)
        end)
        exprs = first.(vals)
        @test any(e -> e isa Expr && e.head == :return, exprs)
        @test any(e -> e isa Expr && e.head == :call, exprs)

        # Both branches have explicit return
        vals = _collect_all_return_values(quote
            if a > 0.5
                return (v = 0.5, data = a)
            else
                return (v = v, data = z)
            end
        end)
        returns = filter(((e, _),) -> e isa Expr && e.head == :return, vals)
        @test length(returns) >= 2

        # Implicit return from if/else branches (no explicit return keyword)
        vals = _collect_all_return_values(quote
            if cond
                v
            else
                sum(v)
            end
        end)
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)
        @test any(e -> e isa Expr && e.head == :call, exprs)

        # elseif branches
        vals = _collect_all_return_values(quote
            if cond1
                v
            elseif cond2
                w
            else
                sum(v)
            end
        end)
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)
        @test any(e -> e isa Symbol && e == :w, exprs)

        # Nested if inside branch
        vals = _collect_all_return_values(quote
            if outer
                if inner
                    v
                else
                    w
                end
            else
                sum(v)
            end
        end)
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)
        @test any(e -> e isa Symbol && e == :w, exprs)

        # Skips nested function definitions
        vals = _collect_all_return_values(quote
            f = function()
                return v  # belongs to inner function, not our scope
            end
            sum(v)
        end)
        returns = filter(((e, _),) -> e isa Expr && e.head == :return, vals)
        @test isempty(returns)

        # Ternary operator (same AST as if/else)
        vals = _collect_all_return_values(quote
            cond ? v : sum(v)
        end)
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)

        # Line numbers are tracked
        vals = _collect_all_return_values(quote
            if cond
                return v  # this will have a line number
            end
            sum(v)
        end)
        explicit_returns = filter(((e, _),) -> e isa Expr && e.head == :return, vals)
        @test !isempty(explicit_returns)
        @test last(explicit_returns[1]) !== nothing  # line is captured
    end

    # ==============================================================================
    # _check_compile_time_escape: branch return detection
    # ==============================================================================

    @testset "Escape detection through branches" begin
        src = LineNumberNode(1, :test)

        # Explicit return inside if branch — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                if cond
                    return v
                end
                sum(v)
            end,
            :pool, src
        )

        # Both branches return, one unsafe — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                if a > 0.5
                    return sum(v)
                else
                    return v
                end
            end,
            :pool, src
        )

        # NamedTuple in branch — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                z = similar!(pool, v)
                if a > 0.5
                    return (v = 0.5, data = a)
                else
                    return (v = v, data = z)
                end
            end,
            :pool, src
        )

        # Both branches safe — no error
        @test_nowarn _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                if cond
                    return sum(v)
                else
                    return length(v)
                end
            end,
            :pool, src
        )

        # Implicit return from if/else branches — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                if cond
                    sum(v)
                else
                    v
                end
            end,
            :pool, src
        )

        # elseif branch — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                if cond1
                    sum(v)
                elseif cond2
                    v
                else
                    length(v)
                end
            end,
            :pool, src
        )

        # Ternary with escape — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                cond ? v : sum(v)
            end,
            :pool, src
        )

        # Early return in loop — caught
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                for i in 1:10
                    if cond
                        return v
                    end
                end
                sum(v)
            end,
            :pool, src
        )

        # Multi-variable across branches: reports all
        err = try _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                w = acquire!(pool, Float64, 5)
                if cond
                    return v
                else
                    return w
                end
            end,
            :pool, src
        ) catch e; e end
        @test err isa PoolEscapeError
        @test :v in err.vars
        @test :w in err.vars
    end

    # ==============================================================================
    # Integration: compile-time error via @macroexpand (branch scenarios)
    # ==============================================================================

    @testset "Branch escape detection through macro pipeline" begin
        # if/else with explicit return in one branch — caught
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            if rand() > 0.5
                return sum(v)
            else
                return v
            end
        end

        # Function form with branch returns — caught
        @test_throws PoolEscapeError @macroexpand @with_pool pool function branch_fn(n)
            v = acquire!(pool, Float64, n)
            z = similar!(pool, v)
            if n > 0
                return (v = 0.5, data = 1.0)
            else
                return (v = v, data = z)
            end
        end

        # Ternary — caught
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            rand() > 0.5 ? sum(v) : v
        end

        # All branches safe — no error
        expanded = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            if rand() > 0.5
                return sum(v)
            else
                return length(v)
            end
        end
        @test expanded isa Expr

        # Function form, all branches safe
        expanded = @macroexpand @with_pool pool function safe_branch(n)
            v = acquire!(pool, Float64, n)
            if n > 0
                sum(v)
            else
                0.0
            end
        end
        @test expanded isa Expr
    end

    # ==============================================================================
    # Integration: compile-time error via @macroexpand
    # ==============================================================================

    @testset "Compile-time error through macro pipeline" begin
        # Bare variable: macro expansion itself throws
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
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
        @test_throws PoolEscapeError @macroexpand @with_pool pool function test_fn(n)
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
            sum(v)
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

        # Safe: destructuring reassigns v to a non-pool value
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            (result, v) = process(v)
            v
        end

        # Safe: comma destructuring reassigns v
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            idx, v = findmax(some_array)
            v
        end

        # Safe: destructuring with tuple RHS — v gets safe value, w stays tracked
        # but only v is returned
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            (v, w) = (collect(v), acquire!(pool, Float64, 3))
            sum(w) + sum(v)
        end

        # Safe: swap pattern — v gets non-pool value after swap
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            x = zeros(10)
            v, x = x, v
            v
        end
    end

    @testset "Block form: additional escape scenarios" begin
        # zeros! — definite escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = zeros!(pool, 10)
            v
        end

        # trues! — definite escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            bv = trues!(pool, 100)
            bv
        end

        # Explicit return — definite escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            return v
        end

        # Tuple with acquired var → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (v, 42)
        end

        # Array literal → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            [v, nothing]
        end

        # return (v, scalar) → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            return (v, sum(v))
        end

        # Re-acquire reassignment: v still tracked after v = zeros!(pool, ...)
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v = zeros!(pool, 20)
            v
        end

        # NamedTuple with acquired var as VALUE → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (result = v, n = 42)
        end

        # NamedTuple shorthand (v = v) → value IS acquired → escape
        # (key name coincidentally matches, but VALUE is the acquired var)
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (v = v,)
        end

        # Destructuring with acquire RHS: v still tracked → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            (v, w) = (acquire!(pool, Float64, 10), safe())
            v
        end

        # Destructuring doesn't protect if RHS element IS acquire
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            (v, w) = (zeros!(pool, 5), safe())
            v
        end
    end

    # ==============================================================================
    # Integration: @with_pool function definition form
    # ==============================================================================

    @testset "Function form: escape detection" begin
        # Definite — bare variable return
        @test_throws PoolEscapeError @macroexpand @with_pool pool function fn_esc1(n)
            v = acquire!(pool, Float64, n)
            v
        end

        # Definite — explicit return
        @test_throws PoolEscapeError @macroexpand @with_pool pool function fn_esc2(n)
            v = zeros!(pool, n)
            v .= 1.0
            return v
        end

        # Definite — trues!
        @test_throws PoolEscapeError @macroexpand @with_pool pool function fn_esc3(n)
            bv = trues!(pool, n)
            bv
        end

        # Tuple return → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool function fn_warn1(n)
            v = acquire!(pool, Float64, n)
            (v, sum(v))
        end

        # return tuple → escape
        @test_throws PoolEscapeError @macroexpand @with_pool pool function fn_warn2(n)
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
            sum(v)
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

        # Safe: destructuring reassigns v in function form
        @test_nowarn @macroexpand @with_pool pool function fn_safe_destruct(n)
            v = acquire!(pool, Float64, n)
            (result, v) = process(v)
            v
        end
    end

    # ==============================================================================
    # Integration: @maybe_with_pool forms
    # ==============================================================================

    @testset "@maybe_with_pool block form" begin
        # Definite escape → error
        @test_throws PoolEscapeError @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Container escape → error
        @test_throws PoolEscapeError @macroexpand @maybe_with_pool pool begin
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
        @test_throws PoolEscapeError @macroexpand @maybe_with_pool pool function mwp_esc(n)
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
        @test_throws PoolEscapeError @macroexpand @with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        @test_nowarn @macroexpand @with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end
    end

    @testset "@with_pool :cpu function form" begin
        @test_throws PoolEscapeError @macroexpand @with_pool :cpu pool function cpu_esc(n)
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
        @test_throws PoolEscapeError @macroexpand @maybe_with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            v
        end

        # Block — safe
        @test_nowarn @macroexpand @maybe_with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end

        # Function — error
        @test_throws PoolEscapeError @macroexpand @maybe_with_pool :cpu pool function mcpu_esc(n)
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
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
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

    @testset "PoolEscapeError carries variable names, points, and formatted message" begin
        # Single variable: bare return
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:v]
        @test !isempty(err.points)
        @test :v in err.points[1].vars
        msg = sprint(showerror, err)
        @test occursin("collect(v)", msg)
        @test occursin("False positive?", msg)
        @test occursin("Escaping return", msg)

        # Different variable name
        err = try @macroexpand(@with_pool pool begin
            data = zeros!(pool, 10)
            data
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:data]

        # Function form
        err = try @macroexpand(@with_pool pool function msg_fn(n)
            result = acquire!(pool, Float64, n)
            result
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:result]

        # Container: only w escapes, not sum(v)
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            (sum(v), w)
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:w]
        @test :v ∉ err.vars

        # Multi-variable: both appear, sorted
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            (v, w)
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:v, :w]
        msg = sprint(showerror, err)
        @test occursin("collect(v)", msg)
        @test occursin("collect(w)", msg)

        # Source location captured
        @test err.file !== nothing

        # Branch escapes: points track per-return-point info
        err = try @macroexpand(@with_pool pool function branch_msg(n)
            v = acquire!(pool, Float64, n)
            if n > 0
                return sum(v)
            else
                return v
            end
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:v]
        @test length(err.points) == 1  # only the unsafe return
        @test :v in err.points[1].vars
        @test err.points[1].line !== nothing

        # Multiple escape points across branches
        err = try @macroexpand(@with_pool pool function multi_pt(n)
            v = acquire!(pool, Float64, n)
            w = acquire!(pool, Float64, n)
            if n > 0
                return v
            else
                return w
            end
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:v, :w]
        @test length(err.points) == 2
        msg = sprint(showerror, err)
        @test occursin("[1]", msg)
        @test occursin("[2]", msg)

        # Rendered expression shows highlighted vars
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            z = similar!(pool, v)
            if rand() > 0.5
                return (v = 0.5, data = 1.0)
            else
                return (v = v, data = z)
            end
        end) catch e; e end
        @test err isa PoolEscapeError
        @test err.vars == [:v, :z]
        msg = sprint(showerror, err)
        # The rendered return expression appears in the message
        @test occursin("return", msg)
        @test occursin("data", msg)
    end

    # ==============================================================================
    # Variable classification: view vs array vs bitarray vs container vs alias
    # ==============================================================================

    @testset "_acquire_call_kind classification" begin
        # View-returning functions
        @test _acquire_call_kind(:(acquire!(pool, Float64, 10)), :pool) === :pool_view
        @test _acquire_call_kind(:(zeros!(pool, 10)), :pool) === :pool_view
        @test _acquire_call_kind(:(ones!(pool, Int64, 3)), :pool) === :pool_view
        @test _acquire_call_kind(:(similar!(pool, arr)), :pool) === :pool_view
        @test _acquire_call_kind(:(reshape!(pool, arr, 3, 4)), :pool) === :pool_view

        # Array-returning functions (unsafe_wrap)
        @test _acquire_call_kind(:(unsafe_acquire!(pool, Float64, 10)), :pool) === :pool_array
        @test _acquire_call_kind(:(acquire_array!(pool, Float64, 10)), :pool) === :pool_array

        # BitArray-returning functions
        @test _acquire_call_kind(:(trues!(pool, 100)), :pool) === :pool_bitarray
        @test _acquire_call_kind(:(falses!(pool, 50)), :pool) === :pool_bitarray

        # Non-acquire → nothing
        @test _acquire_call_kind(:(sum(data)), :pool) === nothing
        @test _acquire_call_kind(:(rand(10)), :pool) === nothing

        # Wrong pool → nothing
        @test _acquire_call_kind(:(acquire!(other_pool, Float64, 10)), :pool) === nothing
    end

    @testset "var_info classification in PoolEscapeError" begin
        # Direct pool view
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v
        end) catch e; e end
        @test err.var_info[:v] == (:pool_view, Symbol[])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired view", msg)

        # Direct pool array (unsafe_acquire!)
        err = try @macroexpand(@with_pool pool begin
            v = unsafe_acquire!(pool, Float64, 10)
            v
        end) catch e; e end
        @test err.var_info[:v] == (:pool_array, Symbol[])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired array", msg)

        # Direct pool BitArray
        err = try @macroexpand(@with_pool pool begin
            bv = trues!(pool, 100)
            bv
        end) catch e; e end
        @test err.var_info[:bv] == (:pool_bitarray, Symbol[])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired BitArray", msg)

        # Container wrapping pool variable
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            a = [v, 1]
            a
        end) catch e; e end
        @test err.var_info[:a] == (:container, [:v])
        msg = sprint(showerror, err)
        @test occursin("wraps pool variable (v)", msg)
        # Fix suggests collect(v), not collect(a)
        @test occursin("collect(v)", msg)
        @test occursin("Copy pool variables before wrapping", msg)

        # Container with multiple pool vars
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            a = [v, w]
            a
        end) catch e; e end
        @test err.var_info[:a] == (:container, [:v, :w])
        msg = sprint(showerror, err)
        @test occursin("wraps pool variables (v, w)", msg)

        # Alias of pool variable
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            d = v
            d
        end) catch e; e end
        @test err.var_info[:d] == (:alias, [:v])
        msg = sprint(showerror, err)
        @test occursin("alias of pool variable (v)", msg)

        # Mixed: direct pool var + container in same return
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            a = [v, 1]
            return (v, a)
        end) catch e; e end
        @test err.var_info[:v] == (:pool_view, Symbol[])
        @test err.var_info[:a] == (:container, [:v])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired view", msg)
        @test occursin("wraps pool variable (v)", msg)
        # Fix section deduplicates: only collect(v), not collect(a)
        @test occursin("collect(v)", msg)
        @test !occursin("collect(a)", msg)

        # zeros! classified as view
        err = try @macroexpand(@with_pool pool begin
            data = zeros!(pool, 10)
            data
        end) catch e; e end
        @test err.var_info[:data] == (:pool_view, Symbol[])

        # Tuple container
        err = try @macroexpand(@with_pool pool begin
            v = acquire!(pool, Float64, 10)
            t = (v, 42)
            t
        end) catch e; e end
        @test err.var_info[:t] == (:container, [:v])
    end

end # Compile-Time Escape Detection
