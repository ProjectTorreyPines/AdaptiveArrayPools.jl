import AdaptiveArrayPools: _extract_acquired_vars, _get_last_expression,
    _find_direct_exposure, _check_compile_time_escape,
    _collect_all_return_values, _collect_explicit_returns!, _collect_implicit_return_values!,
    _get_last_expression_with_line, _render_return_expr,
    _acquire_call_kind, _classify_escaped_vars, _is_acquire_call,
    DeclarationSite, _extract_declaration_sites,
    _format_location_str, _format_point_location,
    _find_acquire_call_expr, _literal_contains_acquired,
    _collect_acquired_in_literal,
    _find_first_lnn_index, _ensure_body_has_toplevel_lnn,
    _extract_ordered_acquired,
    _find_reassign_maybe_tainted, _is_safe_copy_call, _rhs_call_contains_sym,
    _extract_container_vars,
    _is_dotted_assign_head, _incidental_exposure, _lint_message, ESCAPE_LINT,
    _report_incidental_escape, PoolEscapeError,
    _poison_block_tails, EscapedPoolArray, EscapedPoolUseError

# Helper: expansion must throw PoolEscapeError (possibly LoadError-wrapped) —
# the severity of function-form and explicit-return escapes.
function _expansion_escape_error(ex)
    try
        macroexpand(@__MODULE__, ex)
        return false
    catch err
        err isa LoadError && (err = err.error)
        return err isa AdaptiveArrayPools.PoolEscapeError
    end
end

# Helper: block-form expansion succeeds AND emits the escape @warn — the
# default severity of block-form implicit tails (the escaping value is
# additionally replaced by an EscapedPoolArray guard in the expansion).
function _expansion_incidental_warns(ex)
    expanded = @test_logs (:warn, r"becomes the scope's return value") match_mode = :any macroexpand(
        @__MODULE__, ex
    )
    return expanded isa Expr
end

function _capture_stderr(f)
    tmpf = tempname()
    open(tmpf, "w") do io
        redirect_stderr(io) do
            f()
        end
    end
    output = read(tmpf, String)
    rm(tmpf; force = true)
    return output
end

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
        @test _get_last_expression(
            quote
                a = 1
                b = 2
                c
            end
        ) == :c

        # Block ending with LineNumberNode → skip it
        block = Expr(:block, :a, LineNumberNode(1, :test))
        @test _get_last_expression(block) == :a

        # Non-block expression
        @test _get_last_expression(:v) == :v
        @test _get_last_expression(42) == 42

        # Nested block
        @test _get_last_expression(
            quote
                begin
                    x
                end
            end
        ) == :x

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
    # _check_compile_time_escape: integration tests
    # ==============================================================================

    @testset "_check_compile_time_escape" begin
        src = LineNumberNode(1, :test)
        # Detection tests below run with `function_form = true` (the always-
        # error severity) so a throw signature pins the DETECTION logic; the
        # block-form default severity (warn + guard) is covered separately.

        # Bare variable return → error "Pool escape"
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            :pool, src; function_form = true
        )

        # Tuple containing acquired var → error
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                w = acquire!(pool, Float64, 5)
                (sum(v), w)
            end,
            :pool, src; function_form = true
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
            :pool, src; function_form = true
        )

        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = ones!(pool, Float32, 10)
                v
            end,
            :pool, src; function_form = true
        )

        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = similar!(pool, some_array)
                v
            end,
            :pool, src; function_form = true
        )

        # trues!/falses! also detected
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                bv = trues!(pool, 100)
                bv
            end,
            :pool, src; function_form = true
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
            :pool, nothing; function_form = true
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

    @testset "_check_compile_time_escape: block-form implicit-tail severity" begin
        src = LineNumberNode(1, :test)
        # Under the default escape_lint = "warn", a block-form implicit tail
        # WARNS instead of throwing (form-based severity). The guard rewrite
        # itself happens in block codegen (`_poison_block_tails`), not here.
        @test_logs (:warn, r"pool-backed array `v` itself") _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            :pool, src
        )
        @test_logs (:warn, r"container literal") _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                (sum(v), v)
            end,
            :pool, src
        )
        # Explicit `return` stays an error even in block-form checking
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                return v
            end,
            :pool, src
        )
    end

    # ==============================================================================
    # _collect_all_return_values: explicit returns + implicit if/else branches
    # ==============================================================================

    @testset "_collect_all_return_values (expr, line) pairs" begin
        # Simple block: implicit return only
        vals = _collect_all_return_values(
            quote
                x = 1
                x
            end
        )
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :x, exprs)

        # Explicit return inside if branch
        vals = _collect_all_return_values(
            quote
                v = acquire!(pool, Float64, 10)
                if cond
                    return v
                end
                sum(v)
            end
        )
        exprs = first.(vals)
        @test any(e -> e isa Expr && e.head == :return, exprs)
        @test any(e -> e isa Expr && e.head == :call, exprs)

        # Both branches have explicit return
        vals = _collect_all_return_values(
            quote
                if a > 0.5
                    return (v = 0.5, data = a)
                else
                    return (v = v, data = z)
                end
            end
        )
        returns = filter(((e, _),) -> e isa Expr && e.head == :return, vals)
        @test length(returns) >= 2

        # Implicit return from if/else branches (no explicit return keyword)
        vals = _collect_all_return_values(
            quote
                if cond
                    v
                else
                    sum(v)
                end
            end
        )
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)
        @test any(e -> e isa Expr && e.head == :call, exprs)

        # elseif branches
        vals = _collect_all_return_values(
            quote
                if cond1
                    v
                elseif cond2
                    w
                else
                    sum(v)
                end
            end
        )
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)
        @test any(e -> e isa Symbol && e == :w, exprs)

        # Nested if inside branch
        vals = _collect_all_return_values(
            quote
                if outer
                    if inner
                        v
                    else
                        w
                    end
                else
                    sum(v)
                end
            end
        )
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)
        @test any(e -> e isa Symbol && e == :w, exprs)

        # Skips nested function definitions
        vals = _collect_all_return_values(
            quote
                f = function ()
                    return v  # belongs to inner function, not our scope
                end
                sum(v)
            end
        )
        returns = filter(((e, _),) -> e isa Expr && e.head == :return, vals)
        @test isempty(returns)

        # Ternary operator (same AST as if/else)
        vals = _collect_all_return_values(
            quote
                cond ? v : sum(v)
            end
        )
        exprs = first.(vals)
        @test any(e -> e isa Symbol && e == :v, exprs)

        # Line numbers are tracked
        vals = _collect_all_return_values(
            quote
                if cond
                    return v  # this will have a line number
                end
                sum(v)
            end
        )
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

        # Implicit return from if/else branches — caught (function-form severity)
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                if cond
                    sum(v)
                else
                    v
                end
            end,
            :pool, src; function_form = true
        )

        # elseif branch — caught (function-form severity)
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
            :pool, src; function_form = true
        )

        # Ternary with escape — caught (function-form severity)
        @test_throws PoolEscapeError _check_compile_time_escape(
            quote
                v = acquire!(pool, Float64, 10)
                cond ? v : sum(v)
            end,
            :pool, src; function_form = true
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
        err = try
            _check_compile_time_escape(
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
            )
        catch e
            e
        end
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

        # Ternary — block form: warns + guard (implicit tail)
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    rand() > 0.5 ? sum(v) : v
                end
            )
        )

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
        # Bare variable, block form: warns + guard (implicit tail)
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        )

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

    # ==============================================================================
    # Scope shadowing: inner scope variables must not taint outer scope
    # ==============================================================================

    @testset "Scope shadowing: no false positives from inner scopes" begin
        # IMPORTANT: `let` is NOT skipped — `return` inside `let` exits the
        # enclosing function, so pool variables acquired there CAN escape.
        # This means `let v = acquire!(...)` with outer `v` IS a false positive,
        # but that's the safer trade-off vs missing real escapes.

        # return inside let MUST be caught (real escape)
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            let
                w = acquire!(pool, Float64, 10)
                return w
            end
        end

        # Anonymous function: inner v is a different scope
        @test_nowarn @macroexpand @with_pool pool begin
            v = 1
            f = (v -> sum(acquire!(pool, Float64, v)))
            f(10)
            v
        end

        # Nested function definition: inner v is scoped
        @test_nowarn @macroexpand @with_pool pool begin
            v = "hello"
            function helper()
                v = acquire!(pool, Float64, 5)
                sum(v)
            end
            helper()
            v
        end

        # do block: inner v is scoped
        @test_nowarn @macroexpand @with_pool pool begin
            v = 0
            map(1:3) do v
                w = acquire!(pool, Float64, v)
                sum(w)
            end
            v
        end

        # IMPORTANT: acquire in if/for/while body IS outer scope (no new scope)
        # These are still detected — block-form implicit tail → warn + guard
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    if true
                        v = acquire!(pool, Float64, 10)
                    end
                    v
                end
            )
        )
    end

    @testset "Block form: additional escape scenarios (warn + guard)" begin
        # Detection coverage across the acquire family and literal shapes.
        # Block-form implicit tails → warn (the escaping value is guarded);
        # explicit `return`s remain hard errors.

        # zeros! — detected
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = zeros!(pool, 10)
                    v
                end
            )
        )

        # trues! — detected
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    bv = trues!(pool, 100)
                    bv
                end
            )
        )

        # Explicit return — definite escape, still an error
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            return v
        end

        # Tuple with acquired var — detected
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    (v, 42)
                end
            )
        )

        # Array literal — detected
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    [v, nothing]
                end
            )
        )

        # return (v, scalar) → definite escape, still an error
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            return (v, sum(v))
        end

        # Re-acquire reassignment: v still tracked after v = zeros!(pool, ...)
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    v = zeros!(pool, 20)
                    v
                end
            )
        )

        # NamedTuple with acquired var as VALUE — detected
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    (result = v, n = 42)
                end
            )
        )

        # NamedTuple shorthand (v = v) → value IS acquired — detected
        # (key name coincidentally matches, but VALUE is the acquired var)
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    (v = v,)
                end
            )
        )

        # Destructuring with acquire RHS: v still tracked — detected
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    (v, w) = (acquire!(pool, Float64, 10), safe())
                    v
                end
            )
        )

        # Destructuring doesn't protect if RHS element IS acquire
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    (v, w) = (zeros!(pool, 5), safe())
                    v
                end
            )
        )
    end

    # ==============================================================================
    # Order-aware taint tracking (_extract_ordered_acquired)
    # ==============================================================================

    @testset "_extract_ordered_acquired: statement ordering" begin
        # Pre-acquire non-pool assignment must NOT clear taint
        vars = _extract_ordered_acquired(
            quote
                v = 1
                v = acquire!(pool, Float64, 3, 3)
            end, :pool,
        )
        @test :v in vars

        # Post-acquire non-pool assignment correctly clears taint
        vars = _extract_ordered_acquired(
            quote
                v = acquire!(pool, Float64, 3, 3)
                v = 1
            end, :pool,
        )
        @test :v ∉ vars

        # Pre-acquire assignment on alias chain must NOT clear taint
        vars = _extract_ordered_acquired(
            quote
                v = acquire!(pool, Float64, 3, 3)
                v2 = 1
                v2 = v
            end, :pool,
        )
        @test :v in vars
        @test :v2 in vars

        # Swap pattern: v becomes safe, x becomes tainted
        vars = _extract_ordered_acquired(
            quote
                v = acquire!(pool, Float64, 10)
                x = zeros(10)
                v, x = x, v
            end, :pool,
        )
        @test :v ∉ vars
        @test :x in vars

        # Container wrapping after pre-acquire assignment
        vars = _extract_ordered_acquired(
            quote
                v = 1
                v = acquire!(pool, Float64, 3, 3)
                wrapper = (v,)
            end, :pool,
        )
        @test :v in vars
        @test :wrapper in vars

        # Re-acquire after non-pool assignment
        vars = _extract_ordered_acquired(
            quote
                v = acquire!(pool, Float64, 10)
                v = 1
                v = acquire!(pool, Float64, 5)
            end, :pool,
        )
        @test :v in vars
    end

    # ==============================================================================
    # Order-aware escape detection through macro pipeline
    # ==============================================================================

    @testset "Order-aware escape detection (false negative fixes)" begin
        # BUG FIX: pre-acquire assignment used to suppress escape detection
        # v=1 before v=acquire!(...) must NOT make v safe
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = 1
            v = acquire!(pool, Float64, 3, 3)
            wrapper = (v,)
            return wrapper
        end

        # BUG FIX: same pattern in function form
        @test_throws PoolEscapeError @macroexpand @with_pool pool function test()
            v = 1
            v = acquire!(pool, Float64, 3, 3)
            wrapper = (v,)
            return wrapper
        end

        # BUG FIX: alias with pre-assignment on the alias variable
        @test_throws PoolEscapeError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 3, 3)
            v2 = 1
            v2 = v
            wrapper = [v2]
            return wrapper
        end

        # BUG FIX: same in function form
        @test_throws PoolEscapeError @macroexpand @with_pool pool function test()
            v = acquire!(pool, Float64, 3, 3)
            v2 = 1
            v2 = v
            wrapper = [v2]
            return wrapper
        end

        # Bare variable with pre-assignment — still detected (block → warn)
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = 1
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        )

        # Control: post-acquire reassignment IS safe (no regression)
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v = 1
            v
        end

        # Control: post-acquire collect IS safe (no regression)
        @test_nowarn @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v = collect(v)
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
        # Implicit tail escape → warn + guard (block form)
        @test _expansion_incidental_warns(
            :(
                @maybe_with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        )

        # Container escape → warn + guard (block form)
        @test _expansion_incidental_warns(
            :(
                @maybe_with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    (v, sum(v))
                end
            )
        )

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
        @test _expansion_incidental_warns(
            :(
                @with_pool :cpu pool begin
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        )

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
        # Block — warn + guard
        @test _expansion_incidental_warns(
            :(
                @maybe_with_pool :cpu pool begin
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        )

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

        # Outer scope escape → warn + guard from outer macro check (block form)
        @test _expansion_incidental_warns(
            :(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    @with_pool pool begin
                        w = acquire!(pool, Float64, 5)
                        w .= 1.0
                        nothing
                    end
                    v  # ← outer implicit-tail escape
                end
            )
        )
    end

    # ==============================================================================
    # Error/warning messages: verify variable names and suggestions
    # ==============================================================================

    @testset "PoolEscapeError carries variable names, points, and formatted message" begin
        # Subjects use FUNCTION form: block-form implicit tails warn + guard
        # under the default severity, so the rich PoolEscapeError rendering is
        # exercised via the always-error form.
        # Single variable: bare return
        err = try
            @macroexpand(
                @with_pool pool function msg_bare_fn()
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:v]
        @test !isempty(err.points)
        @test :v in err.points[1].vars
        msg = sprint(showerror, err)
        @test occursin("collect(v)", msg)
        @test occursin("False positive?", msg)
        @test occursin("Escaping return", msg)
        @test occursin("escapes the @with_pool scope", msg)
        # Declaration sites populated
        @test !isempty(err.declarations)
        @test any(d -> d.var === :v, err.declarations)
        @test occursin("Declarations:", msg)
        @test occursin("acquire!(pool, Float64, 10)", msg)

        # Different variable name
        err = try
            @macroexpand(
                @with_pool pool function msg_data_fn()
                    data = zeros!(pool, 10)
                    data
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:data]
        @test any(d -> d.var === :data, err.declarations)

        # Function form
        err = try
            @macroexpand(
                @with_pool pool function msg_fn(n)
                    result = acquire!(pool, Float64, n)
                    result
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:result]

        # Container: only w escapes, not sum(v)
        err = try
            @macroexpand(
                @with_pool pool function msg_container_fn()
                    v = acquire!(pool, Float64, 10)
                    w = acquire!(pool, Float64, 5)
                    (sum(v), w)
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:w]
        @test :v ∉ err.vars

        # Multi-variable: both appear, sorted
        err = try
            @macroexpand(
                @with_pool pool function msg_multi_fn()
                    v = acquire!(pool, Float64, 10)
                    w = acquire!(pool, Float64, 5)
                    (v, w)
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:v, :w]
        msg = sprint(showerror, err)
        @test occursin("collect(v)", msg)
        @test occursin("collect(w)", msg)
        # Multi-variable: both declarations present
        @test length(err.declarations) == 2
        @test err.declarations[1].var === :v
        @test err.declarations[2].var === :w
        @test occursin("2 variables escape", msg)

        # Source location captured
        @test err.file !== nothing

        # Branch escapes: points track per-return-point info
        err = try
            @macroexpand(
                @with_pool pool function branch_msg(n)
                    v = acquire!(pool, Float64, n)
                    if n > 0
                        return sum(v)
                    else
                        return v
                    end
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:v]
        @test length(err.points) == 1  # only the unsafe return
        @test :v in err.points[1].vars
        @test err.points[1].line !== nothing

        # Multiple escape points across branches
        err = try
            @macroexpand(
                @with_pool pool function multi_pt(n)
                    v = acquire!(pool, Float64, n)
                    w = acquire!(pool, Float64, n)
                    if n > 0
                        return v
                    else
                        return w
                    end
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        @test err.vars == [:v, :w]
        @test length(err.points) == 2
        msg = sprint(showerror, err)
        @test occursin("[1]", msg)
        @test occursin("[2]", msg)

        # Rendered expression shows highlighted vars
        err = try
            @macroexpand(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    z = similar!(pool, v)
                    if rand() > 0.5
                        return (v = 0.5, data = 1.0)
                    else
                        return (v = v, data = z)
                    end
                end
            )
        catch e
            e
        end
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
        @test _acquire_call_kind(:(acquire_view!(pool, Float64, 10)), :pool) === :pool_view

        # Array-returning functions
        @test _acquire_call_kind(:(acquire!(pool, Float64, 10)), :pool) === :pool_array
        @test _acquire_call_kind(:(acquire_array!(pool, Float64, 10)), :pool) === :pool_array
        @test _acquire_call_kind(:(zeros!(pool, 10)), :pool) === :pool_array
        @test _acquire_call_kind(:(ones!(pool, Int64, 3)), :pool) === :pool_array
        @test _acquire_call_kind(:(similar!(pool, arr)), :pool) === :pool_array
        @test _acquire_call_kind(:(reshape!(pool, arr, 3, 4)), :pool) === :pool_array

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
        # Subjects use FUNCTION form (always-error) so the classification is
        # exercised through a thrown PoolEscapeError; block-form implicit
        # tails warn + guard under the default severity instead.
        # Direct pool array (acquire! now returns Array)
        err = try
            @macroexpand(
                @with_pool pool function vi_array_fn()
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        catch e
            e
        end
        @test err.var_info[:v] == (:pool_array, Symbol[])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired array", msg)

        # Direct pool view (acquire_view!)
        err = try
            @macroexpand(
                @with_pool pool function vi_view_fn()
                    v = acquire_view!(pool, Float64, 10)
                    v
                end
            )
        catch e
            e
        end
        @test err.var_info[:v] == (:pool_view, Symbol[])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired view", msg)

        # Direct pool BitArray
        err = try
            @macroexpand(
                @with_pool pool function vi_bit_fn()
                    bv = trues!(pool, 100)
                    bv
                end
            )
        catch e
            e
        end
        @test err.var_info[:bv] == (:pool_bitarray, Symbol[])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired BitArray", msg)

        # Container wrapping pool variable
        err = try
            @macroexpand(
                @with_pool pool function vi_container_fn()
                    v = acquire!(pool, Float64, 10)
                    a = [v, 1]
                    a
                end
            )
        catch e
            e
        end
        @test err.var_info[:a] == (:container, [:v])
        msg = sprint(showerror, err)
        @test occursin("wraps pool variable (v)", msg)
        # Fix suggests collect(v), not collect(a)
        @test occursin("collect(v)", msg)
        @test occursin("Copy pool variables before wrapping", msg)

        # Container with multiple pool vars
        err = try
            @macroexpand(
                @with_pool pool function vi_container2_fn()
                    v = acquire!(pool, Float64, 10)
                    w = acquire!(pool, Float64, 5)
                    a = [v, w]
                    a
                end
            )
        catch e
            e
        end
        @test err.var_info[:a] == (:container, [:v, :w])
        msg = sprint(showerror, err)
        @test occursin("wraps pool variables (v, w)", msg)

        # Alias of pool variable
        err = try
            @macroexpand(
                @with_pool pool function vi_alias_fn()
                    v = acquire!(pool, Float64, 10)
                    d = v
                    d
                end
            )
        catch e
            e
        end
        @test err.var_info[:d] == (:alias, [:v])
        msg = sprint(showerror, err)
        @test occursin("alias of pool variable (v)", msg)

        # Mixed: direct pool var + container in same return
        err = try
            @macroexpand(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    a = [v, 1]
                    return (v, a)
                end
            )
        catch e
            e
        end
        @test err.var_info[:v] == (:pool_array, Symbol[])
        @test err.var_info[:a] == (:container, [:v])
        msg = sprint(showerror, err)
        @test occursin("pool-acquired array", msg)
        @test occursin("wraps pool variable (v)", msg)
        # Fix section deduplicates: only collect(v), not collect(a)
        @test occursin("collect(v)", msg)
        @test !occursin("collect(a)", msg)

        # zeros! classified as array
        err = try
            @macroexpand(
                @with_pool pool function vi_zeros_fn()
                    data = zeros!(pool, 10)
                    data
                end
            )
        catch e
            e
        end
        @test err.var_info[:data] == (:pool_array, Symbol[])

        # Tuple container
        err = try
            @macroexpand(
                @with_pool pool function vi_tuple_fn()
                    v = acquire!(pool, Float64, 10)
                    t = (v, 42)
                    t
                end
            )
        catch e
            e
        end
        @test err.var_info[:t] == (:container, [:v])
    end

    # ==============================================================================
    # Declaration site extraction
    # ==============================================================================

    @testset "_extract_declaration_sites" begin
        # Single acquire: captures var, expr, and line
        sites = _extract_declaration_sites(
            quote
                v = acquire!(pool, Float64, 10)
                v
            end,
            Set([:v])
        )
        @test length(sites) == 1
        @test sites[1].var === :v
        @test sites[1].line !== nothing
        @test string(sites[1].expr) == "v = acquire!(pool, Float64, 10)"

        # Multiple declarations sorted by line
        sites = _extract_declaration_sites(
            quote
                v = acquire!(pool, Float64, 10)
                w = zeros!(pool, 5)
                (v, w)
            end,
            Set([:v, :w])
        )
        @test length(sites) == 2
        @test sites[1].var === :v
        @test sites[2].var === :w
        @test sites[1].line < sites[2].line

        # Container declaration captured
        sites = _extract_declaration_sites(
            quote
                v = acquire!(pool, Float64, 10)
                a = [v, 1]
                a
            end,
            Set([:v, :a])
        )
        @test length(sites) == 2
        @test sites[1].var === :v
        @test sites[2].var === :a

        # Alias declaration captured
        sites = _extract_declaration_sites(
            quote
                v = acquire!(pool, Float64, 10)
                d = v
                d
            end,
            Set([:v, :d])
        )
        @test length(sites) == 2
        @test sites[1].var === :v
        @test sites[2].var === :d

        # Only escaped vars captured (non-escaped ignored)
        sites = _extract_declaration_sites(
            quote
                v = acquire!(pool, Float64, 10)
                w = acquire!(pool, Float64, 5)
                v
            end,
            Set([:v])  # only v escapes
        )
        @test length(sites) == 1
        @test sites[1].var === :v
    end

    # ==============================================================================
    # Formatted message: declarations and escape points with locations
    # ==============================================================================

    @testset "showerror shows declarations and escape locations" begin
        # Container: declarations show both v and a
        err = try
            @macroexpand(
                @with_pool pool begin
                    v = acquire!(pool, Float64, 10)
                    a = [v, 1]
                    return (v, a)
                end
            )
        catch e
            e
        end
        msg = sprint(showerror, err)
        @test occursin("Declarations:", msg)
        @test occursin("Escaping return:", msg)
        @test occursin("acquire!(pool, Float64, 10)", msg)
        @test occursin("[v, 1]", msg)
        @test occursin("return", msg)
    end

    # ==============================================================================
    # Coverage: PoolEscapeError convenience constructors
    # ==============================================================================

    @testset "PoolEscapeError convenience constructors" begin
        pt = EscapePoint(:v, 1, [:v])

        # 4-arg constructor (no var_info, no declarations)
        err4 = PoolEscapeError([:v], "test.jl", 1, [pt])
        @test isempty(err4.var_info)
        @test isempty(err4.declarations)

        # 5-arg constructor (no declarations)
        vi = Dict{Symbol, Tuple{Symbol, Vector{Symbol}}}(:v => (:pool_view, Symbol[]))
        err5 = PoolEscapeError([:v], "test.jl", 1, [pt], vi)
        @test err5.var_info[:v][1] === :pool_view
        @test isempty(err5.declarations)
    end

    # ==============================================================================
    # Coverage: _render_return_expr branches
    # ==============================================================================

    @testset "_render_return_expr branch coverage" begin
        escaped = Set([:v])

        # Non-escaped symbol → plain print (line 54)
        buf = sprint() do io
            _render_return_expr(io, :x, escaped)
        end
        @test buf == "x"

        # Array literal :vect (lines 71-77)
        buf = sprint() do io
            _render_return_expr(io, Expr(:vect, :v, :x), escaped)
        end
        @test occursin("[", buf)
        @test occursin("]", buf)
        @test occursin("x", buf)

        # Fallback Expr (line 79) — e.g. a :ref expression
        buf = sprint() do io
            _render_return_expr(io, :(v[1]), escaped)
        end
        @test occursin("v", buf)

        # Non-Expr, non-Symbol literal (line 82)
        buf = sprint() do io
            _render_return_expr(io, 42, escaped)
        end
        @test buf == "42"
    end

    # ==============================================================================
    # Coverage: showerror with unknown var_info kind (line 121)
    # ==============================================================================

    @testset "showerror pool-backed temporary fallback" begin
        vi = Dict{Symbol, Tuple{Symbol, Vector{Symbol}}}(:v => (:something_else, Symbol[]))
        err = PoolEscapeError(
            [:v], "test.jl", 1,
            [EscapePoint(:v, 1, [:v])],
            vi, DeclarationSite[]
        )
        msg = sprint(showerror, err)
        @test occursin("pool-backed temporary", msg)
    end

    # ==============================================================================
    # Coverage: _format_location_str / _format_point_location edge cases
    # ==============================================================================

    @testset "location formatting edge cases" begin
        # file="none", line present → "line N" (lines 203-204)
        @test _format_location_str("none", 42) == "line 42"
        # file=nothing, line=nothing → nothing (line 206)
        @test _format_location_str(nothing, nothing) === nothing
        # file=nothing, line present → "line N"
        @test _format_location_str(nothing, 7) == "line 7"

        # Same for _format_point_location (lines 214-217)
        @test _format_point_location("none", 42) == "line 42"
        @test _format_point_location(nothing, nothing) === nothing
        @test _format_point_location(nothing, 7) == "line 7"
    end

    # ==============================================================================
    # Coverage: showerror 3-arg (backtrace suppression, line 221)
    # ==============================================================================

    @testset "showerror backtrace suppression" begin
        err = try
            @macroexpand(
                @with_pool pool function bts_fn()
                    v = acquire!(pool, Float64, 10)
                    v
                end
            )
        catch e
            e
        end
        @test err isa PoolEscapeError
        # 3-arg showerror should produce same output as 2-arg
        msg2 = sprint(showerror, err)
        msg3 = sprint() do io
            showerror(io, err, nothing)
        end
        @test msg2 == msg3
    end

    # ==============================================================================
    # Coverage: _is_acquire_call / _acquire_call_kind with qualified names
    # ==============================================================================

    @testset "qualified acquire call detection" begin
        # Module.acquire!(pool, ...) — qualified name (lines 1703-1705)
        @test _is_acquire_call(
            :(AdaptiveArrayPools.acquire!(pool, Float64, 10)), :pool
        )
        @test _is_acquire_call(
            :(SomeModule.zeros!(pool, 10)), :pool
        )
        # Non-acquire qualified call
        @test !_is_acquire_call(
            :(Base.sum(pool, data)), :pool
        )

        # _acquire_call_kind with qualified names
        @test _acquire_call_kind(
            :(M.acquire!(pool, Float64, 10)), :pool
        ) === :pool_array
        @test _acquire_call_kind(
            :(M.acquire_view!(pool, Float64, 10)), :pool
        ) === :pool_view
        @test _acquire_call_kind(
            :(M.trues!(pool, 10)), :pool
        ) === :pool_bitarray

        # Qualified non-acquire → nothing (line 1739)
        @test _acquire_call_kind(
            :(M.sum(pool, data)), :pool
        ) === nothing
    end

    # ==============================================================================
    # Coverage: _find_acquire_call_expr (lines 1458-1467)
    # ==============================================================================

    @testset "_find_acquire_call_expr" begin
        # Direct acquire call
        expr = :(acquire!(pool, Float64, 10))
        @test _find_acquire_call_expr(expr, :pool) === expr

        # Nested inside assignment
        outer = :(v = acquire!(pool, Float64, 10))
        result = _find_acquire_call_expr(outer, :pool)
        @test result !== nothing
        @test result.args[1] === :acquire!

        # No acquire call → nothing
        @test _find_acquire_call_expr(:(sum(data)), :pool) === nothing

        # Non-Expr → nothing
        @test _find_acquire_call_expr(:x, :pool) === nothing
        @test _find_acquire_call_expr(42, :pool) === nothing
    end

    # ==============================================================================
    # Coverage: _literal_contains_acquired — identity / named tuple / kw
    # ==============================================================================

    @testset "_literal_contains_acquired edge cases" begin
        acquired = Set([:v, :w])

        # identity(v) → detected (line 1959)
        @test _literal_contains_acquired(:(identity(v)), acquired)

        # identity(x) → not detected
        @test !_literal_contains_acquired(:(identity(x)), acquired)

        # Named tuple with = syntax: (a=v,) (line 1963-1964)
        @test _literal_contains_acquired(
            Expr(:tuple, Expr(:(=), :a, :v)), acquired
        )

        # Named tuple with kw syntax (line 1965-1966)
        @test _literal_contains_acquired(
            Expr(:tuple, Expr(:kw, :a, :v)), acquired
        )

        # Non-acquired kw
        @test !_literal_contains_acquired(
            Expr(:tuple, Expr(:kw, :a, :x)), acquired
        )
    end

    # ==============================================================================
    # Coverage: _collect_acquired_in_literal — identity / kw (lines 2031-2037)
    # ==============================================================================

    @testset "_collect_acquired_in_literal edge cases" begin
        acquired = Set([:v, :w])

        # identity(v)
        found = _collect_acquired_in_literal(:(identity(v)), acquired)
        @test :v in found

        # Named tuple (a=v,)
        found = _collect_acquired_in_literal(
            Expr(:tuple, Expr(:(=), :a, :v)), acquired
        )
        @test :v in found

        # kw syntax
        found = _collect_acquired_in_literal(
            Expr(:tuple, Expr(:kw, :a, :w)), acquired
        )
        @test :w in found

        # Non-Expr/non-Symbol → empty
        found = _collect_acquired_in_literal(42, acquired)
        @test isempty(found)
    end

    # ==============================================================================
    # Coverage: _find_direct_exposure — identity (line 2012)
    # ==============================================================================

    @testset "_find_direct_exposure identity" begin
        acquired = Set([:v])

        # identity(v) → detected
        found = _find_direct_exposure(:(identity(v)), acquired)
        @test :v in found

        # Base.identity(v)
        found = _find_direct_exposure(:(Base.identity(v)), acquired)
        @test :v in found
    end

    # ==============================================================================
    # Coverage: _find_first_lnn_index / _ensure_body_has_toplevel_lnn
    # ==============================================================================

    @testset "LNN handling edge cases" begin
        # _find_first_lnn_index: :meta then LNN (lines 434-435)
        args = Any[Expr(:meta, :inline), LineNumberNode(1, :test)]
        @test _find_first_lnn_index(args) == 2

        # _find_first_lnn_index: non-meta before LNN → nothing (lines 437-440)
        args2 = Any[:(x = 1), LineNumberNode(1, :test)]
        @test _find_first_lnn_index(args2) === nothing

        # _find_first_lnn_index: empty → nothing
        @test _find_first_lnn_index(Any[]) === nothing

        # _ensure_body_has_toplevel_lnn: source=nothing → identity
        body = Expr(:block, :(x = 1))
        @test _ensure_body_has_toplevel_lnn(body, nothing) === body

        # source.file=:none → identity (line 458)
        @test _ensure_body_has_toplevel_lnn(body, LineNumberNode(1, :none)) === body

        # LNN already points to user file → identity (line 467)
        body_with_lnn = Expr(:block, LineNumberNode(5, :myfile), :(x = 1))
        result = _ensure_body_has_toplevel_lnn(body_with_lnn, LineNumberNode(5, :myfile))
        @test result === body_with_lnn

        # LNN points elsewhere → replaced (line 470-472)
        body_wrong_lnn = Expr(:block, LineNumberNode(1, :macros), :(x = 1))
        result = _ensure_body_has_toplevel_lnn(body_wrong_lnn, LineNumberNode(10, :user))
        @test result.args[1] isa LineNumberNode
        @test result.args[1].file === :user

        # No LNN in block → prepend (lines 476)
        body_no_lnn = Expr(:block, :(x = 1))
        result = _ensure_body_has_toplevel_lnn(body_no_lnn, LineNumberNode(3, :src))
        @test result.args[1] isa LineNumberNode
        @test result.args[1].file === :src

        # Empty block (lines 477-479)
        empty_block = Expr(:block)
        result = _ensure_body_has_toplevel_lnn(empty_block, LineNumberNode(1, :f))
        @test length(result.args) == 1
        @test result.args[1] isa LineNumberNode

        # Non-block body (lines 481-482)
        scalar_body = :(x + 1)
        result = _ensure_body_has_toplevel_lnn(scalar_body, LineNumberNode(1, :f))
        @test result.head === :block
        @test result.args[1] isa LineNumberNode
        @test result.args[2] == scalar_body
    end

    # ==============================================================================
    # PoolReassignEscapeWarning: verify warning output
    # ==============================================================================

    @testset "PoolReassignEscapeWarning output" begin
        # v = f(v) where f is unknown → warning should fire
        warn_output = _capture_stderr() do
            @macroexpand @with_pool pool function test_warn()
                v = acquire!(pool, Float64, 10)
                v = f(v)
                return v
            end
        end
        @test contains(warn_output, "PoolReassignEscapeWarning")
        @test contains(warn_output, "reassigned")

        # v = collect(v) → no warning (safe copy)
        warn_output2 = _capture_stderr() do
            @macroexpand @with_pool pool function test_safe()
                v = acquire!(pool, Float64, 10)
                v = collect(v)
                return v
            end
        end
        @test !contains(warn_output2, "PoolReassignEscapeWarning")

        # v = v .+ 1.0 → no warning (broadcast, new array)
        warn_output3 = _capture_stderr() do
            @macroexpand @with_pool pool function test_bcast()
                v = acquire!(pool, Float64, 10)
                v = v .+ 1.0
                return v
            end
        end
        @test !contains(warn_output3, "PoolReassignEscapeWarning")
    end

    # ==============================================================================
    # _find_reassign_maybe_tainted unit tests
    # ==============================================================================

    @testset "_find_reassign_maybe_tainted" begin
        # v = f(v) after acquire → maybe tainted
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                v = f(v)
            end, :pool,
        )
        @test :v in mt

        # v = collect(v) → NOT maybe tainted (safe copy)
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                v = collect(v)
            end, :pool,
        )
        @test :v ∉ mt

        # v = g(x) where x is unrelated → NOT maybe tainted
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                v = g(x)
            end, :pool,
        )
        @test :v ∉ mt

        # Alias propagation: w = v where v is maybe-tainted
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                v = f(v)
                w = v
            end, :pool,
        )
        @test :v in mt
        @test :w in mt

        # Tuple destructuring with tuple literal RHS: element-wise check
        # (v,) = (f(v),) where f is unknown → v is maybe-tainted
        # BUT opaque destructuring (v,) = transform(v) → NOT maybe-tainted
        # (destructuring implies a transform, not identity)
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                (v,) = transform(v)
            end, :pool,
        )
        @test :v ∉ mt  # opaque destructuring: treated as transform

        # Tuple destructuring with safe copy: (v,) = (collect(v),) → not maybe
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                (v, w) = (collect(v), 1)
            end, :pool,
        )
        @test :v ∉ mt

        # Re-acquire clears maybe-tainted
        mt = _find_reassign_maybe_tainted(
            quote
                v = acquire!(pool, Float64, 10)
                v = f(v)
                v = acquire!(pool, Float64, 5)
            end, :pool,
        )
        @test :v ∉ mt
    end

    # ==============================================================================
    # _is_safe_copy_call unit tests
    # ==============================================================================

    @testset "_is_safe_copy_call" begin
        @test _is_safe_copy_call(:(collect(v)))
        @test _is_safe_copy_call(:(copy(v)))
        @test _is_safe_copy_call(:(deepcopy(v)))
        @test _is_safe_copy_call(:(similar(v)))
        # Array/Vector/Matrix removed — conservative measure (type constructors
        # may behave unpredictably with certain argument types)
        @test !_is_safe_copy_call(:(Array(v)))
        @test !_is_safe_copy_call(:(Vector(v)))
        @test !_is_safe_copy_call(:(Matrix(v)))
        @test _is_safe_copy_call(:(Base.collect(v)))
        # Broadcast: dotted operators
        @test _is_safe_copy_call(Expr(:call, :.+, :v, 1.0))
        @test _is_safe_copy_call(Expr(:call, :.*, :v, :w))
        # Broadcast: f.(v) form
        @test _is_safe_copy_call(Expr(:., :f, Expr(:tuple, :v)))
        # Non-safe calls
        @test !_is_safe_copy_call(:(f(v)))
        @test !_is_safe_copy_call(:(identity(v)))
        @test !_is_safe_copy_call(:(reshape(v, 3, 3)))
        @test !_is_safe_copy_call(:v)
        @test !_is_safe_copy_call(1)
    end

    # ==============================================================================
    # _rhs_call_contains_sym unit tests
    # ==============================================================================

    @testset "_rhs_call_contains_sym" begin
        @test _rhs_call_contains_sym(:(f(v)), :v)
        @test _rhs_call_contains_sym(:(f(g(v))), :v)
        @test !_rhs_call_contains_sym(:(f(w)), :v)
        @test !_rhs_call_contains_sym(:v, :v)  # bare symbol, not a call
        @test !_rhs_call_contains_sym(1, :v)
        @test _rhs_call_contains_sym(:(a[v]), :v)  # ref expression
    end

    # ==============================================================================
    # _extract_container_vars: order-aware container tracking
    # ==============================================================================

    @testset "_extract_container_vars: reassignment clears container taint" begin
        # Basic: vac assigned from tuple with acquire → tracked
        cvars = _extract_container_vars(
            quote
                vac = (wv = zeros!(pool, 10), grri = acquire!(pool, Float64, 5))
            end, :pool,
        )
        @test :vac in cvars

        # Reassigned to non-acquire tuple → should be cleared
        cvars = _extract_container_vars(
            quote
                vac = (wv = zeros!(pool, 10),)
                vac = (name = "ok",)
            end, :pool,
        )
        @test :vac ∉ cvars

        # Reassigned to scalar → should be cleared
        cvars = _extract_container_vars(
            quote
                vac = (wv = acquire!(pool, Float64, 3),)
                vac = nothing
            end, :pool,
        )
        @test :vac ∉ cvars
    end

    @testset "PoolContainerEscapeWarning" begin
        # Positive: container dot-access SHOULD warn
        warn_output = _capture_stderr() do
            @macroexpand @with_pool pool function test_container_warn()
                vac = (wv = acquire!(pool, Float64, 3),)
                return vac.wv
            end
        end
        @test contains(warn_output, "PoolContainerEscapeWarning")
        @test contains(warn_output, "vac")
        @test contains(warn_output, "Declarations:")
        @test contains(warn_output, "Escaping return:")

        # Negative: reassigned container → no warning
        warn_output2 = _capture_stderr() do
            @macroexpand @with_pool pool function test_no_warn()
                vac = (wv = acquire!(pool, Float64, 3),)
                vac = (name = "ok",)
                return vac.name
            end
        end
        @test !contains(warn_output2, "PoolContainerEscapeWarning")
    end

    # ==============================================================================
    # Incidental-tail escape detection: direct acquire-call tails, broadcast-assign
    # tails, and assignment tails. Form-based severity: a function-form tail or an
    # explicit `return` (definite escape to the enclosing function's caller) always
    # errors; a block-form implicit tail (value may be discarded) reports at the
    # ESCAPE_LINT severity — "warn" by default.
    # ==============================================================================

    @static if VERSION >= v"1.12-"
        # (Helpers `_expansion_escape_error` / `_expansion_incidental_warns`
        # are defined at the top of this file — they are also used by the
        # version-independent block-form testsets above.)

        @testset "block-form incidental tails warn at expansion (default)" begin
            # A block's value may simply be discarded by the surrounding code — the
            # macro cannot see its own call site — so under the default
            # escape_lint = "warn" these expand cleanly with a @warn diagnostic.
            # w2: broadcast-assign tail of an acquired var
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                    end
                )
            )

            # w2': dotted op-assign
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        v .+= 1.0
                    end
                )
            )

            # w1: direct acquire-family call tail (no acquired vars at all —
            # regression for the early-return trap in the Stage-2 reachability)
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        acquire!(pool, Float64, 4)
                    end
                )
            )

            # w1 via convenience wrapper
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        zeros!(pool, Float64, 4)
                    end
                )
            )

            # w3: assignment tail whose RHS is an acquire call
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                    end
                )
            )

            # safe variant shares the same block-form severity (kwarg plumbing)
            @test _expansion_incidental_warns(
                :(
                    @safe_with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                    end
                )
            )
        end

        @testset "function-form and explicit-return incidental tails always error" begin
            # A function-form tail IS the function's return value, and an explicit
            # `return` returns from the ENCLOSING function even in block form (a
            # `begin` block is not a function boundary) — definite escapes to a
            # caller, hard errors regardless of the escape_lint severity.
            #
            # function form: implicit x .= v tail is the function's return value.
            # NOTE: uses the 2-arg `@with_pool pool function ... end` form (pool
            # captured via closure) — `@with_pool function f(pool) ... end` (pool as
            # the function's own parameter) is a different, unsupported pattern: the
            # macro always gensyms its internal pool binding for the 1-arg function
            # form, so a same-named parameter shadows it and never matches during
            # escape analysis (confirmed via macroexpand: the two `pool`s are
            # distinct bindings). See docs-plans-symlink / scoping-pitfall memory note.
            @test _expansion_escape_error(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                    end
                )
            )

            # function form, w1: bare acquire-call tail
            @test _expansion_escape_error(
                :(
                    @with_pool pool function f()
                        acquire!(pool, Float64, 4)
                    end
                )
            )

            # function form, w3: assignment tail
            @test _expansion_escape_error(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                    end
                )
            )

            # safe function form shares the always-error severity
            @test _expansion_escape_error(
                :(
                    @safe_with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                    end
                )
            )

            # w1 with explicit `return`: `return acquire!(...)` must be unwrapped
            # just like the implicit-tail form above.
            @test _expansion_escape_error(
                :(
                    @with_pool pool begin
                        return acquire!(pool, Float64, 4)
                    end
                )
            )

            # w1 via convenience wrapper, explicit `return`
            @test _expansion_escape_error(
                :(
                    @with_pool pool begin
                        return zeros!(pool, Float64, 4)
                    end
                )
            )

            # w2 with explicit `return`: `return (v .= 0.0)`
            @test _expansion_escape_error(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        return v .= 0.0
                    end
                )
            )

            # Reviewer's reproduction: nested-if early return, function form.
            @test _expansion_escape_error(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        if true
                            return v .= 0.0
                        end
                        nothing
                    end
                )
            )
        end

        @testset "incidental-tail diagnostics teach the fix" begin
            # Error path (function form): showerror carries the fix hint
            err = try
                macroexpand(
                    @__MODULE__, :(
                        @with_pool pool function f()
                            v = acquire!(pool, Float64, 4)
                            v .= 0.0
                        end
                    )
                )
                nothing
            catch e
                e isa LoadError ? e.error : e
            end
            @test err isa AdaptiveArrayPools.PoolEscapeError
            msg = sprint(showerror, err)
            @test occursin("nothing", msg)          # suggests the one-line fix
            @test occursin("last expression", msg)  # explains block-tail semantics

            # Warn path (block form): the @warn diagnostic carries the same hint
            @test_logs (:warn, r"end the block with `nothing`") match_mode = :any macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                    end
                )
            )
        end

        @testset "safe tails do not error" begin
            # each expands cleanly (no throw): nothing tail, scalar call, owned copy,
            # scalar index, broadcast into a non-pool array
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4); v .= 0.0; nothing
                    end
                )
            ) isa Expr
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4); sum(v)
                    end
                )
            ) isa Expr
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4); collect(v)
                    end
                )
            ) isa Expr
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4); v[1]
                    end
                )
            ) isa Expr
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4); w = zeros(4); w .= 1.0
                    end
                )
            ) isa Expr
            # explicit-return safe cases: bare `return` and `return sum(v)`
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                        return
                    end
                )
            ) isa Expr
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        return sum(v)
                    end
                )
            ) isa Expr
        end

        @testset "explicit-return escape renders correct pattern label (M5)" begin
            # `return (v .= 0.0)` must still classify as broadcast_assign in
            # showerror, not fall through to the acquire_call label — the
            # `:return` wrapper must be unwrapped before pattern-matching,
            # same as `_incidental_exposure` already does for detection.
            err = try
                macroexpand(
                    @__MODULE__, :(
                        @with_pool pool begin
                            v = acquire!(pool, Float64, 4)
                            return (v .= 0.0)
                        end
                    )
                )
                nothing
            catch e
                e isa LoadError ? e.error : e
            end
            @test err isa AdaptiveArrayPools.PoolEscapeError
            msg = sprint(showerror, err)
            @test occursin("pool-backed array `v`", msg)
            @test !occursin("direct acquire call", msg)
        end

        @testset "incidental tail inside if/else implicit branch warns (block form)" begin
            # A common Julia idiom: the block's last expression is an if/else
            # whose branch tail is an incidental escape. `_collect_all_return_values`
            # expands into both branch tails, and Stage 2 must flag the escaping one
            # — at block-form severity (warn), since the branch tail is still an
            # implicit block tail, not an explicit `return`.
            # w2 — broadcast-assign branch tail
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        if true
                            v .= 0.0
                        else
                            nothing
                        end
                    end
                )
            )
            # w1 — direct acquire-call branch tail
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        if true
                            acquire!(pool, Float64, 4)
                        else
                            nothing
                        end
                    end
                )
            )
            # w3 — assignment branch tail
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        if true
                            v = acquire!(pool, Float64, 4)
                        else
                            nothing
                        end
                    end
                )
            )
            # Same branch-tail shape in FUNCTION form is the function's return
            # value — always an error.
            @test _expansion_escape_error(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        if true
                            v .= 0.0
                        else
                            nothing
                        end
                    end
                )
            )
            # Safe: both branch tails are `nothing` — must NOT error
            @test macroexpand(
                @__MODULE__, :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        if true
                            v .= 0.0
                            nothing
                        else
                            nothing
                        end
                    end
                )
            ) isa Expr
        end

        @testset "incidental-tail showerror uses the stored (kind, detail) label" begin
            # After the classification is stored on the EscapePoint at throw time,
            # showerror renders each pattern's own label without re-deriving it.
            # Uses FUNCTION form so the incidental tail still throws (block form
            # only warns under the default severity).
            function _escape_err(ex)
                try
                    macroexpand(@__MODULE__, ex)
                    return nothing
                catch e
                    return e isa LoadError ? e.error : e
                end
            end

            # w1 — acquire-call tail
            e1 = _escape_err(
                :(
                    @with_pool pool function f()
                        acquire!(pool, Float64, 4)
                    end
                )
            )
            @test e1 isa AdaptiveArrayPools.PoolEscapeError
            @test occursin("direct acquire call", sprint(showerror, e1))

            # w2 — broadcast-assign tail names the array
            e2 = _escape_err(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        v .= 0.0
                    end
                )
            )
            @test e2 isa AdaptiveArrayPools.PoolEscapeError
            @test occursin("pool-backed array `v`", sprint(showerror, e2))

            # w3 — assignment tail
            e3 = _escape_err(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                    end
                )
            )
            @test e3 isa AdaptiveArrayPools.PoolEscapeError
            @test occursin("assigns a pool-backed array", sprint(showerror, e3))
        end

        @testset "form-based severity boundary (regression pins)" begin
            # block bare-var tail: warn + guard (implicit tail)
            @test _expansion_incidental_warns(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        v
                    end
                )
            )
            # explicit return of the same var: still a hard error
            @test _expansion_escape_error(
                :(
                    @with_pool pool begin
                        v = acquire!(pool, Float64, 4)
                        return v
                    end
                )
            )
            # function form of the same shape: still a hard error
            @test _expansion_escape_error(
                :(
                    @with_pool pool function f()
                        v = acquire!(pool, Float64, 4)
                        v
                    end
                )
            )
        end

        # ==========================================================================
        # Direct unit tests for _incidental_exposure and _lint_message.
        #
        # ESCAPE_LINT is a load-time constant (like RUNTIME_CHECK); the test session
        # runs under the default "warn". The block-form warn path is integration-
        # tested above; the "error" severity is integration-tested via the
        # function-form / explicit-return paths (always error). The "error"-pref-on-
        # block-tails and "off" branches cannot be flipped in-process, so these unit
        # tests exercise the detection helper and message builder directly instead.
        # ==========================================================================

        @testset "_is_dotted_assign_head" begin
            @test _is_dotted_assign_head(Symbol(".="))
            @test _is_dotted_assign_head(Symbol(".+="))
            @test _is_dotted_assign_head(Symbol(".*="))
            @test !_is_dotted_assign_head(:(=))
            @test !_is_dotted_assign_head(:ref)
            @test !_is_dotted_assign_head(:call)
            @test !_is_dotted_assign_head(1)  # non-Symbol head
        end

        @testset "_incidental_exposure: detects the three incidental patterns" begin
            # w1: direct acquire-family call tail — no tainted vars needed at all
            hit = _incidental_exposure(:(acquire!(pool, Float64, 4)), Set{Symbol}(), :pool)
            @test hit !== nothing
            @test hit[1] === :acquire_call

            hit = _incidental_exposure(:(zeros!(pool, Float64, 4)), Set{Symbol}(), :pool)
            @test hit !== nothing
            @test hit[1] === :acquire_call

            # w2: broadcast-assign tail of a tainted var
            hit = _incidental_exposure(:(v .= 0.0), Set([:v]), :pool)
            @test hit == (:broadcast_assign, :v)

            # w2': dotted op-assign
            hit = _incidental_exposure(:(v .+= 1.0), Set([:v]), :pool)
            @test hit == (:broadcast_assign, :v)

            # w2'': indexed base — x[...] .= v on a tainted base
            hit = _incidental_exposure(:(v[1:2] .= 0.0), Set([:v]), :pool)
            @test hit == (:broadcast_assign, :v)

            # w3: assignment tail whose RHS is a direct acquire call
            hit = _incidental_exposure(:(v = acquire!(pool, Float64, 4)), Set{Symbol}(), :pool)
            @test hit !== nothing
            @test hit[1] === :assign

            # w3': assignment tail whose RHS is an already-tainted var (alias)
            hit = _incidental_exposure(:(d = v), Set([:v]), :pool)
            @test hit == (:assign, :v)
        end

        @testset "_incidental_exposure: safe tails return nothing" begin
            @test _incidental_exposure(:nothing, Set([:v]), :pool) === nothing
            @test _incidental_exposure(1, Set([:v]), :pool) === nothing  # non-Expr
            @test _incidental_exposure(:(v[1]), Set([:v]), :pool) === nothing  # scalar index
            @test _incidental_exposure(:(sum(v)), Set([:v]), :pool) === nothing
            @test _incidental_exposure(:(collect(v)), Set([:v]), :pool) === nothing
            # broadcast into a non-tainted (non-pool) array
            @test _incidental_exposure(:(w .= 1.0), Set([:v]), :pool) === nothing
            # assignment whose RHS is neither an acquire call nor a tainted var
            @test _incidental_exposure(:(d = 1), Set([:v]), :pool) === nothing
            @test _incidental_exposure(:(d = sum(v)), Set([:v]), :pool) === nothing
        end

        @testset "_lint_message: content pins \"last expression\" and \"nothing\"" begin
            acquire_expr = :(acquire!(pool, Float64, 4))
            msg = _lint_message(:acquire_call, acquire_expr, acquire_expr)
            @test occursin("last expression", msg)
            @test occursin("nothing", msg)
            @test occursin("escape_lint", msg)
            @test occursin("safety/compile-time", msg)     # docs pointer
            @test occursin("\"warn\" (default)", msg)       # warn is now the default severity
            @test occursin("compile time", msg)             # "can't tell if the value is used" framing

            broadcast_expr = :(v .= 0.0)
            msg = _lint_message(:broadcast_assign, :v, broadcast_expr)
            @test occursin("last expression", msg)
            @test occursin("nothing", msg)
            @test occursin("v", msg)

            assign_expr = :(v = acquire!(pool, Float64, 4))
            msg = _lint_message(:assign, :v, assign_expr)
            @test occursin("last expression", msg)
            @test occursin("nothing", msg)
        end

        @testset "_report_incidental_escape: both severity branches" begin
            # ESCAPE_LINT is a load-time constant so the call site only ever
            # exercises one branch per session — test both directly here.
            ax = :(acquire!(pool, Float64, 4))
            # "error" throws PoolEscapeError carrying the incidental point
            err = try
                _report_incidental_escape("error", :acquire_call, ax, ax, 10, "f.jl", 1)
                nothing
            catch e
                e
            end
            @test err isa PoolEscapeError
            @test occursin("direct acquire call", sprint(showerror, err))

            # "warn" emits an expansion-time warning (no throw)
            bx = :(v .= 0.0)
            @test_logs (:warn,) _report_incidental_escape("warn", :broadcast_assign, :v, bx, 12, "f.jl", 1)
        end
    end

    # ==============================================================================
    # EscapedPoolArray guard: type behavior, tail rewriting, runtime integration
    # ==============================================================================

    @testset "EscapedPoolArray guard type (traps, show, provenance)" begin
        x = rand(3, 2)
        g = EscapedPoolArray(x, :x, "f.jl", 12)
        @test g isa EscapedPoolArray
        @test g.var === :x
        @test g.arraytype === Matrix{Float64}
        @test g.dims == (3, 2)
        # metadata-only: no field holds the array itself
        @test fieldnames(EscapedPoolArray) == (:var, :arraytype, :dims, :file, :line)

        # trapped operations throw with provenance
        @test_throws EscapedPoolUseError g[1]
        @test_throws EscapedPoolUseError setindex!(g, 0.0, 1)
        @test_throws EscapedPoolUseError size(g)
        @test_throws EscapedPoolUseError length(g)
        @test_throws EscapedPoolUseError axes(g)
        @test_throws EscapedPoolUseError iterate(g)
        @test_throws EscapedPoolUseError copy(g)
        @test_throws EscapedPoolUseError similar(g)
        @test_throws EscapedPoolUseError sum(g)
        @test_throws EscapedPoolUseError collect(g)
        @test_throws EscapedPoolUseError g .+ 1

        # non-throwing surfaces: identity ops and informative show
        @test g === g
        @test !isnothing(g)
        shown = sprint(show, g)
        @test occursin("escaped", shown)
        @test occursin("`x`", shown)
        @test occursin("3×2", shown)
        @test occursin("f.jl:12", shown)

        uerr = try
            g[1]
        catch e
            e
        end
        @test uerr isa EscapedPoolUseError
        m = sprint(showerror, uerr)
        @test occursin("getindex", m)
        @test occursin("`x`", m)
        @test occursin("f.jl:12", m)
        @test occursin("nothing", m)      # teaches the discard fix

        # anonymous / non-array fallbacks
        ga = EscapedPoolArray(42, :expression, nothing, nothing)
        @test ga.dims == ()
        @test occursin("anonymous expression", sprint(show, ga))
    end

    @testset "_poison_block_tails rewrites" begin
        lnn = LineNumberNode(7, Symbol("poison.jl"))
        guard_call(x) = x isa Expr && x.head == :call && x.args[1] === EscapedPoolArray
        lastexpr(b) = last(filter(a -> !(a isa LineNumberNode), b.args))

        # bare tail → guard ctor with the var name + scope location
        out = _poison_block_tails(
            :(
                begin
                    v = acquire!(pool, Float64, 4)
                    v
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test guard_call(tail)
        @test tail.args[2] === :v
        @test tail.args[3] isa QuoteNode && tail.args[3].value === :v
        @test tail.args[4] == "poison.jl" && tail.args[5] == 7

        # dest[...] = v tail → (dest[...] = v; guard(v)): side effect preserved
        out = _poison_block_tails(
            :(
                begin
                    v = acquire!(pool, Float64, 4)
                    dest[:, :, k] = v
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test tail isa Expr && tail.head == :block
        @test Meta.isexpr(tail.args[1], :(=))     # original assignment first
        @test guard_call(tail.args[2])

        # tuple: only pool-backed elements replaced
        out = _poison_block_tails(
            :(
                begin
                    x = 1
                    v = acquire!(pool, Float64, 4)
                    (x, v)
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test Meta.isexpr(tail, :tuple)
        @test tail.args[1] === :x                  # non-pool element untouched
        @test guard_call(tail.args[2])

        # explicit return: never rewritten
        out = _poison_block_tails(
            :(
                begin
                    v = acquire!(pool, Float64, 4)
                    return v
                end
            ), :pool, lnn
        )
        @test lastexpr(out) == :(return v)

        # `_ = acquire!(...)` tail: `_` is write-only, so the rewrite must wrap
        # the WHOLE assignment (which evaluates to its RHS) instead of reading
        # `_` back — regression for the all-underscore rvalue syntax error
        out = _poison_block_tails(
            :(
                begin
                    _ = acquire!(pool, Float64, 4)
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test guard_call(tail)
        @test Meta.isexpr(tail.args[2], :(=))          # ctor arg IS the assignment
        @test tail.args[3] isa QuoteNode && tail.args[3].value === :expression
        # and no bare `_` appears as a ctor value argument
        @test tail.args[2] !== :_

        # identity(v) tail: unwrapped like _find_direct_exposure does
        out = _poison_block_tails(
            :(
                begin
                    v = acquire!(pool, Float64, 4)
                    identity(v)
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test Meta.isexpr(tail, :call) && tail.args[1] === :identity
        @test guard_call(tail.args[2])                 # inner v guarded

        # literal parity branches: nested tuple, identity element, acquire-call element
        out = _poison_block_tails(
            :(
                begin
                    x = 1
                    v = acquire!(pool, Float64, 4)
                    (x, (x, v), identity(v), acquire!(pool, Float64, 2))
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test Meta.isexpr(tail, :tuple)
        @test tail.args[1] === :x                      # untouched
        nested = tail.args[2]                          # (x, v) → (x, guard(v))
        @test Meta.isexpr(nested, :tuple) && nested.args[1] === :x && guard_call(nested.args[2])
        ident = tail.args[3]                           # identity(v) → identity(guard(v))
        @test Meta.isexpr(ident, :call) && ident.args[1] === :identity && guard_call(ident.args[2])
        @test guard_call(tail.args[4])                 # acquire-call element ctor-wrapped
        @test tail.args[4].args[3].value === :expression

        # if/else: only the escaping branch tail rewritten
        out = _poison_block_tails(
            :(
                begin
                    v = acquire!(pool, Float64, 4)
                    if c
                        v
                    else
                        nothing
                    end
                end
            ), :pool, lnn
        )
        tail = lastexpr(out)
        @test Meta.isexpr(tail, :if)
        thenlast = last(filter(a -> !(a isa LineNumberNode), tail.args[2].args))
        elselast = last(filter(a -> !(a isa LineNumberNode), tail.args[3].args))
        @test guard_call(thenlast)
        @test elselast === :nothing    # the identifier `nothing` is a Symbol in the AST

        # safe tail: structurally unchanged
        src = :(
            begin
                v = acquire!(pool, Float64, 4)
                sum(v)
            end
        )
        @test _poison_block_tails(src, :pool, lnn) == src
    end

    @testset "block-tail guard integration (runtime)" begin
        # NOTE: subjects are eval'd at RUNTIME so the expansion-time @warn is
        # captured by @test_logs (a literal @with_pool here would warn once at
        # test-file load instead).

        # bare tail: block value is a guard; first use throws with provenance
        out = @test_logs (:warn, r"becomes the scope's return value") match_mode = :any Core.eval(
            @__MODULE__, :(
                @with_pool pool begin
                    a = acquire!(pool, Float64, 3)
                    a .= 7.0
                    a
                end
            )
        )
        @test out isa EscapedPoolArray
        @test out.var === :a
        @test out.arraytype <: AbstractVector{Float64}
        @test out.dims == (3,)
        @test_throws EscapedPoolUseError out[1]
        @test_throws EscapedPoolUseError sum(out)

        # FLUX shape: dest[...] = v tail — data copied out, guard discarded, silent
        res = @test_logs (:warn, r"becomes the scope's return value") match_mode = :any Core.eval(
            @__MODULE__, :(
                let dest = zeros(2, 3, 2)
                    for k in 1:2
                        @with_pool pool begin
                            v = zeros!(pool, Float64, 2, 3)
                            v .= k
                            dest[:, :, k] = v
                        end
                    end
                    dest
                end
            )
        )
        @test res[:, :, 1] == fill(1.0, 2, 3)
        @test res[:, :, 2] == fill(2.0, 2, 3)

        # partial tuple: non-pool element stays usable, pool element traps
        r = @test_logs (:warn, r"becomes the scope's return value") match_mode = :any Core.eval(
            @__MODULE__, :(
                @with_pool pool begin
                    x = 42
                    a = acquire!(pool, Float64, 3)
                    (x, a)
                end
            )
        )
        @test r isa Tuple
        @test r[1] === 42
        @test r[2] isa EscapedPoolArray
        @test_throws EscapedPoolUseError r[2][1]

        # discarded guard: silent at runtime (warn fires once, at expansion) —
        # the FLUX pattern as a reusable function
        f = @test_logs (:warn, r"becomes the scope's return value") match_mode = :any Core.eval(
            @__MODULE__, :(
                function guard_discard_fn(dest)
                    @with_pool pool begin
                        v = zeros!(pool, Float64, 4)
                        v .= 1.0
                        dest[:] = v
                    end
                    return dest
                end
            )
        )
        dst = zeros(4)
        @test f(dst) == ones(4)     # no error, no warn at runtime
        @test f(dst) == ones(4)     # repeated runs stay silent

        # `_ = acquire!(...)` tail: the canonical discard spelling must expand
        # AND run (regression: the rewrite used to read the write-only `_`)
        u = @test_logs (:warn, r"becomes the scope's return value") match_mode = :any Core.eval(
            @__MODULE__, :(
                @with_pool pool begin
                    _ = acquire!(pool, Float64, 3)
                end
            )
        )
        @test u isa EscapedPoolArray
        @test u.var === :expression
        @test u.dims == (3,)
    end

end # Compile-Time Escape Detection
