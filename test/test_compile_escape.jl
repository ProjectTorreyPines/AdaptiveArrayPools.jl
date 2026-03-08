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

end # Compile-Time Escape Detection
