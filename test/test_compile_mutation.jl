import AdaptiveArrayPools: _extract_acquired_vars, _check_structural_mutation,
    _is_mutation_call, _find_mutation_calls,
    _STRUCTURAL_MUTATION_NAMES,
    MutationPoint

@testset "Compile-Time Structural Mutation Detection" begin

    # ==============================================================================
    # _is_mutation_call: detect dangerous calls on acquired variables
    # ==============================================================================

    @testset "_is_mutation_call" begin
        acquired = Set{Symbol}([:v, :w])

        # Direct mutation calls
        @test _is_mutation_call(:(resize!(v, 200)), acquired) == (:resize!, :v)
        @test _is_mutation_call(:(push!(v, 1.0)), acquired) == (:push!, :v)
        @test _is_mutation_call(:(pop!(w)), acquired) == (:pop!, :w)
        @test _is_mutation_call(:(append!(v, data)), acquired) == (:append!, :v)
        @test _is_mutation_call(:(prepend!(v, data)), acquired) == (:prepend!, :v)
        @test _is_mutation_call(:(deleteat!(v, 1)), acquired) == (:deleteat!, :v)
        @test _is_mutation_call(:(insert!(v, 1, x)), acquired) == (:insert!, :v)
        @test _is_mutation_call(:(splice!(v, 1:2)), acquired) == (:splice!, :v)
        @test _is_mutation_call(:(empty!(v)), acquired) == (:empty!, :v)
        @test _is_mutation_call(:(sizehint!(v, 1000)), acquired) == (:sizehint!, :v)
        @test _is_mutation_call(:(pushfirst!(v, 0.0)), acquired) == (:pushfirst!, :v)
        @test _is_mutation_call(:(popfirst!(v)), acquired) == (:popfirst!, :v)

        # Qualified name: Base.resize!(v, ...)
        @test _is_mutation_call(:(Base.resize!(v, 200)), acquired) == (:resize!, :v)
        @test _is_mutation_call(:(Base.push!(w, 1.0)), acquired) == (:push!, :w)

        # Non-acquired variable → nothing
        @test _is_mutation_call(:(resize!(x, 200)), acquired) === nothing

        # Non-mutation call → nothing
        @test _is_mutation_call(:(sum(v)), acquired) === nothing
        @test _is_mutation_call(:(fill!(v, 0.0)), acquired) === nothing

        # Not a call → nothing
        @test _is_mutation_call(:v, acquired) === nothing
        @test _is_mutation_call(42, acquired) === nothing
    end

    # ==============================================================================
    # _find_mutation_calls: walk AST to find all mutation sites
    # ==============================================================================

    @testset "_find_mutation_calls" begin
        acquired = Set{Symbol}([:v, :w])

        # Single mutation
        points = _find_mutation_calls(
            quote
                v = acquire!(pool, Float64, 100)
                resize!(v, 200)
            end,
            acquired
        )
        @test length(points) == 1
        @test points[1].var == :v
        @test points[1].func == :resize!

        # Multiple mutations
        points = _find_mutation_calls(
            quote
                push!(v, 1.0)
                pop!(w)
                append!(v, [2.0, 3.0])
            end,
            acquired
        )
        @test length(points) == 3

        # No mutations
        points = _find_mutation_calls(
            quote
                v .= 1.0
                sum(v)
                w[1] = 42.0
            end,
            acquired
        )
        @test isempty(points)

        # Mutation inside nested function definition → skipped
        points = _find_mutation_calls(
            quote
                f = function (x)
                    resize!(v, 100)
                end
            end,
            acquired
        )
        @test isempty(points)

        # Mutation inside lambda → skipped
        points = _find_mutation_calls(
            quote
                f = x -> resize!(v, 100)
            end,
            acquired
        )
        @test isempty(points)

        # Mutation inside if-else → detected
        points = _find_mutation_calls(
            quote
                if condition
                    resize!(v, 200)
                else
                    push!(w, 1.0)
                end
            end,
            acquired
        )
        @test length(points) == 2
    end

    # ==============================================================================
    # _check_structural_mutation: full pipeline (throws PoolMutationError)
    # ==============================================================================

    @testset "_check_structural_mutation integration" begin
        # resize! on acquired variable → throws
        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = acquire!(pool, Float64, 100)
                resize!(v, 200)
            end,
            :pool,
            nothing
        )

        # push! on acquired variable → throws
        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = acquire!(pool, Float64, 100)
                push!(v, 1.0)
            end,
            :pool,
            nothing
        )

        # Alias tracking: w = v; resize!(w, ...) → throws
        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = acquire!(pool, Float64, 100)
                w = v
                resize!(w, 200)
            end,
            :pool,
            nothing
        )

        # No mutation → does not throw (returns nothing)
        @test _check_structural_mutation(
            quote
                v = acquire!(pool, Float64, 100)
                v .= 1.0
                sum(v)
            end,
            :pool,
            nothing
        ) === nothing

        # No acquire → does not throw
        @test _check_structural_mutation(
            quote
                v = zeros(100)
                resize!(v, 200)
            end,
            :pool,
            nothing
        ) === nothing

        # Reassigned variable → not tracked (safe)
        @test _check_structural_mutation(
            quote
                v = acquire!(pool, Float64, 100)
                v = zeros(100)
                resize!(v, 200)
            end,
            :pool,
            nothing
        ) === nothing

        # zeros!/ones!/similar!/trues!/falses! → also tracked
        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = zeros!(pool, 100)
                resize!(v, 200)
            end,
            :pool,
            nothing
        )

        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = ones!(pool, Int64, 100)
                push!(v, 42)
            end,
            :pool,
            nothing
        )

        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = trues!(pool, 100)
                push!(v, false)
            end,
            :pool,
            nothing
        )

        # Qualified Base.resize! also detected
        @test_throws PoolMutationError _check_structural_mutation(
            quote
                v = acquire!(pool, Float64, 100)
                Base.resize!(v, 200)
            end,
            :pool,
            nothing
        )
    end

    # ==============================================================================
    # PoolMutationError message quality
    # ==============================================================================

    @testset "PoolMutationError message" begin
        err = try
            _check_structural_mutation(
                quote
                    v = acquire!(pool, Float64, 100)
                    resize!(v, 200)
                    push!(v, 1.0)
                end,
                :pool,
                nothing
            )
        catch e
            e
        end

        @test err isa PoolMutationError
        @test length(err.points) == 2
        @test err.points[1].func == :resize!
        @test err.points[1].var == :v
        @test err.points[2].func == :push!
        @test err.points[2].var == :v

        # showerror produces output
        buf = IOBuffer()
        showerror(buf, err)
        msg = String(take!(buf))
        @test occursin("PoolMutationError", msg)
        @test occursin("resize!", msg)
        @test occursin("push!", msg)
        @test occursin("acquire!(pool, T, n)", msg)
    end

    # ==============================================================================
    # Macro-level integration (actual @with_pool usage)
    # ==============================================================================

    @testset "macro integration" begin
        # @with_pool with resize! → PoolMutationError at macro expansion
        @test_throws PoolMutationError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 100)
            resize!(v, 200)
        end

        # @with_pool with push! → PoolMutationError
        @test_throws PoolMutationError @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 100)
            push!(v, 1.0)
        end

        # @with_pool without mutation → succeeds
        result = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 100)
            v .= 1.0
            sum(v)
        end
        @test result isa Expr

        # @safe_with_pool with mutation → also caught
        @test_throws PoolMutationError @macroexpand @safe_with_pool pool begin
            v = acquire!(pool, Float64, 100)
            resize!(v, 200)
        end

        # @maybe_with_pool with mutation → also caught
        @test_throws PoolMutationError @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 100)
            pop!(v)
        end
    end
end
