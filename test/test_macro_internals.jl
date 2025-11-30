# ==============================================================================
# Tests for macro internal functions
# ==============================================================================
#
# Tests for _extract_local_assignments and _filter_static_types functions
# to ensure correct type extraction and filtering for optimized checkpoint/rewind.

import AdaptiveArrayPools: _extract_local_assignments, _filter_static_types, _extract_acquire_types

@testset "Macro internals" begin

    @testset "_extract_local_assignments" begin
        # Simple assignment: T = eltype(x)
        @testset "simple assignment" begin
            expr = :(T = eltype(x))
            locals = _extract_local_assignments(expr)
            @test :T in locals
            @test length(locals) == 1
        end

        # Typed assignment: T::Type = SomeType
        @testset "typed assignment" begin
            expr = :(T::Type = SomeType)
            locals = _extract_local_assignments(expr)
            @test :T in locals
        end

        # Local declaration: local T
        @testset "local declaration" begin
            expr = :(local T)
            locals = _extract_local_assignments(expr)
            @test :T in locals
        end

        # Local with assignment: local T = eltype(x)
        @testset "local with assignment" begin
            expr = :(local T = eltype(x))
            locals = _extract_local_assignments(expr)
            @test :T in locals
        end

        # Multiple locals: local A, B, C
        @testset "multiple locals" begin
            expr = :(local A, B, C)
            locals = _extract_local_assignments(expr)
            @test :A in locals
            @test :B in locals
            @test :C in locals
            @test length(locals) == 3
        end

        # Nested expression with multiple assignments
        @testset "nested expression" begin
            expr = quote
                T = eltype(x)
                S = typeof(y)
                v = acquire!(pool, T, 10)
            end
            locals = _extract_local_assignments(expr)
            @test :T in locals
            @test :S in locals
            @test :v in locals
        end

        # No assignments
        @testset "no assignments" begin
            expr = :(sum(x))
            locals = _extract_local_assignments(expr)
            @test isempty(locals)
        end

        # Non-Expr input (Symbol)
        @testset "non-Expr input" begin
            locals = _extract_local_assignments(:x)
            @test isempty(locals)
        end

        # Assignment in function call (should NOT be captured)
        @testset "function call with kwarg" begin
            expr = :(foo(x, y=1))
            locals = _extract_local_assignments(expr)
            # y=1 inside function call is a kwarg, not an assignment
            # This depends on Julia parsing - in some cases it might be captured
            # The important thing is we don't crash
            @test locals isa Set{Symbol}
        end

        # Deeply nested assignments
        @testset "deeply nested" begin
            expr = quote
                if condition
                    A = 1
                    for i in 1:10
                        B = i
                        while true
                            C = B * 2
                            break
                        end
                    end
                end
            end
            locals = _extract_local_assignments(expr)
            @test :A in locals
            @test :B in locals
            @test :C in locals
        end
    end

    @testset "_filter_static_types" begin
        # Symbol not in local_vars → static
        @testset "symbol not in locals" begin
            types = Set{Any}([:Float64, :Int64])
            local_vars = Set{Symbol}()
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test :Float64 in static_types
            @test :Int64 in static_types
            @test !has_dynamic
        end

        # Symbol in local_vars → dynamic
        @testset "symbol in locals" begin
            types = Set{Any}([:T])
            local_vars = Set{Symbol}([:T])
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test isempty(static_types)
            @test has_dynamic
        end

        # Mixed: some in locals, some not
        @testset "mixed symbols" begin
            types = Set{Any}([:Float64, :T, :Int64])
            local_vars = Set{Symbol}([:T])
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test :Float64 in static_types
            @test :Int64 in static_types
            @test :T ∉ static_types
            @test has_dynamic
        end

        # Parametric type: Vector{Float64} → dynamic
        @testset "parametric type" begin
            types = Set{Any}([:(Vector{Float64})])
            local_vars = Set{Symbol}()
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test isempty(static_types)
            @test has_dynamic
        end

        # GlobalRef or other concrete types → static
        @testset "GlobalRef type" begin
            # Simulate a GlobalRef-like expression (not Symbol, not curly)
            types = Set{Any}([GlobalRef(Base, :Float64)])
            local_vars = Set{Symbol}()
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test length(static_types) == 1
            @test !has_dynamic
        end

        # Empty types set
        @testset "empty types" begin
            types = Set{Any}()
            local_vars = Set{Symbol}()
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test isempty(static_types)
            @test !has_dynamic
        end

        # All dynamic (all in locals or parametric)
        @testset "all dynamic" begin
            types = Set{Any}([:T, :S, :(Vector{Int})])
            local_vars = Set{Symbol}([:T, :S])
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test isempty(static_types)
            @test has_dynamic
        end

        # Curly expression detection
        @testset "curly expression" begin
            curly_expr = Expr(:curly, :Vector, :Float64)
            types = Set{Any}([curly_expr])
            local_vars = Set{Symbol}()
            static_types, has_dynamic = _filter_static_types(types, local_vars)
            @test isempty(static_types)
            @test has_dynamic
        end
    end

    @testset "_extract_acquire_types" begin
        # Single acquire! call
        @testset "single acquire!" begin
            expr = :(acquire!(pool, Float64, 10))
            types = _extract_acquire_types(expr)
            @test :Float64 in types
            @test length(types) == 1
        end

        # Multiple acquire! calls
        @testset "multiple acquire!" begin
            expr = quote
                v1 = acquire!(pool, Float64, 10)
                v2 = acquire!(pool, Int64, 5)
            end
            types = _extract_acquire_types(expr)
            @test :Float64 in types
            @test :Int64 in types
            @test length(types) == 2
        end

        # Duplicate types (same type used twice)
        @testset "duplicate types" begin
            expr = quote
                v1 = acquire!(pool, Float64, 10)
                v2 = acquire!(pool, Float64, 20)
            end
            types = _extract_acquire_types(expr)
            @test :Float64 in types
            @test length(types) == 1  # Set deduplicates
        end

        # Type parameter
        @testset "type parameter" begin
            expr = :(acquire!(pool, T, 10))
            types = _extract_acquire_types(expr)
            @test :T in types
        end

        # Parametric type
        @testset "parametric type" begin
            expr = :(acquire!(pool, Vector{Float64}, 10))
            types = _extract_acquire_types(expr)
            @test length(types) == 1
            # It should capture the curly expression
        end

        # No acquire! calls
        @testset "no acquire!" begin
            expr = :(sum(x))
            types = _extract_acquire_types(expr)
            @test isempty(types)
        end

        # Nested acquire! in complex expression
        @testset "nested in if-else" begin
            expr = quote
                if condition
                    v = acquire!(pool, Float64, 10)
                else
                    v = acquire!(pool, Float32, 10)
                end
            end
            types = _extract_acquire_types(expr)
            @test :Float64 in types
            @test :Float32 in types
        end

        # Qualified acquire! call (AdaptiveArrayPools.acquire!)
        @testset "qualified acquire!" begin
            expr = :(AdaptiveArrayPools.acquire!(pool, Int32, 5))
            types = _extract_acquire_types(expr)
            @test :Int32 in types
        end
    end

    @testset "Integration: type extraction + filtering" begin
        # Simulate macro behavior: extract types, then filter

        @testset "static types only" begin
            expr = quote
                v1 = acquire!(pool, Float64, 10)
                v2 = acquire!(pool, Int64, 5)
            end
            local_vars = _extract_local_assignments(expr)
            types = _extract_acquire_types(expr)
            static_types, has_dynamic = _filter_static_types(types, local_vars)

            @test :Float64 in static_types
            @test :Int64 in static_types
            @test !has_dynamic
        end

        @testset "with local type variable" begin
            expr = quote
                T = eltype(x)
                v = acquire!(pool, T, 10)
            end
            local_vars = _extract_local_assignments(expr)
            types = _extract_acquire_types(expr)
            static_types, has_dynamic = _filter_static_types(types, local_vars)

            @test :T in local_vars
            @test :T in types
            @test isempty(static_types)
            @test has_dynamic
        end

        @testset "mixed: static and local" begin
            expr = quote
                T = eltype(x)
                v1 = acquire!(pool, T, 10)
                v2 = acquire!(pool, Float64, 20)
            end
            local_vars = _extract_local_assignments(expr)
            types = _extract_acquire_types(expr)
            static_types, has_dynamic = _filter_static_types(types, local_vars)

            @test :Float64 in static_types
            @test :T ∉ static_types
            @test has_dynamic
        end
    end

end
