# ==============================================================================
# Tests for macro internal functions
# ==============================================================================
#
# Tests for _extract_local_assignments and _filter_static_types functions
# to ensure correct type extraction and filtering for optimized checkpoint/rewind.

import AdaptiveArrayPools: _extract_local_assignments, _filter_static_types, _extract_acquire_types, _uses_local_var

@testset "Macro Internals" begin

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

        @testset "_uses_local_var" begin
            @testset "simple symbol - local" begin
                local_vars = Set{Symbol}([:x, :y])
                @test _uses_local_var(:x, local_vars) == true
                @test _uses_local_var(:y, local_vars) == true
            end

            @testset "simple symbol - not local" begin
                local_vars = Set{Symbol}([:x])
                @test _uses_local_var(:z, local_vars) == false
            end

            @testset "field access - base is local" begin
                # cp1d.t_i_average where cp1d is local
                local_vars = Set{Symbol}([:cp1d])
                field_expr = Expr(:., :cp1d, QuoteNode(:t_i_average))
                @test _uses_local_var(field_expr, local_vars) == true
            end

            @testset "field access - base is not local" begin
                # actor.dd where actor is function parameter (not local)
                local_vars = Set{Symbol}([:x])
                field_expr = Expr(:., :actor, QuoteNode(:dd))
                @test _uses_local_var(field_expr, local_vars) == false
            end

            @testset "nested field access - base is local" begin
                # cp1d.grid.rho_tor_norm where cp1d is local
                local_vars = Set{Symbol}([:cp1d])
                inner = Expr(:., :cp1d, QuoteNode(:grid))
                outer = Expr(:., inner, QuoteNode(:rho_tor_norm))
                @test _uses_local_var(outer, local_vars) == true
            end

            @testset "nested field access - base is not local" begin
                # actor.dd.core_profiles where actor is not local
                local_vars = Set{Symbol}([:x])
                inner = Expr(:., :actor, QuoteNode(:dd))
                outer = Expr(:., inner, QuoteNode(:core_profiles))
                @test _uses_local_var(outer, local_vars) == false
            end

            @testset "indexing - base is local" begin
                # arr[i] where arr is local
                local_vars = Set{Symbol}([:arr])
                ref_expr = Expr(:ref, :arr, :i)
                @test _uses_local_var(ref_expr, local_vars) == true
            end

            @testset "indexing - base is not local" begin
                # data[i] where data is not local
                local_vars = Set{Symbol}([:x])
                ref_expr = Expr(:ref, :data, :i)
                @test _uses_local_var(ref_expr, local_vars) == false
            end

            @testset "non-Expr input" begin
                local_vars = Set{Symbol}([:x])
                @test _uses_local_var(42, local_vars) == false
                @test _uses_local_var("string", local_vars) == false
            end

            @testset "function call with local arg" begin
                # foo(x) where x is local
                local_vars = Set{Symbol}([:x])
                call_expr = Expr(:call, :foo, :x)
                @test _uses_local_var(call_expr, local_vars) == true
            end

            @testset "function call with non-local args" begin
                local_vars = Set{Symbol}([:x])
                call_expr = Expr(:call, :foo, :y, :z)
                @test _uses_local_var(call_expr, local_vars) == false
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

            # ==================================================================
            # eltype(x) expression handling
            # ==================================================================

            @testset "eltype(x) with external variable" begin
                # eltype(input_array) where input_array is NOT a local variable
                eltype_expr = Expr(:call, :eltype, :input_array)
                types = Set{Any}([eltype_expr])
                local_vars = Set{Symbol}()  # input_array is not local
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                @test length(static_types) == 1
                @test eltype_expr in static_types
                @test !has_dynamic
            end

            @testset "eltype(x) with local variable" begin
                # eltype(x) where x IS a local variable
                eltype_expr = Expr(:call, :eltype, :x)
                types = Set{Any}([eltype_expr])
                local_vars = Set{Symbol}([:x])  # x is defined locally
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                @test isempty(static_types)
                @test has_dynamic  # Must fall back because x isn't defined at checkpoint time
            end

            @testset "eltype mixed with static types" begin
                eltype_expr = Expr(:call, :eltype, :input)
                types = Set{Any}([:Float64, eltype_expr])
                local_vars = Set{Symbol}()  # input is external
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                @test :Float64 in static_types
                @test eltype_expr in static_types
                @test length(static_types) == 2
                @test !has_dynamic
            end

            @testset "eltype with local causes fallback but keeps static" begin
                eltype_expr = Expr(:call, :eltype, :local_arr)
                types = Set{Any}([:Float64, eltype_expr])
                local_vars = Set{Symbol}([:local_arr])  # local_arr is defined locally
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                @test :Float64 in static_types
                @test eltype_expr ∉ static_types
                @test has_dynamic
            end

            @testset "eltype with complex expression (not symbol)" begin
                # eltype(some_func().result) - inner is not a Symbol
                inner = Expr(:., Expr(:call, :some_func), QuoteNode(:result))
                eltype_expr = Expr(:call, :eltype, inner)
                types = Set{Any}([eltype_expr])
                local_vars = Set{Symbol}()
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                # Should be safe since inner is not a simple Symbol in local_vars
                @test length(static_types) == 1
                @test !has_dynamic
            end

            # ==================================================================
            # Field access with local base variable
            # ==================================================================

            @testset "eltype(cp1d.field) where cp1d is local" begin
                # This is the key case: acquire!(pool, cp1d.t_i_average)
                # where cp1d = dd.core_profiles.profiles_1d[] is defined locally
                field_expr = Expr(:., :cp1d, QuoteNode(:t_i_average))
                eltype_expr = Expr(:call, :eltype, field_expr)
                types = Set{Any}([eltype_expr])
                local_vars = Set{Symbol}([:cp1d])  # cp1d is defined locally
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                # Should fall back because cp1d is local
                @test isempty(static_types)
                @test has_dynamic
            end

            @testset "eltype(actor.dd.field) where actor is NOT local" begin
                # actor is a function parameter, not local
                inner = Expr(:., :actor, QuoteNode(:dd))
                outer = Expr(:., inner, QuoteNode(:core_profiles))
                eltype_expr = Expr(:call, :eltype, outer)
                types = Set{Any}([eltype_expr])
                local_vars = Set{Symbol}()  # actor is not local
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                # Should be safe
                @test length(static_types) == 1
                @test !has_dynamic
            end

            @testset "eltype(arr[i]) where arr is local" begin
                ref_expr = Expr(:ref, :arr, :i)
                eltype_expr = Expr(:call, :eltype, ref_expr)
                types = Set{Any}([eltype_expr])
                local_vars = Set{Symbol}([:arr])
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                @test isempty(static_types)
                @test has_dynamic
            end

            @testset "mixed: local field access and static type" begin
                field_expr = Expr(:., :local_var, QuoteNode(:data))
                eltype_expr = Expr(:call, :eltype, field_expr)
                types = Set{Any}([:Float64, eltype_expr])
                local_vars = Set{Symbol}([:local_var])
                static_types, has_dynamic = _filter_static_types(types, local_vars)
                @test :Float64 in static_types
                @test length(static_types) == 1  # only Float64, not eltype expr
                @test has_dynamic
            end
        end

        @testset "_extract_acquire_types" begin
            # Single acquire! call
            @testset "single acquire!" begin
                expr = :(acquire!(pool, Float64, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1
            end

            # Multiple acquire! calls
            @testset "multiple acquire!" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = acquire!(pool, Int64, 5)
                end
                types = _extract_acquire_types(expr, :pool)
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
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1  # Set deduplicates
            end

            # Type parameter
            @testset "type parameter" begin
                expr = :(acquire!(pool, T, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :T in types
            end

            # Parametric type
            @testset "parametric type" begin
                expr = :(acquire!(pool, Vector{Float64}, 10))
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 1
                # It should capture the curly expression
            end

            # No acquire! calls
            @testset "no acquire!" begin
                expr = :(sum(x))
                types = _extract_acquire_types(expr, :pool)
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
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Float32 in types
            end

            # Qualified acquire! call (AdaptiveArrayPools.acquire!)
            @testset "qualified acquire!" begin
                expr = :(AdaptiveArrayPools.acquire!(pool, Int32, 5))
                types = _extract_acquire_types(expr, :pool)
                @test :Int32 in types
            end

            # AST pollution test: different pool should NOT be extracted
            @testset "different pool (no pollution)" begin
                expr = quote
                    v1 = acquire!(p1, Float64, 10)  # p1 uses Float64
                    v2 = acquire!(p2, Int, 10)       # p2 uses Int
                end
                # Extract for p1 - should only get Float64
                types_p1 = _extract_acquire_types(expr, :p1)
                @test :Float64 in types_p1
                @test :Int ∉ types_p1
                @test length(types_p1) == 1

                # Extract for p2 - should only get Int
                types_p2 = _extract_acquire_types(expr, :p2)
                @test :Int in types_p2
                @test :Float64 ∉ types_p2
                @test length(types_p2) == 1
            end

            # Unmatched pool returns empty
            @testset "unmatched pool" begin
                expr = :(acquire!(other_pool, Float64, 10))
                types = _extract_acquire_types(expr, :pool)
                @test isempty(types)
            end

            # ==================================================================
            # Similar-style form: acquire!(pool, x)
            # ==================================================================

            @testset "similar-style acquire!(pool, x)" begin
                expr = :(acquire!(pool, input_array))
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 1
                # Should generate eltype(input_array) expression
                type_expr = first(types)
                @test type_expr isa Expr
                @test type_expr.head == :call
                @test type_expr.args[1] == :eltype
                @test type_expr.args[2] == :input_array
            end

            @testset "similar-style with complex expression" begin
                expr = :(acquire!(pool, some_func().result))
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 1
                type_expr = first(types)
                @test type_expr.head == :call
                @test type_expr.args[1] == :eltype
            end

            @testset "mixed traditional and similar-style" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = acquire!(pool, input_array)
                end
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 2
                # Should have Float64 and eltype(input_array)
                has_float64 = any(t -> t == :Float64, types)
                has_eltype = any(t -> t isa Expr && t.head == :call && t.args[1] == :eltype, types)
                @test has_float64
                @test has_eltype
            end

            # ==================================================================
            # unsafe_acquire! support
            # ==================================================================

            @testset "unsafe_acquire! single call" begin
                expr = :(unsafe_acquire!(pool, Float64, 100))
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1
            end

            @testset "unsafe_acquire! multiple types" begin
                expr = quote
                    v1 = unsafe_acquire!(pool, Float64, 10, 10)
                    v2 = unsafe_acquire!(pool, Int32, 5)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Int32 in types
                @test length(types) == 2
            end

            @testset "mixed acquire! and unsafe_acquire!" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = unsafe_acquire!(pool, Int64, 20)
                    v3 = acquire!(pool, input)  # similar-style
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Int64 in types
                # Should have eltype(input)
                has_eltype = any(t -> t isa Expr && t.head == :call && t.args[1] == :eltype, types)
                @test has_eltype
                @test length(types) == 3
            end

            @testset "qualified unsafe_acquire!" begin
                expr = :(AdaptiveArrayPools.unsafe_acquire!(pool, Int16, 5))
                types = _extract_acquire_types(expr, :pool)
                @test :Int16 in types
            end

            @testset "unsafe_acquire! different pool (no pollution)" begin
                expr = quote
                    v1 = unsafe_acquire!(p1, Float64, 10)
                    v2 = unsafe_acquire!(p2, Int, 10)
                end
                types_p1 = _extract_acquire_types(expr, :p1)
                @test :Float64 in types_p1
                @test :Int ∉ types_p1

                types_p2 = _extract_acquire_types(expr, :p2)
                @test :Int in types_p2
                @test :Float64 ∉ types_p2
            end

            # ==================================================================
            # Custom types and type parameters
            # ==================================================================

            @testset "custom struct type" begin
                # MyCustomType is a user-defined type (just a symbol at macro time)
                expr = :(acquire!(pool, MyCustomType, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :MyCustomType in types
                @test length(types) == 1
            end

            @testset "multiple custom types" begin
                expr = quote
                    v1 = acquire!(pool, MyData, 10)
                    v2 = acquire!(pool, MyOtherType, 5)
                    v3 = unsafe_acquire!(pool, UserStruct, 3)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :MyData in types
                @test :MyOtherType in types
                @test :UserStruct in types
                @test length(types) == 3
            end

            @testset "type parameter T (from where clause)" begin
                # In: function foo(::Type{T}) where T
                #         @with_pool p begin
                #             v = acquire!(p, T, 10)
                #         end
                #     end
                # T is a type parameter, not a local variable
                expr = :(acquire!(pool, T, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :T in types
            end

            @testset "multiple type parameters" begin
                expr = quote
                    v1 = acquire!(pool, T, 10)
                    v2 = acquire!(pool, S, 5)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :T in types
                @test :S in types
            end

            @testset "mixed: builtin, custom, type parameter" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = acquire!(pool, MyCustomType, 5)
                    v3 = acquire!(pool, T, 3)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :MyCustomType in types
                @test :T in types
                @test length(types) == 3
            end

            # ==================================================================
            # acquire_view! support (alias for acquire!)
            # ==================================================================

            @testset "acquire_view! single call" begin
                expr = :(acquire_view!(pool, Float64, 100))
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1
            end

            @testset "acquire_view! multiple types" begin
                expr = quote
                    v1 = acquire_view!(pool, Float64, 10)
                    v2 = acquire_view!(pool, Int32, 5)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Int32 in types
                @test length(types) == 2
            end

            @testset "acquire_view! similar-style" begin
                expr = :(acquire_view!(pool, input_array))
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 1
                type_expr = first(types)
                @test type_expr isa Expr
                @test type_expr.head == :call
                @test type_expr.args[1] == :eltype
                @test type_expr.args[2] == :input_array
            end

            @testset "qualified acquire_view!" begin
                expr = :(AdaptiveArrayPools.acquire_view!(pool, Int16, 5))
                types = _extract_acquire_types(expr, :pool)
                @test :Int16 in types
            end

            # ==================================================================
            # acquire_array! support (alias for unsafe_acquire!)
            # ==================================================================

            @testset "acquire_array! single call" begin
                expr = :(acquire_array!(pool, Float64, 100))
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1
            end

            @testset "acquire_array! multiple types" begin
                expr = quote
                    v1 = acquire_array!(pool, Float64, 10, 10)
                    v2 = acquire_array!(pool, Int32, 5)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Int32 in types
                @test length(types) == 2
            end

            @testset "qualified acquire_array!" begin
                expr = :(AdaptiveArrayPools.acquire_array!(pool, Int16, 5))
                types = _extract_acquire_types(expr, :pool)
                @test :Int16 in types
            end

            # ==================================================================
            # Mixed: all acquire functions together
            # ==================================================================

            @testset "all acquire functions mixed" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = unsafe_acquire!(pool, Int64, 20)
                    v3 = acquire_view!(pool, Float32, 5)
                    v4 = acquire_array!(pool, Int32, 15)
                    v5 = acquire_view!(pool, input)  # similar-style
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Int64 in types
                @test :Float32 in types
                @test :Int32 in types
                # Should have eltype(input)
                has_eltype = any(t -> t isa Expr && t.head == :call && t.args[1] == :eltype, types)
                @test has_eltype
                @test length(types) == 5
            end

            @testset "aliases different pool (no pollution)" begin
                expr = quote
                    v1 = acquire_view!(p1, Float64, 10)
                    v2 = acquire_array!(p2, Int, 10)
                end
                types_p1 = _extract_acquire_types(expr, :p1)
                @test :Float64 in types_p1
                @test :Int ∉ types_p1

                types_p2 = _extract_acquire_types(expr, :p2)
                @test :Int in types_p2
                @test :Float64 ∉ types_p2
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
                types = _extract_acquire_types(expr, :pool)
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
                types = _extract_acquire_types(expr, :pool)
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
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :T ∉ static_types
                @test has_dynamic
            end

            # Integration test for AST pollution fix
            @testset "multi-pool isolation" begin
                expr = quote
                    v1 = acquire!(p1, Float64, 10)
                    v2 = acquire!(p2, Int, 5)
                    v3 = acquire!(p1, Int32, 3)
                end
                local_vars = _extract_local_assignments(expr)

                # p1 should only have Float64 and Int32
                types_p1 = _extract_acquire_types(expr, :p1)
                static_p1, _ = _filter_static_types(types_p1, local_vars)
                @test :Float64 in static_p1
                @test :Int32 in static_p1
                @test :Int ∉ static_p1

                # p2 should only have Int
                types_p2 = _extract_acquire_types(expr, :p2)
                static_p2, _ = _filter_static_types(types_p2, local_vars)
                @test :Int in static_p2
                @test :Float64 ∉ static_p2
            end

            # ==================================================================
            # Integration tests for new features
            # ==================================================================

            @testset "similar-style with external array" begin
                expr = quote
                    v = acquire!(pool, input_array)  # input_array is function param
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                # Should have eltype(input_array) in static_types
                @test length(static_types) == 1
                @test !has_dynamic
                type_expr = first(static_types)
                @test type_expr isa Expr
                @test type_expr.args[1] == :eltype
            end

            @testset "similar-style with local array (fallback)" begin
                expr = quote
                    local_arr = rand(10)
                    v = acquire!(pool, local_arr)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                # Should fall back because local_arr is defined locally
                @test isempty(static_types)
                @test has_dynamic
            end

            @testset "similar-style with local.field (fallback)" begin
                # Real-world case: cp1d = dd.core_profiles.profiles_1d[]
                # then: acquire!(pool, cp1d.t_i_average)
                expr = quote
                    cp1d = dd.core_profiles.profiles_1d[]
                    v = acquire!(pool, cp1d.t_i_average)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                # Should fall back because cp1d is defined locally
                @test :cp1d in local_vars
                @test isempty(static_types)
                @test has_dynamic
            end

            @testset "similar-style with param.field (no fallback)" begin
                # actor.dd.field where actor is function parameter
                expr = quote
                    v = acquire!(pool, actor.dd.array_field)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                # Should be static because actor is not local
                @test :actor ∉ local_vars
                @test length(static_types) == 1
                @test !has_dynamic
            end

            @testset "chained local assignments (fallback)" begin
                # tmp = A.b; tmp1 = tmp.c; tmp2 = tmp1.d; acquire!(pool, tmp2)
                expr = quote
                    tmp = A.b
                    tmp1 = tmp.c
                    tmp2 = tmp1.d
                    v = acquire!(pool, tmp2)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                # All intermediates should be detected as local
                @test :tmp in local_vars
                @test :tmp1 in local_vars
                @test :tmp2 in local_vars
                # Should fall back because tmp2 is local
                @test isempty(static_types)
                @test has_dynamic
            end

            @testset "unsafe_acquire! integration" begin
                expr = quote
                    v1 = unsafe_acquire!(pool, Float64, 10, 10)
                    v2 = acquire!(pool, Int64, 5)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :Int64 in static_types
                @test !has_dynamic
            end

            @testset "mixed: acquire!, unsafe_acquire!, similar-style" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = unsafe_acquire!(pool, Int32, 5)
                    v3 = acquire!(pool, external_input)  # external
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :Int32 in static_types
                has_eltype = any(t -> t isa Expr && t.head == :call && t.args[1] == :eltype, static_types)
                @test has_eltype
                @test length(static_types) == 3
                @test !has_dynamic
            end

            # ==================================================================
            # Custom types and type parameters integration
            # ==================================================================

            @testset "custom type integration" begin
                expr = quote
                    v = acquire!(pool, MyCustomType, 10)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :MyCustomType in static_types
                @test !has_dynamic
            end

            @testset "type parameter integration" begin
                # T is a type parameter from where clause, not a local variable
                expr = quote
                    v = acquire!(pool, T, 10)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                # T is not in local_vars (it's from where clause), so it's static
                @test :T in static_types
                @test !has_dynamic
            end

            @testset "type parameter vs local variable conflict" begin
                # If someone shadows T with local assignment, it should trigger fallback
                expr = quote
                    T = eltype(x)  # T is now a local variable!
                    v = acquire!(pool, T, 10)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :T in local_vars
                @test :T ∉ static_types  # T is filtered out
                @test has_dynamic  # Falls back to full checkpoint
            end

            @testset "mixed: builtin, custom, type parameter" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = acquire!(pool, MyData, 5)
                    v3 = unsafe_acquire!(pool, T, 3)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :MyData in static_types
                @test :T in static_types
                @test length(static_types) == 3
                @test !has_dynamic
            end

            # ==================================================================
            # Integration tests for alias functions
            # ==================================================================

            @testset "acquire_view! integration" begin
                expr = quote
                    v1 = acquire_view!(pool, Float64, 10)
                    v2 = acquire_view!(pool, Int64, 5)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :Int64 in static_types
                @test !has_dynamic
            end

            @testset "acquire_array! integration" begin
                expr = quote
                    v1 = acquire_array!(pool, Float64, 10, 10)
                    v2 = acquire_array!(pool, Int64, 5)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :Int64 in static_types
                @test !has_dynamic
            end

            @testset "acquire_view! similar-style with external array" begin
                expr = quote
                    v = acquire_view!(pool, input_array)  # input_array is function param
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test length(static_types) == 1
                @test !has_dynamic
                type_expr = first(static_types)
                @test type_expr isa Expr
                @test type_expr.args[1] == :eltype
            end

            @testset "all acquire functions integration" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = unsafe_acquire!(pool, Int64, 20)
                    v3 = acquire_view!(pool, Float32, 5)
                    v4 = acquire_array!(pool, Int32, 15)
                    v5 = acquire_view!(pool, external_input)
                end
                local_vars = _extract_local_assignments(expr)
                types = _extract_acquire_types(expr, :pool)
                static_types, has_dynamic = _filter_static_types(types, local_vars)

                @test :Float64 in static_types
                @test :Int64 in static_types
                @test :Float32 in static_types
                @test :Int32 in static_types
                has_eltype = any(t -> t isa Expr && t.head == :call && t.args[1] == :eltype, static_types)
                @test has_eltype
                @test length(static_types) == 5
                @test !has_dynamic
            end

            # ==================================================================
            # Convenience functions (zeros!, ones!, similar!)
            # ==================================================================

            @testset "zeros! default type (Float64)" begin
                expr = :(v = zeros!(pool, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1
            end

            @testset "zeros! explicit type" begin
                expr = :(v = zeros!(pool, Float32, 10, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :Float32 in types
                @test length(types) == 1
            end

            @testset "ones! default type (Float64)" begin
                expr = :(v = ones!(pool, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test length(types) == 1
            end

            @testset "ones! explicit type" begin
                expr = :(v = ones!(pool, Int64, 5, 5))
                types = _extract_acquire_types(expr, :pool)
                @test :Int64 in types
                @test length(types) == 1
            end

            @testset "similar! same type as template (nargs == 3)" begin
                expr = :(v = similar!(pool, template))
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 1
                type_expr = first(types)
                @test type_expr isa Expr
                @test type_expr.head == :call
                @test type_expr.args[1] == :eltype
                @test type_expr.args[2] == :template
            end

            @testset "similar! explicit type (nargs >= 4, type arg)" begin
                expr = :(v = similar!(pool, template, Float32))
                types = _extract_acquire_types(expr, :pool)
                @test :Float32 in types
                @test length(types) == 1
            end

            @testset "similar! explicit type with dims (nargs >= 4, type + dims)" begin
                expr = :(v = similar!(pool, template, Int64, 10, 10))
                types = _extract_acquire_types(expr, :pool)
                @test :Int64 in types
                @test length(types) == 1
            end

            @testset "similar! same type with different dims (nargs >= 4, dims only)" begin
                expr = :(v = similar!(pool, template, 5, 5))
                types = _extract_acquire_types(expr, :pool)
                @test length(types) == 1
                type_expr = first(types)
                @test type_expr isa Expr
                @test type_expr.head == :call
                @test type_expr.args[1] == :eltype
                @test type_expr.args[2] == :template
            end

            @testset "mixed convenience functions" begin
                expr = quote
                    v1 = zeros!(pool, Float64, 10)
                    v2 = ones!(pool, Float32, 5)
                    v3 = similar!(pool, template)
                    v4 = similar!(pool, template, Int64)
                end
                types = _extract_acquire_types(expr, :pool)
                @test :Float64 in types
                @test :Float32 in types
                @test :Int64 in types
                has_eltype = any(t -> t isa Expr && t.head == :call && t.args[1] == :eltype, types)
                @test has_eltype
                @test length(types) == 4
            end
        end

    end

    # ==========================================================================
    # _transform_acquire_calls Tests
    # ==========================================================================

    @testset "_transform_acquire_calls" begin
        using AdaptiveArrayPools: _transform_acquire_calls, _ACQUIRE_IMPL_REF, _UNSAFE_ACQUIRE_IMPL_REF

        @testset "basic transformation" begin
            @testset "acquire! → _acquire_impl!" begin
                expr = :(acquire!(pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
                @test transformed.args[2] == :pool
                @test transformed.args[3] == :Float64
                @test transformed.args[4] == 10
            end

            @testset "unsafe_acquire! → _unsafe_acquire_impl!" begin
                expr = :(unsafe_acquire!(pool, Float64, 10, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _UNSAFE_ACQUIRE_IMPL_REF
                @test transformed.args[2] == :pool
            end

            @testset "acquire_view! → _acquire_impl!" begin
                expr = :(acquire_view!(pool, Int32, 5))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
            end

            @testset "acquire_array! → _unsafe_acquire_impl!" begin
                expr = :(acquire_array!(pool, Int64, 3, 4))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _UNSAFE_ACQUIRE_IMPL_REF
            end
        end

        @testset "qualified names" begin
            @testset "AdaptiveArrayPools.acquire!" begin
                expr = :(AdaptiveArrayPools.acquire!(pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
                @test transformed.args[2] == :pool
            end

            @testset "user alias: AAP.acquire!" begin
                # User might use: const AAP = AdaptiveArrayPools
                # Then call: AAP.acquire!(pool, Float64, 10)
                expr = :(AAP.acquire!(pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
                @test transformed.args[2] == :pool
            end

            @testset "deep nesting: A.B.acquire!" begin
                expr = :(SomeModule.SubModule.acquire!(pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
            end

            @testset "very deep nesting: A.B.C.D.acquire!" begin
                expr = :(A.B.C.D.acquire!(pool, Int32, 5))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
            end

            @testset "qualified unsafe_acquire!" begin
                expr = :(SomeAlias.unsafe_acquire!(pool, Float64, 10, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _UNSAFE_ACQUIRE_IMPL_REF
            end

            @testset "qualified acquire_view!" begin
                expr = :(Pkg.acquire_view!(pool, Int64, 5))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
            end

            @testset "qualified acquire_array!" begin
                expr = :(MyModule.acquire_array!(pool, Float32, 3, 4, 5))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _UNSAFE_ACQUIRE_IMPL_REF
            end
        end

        @testset "pool name matching" begin
            @testset "different pool name - no transform" begin
                expr = :(acquire!(other_pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                # Should NOT be transformed because pool name doesn't match
                @test transformed.args[1] == :acquire!
            end

            @testset "qualified with different pool - no transform" begin
                expr = :(AAP.acquire!(other_pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                # Should NOT be transformed
                fn = transformed.args[1]
                @test fn isa Expr && fn.head == :.
            end
        end

        @testset "recursive transformation" begin
            @testset "nested in block" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)
                    v2 = unsafe_acquire!(pool, Int64, 5)
                end
                transformed = _transform_acquire_calls(expr, :pool)
                # Find the transformed calls in the block
                calls = filter(x -> x isa Expr && x.head == :(=), transformed.args)
                @test length(calls) >= 2
            end

            @testset "nested in function call" begin
                expr = :(sum(acquire!(pool, Float64, 10)))
                transformed = _transform_acquire_calls(expr, :pool)
                # The inner acquire! should be transformed
                inner_call = transformed.args[2]
                @test inner_call.args[1] == _ACQUIRE_IMPL_REF
            end

            @testset "mixed transformed and untransformed" begin
                expr = quote
                    v1 = acquire!(pool, Float64, 10)  # Should transform
                    v2 = acquire!(other, Int64, 5)    # Should NOT transform
                end
                transformed = _transform_acquire_calls(expr, :pool)
                # One should be transformed, one should not
                has_impl_ref = false
                has_acquire = false
                for arg in transformed.args
                    if arg isa Expr
                        str = string(arg)
                        if occursin("_acquire_impl!", str)
                            has_impl_ref = true
                        end
                        if occursin("acquire!(other", str)
                            has_acquire = true
                        end
                    end
                end
                @test has_impl_ref
                @test has_acquire
            end
        end

        @testset "similar-style transformation" begin
            @testset "acquire!(pool, x)" begin
                expr = :(acquire!(pool, input_array))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
                @test transformed.args[2] == :pool
                @test transformed.args[3] == :input_array
            end

            @testset "qualified similar-style" begin
                expr = :(AAP.acquire!(pool, some_matrix))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ACQUIRE_IMPL_REF
            end
        end

        @testset "convenience function transformation" begin
            using AdaptiveArrayPools: _ZEROS_IMPL_REF, _ONES_IMPL_REF, _SIMILAR_IMPL_REF

            @testset "zeros! → _zeros_impl!" begin
                expr = :(zeros!(pool, Float64, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ZEROS_IMPL_REF
                @test transformed.args[2] == :pool
                @test transformed.args[3] == :Float64
            end

            @testset "ones! → _ones_impl!" begin
                expr = :(ones!(pool, Int64, 5, 5))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ONES_IMPL_REF
                @test transformed.args[2] == :pool
                @test transformed.args[3] == :Int64
            end

            @testset "similar! → _similar_impl!" begin
                expr = :(similar!(pool, template))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _SIMILAR_IMPL_REF
                @test transformed.args[2] == :pool
                @test transformed.args[3] == :template
            end

            @testset "qualified zeros! → _zeros_impl!" begin
                expr = :(AAP.zeros!(pool, Float32, 10))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ZEROS_IMPL_REF
                @test transformed.args[2] == :pool
            end

            @testset "qualified ones! → _ones_impl!" begin
                expr = :(AAP.ones!(pool, Int32, 5))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _ONES_IMPL_REF
                @test transformed.args[2] == :pool
            end

            @testset "qualified similar! → _similar_impl!" begin
                expr = :(AAP.similar!(pool, arr, Float64))
                transformed = _transform_acquire_calls(expr, :pool)
                @test transformed.args[1] == _SIMILAR_IMPL_REF
                @test transformed.args[2] == :pool
            end
        end

        @testset "GlobalRef verification" begin
            using AdaptiveArrayPools: _ZEROS_IMPL_REF, _ONES_IMPL_REF, _SIMILAR_IMPL_REF

            # Verify that GlobalRef points to AdaptiveArrayPools module
            @test _ACQUIRE_IMPL_REF isa GlobalRef
            @test _UNSAFE_ACQUIRE_IMPL_REF isa GlobalRef
            @test _ZEROS_IMPL_REF isa GlobalRef
            @test _ONES_IMPL_REF isa GlobalRef
            @test _SIMILAR_IMPL_REF isa GlobalRef

            @test _ACQUIRE_IMPL_REF.mod == AdaptiveArrayPools
            @test _UNSAFE_ACQUIRE_IMPL_REF.mod == AdaptiveArrayPools
            @test _ZEROS_IMPL_REF.mod == AdaptiveArrayPools
            @test _ONES_IMPL_REF.mod == AdaptiveArrayPools
            @test _SIMILAR_IMPL_REF.mod == AdaptiveArrayPools

            @test _ACQUIRE_IMPL_REF.name == :_acquire_impl!
            @test _UNSAFE_ACQUIRE_IMPL_REF.name == :_unsafe_acquire_impl!
            @test _ZEROS_IMPL_REF.name == :_zeros_impl!
            @test _ONES_IMPL_REF.name == :_ones_impl!
            @test _SIMILAR_IMPL_REF.name == :_similar_impl!
        end
    end

end # Macro Internals