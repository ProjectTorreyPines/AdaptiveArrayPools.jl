# ==============================================================================
# Tests for backend-specific macro expansion (@with_pool :cuda, etc.)
# ==============================================================================
#
# These tests verify the structure of backend-specific macro-generated code
# WITHOUT requiring the actual backend (CUDA, etc.) to be installed.
# This ensures macro logic is correct regardless of extension availability.

@testset "Backend Macro Expansion" begin

    # ==========================================================================
    # Block Form: @with_pool :backend pool begin ... end
    # ==========================================================================

    @testset "Block form expansion" begin

        @testset "Basic structure" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, Float64, 10)
                sum(v)
            end

            @test expr isa Expr
            expr_str = string(expr)

            # Should use _get_pool_for_backend dispatch
            @test occursin("_get_pool_for_backend", expr_str)
            @test occursin("Val{:cuda}", expr_str)

            # Should have checkpoint/rewind
            @test occursin("checkpoint!", expr_str)
            @test occursin("rewind!", expr_str)

            # Should have try-finally
            @test occursin("try", expr_str)
            @test occursin("finally", expr_str)
        end

        @testset "Different backends" begin
            for backend in [:cuda, :rocm, :metal, :oneapi, :custom_backend]
                # Use @eval to dynamically construct the macroexpand call
                expr = @eval @macroexpand @with_pool $(QuoteNode(backend)) pool begin
                    v = acquire!(pool, Float64, 10)
                end

                expr_str = string(expr)
                @test occursin("Val{:$backend}", expr_str)
                @test occursin("_get_pool_for_backend", expr_str)
            end
        end

        @testset "Without pool name (gensym)" begin
            expr = @macroexpand @with_pool :cuda begin
                nothing
            end

            expr_str = string(expr)
            @test occursin("_get_pool_for_backend", expr_str)
            @test occursin("Val{:cuda}", expr_str)
            @test occursin("checkpoint!", expr_str)
            @test occursin("rewind!", expr_str)
        end

        @testset "Type extraction" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v1 = acquire!(pool, Float64, 10)
                v2 = acquire!(pool, Float32, 5)
            end

            expr_str = string(expr)
            @test occursin("Float64", expr_str)
            @test occursin("Float32", expr_str)
        end

        @testset "unsafe_acquire! type extraction" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = unsafe_acquire!(pool, Int64, 100)
            end

            expr_str = string(expr)
            @test occursin("Int64", expr_str)
        end

        @testset "Similar-style acquire!(pool, x)" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, input_array)
            end

            expr_str = string(expr)
            @test occursin("eltype", expr_str)
            @test occursin("input_array", expr_str)
        end

        @testset "Custom types" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, MyCustomType, 10)
            end

            expr_str = string(expr)
            @test occursin("MyCustomType", expr_str)
        end

        @testset "Type parameters" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, T, 10)
            end

            expr_str = string(expr)
            @test occursin(r"\bT\b", expr_str)
        end
    end

    # ==========================================================================
    # Function Form: @with_pool :backend pool function f() ... end
    # ==========================================================================

    @testset "Function form expansion" begin

        @testset "Basic structure" begin
            expr = @macroexpand @with_pool :cuda pool function my_func(n::Int)
                v = acquire!(pool, Float64, n)
                return sum(v)
            end

            @test expr isa Expr

            # Should be a function definition (not a block wrapping a function)
            @test expr.head == :function || (expr.head == :(=) && expr.args[1] isa Expr)

            expr_str = string(expr)

            # Function name should be preserved
            @test occursin("my_func", expr_str)

            # Pool getter should be INSIDE the function body
            @test occursin("_get_pool_for_backend", expr_str)
            @test occursin("Val{:cuda}", expr_str)

            # checkpoint/rewind should be INSIDE the function
            @test occursin("checkpoint!", expr_str)
            @test occursin("rewind!", expr_str)
        end

        @testset "Pool/checkpoint/rewind inside function body" begin
            expr = @macroexpand @with_pool :cuda pool function compute(n)
                A = acquire!(pool, Float32, n, n)
                return sum(A)
            end

            # Verify structure: function definition with body containing pool operations
            @test expr.head == :function

            # The function body (args[2]) should contain the pool operations
            body = expr.args[2]
            body_str = string(body)

            @test occursin("_get_pool_for_backend", body_str)
            @test occursin("checkpoint!", body_str)
            @test occursin("try", body_str)
            @test occursin("finally", body_str)
            @test occursin("rewind!", body_str)
        end

        @testset "Function signature preserved" begin
            expr = @macroexpand @with_pool :cuda pool function typed_func(x::Vector{Float64}, n::Int)::Float64
                v = acquire!(pool, Float64, n)
                return sum(v)
            end

            @test expr.head == :function
            call_expr = expr.args[1]

            # Call expression should have the function name and args
            call_str = string(call_expr)
            @test occursin("typed_func", call_str)
            @test occursin("Vector{Float64}", call_str)
            @test occursin("n::Int", call_str)
        end

        @testset "Short function syntax" begin
            expr = @macroexpand @with_pool :cuda pool f(x) = acquire!(pool, Float64, x)

            # Should still produce a function
            @test expr.head == :(=) || expr.head == :function
            expr_str = string(expr)
            @test occursin("_get_pool_for_backend", expr_str)
        end

        @testset "Type extraction in function form" begin
            expr = @macroexpand @with_pool :cuda pool function multi_type(n)
                A = acquire!(pool, Float64, n)
                B = acquire!(pool, Int32, n)
                C = unsafe_acquire!(pool, Float32, n)
                return sum(A) + sum(B) + sum(C)
            end

            body_str = string(expr.args[2])
            @test occursin("Float64", body_str)
            @test occursin("Int32", body_str)
            @test occursin("Float32", body_str)
        end

        @testset "Different backends with function form" begin
            for backend in [:cuda, :rocm, :metal]
                # Use @eval to dynamically construct the macroexpand call
                expr = @eval @macroexpand @with_pool $(QuoteNode(backend)) pool function backend_func(n)
                    acquire!(pool, Float64, n)
                end

                expr_str = string(expr)
                @test occursin("Val{:$backend}", expr_str)
                @test expr.head == :function
            end
        end

        @testset "Where clause preserved" begin
            expr = @macroexpand @with_pool :cuda pool function generic_func(x::Vector{T}) where T
                v = acquire!(pool, T, length(x))
                return sum(v)
            end

            expr_str = string(expr)
            @test occursin("where", expr_str)
            @test occursin(r"\bT\b", expr_str)
        end
    end

    # ==========================================================================
    # acquire! → _acquire_impl! transformation
    # ==========================================================================

    @testset "acquire! transformation" begin

        @testset "Block form transforms acquire!" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, Float64, 10)
            end

            expr_str = string(expr)
            # Should transform to _acquire_impl!
            @test occursin("_acquire_impl!", expr_str)
        end

        @testset "Function form transforms acquire!" begin
            expr = @macroexpand @with_pool pool function my_func(n)
                v = acquire!(pool, Float64, n)
            end

            expr_str = string(expr)
            @test occursin("_acquire_impl!", expr_str)
        end

        @testset "unsafe_acquire! transforms" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = unsafe_acquire!(pool, Float64, 10, 10)
            end

            expr_str = string(expr)
            @test occursin("_unsafe_acquire_impl!", expr_str)
        end

        @testset "acquire_view! transforms" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire_view!(pool, Float64, 10)
            end

            expr_str = string(expr)
            @test occursin("_acquire_impl!", expr_str)
        end

        @testset "acquire_array! transforms" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire_array!(pool, Float64, 10, 10)
            end

            expr_str = string(expr)
            @test occursin("_unsafe_acquire_impl!", expr_str)
        end
    end

    # ==========================================================================
    # Typed checkpoint/rewind optimization
    # ==========================================================================

    @testset "Typed checkpoint/rewind" begin

        @testset "Single type uses typed checkpoint" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, Float64, 10)
            end

            expr_str = string(expr)
            # Should have Float64 in checkpoint call
            @test occursin("Float64", expr_str)
            @test occursin("checkpoint!", expr_str)
        end

        @testset "Multiple types in checkpoint" begin
            expr = @macroexpand @with_pool :cuda pool begin
                v1 = acquire!(pool, Float64, 10)
                v2 = acquire!(pool, Int64, 5)
                v3 = acquire!(pool, Float32, 3)
            end

            expr_str = string(expr)
            @test occursin("Float64", expr_str)
            @test occursin("Int64", expr_str)
            @test occursin("Float32", expr_str)
        end

        @testset "Local variable causes full checkpoint" begin
            expr = @macroexpand @with_pool :cuda pool begin
                T = eltype(some_array)
                v = acquire!(pool, T, 10)
            end

            expr_str = string(expr)
            # When type is a local variable, should use full checkpoint without type args
            # Check for checkpoint!(pool) pattern - the string form has AdaptiveArrayPools prefix
            @test occursin("checkpoint!", expr_str) && occursin("(pool)", expr_str)
        end

        @testset "Function form typed checkpoint" begin
            expr = @macroexpand @with_pool :cuda pool function typed_checkpoint_func(n)
                v1 = acquire!(pool, Float64, n)
                v2 = acquire!(pool, Float32, n)
            end

            body_str = string(expr.args[2])
            @test occursin("Float64", body_str)
            @test occursin("Float32", body_str)
        end
    end

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    @testset "Edge cases" begin

        @testset "Empty block" begin
            expr = @macroexpand @with_pool :cuda pool begin
            end

            expr_str = string(expr)
            @test occursin("_get_pool_for_backend", expr_str)
        end

        @testset "Nested @with_pool" begin
            expr = @macroexpand @with_pool :cuda outer begin
                v1 = acquire!(outer, Float64, 10)
                @with_pool inner begin
                    v2 = acquire!(inner, Float32, 5)
                end
            end

            expr_str = string(expr)
            # Outer should use backend dispatch
            @test occursin("Val{:cuda}", expr_str)
            # Inner should use task-local pool
            @test occursin("get_task_local_pool", expr_str)
        end

        @testset "Mixed backend and regular pools" begin
            expr = @macroexpand @with_pool outer begin
                v1 = acquire!(outer, Float64, 10)
                @with_pool :cuda inner begin
                    v2 = acquire!(inner, Float32, 5)
                end
            end

            expr_str = string(expr)
            @test occursin("get_task_local_pool", expr_str)
            @test occursin("Val{:cuda}", expr_str)
        end

        @testset "Complex function signature" begin
            expr = @macroexpand @with_pool :cuda pool function complex_func(
                    x::AbstractArray{T},
                    y::AbstractArray{S};
                    tol::Float64 = 1e-6
                ) where {T <: Real, S <: Real}
                v = acquire!(pool, T, size(x))
                return sum(v)
            end

            @test expr.head == :function
            expr_str = string(expr)
            @test occursin("complex_func", expr_str)
            @test occursin("tol", expr_str)
            @test occursin("where", expr_str)
        end
    end

    # ==========================================================================
    # Comparison with regular @with_pool
    # ==========================================================================

    @testset "Backend vs regular @with_pool consistency" begin

        @testset "Block form structure matches" begin
            expr_regular = @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
            end

            expr_backend = @macroexpand @with_pool :cuda pool begin
                v = acquire!(pool, Float64, 10)
            end

            # Both should have checkpoint/rewind/try-finally
            for expr in [expr_regular, expr_backend]
                expr_str = string(expr)
                @test occursin("checkpoint!", expr_str)
                @test occursin("rewind!", expr_str)
                @test occursin("try", expr_str)
                @test occursin("finally", expr_str)
            end
        end

        @testset "Function form structure matches" begin
            expr_regular = @macroexpand @with_pool pool function regular_func(n)
                v = acquire!(pool, Float64, n)
            end

            expr_backend = @macroexpand @with_pool :cuda pool function backend_func(n)
                v = acquire!(pool, Float64, n)
            end

            # Both should be function definitions
            @test expr_regular.head == :function
            @test expr_backend.head == :function

            # Both should have pool operations inside function body
            for expr in [expr_regular, expr_backend]
                body_str = string(expr.args[2])
                @test occursin("checkpoint!", body_str)
                @test occursin("rewind!", body_str)
            end
        end
    end

end # Backend Macro Expansion
