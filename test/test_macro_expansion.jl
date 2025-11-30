# ==============================================================================
# Test macro expansion structure
# ==============================================================================
#
# These tests verify the structure of macro-generated code without execution.
# Useful for ensuring macro logic is correct regardless of runtime flags.

@testset "Macro expansion structure" begin
    # Test @with_pool block mode expansion
    @testset "@with_pool block expansion" begin
        expr = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end

        # Should be a let block or quote block
        @test expr isa Expr

        # Should contain get_global_pool() call
        expr_str = string(expr)
        @test occursin("get_global_pool", expr_str)

        # Should contain checkpoint! and rewind!
        @test occursin("checkpoint!", expr_str)
        @test occursin("rewind!", expr_str)

        # Should have try-finally structure
        @test occursin("try", expr_str)
        @test occursin("finally", expr_str)
    end

    # Test @with_pool function mode expansion
    @testset "@with_pool function expansion" begin
        expr = @macroexpand @with_pool pool function test_func(n)
            v = acquire!(pool, Float64, n)
            sum(v)
        end

        expr_str = string(expr)

        # Should define a function
        @test occursin("function", expr_str)
        @test occursin("test_func", expr_str)

        # Should contain pool management
        @test occursin("get_global_pool", expr_str)
        @test occursin("checkpoint!", expr_str)
        @test occursin("rewind!", expr_str)
    end

    # Test @maybe_with_pool expansion (has MAYBE_POOLING_ENABLED branch)
    @testset "@maybe_with_pool expansion" begin
        expr = @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end

        expr_str = string(expr)

        # Should contain conditional check (MAYBE_POOLING_ENABLED is inlined as RefValue)
        @test occursin("RefValue", expr_str) || occursin("if", expr_str)

        # Should have both branches (pool and nothing)
        @test occursin("get_global_pool", expr_str)
        @test occursin("nothing", expr_str)
    end

    # Test @pool_kwarg expansion
    @testset "@pool_kwarg expansion" begin
        expr = @macroexpand @pool_kwarg pool function layer(x)
            out = acquire!(pool, Float64, length(x))
            out .= x
        end

        expr_str = string(expr)

        # Should define a function with pool kwarg
        @test occursin("function", expr_str)
        @test occursin("layer", expr_str)

        # Should have pool as keyword argument with Union type
        @test occursin("Union", expr_str) || occursin("pool", expr_str)

        # Should NOT contain checkpoint!/rewind! (caller's responsibility)
        @test !occursin("checkpoint!", expr_str)
        @test !occursin("rewind!", expr_str)
    end

    # Test typed checkpoint optimization
    @testset "Typed checkpoint optimization" begin
        # Single type should use typed checkpoint
        expr = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end

        expr_str = string(expr)

        # Should use typed checkpoint with Float64
        # The macro detects Float64 and generates checkpoint!(pool, Float64)
        @test occursin("Float64", expr_str)
    end

    # Test multiple types
    @testset "Multiple types checkpoint" begin
        expr = @macroexpand @with_pool pool begin
            v1 = acquire!(pool, Float64, 10)
            v2 = acquire!(pool, Int64, 5)
            sum(v1) + sum(v2)
        end

        expr_str = string(expr)

        # Should contain both types
        @test occursin("Float64", expr_str)
        @test occursin("Int64", expr_str)
    end

    # Test POOL_DEBUG validation in block mode
    @testset "POOL_DEBUG validation in expansion" begin
        expr = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            sum(v)
        end

        expr_str = string(expr)

        # POOL_DEBUG is inlined as RefValue, but _validate_pool_return should be present
        @test occursin("_validate_pool_return", expr_str)
    end

    # Test short-form function
    @testset "Short-form function expansion" begin
        expr = @macroexpand @with_pool pool f(x) = acquire!(pool, Float64, length(x))

        expr_str = string(expr)

        # Should be a function definition
        @test occursin("get_global_pool", expr_str)
        @test occursin("checkpoint!", expr_str)
    end

    # Test function with where clause
    @testset "Function with where clause" begin
        expr = @macroexpand @with_pool pool function typed_func(x::Vector{T}) where {T<:Number}
            v = acquire!(pool, T, length(x))
            v .= x
        end

        expr_str = string(expr)

        # Should preserve where clause
        @test occursin("where", expr_str)
        @test occursin("Number", expr_str)

        # Type parameter T should be in checkpoint (not filtered out)
        @test occursin("checkpoint!", expr_str)
    end

    # Test @pool_kwarg with existing kwargs
    @testset "@pool_kwarg with existing kwargs" begin
        expr = @macroexpand @pool_kwarg pool function with_kwargs(x; scale=1.0)
            v = acquire!(pool, Float64, length(x))
            v .= x .* scale
        end

        expr_str = string(expr)

        # Should preserve existing kwargs
        @test occursin("scale", expr_str)

        # Should add pool kwarg
        @test occursin("pool", expr_str)
    end
end
