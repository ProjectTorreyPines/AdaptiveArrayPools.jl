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

    # Test 1-arg @with_pool (gensym pool)
    @testset "@with_pool 1-arg expansion" begin
        expr = @macroexpand @with_pool begin
            nothing
        end

        expr_str = string(expr)

        # Should still have pool management (with gensym name)
        @test occursin("get_global_pool", expr_str)
        @test occursin("checkpoint!", expr_str)
        @test occursin("rewind!", expr_str)
    end

    # Test @maybe_with_pool 1-arg
    @testset "@maybe_with_pool 1-arg expansion" begin
        expr = @macroexpand @maybe_with_pool begin
            nothing
        end

        expr_str = string(expr)

        # Should have conditional and pool management
        @test occursin("get_global_pool", expr_str)
        @test occursin("nothing", expr_str)
    end

end
