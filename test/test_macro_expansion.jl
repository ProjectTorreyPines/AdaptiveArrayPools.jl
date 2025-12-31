# ==============================================================================
# Test macro expansion structure
# ==============================================================================
#
# These tests verify the structure of macro-generated code without execution.
# Useful for ensuring macro logic is correct regardless of runtime flags.

@testset "Macro Expansion Details" begin

    @testset "Macro expansion structure" begin
        # Test @with_pool block mode expansion
        @testset "@with_pool block expansion" begin
            expr = @macroexpand @with_pool pool begin
                v = acquire!(pool, Float64, 10)
                sum(v)
            end

            # Should be a let block or quote block
            @test expr isa Expr

            # Should contain get_task_local_pool() call
            expr_str = string(expr)
            @test occursin("get_task_local_pool", expr_str)

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

            # Should have both branches (pool getter and DisabledPool fallback)
            @test occursin("get_task_local_pool", expr_str)
            @test occursin("DisabledPool", expr_str)
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
            @test occursin("get_task_local_pool", expr_str)
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
            @test occursin("get_task_local_pool", expr_str)
            @test occursin("nothing", expr_str)
        end

        # ==================================================================
        # New feature expansion tests
        # ==================================================================

        @testset "Similar-style acquire!(pool, x) expansion" begin
            # Note: We test with a symbol that won't be in local_vars
            # to verify eltype() is generated in checkpoint/rewind
            expr = @macroexpand @with_pool pool begin
                # Simulate external input_array (not assigned in block)
                v = acquire!(pool, input_array)
                sum(v)
            end

            expr_str = string(expr)

            # Should contain eltype(input_array) in checkpoint/rewind
            @test occursin("eltype", expr_str)
            @test occursin("input_array", expr_str)
        end

        @testset "Similar-style with local array falls back" begin
            expr = @macroexpand @with_pool pool begin
                local_arr = rand(10)
                v = acquire!(pool, local_arr)
                sum(v)
            end

            expr_str = string(expr)

            # Should use full checkpoint (no type argument)
            # When local_arr is detected as local, it falls back
            # The checkpoint call should NOT have eltype
            # Check that checkpoint! is called (it will be full checkpoint)
            @test occursin("checkpoint!", expr_str)
            @test occursin("rewind!", expr_str)
        end

        @testset "unsafe_acquire! type extraction" begin
            expr = @macroexpand @with_pool pool begin
                v = unsafe_acquire!(pool, Float64, 100)
                sum(v)
            end

            expr_str = string(expr)

            # Should have typed checkpoint with Float64
            @test occursin("checkpoint!", expr_str)
            @test occursin("Float64", expr_str)
        end

        @testset "Mixed acquire! and unsafe_acquire!" begin
            expr = @macroexpand @with_pool pool begin
                v1 = acquire!(pool, Float64, 10)
                v2 = unsafe_acquire!(pool, Int64, 20)
                sum(v1) + sum(v2)
            end

            expr_str = string(expr)

            # Should have both types
            @test occursin("Float64", expr_str)
            @test occursin("Int64", expr_str)
        end

        @testset "Similar-style + traditional + unsafe mixed" begin
            expr = @macroexpand @with_pool pool begin
                v1 = acquire!(pool, Float64, 10)
                v2 = unsafe_acquire!(pool, Int32, 5)
                v3 = acquire!(pool, external_data)  # similar-style
                sum(v1) + sum(v2) + sum(v3)
            end

            expr_str = string(expr)

            # Should have Float64, Int32, and eltype(external_data)
            @test occursin("Float64", expr_str)
            @test occursin("Int32", expr_str)
            @test occursin("eltype", expr_str)
            @test occursin("external_data", expr_str)
        end

        # ==================================================================
        # Custom types and type parameters
        # ==================================================================

        @testset "Custom struct type expansion" begin
            # MyCustomType is treated as a symbol (user-defined type)
            expr = @macroexpand @with_pool pool begin
                v = acquire!(pool, MyCustomType, 10)
                length(v)
            end

            expr_str = string(expr)

            # Should have typed checkpoint with MyCustomType
            @test occursin("checkpoint!", expr_str)
            @test occursin("MyCustomType", expr_str)
        end

        @testset "Type parameter T expansion" begin
            # T represents a type parameter from where clause
            expr = @macroexpand @with_pool pool begin
                v = acquire!(pool, T, 10)
                length(v)
            end

            expr_str = string(expr)

            # Should pass T to checkpoint/rewind
            @test occursin("checkpoint!", expr_str)
            # T should appear in the expanded code
            @test occursin(r"\bT\b", expr_str)
        end

        @testset "Mixed: builtin, custom, type parameter" begin
            expr = @macroexpand @with_pool pool begin
                v1 = acquire!(pool, Float64, 10)
                v2 = acquire!(pool, MyData, 5)
                v3 = unsafe_acquire!(pool, T, 3)
                length(v1) + length(v2) + length(v3)
            end

            expr_str = string(expr)

            # All three types should be in checkpoint/rewind
            @test occursin("Float64", expr_str)
            @test occursin("MyData", expr_str)
            @test occursin(r"\bT\b", expr_str)
        end

    end

    @testset "Convenience functions expansion" begin

        @testset "zeros! default type uses default_eltype(pool)" begin
            expr = @macroexpand @with_pool pool begin
                v = zeros!(pool, 10)
            end

            expr_str = string(expr)

            # Should contain default_eltype(pool) for backend-flexible type detection
            @test occursin("default_eltype", expr_str)
            @test occursin("pool", expr_str)
        end

        @testset "zeros! explicit type uses that type" begin
            expr = @macroexpand @with_pool pool begin
                v = zeros!(pool, Float32, 10)
            end

            expr_str = string(expr)

            # Should contain Float32 directly (not default_eltype)
            @test occursin("Float32", expr_str)
        end

        @testset "ones! default type uses default_eltype(pool)" begin
            expr = @macroexpand @with_pool pool begin
                v = ones!(pool, 10)
            end

            expr_str = string(expr)

            @test occursin("default_eltype", expr_str)
        end

        @testset "unsafe_zeros! default type uses default_eltype(pool)" begin
            expr = @macroexpand @with_pool pool begin
                v = unsafe_zeros!(pool, 10)
            end

            expr_str = string(expr)

            @test occursin("default_eltype", expr_str)
        end

        @testset "unsafe_ones! default type uses default_eltype(pool)" begin
            expr = @macroexpand @with_pool pool begin
                v = unsafe_ones!(pool, 10)
            end

            expr_str = string(expr)

            @test occursin("default_eltype", expr_str)
        end

        @testset "mixed convenience with explicit and default types" begin
            expr = @macroexpand @with_pool pool begin
                v1 = zeros!(pool, Float64, 10)  # explicit
                v2 = ones!(pool, 5)              # default
                v3 = zeros!(pool, Float32, 3)   # explicit
            end

            expr_str = string(expr)

            # Explicit types present
            @test occursin("Float64", expr_str)
            @test occursin("Float32", expr_str)
            # default_eltype for untyped ones!
            @test occursin("default_eltype", expr_str)
        end

    end

end # Macro Expansion Details

# ==============================================================================
# Source Location Preservation Tests (LineNumberNode)
# ==============================================================================
#
# These tests verify that macro-generated code preserves source location
# information for better coverage, stack traces, and debugging.

# Test helper functions for robust AST inspection
"""
    find_linenumbernode_with_line(expr, target_line) -> Union{LineNumberNode, Nothing}

Recursively search for a LineNumberNode matching target_line.
More robust than checking only the first LNN (handles block forms where
`_maybe_add_source_location!` may insert LNN before user code LNN).
"""
function find_linenumbernode_with_line(expr, target_line::Int)
    if expr isa LineNumberNode && expr.line == target_line
        return expr
    elseif expr isa Expr
        for arg in expr.args
            result = find_linenumbernode_with_line(arg, target_line)
            result !== nothing && return result
        end
    end
    return nothing
end

"""
    has_valid_linenumbernode(expr) -> Bool

Check if expr contains any LineNumberNode with valid line info.
"""
function has_valid_linenumbernode(expr)
    if expr isa LineNumberNode
        return expr.line > 0 && expr.file !== :none
    elseif expr isa Expr
        for arg in expr.args
            has_valid_linenumbernode(arg) && return true
        end
    end
    return false
end

"""
    get_function_body(expr) -> Union{Expr, Nothing}

Extract function body from a function definition expression.
Handles both `function f() ... end` and `f() = ...` forms.
"""
function get_function_body(expr)
    if expr isa Expr
        if expr.head === :function && length(expr.args) >= 2
            return expr.args[2]
        elseif expr.head === :(=) && expr.args[1] isa Expr && expr.args[1].head === :call
            return expr.args[2]
        end
        # Recurse for wrapped expressions
        for arg in expr.args
            result = get_function_body(arg)
            result !== nothing && return result
        end
    end
    return nothing
end

@testset "Source Location Preservation" begin
    # Get this test file's path as Symbol for comparison
    this_file = Symbol(@__FILE__)

    # Test 1: @with_pool block form
    @testset "@with_pool block source location" begin
        expected_line = (@__LINE__) + 2
        expr = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
        end
        # Should find LNN matching the macro call line AND pointing to THIS file
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
        @test lnn.file !== :none  # Must have a valid file
        @test lnn.file == this_file  # Must point to the call site, not macros.jl
        # At minimum, there should be some valid LNN
        @test has_valid_linenumbernode(expr)
    end

    # Test 2: @with_pool function form
    # The FIRST LNN in function body must point to user file, not macros.jl
    @testset "@with_pool function source location" begin
        expected_line = (@__LINE__) + 2
        func_expr = @macroexpand @with_pool pool function test_func_source(n)
            acquire!(pool, Float64, n)
        end
        body = get_function_body(func_expr)
        @test body !== nothing
        # Check that FIRST LNN in body points to user file (not macros.jl)
        @test body isa Expr
        @test body.head === :block
        @test !isempty(body.args)
        first_lnn = body.args[1]
        @test first_lnn isa LineNumberNode
        @test first_lnn.file !== :none
        @test first_lnn.file == this_file  # Must point to call site, not macros.jl
    end

    # Test 3: @maybe_with_pool block form
    @testset "@maybe_with_pool block source location" begin
        expected_line = (@__LINE__) + 2
        expr = @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
        end
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
        @test lnn.file !== :none
        @test lnn.file == this_file
    end

    # Test 4: Backend variant (@with_pool :cpu)
    @testset "@with_pool :cpu backend source location" begin
        expected_line = (@__LINE__) + 2
        expr = @macroexpand @with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
        end
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
        @test lnn.file !== :none
        @test lnn.file == this_file
    end

    # Test 5: Without pool name (implicit gensym)
    @testset "@with_pool without pool name source location" begin
        expected_line = (@__LINE__) + 2
        expr = @macroexpand @with_pool begin
            inner_function()
        end
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
        @test lnn.file !== :none
        @test lnn.file == this_file
    end

    # Test 6: Short-form function (f(x) = ...) - LNN이 없는 케이스, __source__로 보정됨
    # The FIRST LNN in function body must point to user file
    @testset "@with_pool short function source location" begin
        func_expr = @macroexpand @with_pool pool test_short_func(x) = acquire!(pool, Float64, x)
        body = get_function_body(func_expr)
        @test body !== nothing
        # Short functions need __source__ fallback since they lack original LNN
        # Check that FIRST LNN in body points to user file (not macros.jl)
        @test body isa Expr
        @test body.head === :block
        @test !isempty(body.args)
        first_lnn = body.args[1]
        @test first_lnn isa LineNumberNode
        @test first_lnn.file !== :none
        @test first_lnn.file == this_file  # Must point to call site
    end

    # Test 7: @maybe_with_pool function form
    # The FIRST LNN in function body must point to user file
    @testset "@maybe_with_pool function source location" begin
        func_expr = @macroexpand @maybe_with_pool pool function maybe_test_func(n)
            acquire!(pool, Float64, n)
        end
        body = get_function_body(func_expr)
        @test body !== nothing
        # Check that FIRST LNN in body points to user file (not macros.jl)
        @test body isa Expr
        @test body.head === :block
        @test !isempty(body.args)
        first_lnn = body.args[1]
        @test first_lnn isa LineNumberNode
        @test first_lnn.file !== :none
        @test first_lnn.file == this_file
    end

    # Test 8: @with_pool :cpu function form
    # The FIRST LNN in function body must point to user file
    @testset "@with_pool :cpu function source location" begin
        func_expr = @macroexpand @with_pool :cpu pool function cpu_test_func(n)
            acquire!(pool, Float64, n)
        end
        body = get_function_body(func_expr)
        @test body !== nothing
        # Check that FIRST LNN in body points to user file (not macros.jl)
        @test body isa Expr
        @test body.head === :block
        @test !isempty(body.args)
        first_lnn = body.args[1]
        @test first_lnn isa LineNumberNode
        @test first_lnn.file !== :none
        @test first_lnn.file == this_file
    end

end # Source Location Preservation