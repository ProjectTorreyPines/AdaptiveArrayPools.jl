# ==============================================================================
# Coverage Tests - Simple tests to cover uncovered code paths
# ==============================================================================

@testset "Coverage Tests" begin

    @testset "DisabledPool convenience functions" begin
        pool = DISABLED_CPU

        # unsafe_zeros! without explicit type (uses default_eltype)
        v = unsafe_zeros!(pool, 5)
        @test v isa Vector{Float64}
        @test all(v .== 0.0)

        v = unsafe_zeros!(pool, (3, 3))
        @test v isa Matrix{Float64}

        # unsafe_ones! without explicit type
        v = unsafe_ones!(pool, 5)
        @test v isa Vector{Float64}
        @test all(v .== 1.0)

        v = unsafe_ones!(pool, (3, 3))
        @test v isa Matrix{Float64}

        # unsafe_similar! with dims
        template = rand(5, 5)
        v = unsafe_similar!(pool, template, 3, 3)
        @test v isa Matrix{Float64}
        @test size(v) == (3, 3)

        v = unsafe_similar!(pool, template, Int, 4, 4)
        @test v isa Matrix{Int}
        @test size(v) == (4, 4)
    end

    @testset "DisabledPool acquire functions" begin
        pool = DISABLED_CPU

        # acquire! with vararg dims
        v = acquire!(pool, Float32, 3, 3)
        @test v isa Array{Float32,2}

        # acquire! with tuple dims
        v = acquire!(pool, Float32, (2, 2))
        @test v isa Array{Float32,2}

        # acquire! with similar
        template = rand(Int32, 4, 4)
        v = acquire!(pool, template)
        @test v isa Array{Int32,2}

        # unsafe_acquire! variants
        v = unsafe_acquire!(pool, Float32, 3, 3)
        @test v isa Array{Float32,2}

        v = unsafe_acquire!(pool, Float32, (2, 2))
        @test v isa Array{Float32,2}

        v = unsafe_acquire!(pool, template)
        @test v isa Array{Int32,2}
    end

    @testset "BackendNotLoadedError" begin
        # Test error message for :cuda backend
        err = AdaptiveArrayPools.BackendNotLoadedError(:cuda)
        io = IOBuffer()
        showerror(io, err)
        msg = String(take!(io))
        @test occursin("cuda", msg)
        @test occursin("CUDA.jl", msg)

        # Test error message for other backends
        err2 = AdaptiveArrayPools.BackendNotLoadedError(:metal)
        io2 = IOBuffer()
        showerror(io2, err2)
        msg2 = String(take!(io2))
        @test occursin("metal", msg2)
        @test occursin("backend package", msg2)

        # Test that errors are thrown for unknown backend
        fake_pool = DisabledPool{:fake_backend}()
        @test try zeros!(fake_pool, 10); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try ones!(fake_pool, 10); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try similar!(fake_pool, rand(3)); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try unsafe_zeros!(fake_pool, 10); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try unsafe_ones!(fake_pool, 10); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try unsafe_similar!(fake_pool, rand(3)); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try acquire!(fake_pool, Float64, 10); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
        @test try unsafe_acquire!(fake_pool, Float64, 10); false catch e; e isa AdaptiveArrayPools.BackendNotLoadedError end
    end

    @testset "_impl! delegators for DisabledPool" begin
        pool = DISABLED_CPU

        # --- _zeros_impl! ---
        # Type + varargs
        v = AdaptiveArrayPools._zeros_impl!(pool, Float64, 5)
        @test v isa Vector{Float64}
        @test all(v .== 0.0)

        v = AdaptiveArrayPools._zeros_impl!(pool, Float32, 3, 4)
        @test v isa Matrix{Float32}
        @test size(v) == (3, 4)

        # No type (default eltype)
        v = AdaptiveArrayPools._zeros_impl!(pool, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._zeros_impl!(pool, 3, 4)
        @test v isa Matrix{Float64}

        # Tuple dims
        v = AdaptiveArrayPools._zeros_impl!(pool, Float64, (2, 3))
        @test v isa Matrix{Float64}
        @test size(v) == (2, 3)

        v = AdaptiveArrayPools._zeros_impl!(pool, (2, 3))
        @test v isa Matrix{Float64}

        # --- _ones_impl! ---
        v = AdaptiveArrayPools._ones_impl!(pool, Float64, 5)
        @test v isa Vector{Float64}
        @test all(v .== 1.0)

        v = AdaptiveArrayPools._ones_impl!(pool, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._ones_impl!(pool, Float64, (2, 3))
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._ones_impl!(pool, (2, 3))
        @test v isa Matrix{Float64}

        # --- _similar_impl! ---
        template = rand(3, 3)
        v = AdaptiveArrayPools._similar_impl!(pool, template)
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._similar_impl!(pool, template, Float32)
        @test v isa Matrix{Float32}

        v = AdaptiveArrayPools._similar_impl!(pool, template, 4, 5)
        @test v isa Matrix{Float64}
        @test size(v) == (4, 5)

        v = AdaptiveArrayPools._similar_impl!(pool, template, Int32, 2, 2)
        @test v isa Matrix{Int32}

        # --- _unsafe_zeros_impl! ---
        v = AdaptiveArrayPools._unsafe_zeros_impl!(pool, Float64, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._unsafe_zeros_impl!(pool, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._unsafe_zeros_impl!(pool, Float64, (2, 3))
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._unsafe_zeros_impl!(pool, (2, 3))
        @test v isa Matrix{Float64}

        # --- _unsafe_ones_impl! ---
        v = AdaptiveArrayPools._unsafe_ones_impl!(pool, Float64, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._unsafe_ones_impl!(pool, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._unsafe_ones_impl!(pool, Float64, (2, 3))
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._unsafe_ones_impl!(pool, (2, 3))
        @test v isa Matrix{Float64}

        # --- _unsafe_similar_impl! ---
        v = AdaptiveArrayPools._unsafe_similar_impl!(pool, template)
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._unsafe_similar_impl!(pool, template, Float32)
        @test v isa Matrix{Float32}

        v = AdaptiveArrayPools._unsafe_similar_impl!(pool, template, 4, 5)
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._unsafe_similar_impl!(pool, template, Int32, 2, 2)
        @test v isa Matrix{Int32}

        # --- _acquire_impl! ---
        v = AdaptiveArrayPools._acquire_impl!(pool, Float64, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._acquire_impl!(pool, Float64, 3, 4)
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._acquire_impl!(pool, Float64, (2, 3))
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._acquire_impl!(pool, template)
        @test v isa Matrix{Float64}

        # --- _unsafe_acquire_impl! ---
        v = AdaptiveArrayPools._unsafe_acquire_impl!(pool, Float64, 5)
        @test v isa Vector{Float64}

        v = AdaptiveArrayPools._unsafe_acquire_impl!(pool, Float64, 3, 4)
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._unsafe_acquire_impl!(pool, Float64, (2, 3))
        @test v isa Matrix{Float64}

        v = AdaptiveArrayPools._unsafe_acquire_impl!(pool, template)
        @test v isa Matrix{Float64}
    end

    @testset "Macro internals" begin
        # Test _disabled_pool_expr for cpu backend
        # Note: Returns the actual DisabledPool instance (const interpolation)
        cpu_result = AdaptiveArrayPools._disabled_pool_expr(:cpu)
        @test cpu_result isa DisabledPool{:cpu}

        # Test _disabled_pool_expr for non-cpu backend (triggers the else branch)
        cuda_result = AdaptiveArrayPools._disabled_pool_expr(:cuda)
        @test cuda_result isa DisabledPool{:cuda}

        # Test _is_function_def
        @test AdaptiveArrayPools._is_function_def(:(function foo() end)) == true
        @test AdaptiveArrayPools._is_function_def(:(foo(x) = x + 1)) == true
        @test AdaptiveArrayPools._is_function_def(:(x = 1)) == false
        @test AdaptiveArrayPools._is_function_def(:(begin; end)) == false

        # Test _filter_static_types
        types = Set{Any}([Float64, Int64])
        static, has_dyn = AdaptiveArrayPools._filter_static_types(types)
        @test Float64 in static
        @test Int64 in static
        @test !has_dyn

        # With local vars
        types2 = Set{Any}([:T, Float64])
        local_vars = Set([:T])
        static2, has_dyn2 = AdaptiveArrayPools._filter_static_types(types2, local_vars)
        @test Float64 in static2
        @test !(:T in static2)
        @test has_dyn2

        # With curly expression (parametric type)
        types3 = Set{Any}([Expr(:curly, :Vector, :Float64)])
        static3, has_dyn3 = AdaptiveArrayPools._filter_static_types(types3)
        @test isempty(static3)
        @test has_dyn3

        # With eltype expression
        types4 = Set{Any}([Expr(:call, :eltype, :x)])
        static4, has_dyn4 = AdaptiveArrayPools._filter_static_types(types4)
        @test length(static4) == 1  # eltype(x) is safe if x is not local
        @test !has_dyn4

        # With eltype of local var
        types5 = Set{Any}([Expr(:call, :eltype, :local_arr)])
        local_vars5 = Set([:local_arr])
        static5, has_dyn5 = AdaptiveArrayPools._filter_static_types(types5, local_vars5)
        @test isempty(static5)
        @test has_dyn5

        # With default_eltype expression
        types6 = Set{Any}([Expr(:call, :default_eltype, :pool)])
        static6, has_dyn6 = AdaptiveArrayPools._filter_static_types(types6)
        @test length(static6) == 1
        @test !has_dyn6

        # With GlobalRef (concrete type reference)
        types7 = Set{Any}([GlobalRef(Base, :Float64)])
        static7, has_dyn7 = AdaptiveArrayPools._filter_static_types(types7)
        @test length(static7) == 1
        @test !has_dyn7

        # Test _generate_typed_checkpoint_call
        pool_expr = :pool
        checkpoint_call = AdaptiveArrayPools._generate_typed_checkpoint_call(pool_expr, [Float64])
        @test checkpoint_call isa Expr

        empty_checkpoint = AdaptiveArrayPools._generate_typed_checkpoint_call(pool_expr, [])
        @test empty_checkpoint isa Expr

        # Test _generate_typed_rewind_call
        rewind_call = AdaptiveArrayPools._generate_typed_rewind_call(pool_expr, [Float64])
        @test rewind_call isa Expr

        empty_rewind = AdaptiveArrayPools._generate_typed_rewind_call(pool_expr, [])
        @test empty_rewind isa Expr
    end

    @testset "pool_stats error handling" begin
        # Test pool_stats(:cuda) without CUDA loaded
        @test_throws MethodError pool_stats(:cuda)
    end

    @testset "set_cache_ways! validation" begin
        # Test invalid range
        @test_throws ArgumentError AdaptiveArrayPools.set_cache_ways!(0)
        @test_throws ArgumentError AdaptiveArrayPools.set_cache_ways!(17)
    end

    @testset "_transform_acquire_calls with qualified names" begin
        # Test qualified name transformation (AdaptiveArrayPools.function!)
        # These test the elseif branches for qualified names in _transform_acquire_calls

        # Qualified unsafe_zeros!
        expr1 = :(AdaptiveArrayPools.unsafe_zeros!(pool, Float64, 10))
        result1 = AdaptiveArrayPools._transform_acquire_calls(expr1, :pool)
        @test result1.args[1] === AdaptiveArrayPools._UNSAFE_ZEROS_IMPL_REF

        # Qualified unsafe_ones!
        expr2 = :(AdaptiveArrayPools.unsafe_ones!(pool, Float64, 10))
        result2 = AdaptiveArrayPools._transform_acquire_calls(expr2, :pool)
        @test result2.args[1] === AdaptiveArrayPools._UNSAFE_ONES_IMPL_REF

        # Qualified unsafe_similar!
        expr3 = :(AdaptiveArrayPools.unsafe_similar!(pool, x))
        result3 = AdaptiveArrayPools._transform_acquire_calls(expr3, :pool)
        @test result3.args[1] === AdaptiveArrayPools._UNSAFE_SIMILAR_IMPL_REF

        # Qualified zeros!
        expr4 = :(AdaptiveArrayPools.zeros!(pool, Float64, 10))
        result4 = AdaptiveArrayPools._transform_acquire_calls(expr4, :pool)
        @test result4.args[1] === AdaptiveArrayPools._ZEROS_IMPL_REF

        # Qualified ones!
        expr5 = :(AdaptiveArrayPools.ones!(pool, Float64, 10))
        result5 = AdaptiveArrayPools._transform_acquire_calls(expr5, :pool)
        @test result5.args[1] === AdaptiveArrayPools._ONES_IMPL_REF

        # Qualified similar!
        expr6 = :(AdaptiveArrayPools.similar!(pool, x))
        result6 = AdaptiveArrayPools._transform_acquire_calls(expr6, :pool)
        @test result6.args[1] === AdaptiveArrayPools._SIMILAR_IMPL_REF

        # Qualified acquire!
        expr7 = :(AdaptiveArrayPools.acquire!(pool, Float64, 10))
        result7 = AdaptiveArrayPools._transform_acquire_calls(expr7, :pool)
        @test result7.args[1] === AdaptiveArrayPools._ACQUIRE_IMPL_REF

        # Qualified unsafe_acquire!
        expr8 = :(AdaptiveArrayPools.unsafe_acquire!(pool, Float64, 10))
        result8 = AdaptiveArrayPools._transform_acquire_calls(expr8, :pool)
        @test result8.args[1] === AdaptiveArrayPools._UNSAFE_ACQUIRE_IMPL_REF

        # Qualified acquire_view! (alias)
        expr9 = :(AdaptiveArrayPools.acquire_view!(pool, Float64, 10))
        result9 = AdaptiveArrayPools._transform_acquire_calls(expr9, :pool)
        @test result9.args[1] === AdaptiveArrayPools._ACQUIRE_IMPL_REF

        # Qualified acquire_array! (alias)
        expr10 = :(AdaptiveArrayPools.acquire_array!(pool, Float64, 10))
        result10 = AdaptiveArrayPools._transform_acquire_calls(expr10, :pool)
        @test result10.args[1] === AdaptiveArrayPools._UNSAFE_ACQUIRE_IMPL_REF
    end

    @testset "_generate_pool_code_with_backend" begin
        # Test that backend-specific code generation works
        # With USE_POOLING=false, it should return DisabledPool expression

        # Test block expression with :cpu backend
        result_cpu = AdaptiveArrayPools._generate_pool_code_with_backend(:cpu, :pool, :(x = 1), true)
        @test result_cpu isa Expr

        # Test block expression with :cuda backend
        result_cuda = AdaptiveArrayPools._generate_pool_code_with_backend(:cuda, :pool, :(x = 1), true)
        @test result_cuda isa Expr

        # Test force_enable=false (maybe_with_pool path)
        result_maybe = AdaptiveArrayPools._generate_pool_code_with_backend(:cpu, :pool, :(x = 1), false)
        @test result_maybe isa Expr

        # Test function definition with backend
        func_expr = :(function foo() end)
        result_func = AdaptiveArrayPools._generate_pool_code_with_backend(:cpu, :pool, func_expr, true)
        @test result_func isa Expr
        @test result_func.head == :function
    end

    @testset "_generate_function_pool_code" begin
        # Test function code generation with disable_pooling=true
        func_expr = :(function bar(x) x + 1 end)
        result = AdaptiveArrayPools._generate_function_pool_code(:pool, func_expr, true, true, :cpu)
        @test result isa Expr
        @test result.head == :function

        # Test with force_enable=false, disable_pooling=false
        result2 = AdaptiveArrayPools._generate_function_pool_code(:pool, func_expr, false, false, :cpu)
        @test result2 isa Expr

        # Test with short form function
        short_func = :(baz(x) = x * 2)
        result3 = AdaptiveArrayPools._generate_function_pool_code(:pool, short_func, true, true, :cpu)
        @test result3 isa Expr
        @test result3.head == :(=)
    end

    @testset "_generate_function_pool_code_with_backend" begin
        # Test function code generation with backend
        func_expr = :(function compute(x) x + 1 end)

        # With disable_pooling=true
        result1 = AdaptiveArrayPools._generate_function_pool_code_with_backend(:cpu, :pool, func_expr, true)
        @test result1 isa Expr
        @test result1.head == :function

        # With disable_pooling=false (generates full checkpoint/rewind)
        result2 = AdaptiveArrayPools._generate_function_pool_code_with_backend(:cuda, :pool, func_expr, false)
        @test result2 isa Expr
        @test result2.head == :function

        # Test with short form function
        short_func = :(fast(x) = x * 2)
        result3 = AdaptiveArrayPools._generate_function_pool_code_with_backend(:cpu, :pool, short_func, true)
        @test result3 isa Expr
        @test result3.head == :(=)
    end

    @testset "_extract_acquire_types with qualified names" begin
        # Test type extraction from qualified function calls
        # Note: _extract_acquire_types returns Symbols, not Types
        expr1 = :(AdaptiveArrayPools.zeros!(pool, Float32, 10))
        types1 = AdaptiveArrayPools._extract_acquire_types(expr1, :pool)
        @test :Float32 in types1

        expr2 = :(AdaptiveArrayPools.ones!(pool, Int64, 5))
        types2 = AdaptiveArrayPools._extract_acquire_types(expr2, :pool)
        @test :Int64 in types2

        expr3 = :(AdaptiveArrayPools.similar!(pool, x, Float64))
        types3 = AdaptiveArrayPools._extract_acquire_types(expr3, :pool)
        @test :Float64 in types3

        # Test acquire! qualified names
        expr4 = :(AdaptiveArrayPools.acquire!(pool, Float64, 10))
        types4 = AdaptiveArrayPools._extract_acquire_types(expr4, :pool)
        @test :Float64 in types4

        # Test acquire_view! alias
        expr5 = :(AdaptiveArrayPools.acquire_view!(pool, Int32, 5))
        types5 = AdaptiveArrayPools._extract_acquire_types(expr5, :pool)
        @test :Int32 in types5

        # Test acquire_array! alias
        expr6 = :(AdaptiveArrayPools.acquire_array!(pool, Float32, 3, 3))
        types6 = AdaptiveArrayPools._extract_acquire_types(expr6, :pool)
        @test :Float32 in types6
    end

    @testset "_looks_like_type" begin
        # Test type-like expressions
        @test AdaptiveArrayPools._looks_like_type(:Float64) == true
        @test AdaptiveArrayPools._looks_like_type(:Int) == true
        @test AdaptiveArrayPools._looks_like_type(:x) == false  # lowercase
        @test AdaptiveArrayPools._looks_like_type(Expr(:curly, :Vector, :Float64)) == true
        @test AdaptiveArrayPools._looks_like_type(GlobalRef(Base, :Float64)) == true
        @test AdaptiveArrayPools._looks_like_type(10) == false
    end

    @testset "_uses_local_var" begin
        local_vars = Set([:x, :y])

        # Direct local var
        @test AdaptiveArrayPools._uses_local_var(:x, local_vars) == true
        @test AdaptiveArrayPools._uses_local_var(:z, local_vars) == false

        # Field access: x.field
        @test AdaptiveArrayPools._uses_local_var(:(x.field), local_vars) == true
        @test AdaptiveArrayPools._uses_local_var(:(z.field), local_vars) == false

        # Indexing: x[i]
        @test AdaptiveArrayPools._uses_local_var(:(x[1]), local_vars) == true
        @test AdaptiveArrayPools._uses_local_var(:(z[1]), local_vars) == false

        # Nested: x.a.b
        @test AdaptiveArrayPools._uses_local_var(:(x.a.b), local_vars) == true

        # Call expression: foo(x)
        @test AdaptiveArrayPools._uses_local_var(:(foo(x)), local_vars) == true
        @test AdaptiveArrayPools._uses_local_var(:(foo(z)), local_vars) == false
    end

    @testset "_extract_local_assignments" begin
        # Simple assignment
        expr1 = :(T = eltype(x))
        locals1 = AdaptiveArrayPools._extract_local_assignments(expr1)
        @test :T in locals1

        # Typed assignment
        expr2 = :(T::Type = Float64)
        locals2 = AdaptiveArrayPools._extract_local_assignments(expr2)
        @test :T in locals2

        # local declaration
        expr3 = :(local T)
        locals3 = AdaptiveArrayPools._extract_local_assignments(expr3)
        @test :T in locals3

        # local with assignment
        expr4 = :(local T = Int)
        locals4 = AdaptiveArrayPools._extract_local_assignments(expr4)
        @test :T in locals4

        # Nested in block
        expr5 = quote
            x = 1
            y = 2
        end
        locals5 = AdaptiveArrayPools._extract_local_assignments(expr5)
        @test :x in locals5
        @test :y in locals5
    end

    @testset "AbstractArrayPool _impl! default type overloads" begin
        # These are called when convenience functions are used without type parameter
        # inside @with_pool macro: unsafe_ones!(pool, 10) → _unsafe_ones_impl!(pool, 10)
        pool = AdaptiveArrayPool()

        # --- _zeros_impl! without type (uses default_eltype) ---
        v = AdaptiveArrayPools._zeros_impl!(pool, 5)
        @test eltype(v) == Float64
        @test length(v) == 5

        v = AdaptiveArrayPools._zeros_impl!(pool, 3, 4)
        @test eltype(v) == Float64
        @test size(v) == (3, 4)

        # --- _ones_impl! without type ---
        v = AdaptiveArrayPools._ones_impl!(pool, 5)
        @test eltype(v) == Float64
        @test length(v) == 5

        v = AdaptiveArrayPools._ones_impl!(pool, 3, 4)
        @test eltype(v) == Float64
        @test size(v) == (3, 4)

        # --- _unsafe_zeros_impl! without type ---
        v = AdaptiveArrayPools._unsafe_zeros_impl!(pool, 5)
        @test v isa Vector{Float64}
        @test all(v .== 0.0)

        v = AdaptiveArrayPools._unsafe_zeros_impl!(pool, 3, 4)
        @test v isa Matrix{Float64}
        @test size(v) == (3, 4)

        # --- _unsafe_ones_impl! without type ---
        v = AdaptiveArrayPools._unsafe_ones_impl!(pool, 5)
        @test v isa Vector{Float64}
        @test all(v .== 1.0)

        v = AdaptiveArrayPools._unsafe_ones_impl!(pool, 3, 4)
        @test v isa Matrix{Float64}
        @test size(v) == (3, 4)

        empty!(pool)
    end
end
