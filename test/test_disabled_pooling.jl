# ==============================================================================
# Test STATIC_POOLING=false branch (requires separate process)
# ==============================================================================
#
# STATIC_POOLING is a compile-time const loaded via Preferences.jl.
# To test the disabled branch, we must run in a separate Julia process
# with the preference set before loading the package.

@testset "STATIC_POOLING=false (separate process)" begin
    # Create test script that will run in separate process
    test_code = """
    # Set preference BEFORE loading package
    using Preferences
    set_preferences!("AdaptiveArrayPools", "use_pooling" => false; force=true)

    # Now load the package - it will see STATIC_POOLING=false
    using AdaptiveArrayPools
    using Test

    # ==================================================================
    # 1. Type hierarchy and basics
    # ==================================================================
    @testset "Type hierarchy" begin
        @test STATIC_POOLING == false
        @test DisabledPool{:cpu} <: AbstractArrayPool
        @test DISABLED_CPU isa AbstractArrayPool
        @test !pooling_enabled(DISABLED_CPU)
    end

    # ==================================================================
    # 2. @with_pool block mode — basic acquire!
    # ==================================================================
    @testset "@with_pool block mode" begin
        result = @with_pool pool begin
            @test pool isa DisabledPool{:cpu}
            @test pool isa AbstractArrayPool
            @test !pooling_enabled(pool)
            v = acquire!(pool, Float64, 10)
            @test v isa Vector{Float64}
            @test length(v) == 10
            v .= 1.0
            sum(v)
        end
        @test result == 10.0
    end

    # ==================================================================
    # 3. @maybe_with_pool block mode
    # ==================================================================
    @testset "@maybe_with_pool block mode" begin
        result = @maybe_with_pool pool begin
            @test pool isa DisabledPool{:cpu}
            v = acquire!(pool, Float64, 5)
            @test v isa Vector{Float64}
            v .= 4.0
            sum(v)
        end
        @test result == 20.0
    end

    # ==================================================================
    # 4. Sub-function passing (the key use case!)
    # ==================================================================
    @testset "Passing pool to sub-functions" begin
        # Untyped parameter
        function _helper_untyped(pool, n)
            v = acquire!(pool, Float64, n)
            fill!(v, 2.0)
            return v
        end

        # Typed as AbstractArrayPool (now works with DisabledPool <: AbstractArrayPool)
        function _helper_typed(pool::AbstractArrayPool, n)
            v = zeros!(pool, Float64, n)
            return v
        end

        # Nested sub-function chain
        function _outer(pool, n)
            return _inner(pool, n)
        end
        function _inner(pool, n)
            return ones!(pool, Float64, n)
        end

        @with_pool pool begin
            # Untyped
            v1 = _helper_untyped(pool, 5)
            @test v1 isa Vector{Float64}
            @test all(v1 .== 2.0)

            # Typed as AbstractArrayPool
            v2 = _helper_typed(pool, 5)
            @test v2 isa Vector{Float64}
            @test all(v2 .== 0.0)

            # Nested chain
            v3 = _outer(pool, 3)
            @test v3 isa Vector{Float64}
            @test all(v3 .== 1.0)
        end
    end

    # ==================================================================
    # 5. Convenience functions: zeros!, ones!, similar!, reshape!
    # ==================================================================
    @testset "Convenience functions" begin
        @with_pool pool begin
            # zeros!
            z = zeros!(pool, Float64, 4)
            @test z isa Vector{Float64}
            @test all(z .== 0.0)

            z2 = zeros!(pool, Float32, 3, 3)
            @test z2 isa Matrix{Float32}
            @test size(z2) == (3, 3)

            # zeros! with default eltype
            z3 = zeros!(pool, 5)
            @test z3 isa Vector{Float64}

            # ones!
            o = ones!(pool, Float64, 4)
            @test o isa Vector{Float64}
            @test all(o .== 1.0)

            o2 = ones!(pool, Int32, 2, 3)
            @test o2 isa Matrix{Int32}

            # similar!
            template = rand(3, 4)
            s = similar!(pool, template)
            @test s isa Matrix{Float64}
            @test size(s) == (3, 4)

            s2 = similar!(pool, template, Float32)
            @test s2 isa Matrix{Float32}

            # reshape!
            a = acquire!(pool, Float64, 12)
            r = reshape!(pool, a, 3, 4)
            @test r isa AbstractMatrix{Float64}
            @test size(r) == (3, 4)

            # unsafe variants
            uz = unsafe_zeros!(pool, Float64, 5)
            @test uz isa Vector{Float64}
            @test all(uz .== 0.0)

            uo = unsafe_ones!(pool, Float64, 5)
            @test uo isa Vector{Float64}
            @test all(uo .== 1.0)

            us = unsafe_similar!(pool, template)
            @test us isa Matrix{Float64}
        end
    end

    # ==================================================================
    # 6. Bit type support
    # ==================================================================
    @testset "Bit type" begin
        @with_pool pool begin
            # acquire! with Bit
            b = acquire!(pool, Bit, 10)
            @test b isa BitVector
            @test length(b) == 10

            # trues! / falses!
            t = trues!(pool, 8)
            @test t isa BitVector
            @test all(t)

            f = falses!(pool, 8)
            @test f isa BitVector
            @test !any(f)

            # zeros!/ones! with Bit
            zb = zeros!(pool, Bit, 5)
            @test zb isa BitVector
            @test !any(zb)

            ob = ones!(pool, Bit, 5)
            @test ob isa BitVector
            @test all(ob)
        end
    end

    # ==================================================================
    # 7. State management no-ops
    # ==================================================================
    @testset "State management no-ops" begin
        pool = DISABLED_CPU
        # These should all be no-ops, not errors
        @test checkpoint!(pool) === nothing
        @test rewind!(pool) === nothing
        @test reset!(pool) === nothing
        @test empty!(pool) === nothing
    end

    # ==================================================================
    # 8. Function definition mode
    # ==================================================================
    @testset "@with_pool function definition" begin
        @with_pool pool function _test_func_def(x)
            v = acquire!(pool, Float64, length(x))
            v .= x .* 2
            return sum(v)
        end

        result = _test_func_def([1.0, 2.0, 3.0])
        @test result == 12.0
    end

    # ==================================================================
    # 8b. Function definition + sub-function passing
    # ==================================================================
    @testset "@with_pool function + sub-function passing" begin
        function _disabled_helper_typed(pool::AbstractArrayPool, n)
            return zeros!(pool, Float64, n)
        end

        @with_pool pool function _test_func_with_helper(n)
            v = _disabled_helper_typed(pool, n)
            @test v isa Vector{Float64}
            @test all(v .== 0.0)
            return sum(v .+ 1.0)
        end

        @test _test_func_with_helper(5) == 5.0
    end

    # ==================================================================
    # 8c. @maybe_with_pool function definition
    # ==================================================================
    @testset "@maybe_with_pool function definition" begin
        @maybe_with_pool pool function _test_maybe_func(x)
            @test pool isa DisabledPool{:cpu}
            v = acquire!(pool, Float64, length(x))
            v .= x .* 3
            return sum(v)
        end

        @test _test_maybe_func([1.0, 2.0]) == 9.0
    end

    # ==================================================================
    # 8d. Backend-specific block mode
    # ==================================================================
    @testset "@with_pool :cpu block mode" begin
        result = @with_pool :cpu pool begin
            @test pool isa DisabledPool{:cpu}
            v = acquire!(pool, Float64, 6)
            @test v isa Vector{Float64}
            v .= 3.0
            sum(v)
        end
        @test result == 18.0
    end

    @testset "@maybe_with_pool :cpu block mode" begin
        result = @maybe_with_pool :cpu pool begin
            @test pool isa DisabledPool{:cpu}
            v = zeros!(pool, Float64, 4)
            @test v isa Vector{Float64}
            sum(v .+ 1.0)
        end
        @test result == 4.0
    end

    # ==================================================================
    # 8e. Backend-specific function definition mode
    # ==================================================================
    @testset "@with_pool :cpu function definition" begin
        @with_pool :cpu pool function _test_backend_func(x)
            @test pool isa DisabledPool{:cpu}
            v = acquire!(pool, Float64, length(x))
            v .= x .* 2
            return sum(v)
        end

        @test _test_backend_func([1.0, 2.0, 3.0]) == 12.0
    end

    @testset "@maybe_with_pool :cpu function definition" begin
        @maybe_with_pool :cpu pool function _test_maybe_backend_func(x)
            @test pool isa DisabledPool{:cpu}
            v = acquire!(pool, Float64, length(x))
            v .= x .* 4
            return sum(v)
        end

        @test _test_maybe_backend_func([1.0, 2.0]) == 12.0
    end

    # ==================================================================
    # 9. Multi-dimensional arrays
    # ==================================================================
    @testset "Multi-dimensional" begin
        @with_pool pool begin
            # 2D
            m = acquire!(pool, Float64, 3, 4)
            @test m isa Matrix{Float64}
            @test size(m) == (3, 4)

            # 3D
            a = acquire!(pool, Float32, 2, 3, 4)
            @test a isa Array{Float32, 3}
            @test size(a) == (2, 3, 4)

            # NTuple dims
            z = zeros!(pool, Float64, (2, 3))
            @test z isa Matrix{Float64}
            @test size(z) == (2, 3)
        end
    end

    # Restore preference for other tests
    set_preferences!("AdaptiveArrayPools", "use_pooling" => true; force=true)

    println("All STATIC_POOLING=false tests passed!")
    """

    # Write test script to temp file
    test_file = tempname() * ".jl"
    write(test_file, test_code)

    # Get project path
    project_path = dirname(@__DIR__)

    # Run in separate process
    cmd = `$(Base.julia_cmd()) --project=$project_path $test_file`

    out = IOBuffer()
    err = IOBuffer()
    result = try
        run(pipeline(cmd; stdout = out, stderr = err))
        println("  Output from subprocess:")
        for line in split(String(take!(out)), '\n')
            println("    ", line)
        end
        true
    catch e
        println("  Subprocess failed: ", e)
        err_str = String(take!(err))
        if !isempty(err_str)
            println("  stderr:")
            for line in split(err_str, '\n')
                println("    ", line)
            end
        end
        false
    finally
        rm(test_file; force = true)
    end

    @test result == true
end
