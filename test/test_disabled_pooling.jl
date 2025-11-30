# ==============================================================================
# Test USE_POOLING=false branch (requires separate process)
# ==============================================================================
#
# USE_POOLING is a compile-time const loaded via Preferences.jl.
# To test the disabled branch, we must run in a separate Julia process
# with the preference set before loading the package.

@testset "USE_POOLING=false (separate process)" begin
    # Create test script that will run in separate process
    test_code = """
    # Set preference BEFORE loading package
    using Preferences
    set_preferences!("AdaptiveArrayPools", "use_pooling" => false; force=true)

    # Now load the package - it will see USE_POOLING=false
    using AdaptiveArrayPools
    using Test

    # Verify USE_POOLING is false
    @test USE_POOLING == false
    println("USE_POOLING = ", USE_POOLING)

    # Test @with_pool - should set pool=nothing
    @with_pool pool function test_with_pool_disabled(n)
        @test pool === nothing
        v = acquire!(pool, Float64, n)  # fallback to normal allocation
        @test v isa Vector{Float64}
        @test length(v) == n
        v .= 1.0
        sum(v)
    end

    result1 = test_with_pool_disabled(10)
    @test result1 == 10.0
    println("@with_pool function mode: PASS")

    # Test @with_pool block mode
    result2 = @with_pool pool begin
        @test pool === nothing
        v = acquire!(pool, Float64, 5)
        @test v isa Vector{Float64}
        v .= 2.0
        sum(v)
    end
    @test result2 == 10.0
    println("@with_pool block mode: PASS")

    # Test @maybe_with_pool - should also set pool=nothing
    @maybe_with_pool pool function test_maybe_disabled(n)
        @test pool === nothing
        v = acquire!(pool, Float64, n)
        v .= 3.0
        sum(v)
    end

    result3 = test_maybe_disabled(10)
    @test result3 == 30.0
    println("@maybe_with_pool function mode: PASS")

    # Test @maybe_with_pool block mode
    result4 = @maybe_with_pool pool begin
        @test pool === nothing
        v = acquire!(pool, Float64, 5)
        v .= 4.0
        sum(v)
    end
    @test result4 == 20.0
    println("@maybe_with_pool block mode: PASS")

    # Test @pool_kwarg - should set pool=nothing in function body
    @pool_kwarg pool function test_pool_kwarg_disabled(n)
        @test pool === nothing
        v = acquire!(pool, Float64, n)
        v .= 5.0
        sum(v)
    end

    result5 = test_pool_kwarg_disabled(10)
    @test result5 == 50.0
    println("@pool_kwarg: PASS")

    # Restore preference for other tests
    set_preferences!("AdaptiveArrayPools", "use_pooling" => true; force=true)

    println("All USE_POOLING=false tests passed!")
    """

    # Write test script to temp file
    test_file = tempname() * ".jl"
    write(test_file, test_code)

    # Get project path
    project_path = dirname(@__DIR__)

    # Run in separate process
    cmd = `$(Base.julia_cmd()) --project=$project_path $test_file`

    result = try
        output = read(cmd, String)
        println("  Output from subprocess:")
        for line in split(output, '\n')
            println("    ", line)
        end
        true
    catch e
        println("  Subprocess failed: ", e)
        false
    finally
        rm(test_file; force=true)
    end

    @test result == true
end
