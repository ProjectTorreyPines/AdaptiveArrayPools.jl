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

    # Test @with_pool block mode - should set pool=nothing
    result1 = @with_pool pool begin
        @test pool === nothing
        v = acquire!(pool, Float64, 10)  # fallback to normal allocation
        @test v isa Vector{Float64}
        @test length(v) == 10
        v .= 1.0
        sum(v)
    end
    @test result1 == 10.0
    println("@with_pool block mode: PASS")

    # Test @maybe_with_pool block mode - should also set pool=nothing
    result2 = @maybe_with_pool pool begin
        @test pool === nothing
        v = acquire!(pool, Float64, 5)
        @test v isa Vector{Float64}
        v .= 4.0
        sum(v)
    end
    @test result2 == 20.0
    println("@maybe_with_pool block mode: PASS")

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
