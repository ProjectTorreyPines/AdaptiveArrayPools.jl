# Metal Extension Test Suite
# =========================
# Entry point for all Metal-related tests.
#
# Usage:
#   - From main test suite: automatically included when Metal is available
#   - Direct execution: julia --project test/metal/runtests.jl
#   - Skip Metal tests: TEST_METAL=false julia --project -e 'using Pkg; Pkg.test()'

using Test

# GPU pooling requires Julia 1.11+
@static if VERSION < v"1.11-"
    @info "Metal extension tests skipped (requires Julia 1.11+)"
    @testset "Metal (skipped — Julia < 1.11)" begin end
else
    # Check Metal availability (requires macOS + Apple Silicon)
    const METAL_AVAILABLE = try
        Sys.isapple() || error("Not macOS")
        using Metal
        Metal.functional()
    catch
        false
    end

    if !METAL_AVAILABLE
        @info "Metal not available or not functional, skipping Metal tests"
    else
        @info "Running Metal extension tests on device: $(Metal.device())"

        # Load dependencies
        using AdaptiveArrayPools
        using AdaptiveArrayPools: checkpoint!, rewind!, get_typed_pool!, get_view!, foreach_fixed_slot

        # Extension types (only needed for type checks in tests)
        const ext = Base.get_extension(AdaptiveArrayPools, :AdaptiveArrayPoolsMetalExt)
        const MetalTypedPool = ext.MetalTypedPool
        const MetalAdaptiveArrayPool = ext.MetalAdaptiveArrayPool
        const METAL_FIXED_SLOT_FIELDS = ext.METAL_FIXED_SLOT_FIELDS

        # Include all Metal test files
        include("test_extension.jl")
        include("test_allocation.jl")
        include("test_display.jl")
        include("test_convenience.jl")
        include("test_disabled_pool.jl")
        include("test_metal_safety.jl")
        include("test_reshape.jl")
        include("test_task_local_pool.jl")
    end
end
