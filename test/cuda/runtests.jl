# CUDA Extension Test Suite
# =========================
# Entry point for all CUDA-related tests.
#
# Usage:
#   - From main test suite: automatically included when CUDA is available
#   - Direct execution: julia --project test/cuda/runtests.jl
#   - Skip CUDA tests: TEST_CUDA=false julia --project -e 'using Pkg; Pkg.test()'

using Test

# Check CUDA availability (separate from test execution)
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !CUDA_AVAILABLE
    @info "CUDA not available or not functional, skipping CUDA tests"
    # Return early - no tests to run
else
    @info "Running CUDA extension tests on device: $(CUDA.name(CUDA.device()))"

    # Load dependencies
    using AdaptiveArrayPools
    using AdaptiveArrayPools: checkpoint!, rewind!, get_typed_pool!, get_view!, foreach_fixed_slot

    # Get extension module
    const ext = Base.get_extension(AdaptiveArrayPools, :AdaptiveArrayPoolsCUDAExt)
    const CuTypedPool = ext.CuTypedPool
    const CuAdaptiveArrayPool = ext.CuAdaptiveArrayPool
    const get_task_local_cuda_pool = ext.get_task_local_cuda_pool
    const get_task_local_cuda_pools = ext.get_task_local_cuda_pools
    const GPU_FIXED_SLOT_FIELDS = ext.GPU_FIXED_SLOT_FIELDS

    # Include all CUDA test files
    @testset "CUDA Extension Tests" begin
        include("test_extension.jl")
        # Future CUDA tests can be added here:
        # include("test_nway_cache.jl")
        # include("test_performance.jl")
        # include("test_multi_gpu.jl")
    end
end
