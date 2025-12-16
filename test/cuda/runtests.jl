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

    # Load dependencies - functions work via dispatch, no need to access extension directly
    using AdaptiveArrayPools
    using AdaptiveArrayPools: checkpoint!, rewind!, get_typed_pool!, get_view!, foreach_fixed_slot

    # Extension types (only needed for type checks in tests)
    const ext = Base.get_extension(AdaptiveArrayPools, :AdaptiveArrayPoolsCUDAExt)
    const CuTypedPool = ext.CuTypedPool
    const CuAdaptiveArrayPool = ext.CuAdaptiveArrayPool
    const GPU_FIXED_SLOT_FIELDS = ext.GPU_FIXED_SLOT_FIELDS
    # get_task_local_cuda_pool, get_task_local_cuda_pools are exported from AdaptiveArrayPools

    # Include all CUDA test files
    include("test_extension.jl")
    include("test_allocation.jl")
    include("test_nway_cache.jl")
    include("test_display.jl")
    include("test_convenience.jl")
end
