using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!
import AdaptiveArrayPools: checkpoint!, rewind!  

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    include("test_basic.jl")
    include("test_state.jl")
    include("test_multidimensional.jl")
    include("test_macros.jl")
    include("test_task_local_pool.jl")
    include("test_utils.jl")
    include("test_macro_expansion.jl")
    include("test_macro_internals.jl")
    include("test_zero_allocation.jl")
    include("test_disabled_pooling.jl")
    include("test_aliases.jl")
    include("test_nway_cache.jl")
    include("test_fixed_slots.jl")
    include("test_backend_macro_expansion.jl")

    # CUDA extension tests (auto-detect, skip with TEST_CUDA=false)
    if get(ENV, "TEST_CUDA", "true") != "false"
        include("cuda/runtests.jl")
    else
        @info "CUDA tests disabled via TEST_CUDA=false"
    end
end
