using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!
import AdaptiveArrayPools: checkpoint!, rewind!  # v2 API (not exported)

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    @testset "AdaptiveArrayPools" begin
        include("test_basic.jl")
        include("test_state.jl")
        include("test_multidimensional.jl")
        include("test_macros.jl")
        include("test_task_local_pool.jl")
        include("test_utils.jl")
        include("test_macro_expansion.jl")
        include("test_macro_internals.jl")
        include("test_disabled_pooling.jl")
    end
end
