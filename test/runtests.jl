using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!
import AdaptiveArrayPools: checkpoint!, rewind!

# AUTO_MANAGE defaults on, so __init__ auto-starts the background timer. Stop it so the
# suite is deterministic; testsets that exercise auto-manageion enable it explicitly.
AdaptiveArrayPools.disable_auto_manage!()

# Version-specific helpers (always defined, even for ARGS path)
@static if VERSION >= v"1.12-"
    _test_nd_cache_preserved(tp) = !isempty(tp.arr_wrappers)
else
    _test_nd_cache_preserved(tp) = length(tp.nd_arrays) >= 1
end

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Version-specific test file selection
    @static if VERSION >= v"1.12-"
        include("test_aqua.jl")
        include("test_basic.jl")
        include("test_state.jl")
        include("test_trim.jl")
        include("test_compact.jl")
        include("test_auto_manage.jl")
        include("test_auto_trim.jl")
        include("test_auto_integration.jl")
        include("test_multidimensional.jl")
        include("test_macros.jl")
        include("test_task_local_pool.jl")
        include("test_utils.jl")
        include("test_debug.jl")
        include("test_borrow_registry.jl")
        include("test_safety.jl")
        include("test_compile_escape.jl")
        include("test_compile_mutation.jl")
        include("test_runtime_mutation.jl")
        include("test_macro_expansion.jl")
        include("test_macro_internals.jl")
        include("test_zero_allocation.jl")
        include("test_disabled_pooling.jl")
        include("test_aliases.jl")
        include("test_nway_cache.jl")
        include("test_fixed_slots.jl")
        include("test_backend_macro_expansion.jl")
        include("test_convenience.jl")
        include("test_random.jl")
        include("test_reshape.jl")
        include("test_bitarray.jl")
        include("test_coverage.jl")
        include("test_allocation.jl")
        include("test_fallback_reclamation.jl")
        include("test_scope_depth_validation.jl")
    else
        include("test_aqua.jl")
        include("test_basic.jl")
        include("test_state.jl")
        include("test_trim_legacy.jl")
        include("test_compact_legacy.jl")
        include("test_auto_manage_legacy.jl")
        include("test_multidimensional.jl")
        include("test_macros.jl")
        include("test_task_local_pool.jl")
        include("test_utils.jl")
        include("test_debug.jl")
        include("test_safety.jl")
        include("test_compile_escape.jl")
        include("test_compile_mutation.jl")
        include("test_macro_expansion.jl")
        include("test_macro_internals.jl")
        include("test_zero_allocation.jl")
        include("test_disabled_pooling.jl")
        include("test_aliases.jl")
        include("legacy/test_nway_cache.jl")
        include("test_fixed_slots.jl")
        include("test_backend_macro_expansion.jl")
        include("test_convenience.jl")
        include("test_random.jl")
        include("test_reshape.jl")
        include("test_bitarray.jl")
        include("test_coverage.jl")
        include("test_allocation.jl")
        include("test_fallback_reclamation.jl")
        include("test_scope_depth_validation.jl")
    end

    # CUDA extension tests (auto-detect, skip with TEST_CUDA=false)
    if get(ENV, "TEST_CUDA", "true") != "false"
        include("cuda/runtests.jl")
    else
        @info "CUDA tests disabled via TEST_CUDA=false"
    end

    # Metal extension tests (auto-detect, skip with TEST_METAL=false)
    if get(ENV, "TEST_METAL", "true") != "false"
        include("metal/runtests.jl")
    else
        @info "Metal tests disabled via TEST_METAL=false"
    end
end
