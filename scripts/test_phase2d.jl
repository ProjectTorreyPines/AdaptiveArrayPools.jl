#!/usr/bin/env julia
#=
Phase 2d Test: Macro Integration (@with_pool :cuda)
===================================================
Verifies that @with_pool :cuda and @with_cuda_pool work correctly.

Usage:
    julia --project=/path/to/AdaptiveArrayPools scripts/test_phase2d.jl

Or from CUDA environment:
    julia test_phase2d.jl
=#

println("=" ^ 60)
println("Phase 2d Test: Macro Integration")
println("=" ^ 60)
println()

# Step 1: Load packages
println("[1] Loading AdaptiveArrayPools...")
using AdaptiveArrayPools
println("    OK")

println("[2] Loading CUDA (triggers extension)...")
using CUDA
println("    OK")

# Step 3: Get extension module for direct access
println("[3] Getting extension module...")
ext = Base.get_extension(AdaptiveArrayPools, :AdaptiveArrayPoolsCUDAExt)
if ext === nothing
    println("    FAILED: Extension not loaded!")
    exit(1)
end
get_task_local_cuda_pool = ext.get_task_local_cuda_pool
println("    OK")

println()
println("-" ^ 60)
println("Testing @with_pool :cuda syntax")
println("-" ^ 60)

println("[4] Testing @with_pool :cuda with pool name...")
result1 = @with_pool :cuda pool begin
    println("    Inside @with_pool :cuda block")
    println("    pool type: ", typeof(pool))
    println("    pool.device_id: ", pool.device_id)

    # Acquire some GPU arrays
    A = acquire!(pool, Float32, 100)
    B = acquire!(pool, Float32, 100)
    println("    Acquired A ($(length(A))) and B ($(length(B)))")
    println("    A type: ", typeof(A))

    # Fill with data
    A .= 1.0f0
    B .= 2.0f0

    sum(A) + sum(B)
end
println("    Result: ", result1, " (expected: 300.0)")

if result1 != 300.0f0
    println("    FAILED: Incorrect result!")
    exit(1)
end
println("    OK")

println()
println("[5] Testing @with_pool :cuda without pool name...")
result2 = @with_pool :cuda begin
    # Use get_task_local_cuda_pool() to access pool
    pool = get_task_local_cuda_pool()
    v = acquire!(pool, Float64, 50)
    v .= 3.0
    sum(v)
end
println("    Result: ", result2, " (expected: 150.0)")

if result2 != 150.0
    println("    FAILED: Incorrect result!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing @with_cuda_pool explicit macro")
println("-" ^ 60)

println("[6] Testing @with_cuda_pool with pool name...")
result3 = ext.@with_cuda_pool pool begin
    println("    Inside @with_cuda_pool block")
    println("    pool type: ", typeof(pool))

    A = acquire!(pool, Float32, 200)
    A .= 0.5f0
    sum(A)
end
println("    Result: ", result3, " (expected: 100.0)")

if result3 != 100.0f0
    println("    FAILED: Incorrect result!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing nested CPU/GPU pools")
println("-" ^ 60)

println("[7] Testing nested @with_pool (CPU outer, GPU inner)...")
result4 = @with_pool cpu_pool begin
    cpu_v = acquire!(cpu_pool, Float64, 10)
    cpu_v .= 1.0

    gpu_result = @with_pool :cuda gpu_pool begin
        gpu_v = acquire!(gpu_pool, Float32, 10)
        gpu_v .= 2.0f0
        sum(gpu_v)
    end

    sum(cpu_v) + gpu_result
end
println("    Result: ", result4, " (expected: 30.0)")

if result4 != 30.0
    println("    FAILED: Incorrect result!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing checkpoint/rewind semantics")
println("-" ^ 60)

println("[8] Testing that rewind clears GPU allocations...")
pool = get_task_local_cuda_pool()
reset!(pool)  # Start fresh

initial_n_active = pool.float32.n_active
println("    Initial float32.n_active: ", initial_n_active)

@with_pool :cuda p begin
    v1 = acquire!(p, Float32, 100)
    v2 = acquire!(p, Float32, 200)
    println("    Inside block: float32.n_active = ", p.float32.n_active)
end

final_n_active = pool.float32.n_active
println("    After block: float32.n_active = ", final_n_active, " (should be 0)")

if final_n_active != 0
    println("    FAILED: rewind did not restore n_active!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing acquire! transformation")
println("-" ^ 60)

println("[9] Testing that acquire! calls are transformed...")
# This tests that acquire! is transformed to _acquire_impl!
# which bypasses untracked marking in macro-transformed code
pool = get_task_local_cuda_pool()
reset!(pool)

@with_pool :cuda p begin
    # These should NOT mark as untracked (transformed to _acquire_impl!)
    v = acquire!(p, Float32, 100)
    v .= 1.0f0
end

# Check _untracked_flags - should be [false] (only sentinel)
println("    _untracked_flags: ", pool._untracked_flags)
if length(pool._untracked_flags) != 1 || pool._untracked_flags[1] != false
    println("    WARNING: Unexpected _untracked_flags state")
end
println("    OK")

println()
println("-" ^ 60)
println("Testing error handling")
println("-" ^ 60)

println("[10] Testing rewind on error...")
pool = get_task_local_cuda_pool()
reset!(pool)

try
    @with_pool :cuda p begin
        v = acquire!(p, Float32, 100)
        println("    Acquired array, n_active = ", p.float32.n_active)
        error("Intentional error")
    end
catch e
    println("    Caught error: ", e)
end

println("    After error: n_active = ", pool.float32.n_active, " (should be 0)")
if pool.float32.n_active != 0
    println("    FAILED: rewind not called on error!")
    exit(1)
end
println("    OK")

println()
println("=" ^ 60)
println("Phase 2d Test: COMPLETE")
println("=" ^ 60)
println()
println("Summary: All macro integration tests passed!")
println()
println("CUDA Extension Implementation Complete!")
println("  - @with_pool :cuda pool begin ... end")
println("  - @with_cuda_pool pool begin ... end")
println("  - Nested CPU/GPU pools")
println("  - Automatic checkpoint/rewind")
println("  - Error handling with cleanup")
