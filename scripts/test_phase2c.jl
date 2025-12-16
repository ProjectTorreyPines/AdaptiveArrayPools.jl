#!/usr/bin/env julia
#=
Phase 2c Test: Task-Local Pool + checkpoint/rewind
===================================================
Verifies task-local GPU pool management and state functions.

Usage:
    julia --project=/path/to/AdaptiveArrayPools scripts/test_phase2c.jl

Or from CUDA environment:
    julia test_phase2c.jl
=#

println("=" ^ 60)
println("Phase 2c Test: Task-Local Pool + checkpoint/rewind")
println("=" ^ 60)
println()

# Step 1: Load packages
println("[1] Loading AdaptiveArrayPools...")
using AdaptiveArrayPools
println("    OK")

println("[2] Loading CUDA (triggers extension)...")
using CUDA
println("    OK")

# Step 3: Get extension module
println("[3] Getting extension module...")
ext = Base.get_extension(AdaptiveArrayPools, :AdaptiveArrayPoolsCUDAExt)
if ext === nothing
    println("    FAILED: Extension not loaded!")
    exit(1)
end
get_task_local_cuda_pool = ext.get_task_local_cuda_pool
get_task_local_cuda_pools = ext.get_task_local_cuda_pools
CuTypedPool = ext.CuTypedPool
CuAdaptiveArrayPool = ext.CuAdaptiveArrayPool
println("    OK")

println()
println("-" ^ 60)
println("Testing get_task_local_cuda_pool")
println("-" ^ 60)

# Test task-local pool
println("[4] Testing get_task_local_cuda_pool...")
pool1 = get_task_local_cuda_pool()
println("    Type: ", typeof(pool1))
println("    Is CuAdaptiveArrayPool? ", pool1 isa CuAdaptiveArrayPool)
println("    device_id: ", pool1.device_id)
println("    _current_depth: ", pool1._current_depth)

# Same pool on second call?
pool2 = get_task_local_cuda_pool()
println("    Same pool on second call? ", pool1 === pool2)

if !(pool1 isa CuAdaptiveArrayPool) || pool1 !== pool2
    println("    FAILED!")
    exit(1)
end
println("    OK")

# Test pools dictionary
println("[5] Testing get_task_local_cuda_pools...")
pools_dict = get_task_local_cuda_pools()
println("    Type: ", typeof(pools_dict))
println("    Keys (device IDs): ", collect(keys(pools_dict)))
println("    Current device pool in dict? ", haskey(pools_dict, pool1.device_id)
)
println("    OK")

println()
println("-" ^ 60)
println("Testing checkpoint!/rewind! cycle")
println("-" ^ 60)

println("[6] Testing basic checkpoint/rewind...")
pool = get_task_local_cuda_pool()

# Initial state
println("    Initial _current_depth: ", pool._current_depth)
println("    Initial float32.n_active: ", pool.float32.n_active)

# Checkpoint
checkpoint!(pool)
println("    After checkpoint!:")
println("      _current_depth: ", pool._current_depth)
println("      float32._checkpoint_depths: ", pool.float32._checkpoint_depths)

# Acquire some arrays
tp = pool.float32
v1 = AdaptiveArrayPools.get_view!(tp, 100)
v2 = AdaptiveArrayPools.get_view!(tp, 200)
println("    After acquiring 2 arrays:")
println("      float32.n_active: ", tp.n_active)
println("      vectors count: ", length(tp.vectors))

# Rewind
rewind!(pool)
println("    After rewind!:")
println("      _current_depth: ", pool._current_depth)
println("      float32.n_active: ", tp.n_active, " (should be 0)")
println("      vectors count: ", length(tp.vectors), " (memory preserved)")

if pool._current_depth != 1 || tp.n_active != 0
    println("    FAILED: rewind! did not restore state correctly!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing nested checkpoint/rewind")
println("-" ^ 60)

println("[7] Testing nested scopes...")
pool = get_task_local_cuda_pool()
reset!(pool)  # Start fresh

# Outer checkpoint
checkpoint!(pool)
println("    After outer checkpoint: depth=", pool._current_depth)

v1 = AdaptiveArrayPools.get_view!(pool.float32, 50)
println("    Acquired v1, n_active=", pool.float32.n_active)

# Inner checkpoint
checkpoint!(pool)
println("    After inner checkpoint: depth=", pool._current_depth)

v2 = AdaptiveArrayPools.get_view!(pool.float32, 100)
v3 = AdaptiveArrayPools.get_view!(pool.float32, 150)
println("    Acquired v2, v3, n_active=", pool.float32.n_active)

# Inner rewind
rewind!(pool)
println("    After inner rewind: depth=", pool._current_depth, ", n_active=", pool.float32.n_active)

if pool._current_depth != 2 || pool.float32.n_active != 1
    println("    FAILED: inner rewind incorrect!")
    exit(1)
end

# Outer rewind
rewind!(pool)
println("    After outer rewind: depth=", pool._current_depth, ", n_active=", pool.float32.n_active)

if pool._current_depth != 1 || pool.float32.n_active != 0
    println("    FAILED: outer rewind incorrect!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing reset!")
println("-" ^ 60)

println("[8] Testing reset!...")
pool = get_task_local_cuda_pool()

# Acquire some without checkpoint (simulating misuse)
v1 = AdaptiveArrayPools.get_view!(pool.float32, 100)
v2 = AdaptiveArrayPools.get_view!(pool.float64, 200)
println("    After acquiring without checkpoint:")
println("      float32.n_active: ", pool.float32.n_active)
println("      float64.n_active: ", pool.float64.n_active)
println("      float32 vectors: ", length(pool.float32.vectors))

# Reset
reset!(pool)
println("    After reset!:")
println("      float32.n_active: ", pool.float32.n_active, " (should be 0)")
println("      float64.n_active: ", pool.float64.n_active, " (should be 0)")
println("      float32 vectors: ", length(pool.float32.vectors), " (preserved)")
println("      _current_depth: ", pool._current_depth, " (should be 1)")

if pool.float32.n_active != 0 || pool.float64.n_active != 0 || pool._current_depth != 1
    println("    FAILED: reset! did not work correctly!")
    exit(1)
end
if length(pool.float32.vectors) == 0
    println("    WARNING: reset! cleared vectors (should preserve them)")
end
println("    OK")

println()
println("-" ^ 60)
println("Testing empty!")
println("-" ^ 60)

println("[9] Testing empty!...")
pool = get_task_local_cuda_pool()

# Acquire some
v1 = AdaptiveArrayPools.get_view!(pool.float32, 100)
vectors_before = length(pool.float32.vectors)
println("    Before empty!: float32.vectors count = ", vectors_before)

# Empty
empty!(pool)
println("    After empty!:")
println("      float32.n_active: ", pool.float32.n_active)
println("      float32.vectors: ", length(pool.float32.vectors), " (should be 0)")
println("      _current_depth: ", pool._current_depth)

if pool.float32.n_active != 0 || length(pool.float32.vectors) != 0
    println("    FAILED: empty! did not clear storage!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing foreach_fixed_slot")
println("-" ^ 60)

println("[10] Testing foreach_fixed_slot iteration...")
pool = get_task_local_cuda_pool()
slot_count = Ref(0)
AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
    slot_count[] += 1
end
println("    Fixed slot count: ", slot_count[], " (expected: 8)")

if slot_count[] != 8
    println("    FAILED: foreach_fixed_slot did not iterate all slots!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing type-specific checkpoint/rewind")
println("-" ^ 60)

println("[11] Testing checkpoint!/rewind! with specific types...")
pool = get_task_local_cuda_pool()
reset!(pool)

# Checkpoint only Float32
checkpoint!(pool, Float32)
println("    After checkpoint!(pool, Float32): depth=", pool._current_depth)

v1 = AdaptiveArrayPools.get_view!(pool.float32, 100)
v2 = AdaptiveArrayPools.get_view!(pool.float64, 200)  # Untracked for Float64
println("    float32.n_active: ", pool.float32.n_active)
println("    float64.n_active: ", pool.float64.n_active)

rewind!(pool, Float32)
println("    After rewind!(pool, Float32):")
println("      depth: ", pool._current_depth)
println("      float32.n_active: ", pool.float32.n_active, " (should be 0)")
println("      float64.n_active: ", pool.float64.n_active, " (should be restored to 0 via sentinel)")

if pool.float32.n_active != 0
    println("    FAILED: typed rewind did not restore Float32!")
    exit(1)
end
println("    OK")

println()
println("=" ^ 60)
println("Phase 2c Test: COMPLETE")
println("=" ^ 60)
println()
println("Summary: All task-local pool and state management tests passed!")
println()
println("Next: Phase 2d - Macro integration (@with_pool :cuda)")
