#!/usr/bin/env julia
#=
Phase 2a Test: Extension Types
==============================
Verifies that CUDA extension types load and are correctly defined.

Usage:
    julia --project=/path/to/AdaptiveArrayPools scripts/test_phase2a.jl

Or from CUDA environment:
    julia test_phase2a.jl
=#

println("=" ^ 60)
println("Phase 2a Test: CUDA Extension Types")
println("=" ^ 60)
println()

# Step 1: Load AdaptiveArrayPools
println("[1] Loading AdaptiveArrayPools...")
using AdaptiveArrayPools
println("    OK")

# Step 2: Load CUDA (triggers extension)
println("[2] Loading CUDA (triggers extension)...")
using CUDA
println("    OK")

# Step 3: Check extension loaded
println("[3] Checking extension loaded...")
ext_module = Base.get_extension(AdaptiveArrayPools, :AdaptiveArrayPoolsCUDAExt)
if ext_module === nothing
    println("    FAILED: Extension not loaded!")
    exit(1)
end
println("    OK: Extension module = ", ext_module)

# Step 4: Check types are accessible
println("[4] Checking types...")
CuTypedPool = ext_module.CuTypedPool
CuAdaptiveArrayPool = ext_module.CuAdaptiveArrayPool
println("    CuTypedPool: ", CuTypedPool)
println("    CuAdaptiveArrayPool: ", CuAdaptiveArrayPool)

# Step 5: Check CuTypedPool structure (no views field!)
println("[5] Checking CuTypedPool structure...")
tp_fields = fieldnames(CuTypedPool)
println("    Fields: ", tp_fields)

has_vectors = :vectors in tp_fields
has_views = :views in tp_fields
has_view_lengths = :view_lengths in tp_fields
has_n_active = :n_active in tp_fields

println("    Has vectors? ", has_vectors, " (expected: true)")
println("    Has views? ", has_views, " (expected: false - GPU doesn't cache views)")
println("    Has view_lengths? ", has_view_lengths, " (expected: true)")
println("    Has n_active? ", has_n_active, " (expected: true)")

if has_views
    println("    WARNING: CuTypedPool has 'views' field - should be removed per design!")
end

# Step 6: Check CuAdaptiveArrayPool structure
println("[6] Checking CuAdaptiveArrayPool structure...")
pool_fields = fieldnames(CuAdaptiveArrayPool)
println("    Fields: ", pool_fields)

has_float16 = :float16 in pool_fields
has_device_id = :device_id in pool_fields
has_others = :others in pool_fields

println("    Has float16? ", has_float16, " (expected: true - GPU ML support)")
println("    Has device_id? ", has_device_id, " (expected: true - multi-GPU safety)")
println("    Has others? ", has_others, " (expected: true - fallback dict)")

# Step 7: Check inheritance
println("[7] Checking type hierarchy...")
println("    CuTypedPool <: AbstractTypedPool? ", CuTypedPool <: AbstractTypedPool)
println("    CuAdaptiveArrayPool <: AbstractArrayPool? ", CuAdaptiveArrayPool <: AbstractArrayPool)

# Step 8: Create instances
println("[8] Creating instances...")
try
    tp = CuTypedPool{Float32}()
    println("    CuTypedPool{Float32}(): OK")
    println("      n_active = ", tp.n_active)
    println("      vectors length = ", length(tp.vectors))
catch e
    println("    CuTypedPool{Float32}(): FAILED - ", e)
end

try
    pool = CuAdaptiveArrayPool()
    println("    CuAdaptiveArrayPool(): OK")
    println("      device_id = ", pool.device_id)
    println("      _current_depth = ", pool._current_depth)
catch e
    println("    CuAdaptiveArrayPool(): FAILED - ", e)
end

# Step 9: Verify GPU_FIXED_SLOT_FIELDS
println("[9] Checking GPU_FIXED_SLOT_FIELDS...")
gpu_slots = ext_module.GPU_FIXED_SLOT_FIELDS
println("    Slots: ", gpu_slots)
println("    Has :float16? ", :float16 in gpu_slots)
println("    Float32 first? ", first(gpu_slots) == :float32)

println()
println("=" ^ 60)
println("Phase 2a Test: COMPLETE")
println("=" ^ 60)

# Summary
println()
println("Summary:")
all_pass = has_vectors && !has_views && has_view_lengths && has_n_active &&
           has_float16 && has_device_id && has_others
if all_pass
    println("  All structure checks PASSED")
else
    println("  Some checks FAILED - review above")
end
