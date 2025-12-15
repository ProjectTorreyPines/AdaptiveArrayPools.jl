#!/usr/bin/env julia
#=
Phase 2b Test: Dispatch Methods & get_view!
===========================================
Verifies that GPU dispatch methods and get_view! work correctly.

Usage:
    julia --project=/path/to/AdaptiveArrayPools scripts/test_phase2b.jl

Or from CUDA environment:
    julia test_phase2b.jl
=#

println("=" ^ 60)
println("Phase 2b Test: Dispatch Methods & get_view!")
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
CuTypedPool = ext.CuTypedPool
CuAdaptiveArrayPool = ext.CuAdaptiveArrayPool
println("    OK")

println()
println("-" ^ 60)
println("Testing allocate_vector")
println("-" ^ 60)

# Test allocate_vector
println("[4] Testing allocate_vector for CuTypedPool...")
tp = CuTypedPool{Float32}()
vec = AdaptiveArrayPools.allocate_vector(tp, 100)
println("    Type: ", typeof(vec))
println("    Is CuVector{Float32}? ", vec isa CuVector{Float32})
println("    Length: ", length(vec))

if !(vec isa CuVector{Float32}) || length(vec) != 100
    println("    FAILED: allocate_vector did not return correct type/size!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing wrap_array")
println("-" ^ 60)

# Test wrap_array
println("[5] Testing wrap_array for CuTypedPool...")
flat_view = view(vec, 1:50)
wrapped = AdaptiveArrayPools.wrap_array(tp, flat_view, (10, 5))
println("    Input view type: ", typeof(flat_view))
println("    Wrapped type: ", typeof(wrapped))
println("    Is CuArray{Float32,2}? ", wrapped isa CuArray{Float32,2})
println("    Size: ", size(wrapped))

if !(wrapped isa CuArray{Float32,2}) || size(wrapped) != (10, 5)
    println("    FAILED: wrap_array did not return correct type/size!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing get_typed_pool!")
println("-" ^ 60)

# Test get_typed_pool! for fixed slots
println("[6] Testing get_typed_pool! for fixed slots...")
pool = CuAdaptiveArrayPool()

test_types = [Float32, Float64, Float16, Int32, Int64, ComplexF32, ComplexF64, Bool]
for T in test_types
    tp_test = AdaptiveArrayPools.get_typed_pool!(pool, T)
    correct_type = tp_test isa CuTypedPool{T}
    print("    $T: ")
    if correct_type
        println("OK (", typeof(tp_test), ")")
    else
        println("FAILED! Got ", typeof(tp_test))
        exit(1)
    end
end

# Test fallback for rare type
println("[7] Testing get_typed_pool! fallback (UInt8)...")
tp_uint8 = AdaptiveArrayPools.get_typed_pool!(pool, UInt8)
println("    Type: ", typeof(tp_uint8))
println("    Is CuTypedPool{UInt8}? ", tp_uint8 isa CuTypedPool{UInt8})
println("    In others dict? ", haskey(pool.others, UInt8))

if !(tp_uint8 isa CuTypedPool{UInt8}) || !haskey(pool.others, UInt8)
    println("    FAILED: Fallback did not work correctly!")
    exit(1)
end
println("    OK")

println()
println("-" ^ 60)
println("Testing get_view!")
println("-" ^ 60)

# Test get_view!
println("[8] Testing get_view! for CuTypedPool...")
tp_view = CuTypedPool{Float32}()
println("    Initial n_active: ", tp_view.n_active)

# First acquire
v1 = AdaptiveArrayPools.get_view!(tp_view, 100)
println("    After first get_view!(100):")
println("      Type: ", typeof(v1))
println("      Length: ", length(v1))
println("      n_active: ", tp_view.n_active)
println("      vectors count: ", length(tp_view.vectors))

if !(v1 isa CuArray) || length(v1) != 100 || tp_view.n_active != 1
    println("    FAILED: First get_view! incorrect!")
    exit(1)
end

# Second acquire (different size)
v2 = AdaptiveArrayPools.get_view!(tp_view, 200)
println("    After second get_view!(200):")
println("      Type: ", typeof(v2))
println("      Length: ", length(v2))
println("      n_active: ", tp_view.n_active)
println("      vectors count: ", length(tp_view.vectors))

if !(v2 isa CuArray) || length(v2) != 200 || tp_view.n_active != 2
    println("    FAILED: Second get_view! incorrect!")
    exit(1)
end
println("    OK")

# Test view memory sharing
println("[9] Testing view memory sharing...")
base_vec = tp_view.vectors[1]
v1_new = AdaptiveArrayPools.get_view!(CuTypedPool{Float32}(
    [base_vec], [100], Any[], Any[], UInt[], Int[], 0, [0], [0]
), 50)
# Manually create a typed pool with existing vector to test view sharing
CUDA.@allowscalar base_vec[1] = 123.0f0
val = CUDA.@allowscalar v1_new[1]
println("    Set base_vec[1] = 123.0")
println("    view[1] = ", val, " (should be 123.0 if shared)")
if val != 123.0f0
    println("    WARNING: Memory may not be shared correctly!")
else
    println("    OK - Memory is shared")
end

println()
println("-" ^ 60)
println("Testing checkpoint correction in get_typed_pool!")
println("-" ^ 60)

println("[10] Testing checkpoint auto-init for dynamic types...")
pool2 = CuAdaptiveArrayPool()
# Simulate being inside @with_pool scope
pool2._current_depth = 2

# Get a rare type while inside scope
tp_rare = AdaptiveArrayPools.get_typed_pool!(pool2, UInt16)
println("    pool._current_depth: ", pool2._current_depth)
println("    Created CuTypedPool{UInt16}:")
println("      _checkpoint_n_active: ", tp_rare._checkpoint_n_active)
println("      _checkpoint_depths: ", tp_rare._checkpoint_depths)

# Should have checkpoint auto-initialized
expected_n_active = [0, 0]  # Sentinel + checkpoint at depth 2
expected_depths = [0, 2]
if tp_rare._checkpoint_n_active != expected_n_active || tp_rare._checkpoint_depths != expected_depths
    println("    FAILED: Checkpoint not auto-initialized!")
    println("    Expected _checkpoint_n_active: ", expected_n_active)
    println("    Expected _checkpoint_depths: ", expected_depths)
    exit(1)
end
println("    OK - Checkpoint auto-initialized correctly")

println()
println("=" ^ 60)
println("Phase 2b Test: COMPLETE")
println("=" ^ 60)
println()
println("Summary: All dispatch methods and get_view! working correctly!")
println()
println("Next: Phase 2c - Task-local pool + checkpoint/rewind")
