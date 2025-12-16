#!/usr/bin/env julia
#=
CUDA Extension Design Verification Script
==========================================
Run this script in a CUDA-enabled environment and share the output.

Usage:
    julia cuda_design_check.jl

This checks key assumptions for AdaptiveArrayPools CUDA extension design.
=#

println("=" ^ 70)
println("CUDA Extension Design Verification")
println("=" ^ 70)
println()

# Check CUDA availability
try
    using CUDA
    println("[OK] CUDA.jl loaded successfully")
    println("  CUDA versioninfo: ", CUDA.versioninfo())
    println("  Device: ", CUDA.name(CUDA.device()))
    println()
catch e
    println("[ERROR] Failed to load CUDA.jl: ", e)
    exit(1)
end

println("-" ^ 70)
println("1. VIEW TYPE CHECK")
println("-" ^ 70)

# Test view on CuVector
cu_vec = CUDA.zeros(Float32, 100)
cu_view = view(cu_vec, 1:50)

println("  CuVector type: ", typeof(cu_vec))
println("  view(CuVector, 1:50) type: ", typeof(cu_view))
println()
println("  Is view a SubArray? ", cu_view isa SubArray)
println("  Is view a CuArray? ", cu_view isa CuArray)
println("  Is view an AbstractGPUArray? ", cu_view isa CUDA.AbstractGPUArray)
println()

# Check if they share memory (use allowscalar for testing)
CUDA.@allowscalar cu_vec[1] = 999.0f0
println("  Memory sharing test:")
println("    Set cu_vec[1] = 999.0")
println("    cu_view[1] = ", CUDA.@allowscalar(cu_view[1]), " (should be 999.0 if shared)")
println()

# Nested view
cu_view2 = view(cu_view, 1:25)
println("  Nested view(view, 1:25) type: ", typeof(cu_view2))
println()

println("-" ^ 70)
println("2. RESHAPE TYPE CHECK")
println("-" ^ 70)

# Test reshape on CuVector
reshaped = reshape(cu_vec, 10, 10)
println("  reshape(CuVector, 10, 10) type: ", typeof(reshaped))
println("  Is ReshapedArray? ", reshaped isa Base.ReshapedArray)
println("  Is CuArray? ", reshaped isa CuArray)
println()

# Test reshape on view
reshaped_view = reshape(cu_view, 10, 5)
println("  reshape(view_of_CuVector, 10, 5) type: ", typeof(reshaped_view))
println("  Is ReshapedArray? ", reshaped_view isa Base.ReshapedArray)
println("  Is CuArray? ", reshaped_view isa CuArray)
println()

println("-" ^ 70)
println("3. RESIZE! BEHAVIOR CHECK")
println("-" ^ 70)

# Test resize!
test_vec = CUDA.zeros(Float32, 10)
copyto!(test_vec, 1, CuArray(Float32.([1,2,3,4,5])), 1, 5)
println("  Original CuVector: size=$(size(test_vec)), first 5 elements=$(Array(test_vec[1:5]))")

original_ptr = pointer(test_vec)
resize!(test_vec, 20)
new_ptr = pointer(test_vec)

println("  After resize!(vec, 20): size=$(size(test_vec))")
println("  First 5 elements preserved? $(Array(test_vec[1:5]))")
println("  Pointer changed? $(original_ptr != new_ptr) ($(original_ptr) -> $(new_ptr))")
println()

# Test shrink
resize!(test_vec, 5)
shrink_ptr = pointer(test_vec)
println("  After resize!(vec, 5): size=$(size(test_vec))")
println("  Pointer changed on shrink? $(new_ptr != shrink_ptr)")
println()

println("-" ^ 70)
println("4. DEVICE ID API CHECK")
println("-" ^ 70)

dev = CUDA.device()
println("  CUDA.device() type: ", typeof(dev))
println()

# Check different ways to get device ID
println("  Available device ID methods:")
if hasproperty(dev, :handle)
    println("    dev.handle = ", dev.handle, " (internal field)")
end
try
    did = CUDA.deviceid(dev)
    println("    CUDA.deviceid(dev) = ", did, " (public API)")
catch e
    println("    CUDA.deviceid(dev) = ERROR: ", e)
end
try
    did = CUDA.deviceid()
    println("    CUDA.deviceid() = ", did, " (no argument)")
catch e
    println("    CUDA.deviceid() = ERROR: ", e)
end
println()

println("-" ^ 70)
println("5. MEMORY & ALLOCATION CHECK")
println("-" ^ 70)

# Check allocation
println("  Allocation test:")
@time "  CuVector{Float32}(undef, 1000)" begin
    for _ in 1:100
        _ = CuVector{Float32}(undef, 1000)
    end
end

# View creation overhead
vec = CUDA.zeros(Float32, 1000)
@time "  view(CuVector, 1:500) x100" begin
    for _ in 1:100
        _ = view(vec, 1:500)
    end
end
println()

println("-" ^ 70)
println("6. TASK LOCAL STORAGE CHECK")
println("-" ^ 70)

# Check task local storage works with CuArrays
const TLS_KEY = :test_cuda_pool

function test_tls()
    d = get(task_local_storage(), TLS_KEY, nothing)
    if d === nothing
        d = Dict{Int, CuVector{Float32}}()
        task_local_storage(TLS_KEY, d)
    end
    return d
end

tls_dict = test_tls()
tls_dict[1] = CUDA.zeros(Float32, 10)
println("  Task-local CuVector storage: OK")
println("  Retrieved type: ", typeof(test_tls()[1]))
println()

println("-" ^ 70)
println("7. SUBARRAYS & CONTIGUOUS CHECK")
println("-" ^ 70)

# Check if non-contiguous view returns SubArray
cu_mat = CUDA.zeros(Float32, 10, 10)
col_view = view(cu_mat, :, 1)  # Contiguous column
row_view = view(cu_mat, 1, :)  # Non-contiguous row (in column-major)

println("  Matrix shape: ", size(cu_mat))
println("  view(mat, :, 1) [column] type: ", typeof(col_view))
println("  view(mat, 1, :) [row] type: ", typeof(row_view))
println()

# Strided view
strided_view = view(cu_vec, 1:2:50)
println("  view(vec, 1:2:50) [strided] type: ", typeof(strided_view))
println()

println("-" ^ 70)
println("8. VERSION INFO")
println("-" ^ 70)

println("  Julia version: ", VERSION)
println("  CUDA.jl version: ", pkgversion(CUDA))
try
    using GPUArrays
    println("  GPUArrays.jl version: ", pkgversion(GPUArrays))
catch
    println("  GPUArrays.jl: not directly loaded")
end
println()

println("=" ^ 70)
println("VERIFICATION COMPLETE")
println("=" ^ 70)
