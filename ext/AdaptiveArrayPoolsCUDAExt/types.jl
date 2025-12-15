# ==============================================================================
# CUDA Pool Types
# ==============================================================================

# Note: Unlike CPU, view(CuVector, 1:n) returns CuVector (via GPUArrays derive()),
# NOT SubArray. Therefore, we don't cache view objects - just create fresh views
# each time (O(1) metadata operation, no GPU allocation).

"""
    CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}

GPU memory pool for element type `T`. Similar to `TypedPool` but without
view caching since `view(CuVector, 1:n)` returns a `CuVector`, not `SubArray`.

## Fields
- `vectors`: Backing `CuVector{T}` storage
- `view_lengths`: Cached lengths for resize decision (no view object cache)
- `nd_*`: N-D array cache (same structure as CPU)
- State management fields (same as CPU)

## Design Note
View creation on GPU is O(1) metadata operation, so caching provides no benefit.
"""
mutable struct CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}
    # --- Storage ---
    vectors::Vector{CuVector{T}}

    # --- Length tracking (no view cache!) ---
    view_lengths::Vector{Int}

    # --- N-D Array Cache (N-way set associative, same as CPU) ---
    nd_arrays::Vector{Any}
    nd_dims::Vector{Any}
    nd_ptrs::Vector{UInt}
    nd_next_way::Vector{Int}

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

function CuTypedPool{T}() where {T}
    CuTypedPool{T}(
        CuVector{T}[],      # vectors
        Int[],              # view_lengths (no views vector!)
        Any[], Any[], UInt[], Int[],  # N-D cache
        0, [0], [0]         # State (1-based sentinel)
    )
end

# ==============================================================================
# GPU Fixed Slot Configuration
# ==============================================================================

"""
GPU-optimized fixed slots. Differs from CPU:
- Float32 first (GPU-preferred precision)
- Float16 added (ML/inference workloads)
"""
const GPU_FIXED_SLOT_FIELDS = (
    :float32,       # Primary GPU type
    :float64,       # Precision when needed
    :float16,       # ML inference
    :int32,         # GPU-preferred indexing
    :int64,         # Large indices
    :complexf32,    # FFT, signal processing
    :complexf64,    # High-precision complex
    :bool,          # Masks
)

# ==============================================================================
# CuAdaptiveArrayPool
# ==============================================================================

"""
    CuAdaptiveArrayPool <: AbstractArrayPool

Multi-type GPU memory pool. Task-local and device-specific.

## Device Safety
Each pool is bound to a specific GPU device. Using a pool on the wrong device
causes undefined behavior. The `device_id` field tracks ownership.

## Fields
- Fixed slots for common GPU types (Float32 priority, includes Float16)
- `others`: IdDict fallback for rare types
- `device_id`: The GPU device this pool belongs to
"""
mutable struct CuAdaptiveArrayPool <: AbstractArrayPool
    # Fixed Slots (GPU-optimized order)
    float32::CuTypedPool{Float32}
    float64::CuTypedPool{Float64}
    float16::CuTypedPool{Float16}
    int32::CuTypedPool{Int32}
    int64::CuTypedPool{Int64}
    complexf32::CuTypedPool{ComplexF32}
    complexf64::CuTypedPool{ComplexF64}
    bool::CuTypedPool{Bool}

    # Fallback for rare types
    others::IdDict{DataType, Any}

    # State management (same as CPU)
    _current_depth::Int
    _untracked_flags::Vector{Bool}

    # Device tracking (safety)
    device_id::Int
end

function CuAdaptiveArrayPool()
    dev = CUDA.device()
    CuAdaptiveArrayPool(
        CuTypedPool{Float32}(),
        CuTypedPool{Float64}(),
        CuTypedPool{Float16}(),
        CuTypedPool{Int32}(),
        CuTypedPool{Int64}(),
        CuTypedPool{ComplexF32}(),
        CuTypedPool{ComplexF64}(),
        CuTypedPool{Bool}(),
        IdDict{DataType, Any}(),
        1,              # _current_depth (1 = global scope)
        [false],        # _untracked_flags sentinel
        CUDA.deviceid(dev)  # Use public API
    )
end
