# ==============================================================================
# CUDA Pool Types
# ==============================================================================

# Note: Unlike CPU, view(CuVector, 1:n) returns CuVector (via GPUArrays derive()),
# NOT SubArray. However, we still cache view objects to avoid CPU heap allocation
# (~80 bytes per call) for the CuVector metadata wrapper.

# Note: Uses shared CACHE_WAYS constant from main module for consistency.

"""
    CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}

GPU memory pool for element type `T`. Uses unified N-way view caching for all dimensions.

## Fields
- `vectors`: Backing `CuVector{T}` storage (one per slot)
- `views`: Flat N-way cache storing CuArray of any dimension
  - Layout: `views[(slot-1)*CACHE_WAYS + way]` for way ∈ 1:CACHE_WAYS
- `view_dims`: Cached dims corresponding to views
- `next_way`: Round-robin counter per slot for cache replacement
- State management fields (same as CPU)

## Design Note
Unlike CPU where view() returns SubArray and reshape() returns ReshapedArray,
CUDA returns CuArray for both operations. This allows a unified cache that
stores CuArray{T,N} for any N, eliminating the need for separate 1D/N-D caches.

GPU view/reshape creation allocates ~80 bytes on CPU heap for the CuArray
wrapper object. N-way caching with for-loop lookup eliminates this allocation
when the same dimensions pattern is requested again.
"""
mutable struct CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}
    # --- Storage ---
    vectors::Vector{CuVector{T}}

    # --- Unified N-Way View Cache (flat layout) ---
    # Length = n_slots * CACHE_WAYS
    views::Vector{Any}       # CuArray{T,N} for any N
    view_dims::Vector{Any}   # NTuple{N,Int} or nothing

    # --- Cache Replacement (round-robin per slot) ---
    next_way::Vector{Int}    # next_way[slot] ∈ 1:CACHE_WAYS

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

function CuTypedPool{T}() where {T}
    CuTypedPool{T}(
        CuVector{T}[],      # vectors
        Any[],              # views (N-way flat cache)
        Any[],              # view_dims
        Int[],              # next_way (round-robin counters)
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
