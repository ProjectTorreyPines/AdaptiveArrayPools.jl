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
    return CuTypedPool{T}(
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
    CuAdaptiveArrayPool{S} <: AbstractArrayPool

Multi-type GPU memory pool, parameterized by safety level `S` (0–3).

## Safety Levels (CUDA-specific)
- `S=0`: Zero overhead — all safety branches eliminated by dead-code elimination
- `S=1`: Guard — poisoning (NaN/sentinel fill on released vectors) + cache invalidation
         (CUDA equivalent of CPU's resize! structural invalidation)
- `S=2`: Full — poisoning + escape detection (`_validate_pool_return`)
- `S=3`: Debug — full + borrow call-site registry + debug messages

## Device Safety
Each pool is bound to a specific GPU device. Using a pool on the wrong device
causes undefined behavior. The `device_id` field tracks ownership.

## Fields
- Fixed slots for common GPU types (Float32 priority, includes Float16)
- `others`: IdDict fallback for rare types
- `device_id`: The GPU device this pool belongs to
- Borrow tracking fields (required by macro-injected field access at all S levels)
"""
mutable struct CuAdaptiveArrayPool{S} <: AbstractArrayPool
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
    _touched_type_masks::Vector{UInt16}  # Per-depth: which fixed slots were touched + mode flags
    _touched_has_others::Vector{Bool}    # Per-depth: any non-fixed-slot type touched?

    # Device tracking (safety)
    device_id::Int

    # Borrow tracking (required: macro injects pool._pending_callsite = "..." as raw AST)
    _pending_callsite::String
    _pending_return_site::String
    _borrow_log::Union{Nothing, IdDict{Any, String}}
end

function CuAdaptiveArrayPool{S}() where {S}
    dev = CUDA.device()
    return CuAdaptiveArrayPool{S}(
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
        [UInt16(0)],    # _touched_type_masks: sentinel (no bits set)
        [false],        # _touched_has_others: sentinel (no others)
        CUDA.deviceid(dev),
        "",             # _pending_callsite
        "",             # _pending_return_site
        nothing         # _borrow_log: lazily created at S >= 3
    )
end

"""Create pool at the current `POOL_SAFETY_LV[]` level."""
CuAdaptiveArrayPool() = _make_cuda_pool(AdaptiveArrayPools.POOL_SAFETY_LV[])

# ==============================================================================
# Safety Level Dispatch
# ==============================================================================

"""
    _safety_level(pool::CuAdaptiveArrayPool{S}) -> Int

Return compile-time constant safety level for CUDA pools.
"""
@inline AdaptiveArrayPools._safety_level(::CuAdaptiveArrayPool{S}) where {S} = S

"""
    _make_cuda_pool(s::Int) -> CuAdaptiveArrayPool{s}

Function barrier: converts runtime `Int` to concrete `CuAdaptiveArrayPool{S}`.
Levels outside 0-3 are clamped (≤0 → 0, ≥3 → 3).
"""
@noinline function _make_cuda_pool(s::Int)
    s <= 0 && return CuAdaptiveArrayPool{0}()
    s == 1 && return CuAdaptiveArrayPool{1}()
    s == 2 && return CuAdaptiveArrayPool{2}()
    return CuAdaptiveArrayPool{3}()
end

"""
    _make_cuda_pool(s::Int, old::CuAdaptiveArrayPool) -> CuAdaptiveArrayPool{s}

Create a new pool at safety level `s`, transferring cached arrays and scope state
from `old`. Only reference copies — no memory allocation for underlying GPU buffers.

Transferred: all CuTypedPool slots, `others`, depth & touch tracking, device_id.
Reset: `_pending_callsite/return_site` (transient macro state),
       `_borrow_log` (created fresh when `s >= 3`).
"""
@noinline function _make_cuda_pool(s::Int, old::CuAdaptiveArrayPool)
    _new(::Val{V}) where {V} = CuAdaptiveArrayPool{V}(
        old.float32, old.float64, old.float16,
        old.int32, old.int64,
        old.complexf32, old.complexf64, old.bool,
        old.others,
        old._current_depth,
        old._touched_type_masks,
        old._touched_has_others,
        old.device_id,
        "",       # _pending_callsite: reset
        "",       # _pending_return_site: reset
        V >= 3 ? IdDict{Any, String}() : nothing  # _borrow_log
    )
    s <= 0 && return _new(Val(0))
    s == 1 && return _new(Val(1))
    s == 2 && return _new(Val(2))
    return _new(Val(3))
end

"""Human-readable safety level label."""
function _cuda_safety_label(s::Int)
    s <= 0 && return "off"
    s == 1 && return "guard"
    s == 2 && return "full"
    return "debug"
end
