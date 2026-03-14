# ==============================================================================
# Metal Type Definitions
# ==============================================================================
#
# Note: Unlike CPU, view(MtlVector, 1:n) returns MtlVector (via GPUArrays derive()),
# NOT SubArray. GPU view/reshape creation allocates ~80 bytes on CPU heap for the
# MtlArray wrapper. We cache wrappers via arr_wrappers to achieve zero-allocation
# on cache hit (same approach as CPU's setfield!-based Array wrapper reuse).

using AdaptiveArrayPools: RUNTIME_CHECK

const METAL_STORAGE = Metal.PrivateStorage

"""
    MetalTypedPool{T, S} <: AbstractTypedPool{T, MtlArray{T, 1, S}}

GPU memory pool for element type `T` with Metal storage mode `S`.
Uses `arr_wrappers`-based MtlArray reuse for zero-allocation acquire
(same design as CPU TypedPool on Julia 1.11+ and CUDA CuTypedPool).

## Fields
- `vectors`: Backing `MtlArray{T,1,S}` storage (one per slot)
- `arr_wrappers`: `Vector{Union{Nothing, Vector{Any}}}` — indexed by N (dimensionality),
  each entry is a per-slot cached `MtlArray{T,N,S}` wrapper. Uses `setfield!(wrapper, :dims, dims)`
  for zero-allocation reuse of unlimited dimension patterns within the same N.
  When the backing vector's GPU buffer changes (rare: only on grow beyond capacity),
  the wrapper's `:data` field is updated via DataRef refcount management.
- State management fields (same as CPU)
"""
mutable struct MetalTypedPool{T, S} <: AbstractTypedPool{T, MtlArray{T, 1, S}}
    # --- Storage ---
    vectors::Vector{MtlArray{T, 1, S}}

    # --- N-D Wrapper Cache (setfield!-based reuse, matches CPU TypedPool) ---
    arr_wrappers::Vector{Union{Nothing, Vector{Any}}}  # index=N (dimensionality), value=per-slot MtlArray{T,N,S}

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

function MetalTypedPool{T, S}() where {T, S}
    return MetalTypedPool{T, S}(
        MtlArray{T, 1, S}[],                # vectors
        Union{Nothing, Vector{Any}}[],       # arr_wrappers (indexed by N)
        0, [0], [0],                         # State (1-based sentinel)
    )
end

# ==============================================================================
# Metal Fixed Slot Configuration
# ==============================================================================

"""
Metal-optimized fixed slots. Differs from CUDA:
- No Float64, ComplexF64 (unsupported by Metal hardware)
- Float32 first (GPU-preferred precision)
- Float16 added (ML/inference workloads)
"""
const METAL_FIXED_SLOT_FIELDS = (
    :float32,       # Primary GPU type
    :float16,       # ML inference
    :int32,         # GPU-preferred indexing
    :int64,         # Large indices
    :complexf32,    # FFT, signal processing
    :bool,          # Masks
)

# ==============================================================================
# MetalAdaptiveArrayPool
# ==============================================================================

"""
    MetalAdaptiveArrayPool{R, S} <: AbstractArrayPool

Multi-type Metal GPU memory pool, parameterized by runtime check level `R` (binary: 0 or 1)
and Metal storage mode `S`.

## Runtime Check Levels
- `R=0`: Zero overhead — all safety branches eliminated by dead-code elimination
- `R=1`: Full checks — poisoning + structural invalidation + escape detection + borrow tracking

## Device Safety
Each pool is bound to a specific Metal device. Using a pool on the wrong device
causes undefined behavior. The `device_key` field tracks ownership.

## Fields
- Fixed slots for common GPU types (Float32 priority, includes Float16, no Float64/ComplexF64)
- `others`: IdDict fallback for rare types
- `device_key`: The Metal device this pool belongs to
- Borrow tracking fields (required by macro-injected field access at all R levels)
"""
mutable struct MetalAdaptiveArrayPool{R, S} <: AbstractArrayPool
    # Fixed Slots (Metal-optimized order, no Float64/ComplexF64)
    float32::MetalTypedPool{Float32, S}
    float16::MetalTypedPool{Float16, S}
    int32::MetalTypedPool{Int32, S}
    int64::MetalTypedPool{Int64, S}
    complexf32::MetalTypedPool{ComplexF32, S}
    bool::MetalTypedPool{Bool, S}

    # Fallback for rare types
    others::IdDict{DataType, Any}

    # State management (same as CPU)
    _current_depth::Int
    _touched_type_masks::Vector{UInt16}  # Per-depth: which fixed slots were touched + mode flags
    _touched_has_others::Vector{Bool}    # Per-depth: any non-fixed-slot type touched?

    # Device tracking (safety)
    device_key::Any

    # Borrow tracking (required: macro injects pool._pending_callsite = "..." as raw AST)
    _pending_callsite::String
    _pending_return_site::String
    _borrow_log::Union{Nothing, IdDict{Any, String}}
end

function MetalAdaptiveArrayPool{R, S}() where {R, S}
    return MetalAdaptiveArrayPool{R, S}(
        MetalTypedPool{Float32, S}(),
        MetalTypedPool{Float16, S}(),
        MetalTypedPool{Int32, S}(),
        MetalTypedPool{Int64, S}(),
        MetalTypedPool{ComplexF32, S}(),
        MetalTypedPool{Bool, S}(),
        IdDict{DataType, Any}(),
        1,              # _current_depth (1 = global scope)
        [UInt16(0)],    # _touched_type_masks: sentinel (no bits set)
        [false],        # _touched_has_others: sentinel (no others)
        Metal.device(),
        "",             # _pending_callsite
        "",             # _pending_return_site
        nothing,        # _borrow_log: lazily created when R >= 1
    )
end

"""Create pool with the default `RUNTIME_CHECK` level and PrivateStorage."""
MetalAdaptiveArrayPool() = MetalAdaptiveArrayPool{RUNTIME_CHECK, METAL_STORAGE}()

# ==============================================================================
# Runtime Check Dispatch
# ==============================================================================

"""
    _runtime_check(pool::MetalAdaptiveArrayPool) -> Bool

Return compile-time constant indicating whether runtime safety checks are enabled.
`R >= 1` enables checks; `R == 0` disables (dead-code-eliminated).
"""
@inline AdaptiveArrayPools._runtime_check(::MetalAdaptiveArrayPool{0}) = false
@inline AdaptiveArrayPools._runtime_check(::MetalAdaptiveArrayPool) = true  # R >= 1

"""
    _make_metal_pool(level) -> MetalAdaptiveArrayPool

Function barrier: converts runtime check level to concrete `MetalAdaptiveArrayPool{R,S}`.
Accepts `Bool` (`true`->1, `false`->0) or `Int` (used directly as R).
"""
_make_metal_pool(runtime_check::Bool) = _make_metal_pool(Int(runtime_check))
@noinline function _make_metal_pool(R::Int)
    R == 0 && return MetalAdaptiveArrayPool{0, METAL_STORAGE}()
    return MetalAdaptiveArrayPool{1, METAL_STORAGE}()
end

"""Human-readable runtime check label."""
function _metal_check_label(r::Int)
    r <= 0 && return "off"
    return "on"
end
