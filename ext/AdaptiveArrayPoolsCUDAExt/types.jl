# ==============================================================================
# CUDA Pool Types
# ==============================================================================

# Note: Unlike CPU, view(CuVector, 1:n) returns CuVector (via GPUArrays derive()),
# NOT SubArray. GPU view/reshape creation allocates ~80 bytes on CPU heap for the
# CuArray wrapper. We cache wrappers via arr_wrappers to achieve zero-allocation
# on cache hit (same approach as CPU's setfield!-based Array wrapper reuse).

"""
    CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}

GPU memory pool for element type `T`. Uses `arr_wrappers`-based CuArray reuse
for zero-allocation acquire (same design as CPU TypedPool on Julia 1.11+).

## Fields
- `vectors`: Backing `CuVector{T}` storage (one per slot)
- `arr_wrappers`: `Vector{Union{Nothing, Vector{Any}}}` — indexed by N (dimensionality),
  each entry is a per-slot cached `CuArray{T,N}` wrapper. Uses `setfield!(wrapper, :dims, dims)`
  for zero-allocation reuse of unlimited dimension patterns within the same N.
  When the backing vector's GPU buffer changes (rare: only on grow beyond capacity),
  the wrapper's `:data` field is updated via DataRef refcount management.
- State management fields (same as CPU)

## Design Note
Unlike CPU where `setfield!(:ref, MemoryRef)` is free (GC-managed),
CuArray's `:data` field is `DataRef` with manual refcounting. We minimize this cost
via `wrapper.data.rc !== vec.data.rc` identity check (~2ns): only update `:data`
when the backing vector's GPU buffer actually changed. The common case (same buffer)
is a simple `setfield!(:dims)` — truly zero-allocation.
"""
mutable struct CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}
    # --- Storage ---
    vectors::Vector{CuVector{T}}

    # --- N-D Wrapper Cache (setfield!-based reuse, matches CPU TypedPool) ---
    arr_wrappers::Vector{Union{Nothing, Vector{Any}}}  # index=N (dimensionality), value=per-slot CuArray{T,N}

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

function CuTypedPool{T}() where {T}
    return CuTypedPool{T}(
        CuVector{T}[],                   # vectors
        Union{Nothing, Vector{Any}}[],   # arr_wrappers (indexed by N)
        0, [0], [0]                      # State (1-based sentinel)
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

Multi-type GPU memory pool, parameterized by runtime check level `S` (binary: 0 or 1).

## Runtime Check Levels
- `S=0`: Zero overhead — all safety branches eliminated by dead-code elimination
- `S=1`: Full checks — poisoning + structural invalidation + escape detection + borrow tracking

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
        nothing         # _borrow_log: lazily created when S >= 1
    )
end

"""Create pool with the default `RUNTIME_CHECK` setting."""
CuAdaptiveArrayPool() = _make_cuda_pool(AdaptiveArrayPools.RUNTIME_CHECK)

# ==============================================================================
# Runtime Check Dispatch
# ==============================================================================

"""
    _runtime_check(pool::CuAdaptiveArrayPool) -> Bool

Return compile-time constant indicating whether runtime safety checks are enabled.
`S >= 1` enables checks; `S == 0` disables (dead-code-eliminated).
"""
@inline AdaptiveArrayPools._runtime_check(::CuAdaptiveArrayPool{0}) = false
@inline AdaptiveArrayPools._runtime_check(::CuAdaptiveArrayPool{1}) = true

"""
    _make_cuda_pool(runtime_check::Bool) -> CuAdaptiveArrayPool

Function barrier: converts runtime `Bool` to concrete `CuAdaptiveArrayPool{S}`.
`false` → `CuAdaptiveArrayPool{0}`, `true` → `CuAdaptiveArrayPool{1}`.
"""
@noinline function _make_cuda_pool(runtime_check::Bool)
    runtime_check && return CuAdaptiveArrayPool{1}()
    return CuAdaptiveArrayPool{0}()
end

"""
    _make_cuda_pool(runtime_check::Bool, old::CuAdaptiveArrayPool) -> CuAdaptiveArrayPool

Create a new CUDA pool, transferring cached arrays and scope state from `old`.
Only reference copies — no memory allocation for underlying GPU buffers.

Transferred: all CuTypedPool slots, `others`, depth & touch tracking, device_id.
Reset: `_pending_callsite/return_site` (transient macro state),
       `_borrow_log` (created fresh when `runtime_check = true`).
"""
@noinline function _make_cuda_pool(runtime_check::Bool, old::CuAdaptiveArrayPool)
    runtime_check && return _transfer_cuda_pool(Val(1), old)
    return _transfer_cuda_pool(Val(0), old)
end

"""Transfer cached arrays and scope state from `old` pool into a new `CuAdaptiveArrayPool{V}`."""
function _transfer_cuda_pool(::Val{V}, old::CuAdaptiveArrayPool) where {V}
    return CuAdaptiveArrayPool{V}(
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
        V >= 1 ? IdDict{Any, String}() : nothing  # _borrow_log
    )
end

"""Human-readable runtime check label."""
function _cuda_check_label(s::Int)
    s <= 0 && return "off"
    return "on"
end
