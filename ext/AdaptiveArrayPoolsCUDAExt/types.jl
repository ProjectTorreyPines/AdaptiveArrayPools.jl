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
for zero-allocation acquire (same design as CPU TypedPool on Julia 1.12+).

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

    # --- Per-slot current logical extent (parallel to `vectors`, matches CPU TypedPool) ---
    # Size of each slot's most recent `_cuda_claim_slot!`; `compact!` reads it
    # (`_slot_used`) to know how much of an over-allocated device buffer is in use.
    slot_extents::Vector{Int}

    # --- State Management ---
    # Checkpoint bookkeeping, extracted into a concrete shared struct (see CPU
    # PoolCheckpointState docstring). `const`: the reference never changes after
    # construction. Accessed as tp.n_active / tp._checkpoint_* via forwarding below.
    const state::PoolCheckpointState

    # --- Auto-trim telemetry (parity with CPU TypedPool; see its docstring) ---
    # Peak `n_active` since the last auto-trim — the recent working-set width. Written on the
    # hot path (`_cuda_claim_slot!`, one `max`, gated by AUTO_MANAGE → DCE'd off); auto-trim
    # reads it as the slot count to keep, then resets it to 0. Owner-only (non-atomic).
    _am_peak_n_active::Int
end

function CuTypedPool{T}() where {T}
    return CuTypedPool{T}(
        CuVector{T}[],                   # vectors
        Union{Nothing, Vector{Any}}[],   # arr_wrappers (indexed by N)
        Int[],                           # slot_extents (parallel to vectors)
        PoolCheckpointState(),           # state (1-based sentinel)
        0,                               # _am_peak_n_active: no usage observed yet
    )
end

# Checkpoint-state property forwarding (mirror of CPU src/types.jl:294-307).
@inline function Base.getproperty(tp::CuTypedPool, f::Symbol)
    f === :n_active && return getfield(tp, :state).n_active
    f === :_checkpoint_n_active && return getfield(tp, :state)._checkpoint_n_active
    f === :_checkpoint_depths && return getfield(tp, :state)._checkpoint_depths
    return getfield(tp, f)
end

@inline function Base.setproperty!(tp::CuTypedPool, f::Symbol, v)
    f === :n_active && return setfield!(getfield(tp, :state), :n_active, convert(Int, v))
    return setfield!(tp, f, convert(fieldtype(typeof(tp), f), v))
end

Base.propertynames(tp::CuTypedPool) =
    (fieldnames(typeof(tp))..., :n_active, :_checkpoint_n_active, :_checkpoint_depths)

# Route the generic checkpoint/rewind cores at the concrete state (zero-dispatch
# drain); without this, CuTypedPool falls back to _cp_state(tp) = tp and
# _touch_fallback_pool!'s ::PoolCheckpointState assert throws.
@inline AdaptiveArrayPools._cp_state(tp::CuTypedPool) = getfield(tp, :state)

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

    # Touched-others tracking (depth-tagged, concrete) — mirror of CPU
    # src/types.jl:434-453. Checkpoint variants push NOTHING; producers push one
    # (state, depth) entry per first touch; rewind pops while the top tag matches.
    # _touched_others_pools is populated only when S >= 1 (slot invalidation).
    _touched_others_states::Vector{PoolCheckpointState}
    _touched_others_depths::Vector{Int}
    _touched_others_pools::Vector{Any}

    # Last-lookup memo for the fallback registry (mirror of CPU). Set on every
    # slow-path lookup; cleared by empty! (identities die), preserved by
    # reset!/trim!/compact! (identities survive). Task-local pool → no races.
    _lookup_memo_type::Any
    _lookup_memo_tp::Any

    # Device tracking (safety)
    device_id::Int

    # Borrow tracking (required: macro injects pool._pending_callsite = "..." as raw AST)
    _pending_callsite::String
    _pending_return_site::String
    _borrow_log::Union{Nothing, IdDict{Any, String}}

    # Auto-manage request flags (parity with CPU AdaptiveArrayPool): set by the base
    # module's background Timer sweep (a different thread), read + reset by the owner task
    # at the `@with_pool :cuda` entry safepoint. Atomic for the cross-thread handoff.
    # `_compact_requested` is set every tick (→ compact!); `_trim_requested` every
    # `trim_interval` (→ auto-trim the tail). The entry hook fires on `_compact_requested`
    # (set on the same sweep), so the generic `_run_auto_manage!` services both at once.
    @atomic _compact_requested::Bool
    @atomic _trim_requested::Bool
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
        PoolCheckpointState[],  # _touched_others_states: no fallback touches yet
        Int[],                   # _touched_others_depths
        Any[],                   # _touched_others_pools
        nothing,                 # _lookup_memo_type
        nothing,                 # _lookup_memo_tp
        CUDA.deviceid(dev),
        "",             # _pending_callsite
        "",             # _pending_return_site
        nothing,        # _borrow_log: lazily created when S >= 1
        false,          # _compact_requested: no pending compact request
        false           # _trim_requested: no pending auto-trim request
    )
end

"""Create pool with the default `RUNTIME_CHECK` level."""
CuAdaptiveArrayPool() = CuAdaptiveArrayPool{AdaptiveArrayPools.RUNTIME_CHECK}()

# ==============================================================================
# Runtime Check Dispatch
# ==============================================================================

"""
    _runtime_check(pool::CuAdaptiveArrayPool) -> Bool

Return compile-time constant indicating whether runtime safety checks are enabled.
`S >= 1` enables checks; `S == 0` disables (dead-code-eliminated).
"""
@inline AdaptiveArrayPools._runtime_check(::CuAdaptiveArrayPool{0}) = false
@inline AdaptiveArrayPools._runtime_check(::CuAdaptiveArrayPool) = true  # S >= 1

"""
    _check_level(pool::CuAdaptiveArrayPool) -> Int

Runtime-check level as an Int (mirror of CPU `src/types.jl:513`), for
backend-shared code that forwards it to `_invalidate_released_slots!` /
`_rewind_typed_pool!`. Compile-time constant per concrete pool type.
"""
@inline AdaptiveArrayPools._check_level(::CuAdaptiveArrayPool{S}) where {S} = S

"""
    _make_cuda_pool(level) -> CuAdaptiveArrayPool

Function barrier: converts runtime check level to concrete `CuAdaptiveArrayPool{S}`.
Accepts `Bool` (`true`→1, `false`→0) or `Int` (used directly as S).
"""
_make_cuda_pool(runtime_check::Bool) = _make_cuda_pool(Int(runtime_check))
@noinline function _make_cuda_pool(S::Int)
    S == 0 && return CuAdaptiveArrayPool{0}()
    return CuAdaptiveArrayPool{1}()
end

# NOTE: a former 2-arg `_make_cuda_pool(level, old)` overload migrated a pool
# across S while transferring caches by reference. Removed (no production
# callers): the touched-others stack's shape is S-dependent (pools entries
# exist only at S >= 1), so a mid-scope migration would desync the next
# rewind. If runtime S switching is ever needed, reintroduce it with an
# explicit global-scope guard.

"""Human-readable runtime check label."""
function _cuda_check_label(s::Int)
    s <= 0 && return "off"
    return "on"
end
