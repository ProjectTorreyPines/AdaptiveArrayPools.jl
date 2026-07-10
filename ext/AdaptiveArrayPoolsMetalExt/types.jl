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
(same design as CPU TypedPool on Julia 1.12+ and CUDA CuTypedPool).

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

    # --- Per-slot current logical extent (parallel to `vectors`, matches CPU TypedPool) ---
    # Size of each slot's most recent `_metal_claim_slot!`; `compact!` reads it
    # (`_slot_used`) to know how much of an over-allocated device buffer is in use.
    slot_extents::Vector{Int}

    # --- State Management ---
    # Checkpoint bookkeeping, extracted into a concrete shared struct (see CPU
    # PoolCheckpointState docstring). `const`: the reference never changes after
    # construction. Accessed as tp.n_active / tp._checkpoint_* via forwarding below.
    const state::PoolCheckpointState

    # --- Auto-trim telemetry (parity with CPU TypedPool; see its docstring) ---
    # Peak `n_active` since the last auto-trim — the recent working-set width. Written on the
    # hot path (`_metal_claim_slot!`, one `max`, gated by AUTO_MANAGE → DCE'd off); auto-trim
    # reads it as the slot count to keep, then resets it to 0. Owner-only (non-atomic).
    _am_peak_n_active::Int
end

function MetalTypedPool{T, S}() where {T, S}
    return MetalTypedPool{T, S}(
        MtlArray{T, 1, S}[],                # vectors
        Union{Nothing, Vector{Any}}[],       # arr_wrappers (indexed by N)
        Int[],                               # slot_extents (parallel to vectors)
        PoolCheckpointState(),               # state (1-based sentinel)
        0,                                   # _am_peak_n_active: no usage observed yet
    )
end

# Checkpoint-state property forwarding (mirror of CPU src/types.jl:294-307).
@inline function Base.getproperty(tp::MetalTypedPool, f::Symbol)
    f === :n_active && return getfield(tp, :state).n_active
    f === :_checkpoint_n_active && return getfield(tp, :state)._checkpoint_n_active
    f === :_checkpoint_depths && return getfield(tp, :state)._checkpoint_depths
    return getfield(tp, f)
end

@inline function Base.setproperty!(tp::MetalTypedPool, f::Symbol, v)
    f === :n_active && return setfield!(getfield(tp, :state), :n_active, convert(Int, v))
    return setfield!(tp, f, convert(fieldtype(typeof(tp), f), v))
end

Base.propertynames(tp::MetalTypedPool) =
    (fieldnames(typeof(tp))..., :n_active, :_checkpoint_n_active, :_checkpoint_depths)

# Route the generic checkpoint/rewind cores at the concrete state (zero-dispatch
# drain); without this, MetalTypedPool falls back to _cp_state(tp) = tp and
# _touch_fallback_pool!'s ::PoolCheckpointState assert throws.
@inline AdaptiveArrayPools._cp_state(tp::MetalTypedPool) = getfield(tp, :state)

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

    # Touched-others tracking (depth-tagged, concrete) — mirror of CPU
    # src/types.jl:434-453. Checkpoint variants push NOTHING; producers push one
    # (state, depth) entry per first touch; rewind pops while the top tag matches.
    # _touched_others_pools is populated only when R >= 1 (slot invalidation).
    _touched_others_states::Vector{PoolCheckpointState}
    _touched_others_depths::Vector{Int}
    _touched_others_pools::Vector{Any}

    # Last-lookup memo for the fallback registry (mirror of CPU). Set on every
    # slow-path lookup; cleared by empty! (identities die), preserved by
    # reset!/trim!/compact! (identities survive). Task-local pool → no races.
    _lookup_memo_type::Any
    _lookup_memo_tp::Any

    # Device tracking (safety)
    device_key::Any

    # Borrow tracking (required: macro injects pool._pending_callsite = "..." as raw AST)
    _pending_callsite::String
    _pending_return_site::String
    _borrow_log::Union{Nothing, IdDict{Any, String}}

    # Auto-manage request flags (parity with CPU AdaptiveArrayPool): set by the base
    # module's background Timer sweep (a different thread), read + reset by the owner task
    # at the `@with_pool :metal` entry safepoint. Atomic for the cross-thread handoff.
    # `_compact_requested` is set every tick (→ compact!); `_trim_requested` every
    # `trim_interval` (→ auto-trim the tail). The entry hook fires on `_compact_requested`
    # (set on the same sweep), so the generic `_run_auto_manage!` services both at once.
    @atomic _compact_requested::Bool
    @atomic _trim_requested::Bool
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
        PoolCheckpointState[],  # _touched_others_states: no fallback touches yet
        Int[],                  # _touched_others_depths
        Any[],                  # _touched_others_pools
        nothing,                # _lookup_memo_type
        nothing,                # _lookup_memo_tp
        Metal.device(),
        "",             # _pending_callsite
        "",             # _pending_return_site
        nothing,        # _borrow_log: lazily created when R >= 1
        false,          # _compact_requested: no pending compact request
        false,          # _trim_requested: no pending auto-trim request
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
    _check_level(pool::MetalAdaptiveArrayPool) -> Int

Runtime-check level as an Int (mirror of CPU `src/types.jl:513`), for
backend-shared code that forwards it to `_invalidate_released_slots!` /
`_rewind_typed_pool!`. Compile-time constant per concrete pool type.
"""
@inline AdaptiveArrayPools._check_level(::MetalAdaptiveArrayPool{R, S}) where {R, S} = R

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
