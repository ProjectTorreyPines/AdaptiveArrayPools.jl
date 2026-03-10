# ==============================================================================
# Abstract Type Hierarchy (for extensibility)
# ==============================================================================

"""
    AbstractTypedPool{T, V<:AbstractVector{T}}

Abstract base for type-specific memory pools.
"""
abstract type AbstractTypedPool{T, V <: AbstractVector{T}} end

"""
    AbstractArrayPool

Abstract base for multi-type array pools.
"""
abstract type AbstractArrayPool end

# ==============================================================================
# Disabled Pool Sentinel Types
# ==============================================================================

"""
    DisabledPool{Backend}

Sentinel type for disabled pooling that preserves backend context.
When `STATIC_POOLING=false` (compile-time) or `MAYBE_POOLING[]=false` (runtime),
macros return `DisabledPool{backend}()` instead of `nothing`.

Backend symbols:
- `:cpu` - Standard Julia arrays
- `:cuda` - CUDA.jl CuArrays (defined in extension)

This enables `@with_pool :cuda` to return correct array types even when pooling is off.

## Example
```julia
# When STATIC_POOLING=false:
@with_pool :cuda pool begin
    v = zeros!(pool, 10)  # Returns CuArray{Float32}, not Array{Float64}!
end
```

See also: [`pooling_enabled`](@ref), [`DISABLED_CPU`](@ref)
"""
struct DisabledPool{Backend} <: AbstractArrayPool end

"""
    DISABLED_CPU

Singleton instance for disabled CPU pooling.
Used by macros when `STATIC_POOLING=false` without backend specification.
"""
const DISABLED_CPU = DisabledPool{:cpu}()

"""
    pooling_enabled(pool) -> Bool

Returns `true` if `pool` is an active pool, `false` if pooling is disabled.

## Examples
```julia
@maybe_with_pool pool begin
    if pooling_enabled(pool)
        # Using pooled memory
    else
        # Using standard allocation
    end
end
```

See also: [`DisabledPool`](@ref)
"""
pooling_enabled(::AbstractArrayPool) = true
pooling_enabled(::DisabledPool) = false

# ==============================================================================
# Core Data Structures
# ==============================================================================

# 1-Based Sentinel Pattern: Arrays start with sentinel values to eliminate
# isempty() checks in hot paths. See docstrings for details.

"""
    TypedPool{T} <: AbstractTypedPool{T, Vector{T}}

Internal structure managing pooled vectors for a specific element type `T`.

## Fields

### Storage
- `vectors`: Backing `Vector{T}` storage (actual memory allocation)

### N-D Wrapper Cache (Julia 1.11+, setfield!-based reuse)
- `arr_wrappers`: `Vector{Union{Nothing, Vector{Any}}}` — indexed by N (dimensionality),
  each entry is a per-slot cached `Array{T,N}` wrapper. Uses `setfield!(wrapper, :size, dims)`
  and `setfield!(wrapper, :ref, parent)` for zero-allocation reuse of unlimited dim patterns.

### State Management (1-based sentinel pattern)
- `n_active`: Count of currently active (checked-out) arrays
- `_checkpoint_n_active`: Saved n_active values at each checkpoint (sentinel: `[0]`)
- `_checkpoint_depths`: Depth of each checkpoint entry (sentinel: `[0]`)

## Design Notes
- 1D views (`SubArray`) are created fresh on every `acquire!` call — SubArray is stack-allocated
  via SROA in modern Julia, making caching unnecessary (and slower due to memory indirection).
- `unsafe_acquire!` uses `setfield!` wrapper reuse — unlimited dim patterns, 0-alloc after warmup.
- Slot management is unified via `_claim_slot!` — the shared primitive for all acquisition paths.
"""
mutable struct TypedPool{T} <: AbstractTypedPool{T, Vector{T}}
    # --- Storage ---
    vectors::Vector{Vector{T}}

    # --- N-D Wrapper Cache (setfield!-based reuse) ---
    arr_wrappers::Vector{Union{Nothing, Vector{Any}}}  # index=N (dimensionality), value=per-slot Array{T,N}

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}   # Saved n_active at each checkpoint
    _checkpoint_depths::Vector{Int}     # Depth of each checkpoint
end

TypedPool{T}() where {T} = TypedPool{T}(
    # Storage
    Vector{T}[],
    # N-D Wrapper Cache
    Union{Nothing, Vector{Any}}[],
    # State Management (1-based sentinel pattern: guaranteed non-empty)
    0,          # n_active
    [0],        # _checkpoint_n_active: sentinel (n_active=0 at depth=0)
    [0]         # _checkpoint_depths: sentinel (depth=0 = no checkpoint)
)

# ==============================================================================
# Bit Sentinel Type
# ==============================================================================

"""
    Bit

Sentinel type for bit-packed boolean storage via `BitVector`.

Use `Bit` instead of `Bool` in pool operations to get memory-efficient
bit-packed arrays (1 bit per element vs 1 byte for `Vector{Bool}`).

## Usage
```julia
@with_pool pool begin
    # BitVector (1 bit per element, ~8x memory savings)
    bv = acquire!(pool, Bit, 1000)

    # vs Vector{Bool} (1 byte per element)
    vb = acquire!(pool, Bool, 1000)

    # Convenience functions work too
    mask = falses!(pool, 100)       # BitVector filled with false
    flags = trues!(pool, 100)       # BitVector filled with true
end
```

## Return Types (Unified for Performance)
Unlike other types, `Bit` always returns native `BitVector`/`BitArray`:
- **1D**: `BitVector` (both `acquire!` and `unsafe_acquire!`)
- **N-D**: `BitArray{N}` (reshaped, preserves SIMD optimization)

This design ensures users always get SIMD-optimized performance without
needing to remember which API to use.

## Performance
`BitVector` operations like `count()`, `sum()`, and bitwise operations are
~(10x ~ 100x) faster than equivalent operations on `SubArray{Bool}` because they
use SIMD-optimized algorithms on packed 64-bit chunks.

```julia
@with_pool pool begin
    bv = acquire!(pool, Bit, 10000)
    fill!(bv, true)
    count(bv)  # Uses fast SIMD path automatically
end
```

## Memory Safety
The returned `BitVector` shares its internal `chunks` array with the pool.
It is only valid within the `@with_pool` scope - using it after the scope
ends leads to undefined behavior (use-after-free risk).

See also: [`trues!`](@ref), [`falses!`](@ref), [`BitTypedPool`](@ref)
"""
struct Bit end

# ==============================================================================
# BitTypedPool - Specialized pool for BitVector/BitArray
# ==============================================================================

"""
    BitTypedPool <: AbstractTypedPool{Bool, BitVector}

Specialized pool for `BitVector` arrays with memory reuse.

Unlike `TypedPool{Bool}` which stores `Vector{Bool}` (1 byte per element),
this pool stores `BitVector` (1 bit per element, ~8x memory efficiency).

## Unified API (Always Returns BitVector)
Unlike other types, both `acquire!` and `unsafe_acquire!` return `BitVector`
for the `Bit` type. This design ensures users always get SIMD-optimized
performance without needing to choose between APIs.

- `acquire!(pool, Bit, n)` → `BitVector` (SIMD optimized)
- `unsafe_acquire!(pool, Bit, n)` → `BitVector` (same behavior)
- `trues!(pool, n)` → `BitVector` filled with `true`
- `falses!(pool, n)` → `BitVector` filled with `false`

## Fields
- `vectors`: Backing `BitVector` storage
- `arr_wrappers`: `Vector{Union{Nothing, Vector{Any}}}` — setfield!-based cache (Julia 1.11+)
- `n_active`: Count of currently active arrays
- `_checkpoint_*`: State management stacks (1-based sentinel pattern)

## Performance
Operations like `count()`, `sum()`, and bitwise operations are ~(10x ~ 100x) faster
than equivalent operations on `SubArray{Bool}` because `BitVector` uses
SIMD-optimized algorithms on packed 64-bit chunks.

See also: [`trues!`](@ref), [`falses!`](@ref), [`Bit`](@ref)
"""
mutable struct BitTypedPool <: AbstractTypedPool{Bool, BitVector}
    # --- Storage ---
    vectors::Vector{BitVector}

    # --- N-D Wrapper Cache (setfield!-based reuse) ---
    arr_wrappers::Vector{Union{Nothing, Vector{Any}}}  # index=N (dimensionality), value=per-slot BitArray{N}

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

BitTypedPool() = BitTypedPool(
    # Storage
    BitVector[],
    # N-D Wrapper Cache
    Union{Nothing, Vector{Any}}[],
    # State Management (1-based sentinel pattern)
    0,          # n_active
    [0],        # _checkpoint_n_active: sentinel
    [0]         # _checkpoint_depths: sentinel
)

# ==============================================================================
# Fixed Slot Configuration
# ==============================================================================

"""
    FIXED_SLOT_FIELDS

Field names for fixed slot TypedPools. Single source of truth for `foreach_fixed_slot`.

When modifying, also update: struct definition, `get_typed_pool!` dispatches, constructor.
Tests verify synchronization automatically.
"""
const FIXED_SLOT_FIELDS = (:float64, :float32, :int64, :int32, :complexf64, :complexf32, :bool, :bits)

# ==============================================================================
# Bitmask Mode Constants
# ==============================================================================
# Bits 0-7: fixed-slot type touch tracking (one bit per type)
# Bits 14-15: mode flags set during checkpoint to control lazy behavior

const _LAZY_MODE_BIT = UInt16(0x8000)  # bit 15: lazy (dynamic-selective) checkpoint mode
const _TYPED_LAZY_BIT = UInt16(0x4000)  # bit 14: typed lazy-fallback mode
const _MODE_BITS_MASK = UInt16(0xC000)  # bits 14-15: all mode flags
const _TYPE_BITS_MASK = UInt16(0x00FF)  # bits 0-7: fixed-slot type bits

# ==============================================================================
# Fixed-Slot Bit Mapping (for type touch tracking)
# ==============================================================================
# Maps each fixed-slot type to a unique bit in a UInt16 bitmask.
# Bit ordering matches FIXED_SLOT_FIELDS. Non-fixed types return UInt16(0).

@inline _fixed_slot_bit(::Type{Float64}) = UInt16(1) << 0
@inline _fixed_slot_bit(::Type{Float32}) = UInt16(1) << 1
@inline _fixed_slot_bit(::Type{Int64}) = UInt16(1) << 2
@inline _fixed_slot_bit(::Type{Int32}) = UInt16(1) << 3
@inline _fixed_slot_bit(::Type{ComplexF64}) = UInt16(1) << 4
@inline _fixed_slot_bit(::Type{ComplexF32}) = UInt16(1) << 5
@inline _fixed_slot_bit(::Type{Bool}) = UInt16(1) << 6
@inline _fixed_slot_bit(::Type{Bit}) = UInt16(1) << 7
@inline _fixed_slot_bit(::Type) = UInt16(0)  # non-fixed-slot → triggers has_others

# Check whether a type's bit is set in a bitmask (e.g. _touched_type_masks or combined).
@inline _has_bit(mask::UInt16, ::Type{T}) where {T} = (mask & _fixed_slot_bit(T)) != 0

# ==============================================================================
# Safety Configuration
# ==============================================================================
#
# Safety is controlled per-pool via the type parameter S in AdaptiveArrayPool{S}.
# S encodes the safety level (0-3), enabling dead-code elimination at compile time.
#
#   0 = off   — zero overhead (default)
#   1 = guard — structural invalidation on rewind (resize + setfield!)
#   2 = full  — guard + escape detection + poisoning
#   3 = debug — full + borrow registry (call-site tracking)
#
# DEFAULT_SAFETY_LV sets the initial S for new pools.
# POOL_SAFETY_LV is the runtime Ref read by AdaptiveArrayPool() constructor.
# Use set_safety_level!() to replace the task-local pool at runtime.

using Preferences: @load_preference

"""
    DEFAULT_SAFETY_LV::Int

Compile-time default safety level for new pool instances.

Set via `LocalPreferences.toml`:
```toml
[AdaptiveArrayPools]
safety_level = 1
```

Levels: `0` (off, default), `1` (guard), `2` (full), `3` (debug).

Legacy fallback: reads `pool_checks` preference if `safety_level` is not set
(`true` → 1, `false` → 0).
"""
const DEFAULT_SAFETY_LV = let
    sl = @load_preference("safety_level", nothing)
    if sl !== nothing
        sl::Int
    else
        # Legacy: pool_checks=true → safety_level=1
        pc = @load_preference("pool_checks", nothing)
        pc !== nothing ? (pc::Bool ? 1 : 0) : 0
    end
end

"""
    STATIC_POOL_CHECKS::Bool

Compile-time constant derived from `DEFAULT_SAFETY_LV > 0`.
Retained for backward compatibility. New code should use `_safety_level(pool) >= N`.
"""
const STATIC_POOL_CHECKS = DEFAULT_SAFETY_LV > 0

"""
    POOL_SAFETY_LV

Runtime safety level. Read by `AdaptiveArrayPool()` constructor to determine
the `S` parameter of newly created pools.

- `0`: Off — zero overhead
- `1`: Guard — structural invalidation on rewind (resize + setfield!)
- `2`: Full — guard + escape detection on scope exit + poisoning
- `3`: Debug — full + borrow registry (acquire call-site tracking)

Initial value: `DEFAULT_SAFETY_LV` (default `0`).
Use `set_safety_level!()` to change the task-local pool at runtime.
"""
const POOL_SAFETY_LV = Ref{Int}(DEFAULT_SAFETY_LV)

# ==============================================================================
# AdaptiveArrayPool
# ==============================================================================

"""
    AdaptiveArrayPool{S}

Multi-type memory pool with fixed slots for common types and IdDict fallback for others.
Zero allocation after warmup. NOT thread-safe - use one pool per Task.

The type parameter `S::Int` encodes the safety level (0-3). Inside `@inline` call chains,
`S` is a compile-time constant — safety checks like `S >= 1` are eliminated by dead-code
elimination when `S = 0`, achieving true zero overhead.

See also: [`_safety_level`], [`_make_pool`], [`set_safety_level!`]
"""
mutable struct AdaptiveArrayPool{S} <: AbstractArrayPool
    # Fixed Slots: common types with zero lookup overhead
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    complexf32::TypedPool{ComplexF32}
    bool::TypedPool{Bool}
    bits::BitTypedPool              # BitVector pool (1 bit per element)

    # Fallback: rare types
    others::IdDict{DataType, Any}

    # Type touch tracking (1-based sentinel pattern)
    _current_depth::Int             # Current scope depth (1 = global scope)
    _touched_type_masks::Vector{UInt16}  # Per-depth: which fixed slots were touched + mode flags
    _touched_has_others::Vector{Bool}    # Per-depth: any non-fixed-slot type touched?

    # Borrow registry (POOL_SAFETY_LV >= 3 only)
    _pending_callsite::String                        # "" = no pending; set by macro before acquire
    _pending_return_site::String                     # "" = no pending; set by macro before validate
    _borrow_log::Union{Nothing, IdDict{Any, String}} # vector_obj => callsite string
end

function AdaptiveArrayPool{S}() where {S}
    return AdaptiveArrayPool{S}(
        TypedPool{Float64}(),
        TypedPool{Float32}(),
        TypedPool{Int64}(),
        TypedPool{Int32}(),
        TypedPool{ComplexF64}(),
        TypedPool{ComplexF32}(),
        TypedPool{Bool}(),
        BitTypedPool(),
        IdDict{DataType, Any}(),
        1,              # _current_depth: 1 = global scope (sentinel)
        [UInt16(0)],    # _touched_type_masks: sentinel (no bits set)
        [false],        # _touched_has_others: sentinel (no others)
        "",             # _pending_callsite: no pending
        "",             # _pending_return_site: no pending
        nothing         # _borrow_log: lazily created at LV >= 3
    )
end

"""Create pool at the current `POOL_SAFETY_LV[]` level."""
AdaptiveArrayPool() = _make_pool(POOL_SAFETY_LV[])

"""
    _safety_level(pool) -> Int

Return the safety level of a pool. Compile-time constant for `AdaptiveArrayPool{S}`.
"""
@inline _safety_level(::AdaptiveArrayPool{S}) where {S} = S
@inline _safety_level(::AbstractArrayPool) = POOL_SAFETY_LV[]  # fallback

"""
    _make_pool(s::Int) -> AdaptiveArrayPool{s}

Function barrier: converts runtime `Int` to concrete `AdaptiveArrayPool{S}`.
Levels outside 0-3 are clamped (≤0 → 0, ≥3 → 3).
"""
@noinline function _make_pool(s::Int)
    s <= 0 && return AdaptiveArrayPool{0}()
    s == 1 && return AdaptiveArrayPool{1}()
    s == 2 && return AdaptiveArrayPool{2}()
    return AdaptiveArrayPool{3}()
end

# ==============================================================================
# Type Dispatch (Zero-cost for Fixed Slots)
# ==============================================================================

# Fast Path: compile-time dispatch, fully inlined
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Float64}) = p.float64
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Float32}) = p.float32
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Int64}) = p.int64
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Int32}) = p.int32
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{ComplexF64}) = p.complexf64
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{ComplexF32}) = p.complexf32
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Bool}) = p.bool
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Bit}) = p.bits

# Slow Path: rare types via IdDict
@inline function get_typed_pool!(p::AdaptiveArrayPool, ::Type{T}) where {T}
    return get!(p.others, T) do
        tp = TypedPool{T}()
        # If inside a checkpoint scope (_current_depth > 1 means inside @with_pool),
        # auto-checkpoint the new pool to prevent issues on rewind
        if p._current_depth > 1
            push!(tp._checkpoint_n_active, 0)  # n_active starts at 0
            push!(tp._checkpoint_depths, p._current_depth)
            # Signal that a fallback type was touched so lazy/typed-lazy rewind
            # iterates pool.others. Without this, _acquire_impl! (which bypasses
            # _record_type_touch!) would leave has_others=false, causing the
            # rewind to skip pool.others entirely and leak this new type's n_active.
            @inbounds p._touched_has_others[p._current_depth] = true
        end
        tp
    end::TypedPool{T}
end

# ==============================================================================
# Zero-Allocation Iteration
# ==============================================================================

"""
    foreach_fixed_slot(f, pool::AdaptiveArrayPool)

Apply `f` to each fixed slot TypedPool. Zero allocation via compile-time unrolling.
"""
@generated function foreach_fixed_slot(f::F, pool::AdaptiveArrayPool) where {F}
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in FIXED_SLOT_FIELDS]
    return quote
        Base.@_inline_meta
        $(exprs...)
        nothing
    end
end

# ==============================================================================
# Safety Dispatch (via pool type parameter S)
# ==============================================================================
#
# Safety checks dispatch on the pool's type parameter S.
# When S < threshold, the check compiles to nothing (dead code elimination).
# No tag structs needed — the type parameter IS the compile-time tag.

"""
    _set_pending_callsite!(pool, msg::String)

Record a pending callsite string for borrow tracking (safety level ≥ 3).
Only sets the callsite if no prior callsite is pending (macro-injected ones take priority).
Compiles to no-op when `S < 3`.
"""
@inline function _set_pending_callsite!(pool::AdaptiveArrayPool{S}, msg::String) where {S}
    S >= 3 && isempty(pool._pending_callsite) && (pool._pending_callsite = msg)
    return nothing
end
@inline _set_pending_callsite!(::AbstractArrayPool, ::String) = nothing

"""
    _maybe_record_borrow!(pool, tp::AbstractTypedPool)

Flush the pending callsite into the borrow log (safety level ≥ 3).
Delegates to `_record_borrow_from_pending!` (defined in `debug.jl`).
Compiles to no-op when `S < 3`.
"""
@inline function _maybe_record_borrow!(pool::AdaptiveArrayPool{S}, tp::AbstractTypedPool) where {S}
    S >= 3 && _record_borrow_from_pending!(pool, tp)
    return nothing
end
@inline _maybe_record_borrow!(::AbstractArrayPool, ::AbstractTypedPool) = nothing
