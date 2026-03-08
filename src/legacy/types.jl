# ==============================================================================
# Constants (Configurable via Preferences) — Legacy (Julia ≤1.10)
# ==============================================================================

using Preferences

"""
Number of cache ways per slot for N-way set associative cache.
Supports up to `CACHE_WAYS` different dimension patterns per slot without thrashing.

Default: 4 (handles most use cases well)

## Configuration
```julia
using AdaptiveArrayPools
AdaptiveArrayPools.set_cache_ways!(8)  # Restart Julia to take effect
```

Or manually in `LocalPreferences.toml`:
```toml
[AdaptiveArrayPools]
cache_ways = 8
```

Valid range: 1-16 (higher values increase memory but reduce eviction)
"""
const CACHE_WAYS = let
    ways = @load_preference("cache_ways", 4)::Int
    if ways < 1 || ways > 16
        @warn "CACHE_WAYS=$ways out of range [1,16], using default 4"
        4
    else
        ways
    end
end

"""
    set_cache_ways!(n::Int)

Set the number of cache ways for N-D array caching.
**Requires Julia restart to take effect.**

Higher values reduce cache eviction but increase memory usage per slot.

## Arguments
- `n::Int`: Number of cache ways (valid range: 1-16)

## Example
```julia
using AdaptiveArrayPools
AdaptiveArrayPools.set_cache_ways!(8)  # Double the default
# Restart Julia to apply the change
```
"""
function set_cache_ways!(n::Int)
    if n < 1 || n > 16
        throw(ArgumentError("cache_ways must be in range [1, 16], got $n"))
    end
    @set_preferences!("cache_ways" => n)
    @info "CACHE_WAYS set to $n. Restart Julia to apply."
    return n
end

# ==============================================================================
# Safety Configuration (2-Tier Toggle)
# ==============================================================================
#   Tier 1: STATIC_POOL_CHECKS (compile-time const)
#   Tier 2: POOL_SAFETY_LV (runtime Ref{Int}, levels 0/1/2)

const STATIC_POOL_CHECKS = @load_preference("pool_checks", true)::Bool

"""
    POOL_SAFETY_LV

Runtime safety level for pool operations. Only effective when `STATIC_POOL_CHECKS` is `true`.

- `0`: Off — no safety checks (Ref read only, ~1ns)
- `1`: Guard — structural invalidation on rewind (resize + setfield!, ~1ns/slot)
- `2`: Full — guard + escape detection on scope exit (future: + poisoning)

Default: `1` (guard mode)
"""
const POOL_SAFETY_LV = Ref(1)

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
When `USE_POOLING=false` (compile-time) or `MAYBE_POOLING_ENABLED[]=false` (runtime),
macros return `DisabledPool{backend}()` instead of `nothing`.

Backend symbols:
- `:cpu` - Standard Julia arrays
- `:cuda` - CUDA.jl CuArrays (defined in extension)

This enables `@with_pool :cuda` to return correct array types even when pooling is off.

## Example
```julia
# When USE_POOLING=false:
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
Used by macros when `USE_POOLING=false` without backend specification.
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
# Core Data Structures — Legacy (Julia ≤1.10, N-way set-associative cache)
# ==============================================================================

# 1-Based Sentinel Pattern: Arrays start with sentinel values to eliminate
# isempty() checks in hot paths. See docstrings for details.

"""
    TypedPool{T} <: AbstractTypedPool{T, Vector{T}}

Internal structure managing pooled vectors for a specific element type `T`.

## Fields

### Storage
- `vectors`: Backing `Vector{T}` storage (actual memory allocation)

### 1D Cache (for `acquire!(pool, T, n)`)
- `views`: Cached `SubArray` views for zero-allocation 1D access
- `view_lengths`: Cached lengths for fast Int comparison (SoA pattern)

### N-D Array Cache (N-way set-associative, Julia ≤1.10)
- `nd_arrays`: Cached N-D `Array` objects (length = slots × CACHE_WAYS)
- `nd_dims`: Cached dimension tuples for cache hit validation
- `nd_ptrs`: Cached pointer values to detect backing vector resize
- `nd_next_way`: Round-robin counter per slot (length = slots)

### State Management (1-based sentinel pattern)
- `n_active`: Count of currently active (checked-out) arrays
- `_checkpoint_n_active`: Saved n_active values at each checkpoint (sentinel: `[0]`)
- `_checkpoint_depths`: Depth of each checkpoint entry (sentinel: `[0]`)
"""
mutable struct TypedPool{T} <: AbstractTypedPool{T, Vector{T}}
    # --- Storage ---
    vectors::Vector{Vector{T}}

    # --- 1D Cache (1:1 mapping) ---
    views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}
    view_lengths::Vector{Int}

    # --- N-D Array Cache (N-way set associative) ---
    nd_arrays::Vector{Any}      # length = slots × CACHE_WAYS
    nd_dims::Vector{Any}        # dimension tuples
    nd_ptrs::Vector{UInt}       # pointer validation
    nd_next_way::Vector{Int}    # round-robin counter per slot

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}   # Saved n_active at each checkpoint
    _checkpoint_depths::Vector{Int}     # Depth of each checkpoint
end

TypedPool{T}() where {T} = TypedPool{T}(
    # Storage
    Vector{T}[],
    # 1D Cache
    SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}[],
    Int[],
    # N-D Array Cache
    Any[], Any[], UInt[], Int[],
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
# BitTypedPool — Legacy (Julia ≤1.10, N-way set-associative cache)
# ==============================================================================

"""
    BitTypedPool <: AbstractTypedPool{Bool, BitVector}

Specialized pool for `BitVector` arrays with memory reuse.

## Fields
- `vectors`: Backing `BitVector` storage
- `nd_arrays`, `nd_dims`, `nd_ptrs`, `nd_next_way`: N-way cache (Julia ≤1.10)
- `n_active`: Count of currently active arrays
- `_checkpoint_*`: State management stacks (1-based sentinel pattern)

See also: [`trues!`](@ref), [`falses!`](@ref), [`Bit`](@ref)
"""
mutable struct BitTypedPool <: AbstractTypedPool{Bool, BitVector}
    # --- Storage ---
    vectors::Vector{BitVector}

    # --- N-D BitArray Cache (N-way set associative) ---
    nd_arrays::Vector{Any}      # Cached BitArray{N} instances
    nd_dims::Vector{Any}        # Cached dims (NTuple{N,Int})
    nd_ptrs::Vector{UInt}       # pointer validation
    nd_next_way::Vector{Int}    # round-robin counter per slot

    # --- State Management (1-based sentinel pattern) ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

BitTypedPool() = BitTypedPool(
    # Storage
    BitVector[],
    # N-D BitArray Cache
    Any[], Any[], UInt[], Int[],
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
# AdaptiveArrayPool
# ==============================================================================

"""
    AdaptiveArrayPool

Multi-type memory pool with fixed slots for common types and IdDict fallback for others.
Zero allocation after warmup. NOT thread-safe - use one pool per Task.
"""
mutable struct AdaptiveArrayPool <: AbstractArrayPool
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
end

function AdaptiveArrayPool()
    return AdaptiveArrayPool(
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
        [false]         # _touched_has_others: sentinel (no others)
    )
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
