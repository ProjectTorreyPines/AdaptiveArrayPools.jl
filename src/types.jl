# ==============================================================================
# Constants (Configurable via Preferences)
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
# Abstract Type Hierarchy (for extensibility)
# ==============================================================================

"""
    AbstractTypedPool{T, V<:AbstractVector{T}}

Abstract base for type-specific memory pools.
"""
abstract type AbstractTypedPool{T, V<:AbstractVector{T}} end

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
struct DisabledPool{Backend} end

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

### 1D Cache (for `acquire!(pool, T, n)`)
- `views`: Cached `SubArray` views for zero-allocation 1D access
- `view_lengths`: Cached lengths for fast Int comparison (SoA pattern)

### N-D Array Cache (for `unsafe_acquire!` only, N-way set associative)
- `nd_arrays`: Cached N-D `Array` objects (length = slots × CACHE_WAYS)
- `nd_dims`: Cached dimension tuples for cache hit validation
- `nd_ptrs`: Cached pointer values to detect backing vector resize
- `nd_next_way`: Round-robin counter per slot (length = slots)

### State Management (1-based sentinel pattern)
- `n_active`: Count of currently active (checked-out) arrays
- `_checkpoint_n_active`: Saved n_active values at each checkpoint (sentinel: `[0]`)
- `_checkpoint_depths`: Depth of each checkpoint entry (sentinel: `[0]`)

## Note
`acquire!` for N-D returns `ReshapedArray` (zero creation cost), so no caching needed.
Only `unsafe_acquire!` benefits from N-D caching since `unsafe_wrap` allocates 112 bytes.
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
    # N-D Array Cache (N-way)
    Any[],
    Any[],
    UInt[],
    Int[],
    # State Management (1-based sentinel pattern: guaranteed non-empty)
    0,          # n_active
    [0],        # _checkpoint_n_active: sentinel (n_active=0 at depth=0)
    [0]         # _checkpoint_depths: sentinel (depth=0 = no checkpoint)
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
const FIXED_SLOT_FIELDS = (:float64, :float32, :int64, :int32, :complexf64, :complexf32, :bool)

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

    # Fallback: rare types
    others::IdDict{DataType, Any}

    # Untracked acquire detection (1-based sentinel pattern)
    _current_depth::Int             # Current scope depth (1 = global scope)
    _untracked_flags::Vector{Bool}  # Per-depth flag: true if untracked acquire occurred
end

function AdaptiveArrayPool()
    AdaptiveArrayPool(
        TypedPool{Float64}(),
        TypedPool{Float32}(),
        TypedPool{Int64}(),
        TypedPool{Int32}(),
        TypedPool{ComplexF64}(),
        TypedPool{ComplexF32}(),
        TypedPool{Bool}(),
        IdDict{DataType, Any}(),
        1,              # _current_depth: 1 = global scope (sentinel)
        [false]         # _untracked_flags: sentinel for global scope
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

# Slow Path: rare types via IdDict
@inline function get_typed_pool!(p::AdaptiveArrayPool, ::Type{T}) where {T}
    get!(p.others, T) do
        tp = TypedPool{T}()
        # If inside a checkpoint scope (_current_depth > 1 means inside @with_pool),
        # auto-checkpoint the new pool to prevent issues on rewind
        if p._current_depth > 1
            push!(tp._checkpoint_n_active, 0)  # n_active starts at 0
            push!(tp._checkpoint_depths, p._current_depth)
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
    quote
        Base.@_inline_meta
        $(exprs...)
        nothing
    end
end
