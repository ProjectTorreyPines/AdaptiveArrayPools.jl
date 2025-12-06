# ==============================================================================
# Core Data Structures
# ==============================================================================

"""
N-way set associative cache size for N-D Array caching.

Each slot can cache up to `CACHE_WAYS` different dimension patterns,
preventing thrashing when alternating between different array shapes.
"""
const CACHE_WAYS = 4

"""
    TypedPool{T}

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

### State Management
- `n_active`: Count of currently active (checked-out) arrays
- `saved_stack`: Stack for nested `checkpoint!/rewind!` (zero-alloc after warmup)

## Note
`acquire!` for N-D returns `ReshapedArray` (zero creation cost), so no caching needed.
Only `unsafe_acquire!` benefits from N-D caching since `unsafe_wrap` allocates 112 bytes.
"""
mutable struct TypedPool{T}
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

    # --- State Management ---
    n_active::Int
    saved_stack::Vector{Int}
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
    # State Management
    0,
    Int[]
)

# ==============================================================================
# AdaptiveArrayPool
# ==============================================================================

"""
    AdaptiveArrayPool

A high-performance memory pool supporting multiple data types.

## Features
- **Fixed Slots**: `Float64`, `Float32`, `Int64`, `Int32`, `ComplexF64`, `Bool` have dedicated fields (zero Dict lookup)
- **Fallback**: Other types use `IdDict` (still fast, but with lookup overhead)
- **Zero Allocation**: `checkpoint!/rewind!` use internal stacks, no allocation after warmup

## Thread Safety
This pool is **NOT thread-safe**. Use one pool per Task via `get_task_local_pool()`.
"""
mutable struct AdaptiveArrayPool
    # Fixed Slots: common types with zero lookup overhead
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    bool::TypedPool{Bool}

    # Fallback: rare types
    others::IdDict{DataType, Any}
end

function AdaptiveArrayPool()
    AdaptiveArrayPool(
        TypedPool{Float64}(),
        TypedPool{Float32}(),
        TypedPool{Int64}(),
        TypedPool{Int32}(),
        TypedPool{ComplexF64}(),
        TypedPool{Bool}(),
        IdDict{DataType, Any}()
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
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Bool}) = p.bool

# Slow Path: rare types via IdDict
@inline function get_typed_pool!(p::AdaptiveArrayPool, ::Type{T}) where {T}
    get!(p.others, T) do
        TypedPool{T}()
    end::TypedPool{T}
end
