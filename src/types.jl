# ==============================================================================
# Core Data Structures
# ==============================================================================

"""
    TypedPool{T}

Internal structure managing pooled vectors for a specific element type `T`.

## Fields

### Storage
- `vectors`: Backing `Vector{T}` storage (actual memory allocation)

### 1D Cache (for `acquire!(pool, T, n)`)
- `views`: Cached `SubArray` views for zero-allocation 1D access
- `view_lengths`: Cached lengths for fast Int comparison (SoA pattern)

### N-D Cache (for `acquire!(pool, T, dims...)` and `unsafe_acquire!`)
- `nd_views`: Cached N-D `SubArray` objects (returned by `acquire!`)
- `nd_arrays`: Cached N-D `Array` objects (returned by `unsafe_acquire!`)
- `nd_dims`: Cached dimension tuples for cache hit validation
- `nd_ptrs`: Cached pointer values to detect backing vector resize

### State Management
- `n_active`: Count of currently active (checked-out) arrays
- `saved_stack`: Stack for nested `checkpoint!/rewind!` (zero-alloc after warmup)
"""
mutable struct TypedPool{T}
    # --- Storage ---
    vectors::Vector{Vector{T}}

    # --- 1D Cache ---
    views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}
    view_lengths::Vector{Int}

    # --- N-D Cache ---
    nd_views::Vector{Any}
    nd_arrays::Vector{Any}
    nd_dims::Vector{Any}
    nd_ptrs::Vector{UInt}

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
    # N-D Cache
    Any[],
    Any[],
    Any[],
    UInt[],
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
