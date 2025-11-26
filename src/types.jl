# ==============================================================================
# Core Data Structures (v3: View Caching + SoA for zero-allocation hot path)
# ==============================================================================

"""
    TypedPool{T}

Internal structure managing a list of vectors for a specific type `T`.

## v3 Features (View Caching + SoA)
- `views`: Cached SubArray objects for zero-allocation hot path
- `view_lengths`: Separate length tracking for cache-friendly comparison (SoA pattern)
- `saved_stack`: Nested mark/reset support with zero allocation
"""
mutable struct TypedPool{T}
    vectors::Vector{Vector{T}}   # Actual memory storage
    views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}  # Cached views
    view_lengths::Vector{Int}    # SoA: cached view lengths (cache-friendly Int comparison)
    in_use::Int                  # Number of currently checked-out vectors
    saved_stack::Vector{Int}     # Stack for nested mark/reset (zero alloc after warmup)
end

TypedPool{T}() where {T} = TypedPool{T}(
    Vector{T}[],
    SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}[],
    Int[],
    0,
    Int[]
)

# ==============================================================================
# AdaptiveArrayPool (v2: Fixed Slots + Fallback)
# ==============================================================================

"""
    AdaptiveArrayPool

A high-performance memory pool supporting multiple data types.

## v2 Features
- **Fixed Slots**: Float64, Float32, Int64, Int32, ComplexF64, Bool have dedicated fields (zero Dict lookup)
- **Fallback**: Other types use IdDict (still fast, but with lookup overhead)
- **Zero Allocation**: mark!/reset! use internal stacks, no allocation after warmup

## Thread Safety
This pool is **NOT thread-safe**. Use one pool per Task via `get_global_pool()`.
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
