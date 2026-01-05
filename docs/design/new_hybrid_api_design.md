# Design Spec: Hybrid N-way Cache & ReshapedArray Strategy

> **Note**: This document was written as a clear, concrete specification that another AI can
> implement mechanically without needing to think through the design.

---

## 1. Objective
Refactor AdaptiveArrayPools.jl to implement a **Hybrid Allocation Strategy**:
1.  **`acquire!` (Default)**: Return `ReshapedArray` (Zero-Allocation, Stack-allocated). Remove N-D caching logic for this path.
2.  **`unsafe_acquire!` (Special)**: Return `Array` (via `unsafe_wrap`). Implement **N-way Set Associative Cache** to minimize `unsafe_wrap` overhead (112 bytes) and support interleaved access patterns.

## 2. Data Structure Changes (types.jl)

### Constants
Define the cache associativity level.
```julia
const CACHE_WAYS = 4
```

### `TypedPool{T}` Struct
Modify fields to support N-way caching for Arrays, while removing unused View caching.

*   **Remove**: `nd_views` (No longer needed as `acquire!` returns `ReshapedArray`).
*   **Update**: `nd_arrays`, `nd_dims`, `nd_ptrs`. These vectors must store `CACHE_WAYS` items per active slot.
*   **Add**: `nd_next_way::Vector{Int}` (To track Round-Robin replacement index for each slot).

**Updated Layout:**
```julia
mutable struct TypedPool{T}
    # --- Backing Storage ---
    vectors::Vector{Vector{T}}

    # --- 1D Cache (Simple 1-way or Direct) ---
    views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    view_lengths::Vector{Int}

    # --- N-D Array Cache (N-way Set Associative) ---
    # Layout: Flat Vector. Index = (slot_idx - 1) * CACHE_WAYS + way_idx
    nd_arrays::Vector{Any}      # Stores Array{T, N}
    nd_dims::Vector{Any}        # Stores NTuple{N, Int}
    nd_ptrs::Vector{UInt}       # Stores objectid/pointer for validation
    nd_next_way::Vector{Int}    # Round-Robin counter per slot (1 per slot)

    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end
```

### Initialization
Ensure `nd_arrays`, `nd_dims`, `nd_ptrs` are initialized with `nothing` or empty values, and `nd_next_way` with `0` or `1`.

## 3. Logic Implementation (core.jl)

### A. `acquire!` (The Fast Path)
**Goal**: Always return `ReshapedArray`. No N-D cache lookup.

**Implementation**:
Modify `get_nd_view!` to:
1.  Calculate total length (`prod(dims)`).
2.  Call `get_view!(tp, len)` to get a 1D `SubArray`.
3.  Return `reshape(flat_view, dims)`.

```julia
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    len = safe_prod(dims)
    flat_view = get_view!(tp, len)
    return reshape(flat_view, dims)
end
```

### B. `unsafe_acquire!` (The N-way Path)
**Goal**: Return `Array`. Use N-way cache to avoid `unsafe_wrap`.

**Implementation**:
Modify `get_nd_array!` to use **Linear Search + Round-Robin Replacement**.

**Algorithm**:
1.  Get 1D view: `flat_view = get_view!(tp, prod(dims))`.
2.  Get current pointer: `current_ptr = UInt(pointer(flat_view))`.
3.  Calculate Base Index: `base = (tp.n_active - 1) * CACHE_WAYS`.
4.  **Search (Hit Check)**:
    *   Loop `k` from `1` to `CACHE_WAYS`.
    *   Check if `nd_dims[base + k] == dims` **AND** `nd_ptrs[base + k] == current_ptr`.
    *   If match: Return `nd_arrays[base + k]`.
5.  **Miss (Replacement)**:
    *   Get victim way from `nd_next_way[tp.n_active]`.
    *   Target Index: `target = base + victim_way + 1`.
    *   Create Array: `arr = unsafe_wrap(Array{T, N}, pointer(flat_view), dims)`.
    *   **Update Cache**:
        *   `nd_arrays[target] = arr`
        *   `nd_dims[target] = dims`
        *   `nd_ptrs[target] = current_ptr`
    *   **Update Round-Robin**: Increment `nd_next_way` (modulo `CACHE_WAYS`).
    *   Return `arr`.

## 4. API & Aliases (AdaptiveArrayPools.jl)

Add explicit aliases for clarity.

```julia
# Main APIs
export acquire!, unsafe_acquire!

# Explicit Aliases
export acquire_view!, acquire_array!

"""Alias for [`acquire!`](@ref). Returns a ReshapedArray (View)."""
const acquire_view! = acquire!

"""Alias for [`unsafe_acquire!`](@ref). Returns an Array (via unsafe_wrap)."""
const acquire_array! = unsafe_acquire!
```

## 5. Client Integration (`TurbulentTransport.jl`)

Update tglf_nn.jl to use the Array-returning API to avoid dynamic dispatch boxing.

**File**: tglf_nn.jl
**Function**: `flux_array!`
**Change**:
```julia
# Before
xx = acquire!(pool, T, size(x))

# After
xx = unsafe_acquire!(pool, T, size(x))
# OR
xx = acquire_array!(pool, T, size(x))
```

## 6. Verification Checklist

1.  **Type Check**: `acquire!` must return `ReshapedArray`. `unsafe_acquire!` must return `Array`.
2.  **Allocation Check**:
    *   `acquire!`: 0 allocations always.
    *   `unsafe_acquire!`: 0 allocations on cache hit.
    *   `unsafe_acquire!`: 0 allocations on interleaved access (e.g., alternating 10x10 and 20x20) thanks to N-way cache.
3.  **Safety**: Ensure `unsafe_acquire!` validates pointers (re-wraps if the backing vector was resized).
