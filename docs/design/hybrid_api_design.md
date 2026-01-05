# Hybrid API Design: acquire! vs unsafe_acquire!

## Executive Summary

Redesigning `AdaptiveArrayPools.jl`'s N-D array acquisition API with a **Two Tools Strategy**:

| API | Return Type | Use Case | Allocation Characteristics |
|-----|-------------|----------|---------------------------|
| `acquire!` | `ReshapedArray` (fixed) | General use, Static dispatch | No cache needed, relies on compiler optimization |
| `unsafe_acquire!` | `Array` (fixed) | Dynamic dispatch, FFI | Cache hit: 0, miss: 112 bytes |

**Core Principle**: Return type does not change based on state (Type Stability guaranteed)

> **Note**: ReshapedArray's "0 allocation" depends on compiler's SROA (Scalar Replacement of Aggregates) and
> escape analysis. Not always guaranteed - heap allocation may occur if the value escapes from the function.

---

## Problem Statement

### Current State (v0.2.0)

```
acquire!(pool, T, dims...)
  └─> get_nd_view!()
        └─> get_nd_array!()  ← uses unsafe_wrap
              └─> 112 bytes on cache miss!
```

- Both `acquire!` and `unsafe_acquire!` internally use `unsafe_wrap`
- Always 112 bytes allocation on cache miss
- Tried to reduce miss rate with N-way cache, but 100% miss on cyclic patterns

### v0.1.2 Approach

```
acquire!(pool, T, dims...)
  └─> get_view!(tp, total_len)  ← 1D view (cached)
        └─> reshape(view, dims)  ← 0 bytes always!
```

- `reshape(view, dims)` creates a wrapper object, but heap allocation can be avoided via compiler optimization (SROA/escape analysis)
- Simple and predictable

---

## Why Not Mixed Return Types?

### Proposed (but rejected) Approach

```julia
# ❌ BAD: Array on cache hit, View on miss
function acquire!(pool, T, dims...)
    if cache_hit
        return cached_array::Array{T,N}
    else
        return reshape(view, dims)::ReshapedArray{...}
    end
end
```

### Problem: Type Instability

| Aspect | Impact |
|--------|--------|
| **Compiler inference** | `Union{Array, ReshapedArray}` → Union splitting or dynamic dispatch |
| **Performance** | Execution slowdown while trying to achieve zero-alloc |
| **API semantics** | Same function returning different types → confusion |
| **Module boundaries** | Inference widens when storing result or passing to other modules |

**AI Feedback Quote**:
> "State-dependent returns become Union{Array, ReshapedArray} from external view, breaking API-level type stability."

---

## Recommended Design: Two Tools Strategy

### Principles

1. **Fixed return type**: Each API always returns the same type
2. **Purpose separation**: Users choose API based on situation
3. **Simple implementation**: Minimize complex cache logic

### API Design

#### 1. `acquire!` → ReshapedArray (regression to v0.1.2 style)

```julia
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)  # 1D view (cached, 0 alloc)
    return reshape(flat_view, dims)        # ReshapedArray (0 alloc always!)
end
```

**Characteristics**:
- Always returns `ReshapedArray{T, N, SubArray{T, 1, Vector{T}, ...}, ...}`
- No `unsafe_wrap` call → no Array header creation cost (112B) even on cache miss
- N-way cache unnecessary (simple 1D view cache sufficient)

**Use Cases**:
- General `Flux` layers (`mul!`, `broadcast`)
- Code where static dispatch is guaranteed
- Most use cases

**Constraints**:
- Escape optimization may fail in type-unspecified call paths, causing wrapper allocation
- Incompatible with APIs requiring strict `Array` type (rare)

#### 2. `unsafe_acquire!` → Array (maintains v0.2.0 + N-way Cache)

```julia
@inline function unsafe_acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    tp = get_typed_pool!(pool, T)
    return get_nd_array!(tp, dims)  # Array with slot-based + N-way cache
end
```

**Characteristics**:
- Always returns `Array{T, N}`
- Cache hit: 0 bytes, Cache miss: 112 bytes
- Maintains existing N-way cache (4-way)

**Use Cases**:
- Type-unspecified call paths (e.g., `TGLFNNmodel._pooled_chain` - no concrete type parameters)
- FFI / ccall
- Special APIs that strictly require `StridedArray`
- Places requiring strict `Array` type

**Benefits**:
- Array is already a heap-allocated object → no additional allocation when reusing cached instance
- Avoids wrapper object optimization issues in type-unspecified paths

### API Aliases

For clarity, explicit aliases are provided:

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

---

## Comparison Matrix

| Strategy | Return Type | Cache Miss Cost | Type-Unspecified Path* | Type Stable |
|----------|-------------|-----------------|------------------------|-------------|
| **acquire! (new)** | `ReshapedArray` | **0 bytes** (no unsafe_wrap) | May allocate wrapper | **✓** |
| **unsafe_acquire!** | `Array` | 112 bytes | **0 bytes** (on cache hit) | **✓** |
| ~~Mixed (rejected)~~ | `Union{...}` | 0 bytes | Unspecified | **✗** |

*Type-unspecified path: Calls through abstract fields without concrete type parameters, etc. Compiler cannot apply escape optimization, causing wrapper object heap allocation.

### Recommended API by Situation

| Situation | acquire! | unsafe_acquire! | Recommendation |
|-----------|----------|-----------------|----------------|
| Type-specified call path | Optimizable | 0 bytes (hit) | `acquire!` |
| Variable dims (cyclic pattern) | Optimizable | cache miss occurs | `acquire!` |
| Type-unspecified path | Wrapper alloc | **0 bytes** (hit) | **`unsafe_acquire!`** |
| FFI / raw pointer | N/A | 0 bytes | `unsafe_acquire!` |

---

## Implementation Specification

### Data Structure Changes (types.jl)

#### Constants
```julia
const CACHE_WAYS = 4
```

#### TypedPool Struct Layout

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

    # --- State ---
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end
```

**Implemented Changes** (compared to initial design):
- **Removed**: `nd_views` (No longer needed as `acquire!` returns `ReshapedArray`) ✓
- **Updated**: `nd_arrays`, `nd_dims`, `nd_ptrs` store `CACHE_WAYS` items per active slot ✓
- **Added**: `nd_next_way::Vector{Int}` for Round-Robin replacement index per slot ✓

### Logic Implementation (core.jl)

#### `acquire!` (The Fast Path)

**Goal**: Always return `ReshapedArray`. No N-D cache lookup.

```julia
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    len = safe_prod(dims)
    flat_view = get_view!(tp, len)
    return reshape(flat_view, dims)
end
```

#### `unsafe_acquire!` (The N-way Cache Path)

**Goal**: Return `Array`. Use N-way cache with Linear Search + Round-Robin Replacement.

**Algorithm**:
1. Get 1D view: `flat_view = get_view!(tp, prod(dims))`
2. Get current pointer: `current_ptr = UInt(pointer(flat_view))`
3. Calculate Base Index: `base = (tp.n_active - 1) * CACHE_WAYS`
4. **Search (Hit Check)**:
   - Loop `k` from `1` to `CACHE_WAYS`
   - Check if `nd_dims[base + k] == dims` **AND** `nd_ptrs[base + k] == current_ptr`
   - If match: Return `nd_arrays[base + k]`
5. **Miss (Replacement)**:
   - Get victim way from `nd_next_way[tp.n_active]`
   - Target Index: `target = base + victim_way + 1`
   - Create Array: `arr = unsafe_wrap(Array{T, N}, pointer(flat_view), dims)`
   - **Update Cache**:
     - `nd_arrays[target] = arr`
     - `nd_dims[target] = dims`
     - `nd_ptrs[target] = current_ptr`
   - **Update Round-Robin**: Increment `nd_next_way` (modulo `CACHE_WAYS`)
   - Return `arr`

---

## Implementation Plan

### Phase 1: Simplify `acquire!` N-D Path

**File**: `src/core.jl`

**Before**:
```julia
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    arr = get_nd_array!(tp, dims)  # uses unsafe_wrap
    idx = tp.n_active
    # ... complex caching logic
    new_view = view(arr, ntuple(_ -> Colon(), Val(N))...)
    return new_view  # SubArray{T,N,Array{T,N}}
end
```

**After**:
```julia
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)  # 1D view (cached)
    return reshape(flat_view, dims)        # ReshapedArray (0 alloc!)
end
```

### Phase 2: Maintain `unsafe_acquire!` Cache

**No changes** - maintain current implementation:
- `get_nd_array!` → `unsafe_wrap` + slot-based cache
- Maintain N-way cache (4-way)
- 112 bytes allocation on cache miss

### Phase 3: TypedPool Field Updates

Update struct as specified in Implementation Specification above.

### Phase 4: Test Updates

**Files**: `test/test_nway_cache.jl`, `test/test_zero_allocation.jl`

```julia
@testset "acquire! returns ReshapedArray" begin
    pool = AdaptiveArrayPool()
    @with_pool pool begin
        m = acquire!(pool, Float64, 10, 10)
        @test m isa Base.ReshapedArray
        @test size(m) == (10, 10)
    end
end

@testset "acquire! is always zero-allocation" begin
    pool = AdaptiveArrayPool()

    # 5-way cycling (exceeds any cache) - still 0 alloc!
    function test_5way!(p)
        dims_list = ((5, 10), (10, 5), (7, 7), (3, 16), (4, 12))
        for dims in dims_list
            checkpoint!(p)
            acquire!(p, Float64, dims...)  # ReshapedArray
            rewind!(p)
        end
    end

    test_5way!(pool); test_5way!(pool)
    allocs = @allocated test_5way!(pool)
    @test allocs == 0  # Always zero, regardless of pattern!
end

@testset "unsafe_acquire! returns Array" begin
    pool = AdaptiveArrayPool()
    @with_pool pool begin
        m = unsafe_acquire!(pool, Float64, 10, 10)
        @test m isa Array
        @test size(m) == (10, 10)
    end
end
```

### Phase 5: Documentation

**CHANGELOG.md** (not a breaking change, behavior improvement):
```markdown
## [Unreleased]
### Changed
- `acquire!` N-D path now returns `ReshapedArray` instead of `SubArray{Array}`
  - Always zero-allocation, regardless of cache hit/miss
  - Simpler implementation, no N-D cache dependency
- `unsafe_acquire!` continues to return `Array` with N-way cache
  - Use this when dynamic dispatch or raw Array is needed
```

**Docstring Updates**:
```julia
"""
    acquire!(pool, Type{T}, dims...) -> ReshapedArray{T,N,...}

Acquire a view with dimensions `dims` from the pool.

Returns a `ReshapedArray` backed by pool memory. **Zero creation cost** - no
`unsafe_wrap` call needed. Compiler may optimize away heap allocation via
SROA/escape analysis in type-specified paths.

For type-unspecified paths (struct fields without concrete type parameters),
use [`unsafe_acquire!`](@ref) instead - cached Array instances can be reused.

## Example
```julia
@with_pool pool begin
    m = acquire!(pool, Float64, 64, 100)  # ReshapedArray
    m .= 1.0
    result = sum(m)
end
```
"""
```

```julia
"""
    unsafe_acquire!(pool, Type{T}, dims...) -> Array{T,N}

Acquire a raw `Array` backed by pool memory.

Returns an `Array` object. Since Array is already heap-allocated, the cached
instance can be reused without wrapper allocation overhead.

## When to use
- Type-unspecified paths (e.g., struct fields without concrete type parameters)
- FFI / ccall requiring raw pointers
- APIs that strictly require `Array` type

## Allocation behavior
- Cache hit: 0 bytes (cached Array instance reused)
- Cache miss: 112 bytes (Array header creation)

## Example
```julia
@with_pool pool begin
    m = unsafe_acquire!(pool, Float64, 64, 100)  # Matrix{Float64}
    # Safe for type-unspecified paths
    some_abstract_field.process(m)  # 0 bytes - cached instance reused
end
```
"""
```

---

## Verification Checklist

1. **Type Check**: `acquire!` must return `ReshapedArray`. `unsafe_acquire!` must return `Array`.
2. **Allocation Check**:
   - `acquire!`: 0 allocations always
   - `unsafe_acquire!`: 0 allocations on cache hit
   - `unsafe_acquire!`: 0 allocations on interleaved access (e.g., alternating 10x10 and 20x20) thanks to N-way cache
3. **Safety**: Ensure `unsafe_acquire!` validates pointers (re-wraps if the backing vector was resized)

---

## TurbulentTransport Integration

### Changed File: `src/tglf_nn.jl`

**Already Applied** (line 277):
```julia
@with_pool pool function flux_array!(out_y::AbstractMatrix{T}, fluxmodel::TGLFNNmodel, x::AbstractMatrix{T}; ...) where {T<:Real}
    # ...
    # NOTE: Use unsafe_acquire! (returns Array) instead of acquire! (returns ReshapedArray)
    # because _pooled_chain field lacks concrete type parameters, causing
    # escape optimization failure. Array (cached instance) avoids wrapper allocation.
    xx = unsafe_acquire!(pool, T, size(x))
    # ...
    fluxmodel._pooled_chain(out_y, xx)  # 0 bytes - cached Array instance reused
end
```

### No Change Needed: `src/pooled_layers.jl`

`PooledDense`, `PooledActivation` are in **static dispatch** environment:
- Types are known at compile time
- Maintain use of `acquire!` (ReshapedArray)
- ReshapedArray is also 0 bytes in static dispatch

```julia
@inline function _pooled_dense_forward!(pd::PooledDense, x::AbstractVecOrMat)
    pool = get_task_local_pool()
    # acquire! usage OK - static dispatch environment
    out = acquire!(pool, Float64, size(d.weight, 1), size(xT, 2))
    mul!(out, d.weight, xT)  # ReshapedArray is StridedArray ✓
    return Flux.NNlib.bias_act!(d.σ, out, d.bias)
end
```

---

## Summary

### Before (v0.2.0)

```
┌─────────────────────────────────────────────────────────────┐
│  acquire!() ──┬──> get_nd_view!() ──> get_nd_array!()       │
│               │         │                  │                │
│               │         │           unsafe_wrap (112B miss) │
│               │         │                  ↓                │
│               │         └──────> SubArray{Array} ←──────────┘
│               │                                             │
│  unsafe_acquire!() ──> get_nd_array!() ──> Array            │
│                               │                             │
│                        unsafe_wrap (112B miss)              │
└─────────────────────────────────────────────────────────────┘
```

### After (Hybrid)

```
┌─────────────────────────────────────────────────────────────┐
│  acquire!() ──> get_view!() ──> reshape() ──> ReshapedArray │
│                      │              │                       │
│                  1D cache      0 bytes always!              │
│                  (0 alloc)                                  │
│                                                             │
│  unsafe_acquire!() ──> get_nd_array!() ──> Array            │
│                               │                             │
│                        unsafe_wrap + N-way cache            │
│                        (0B hit, 112B miss)                  │
└─────────────────────────────────────────────────────────────┘
```

### Decision Matrix for Users

```
┌─────────────────────────────────────────────────────────────┐
│                    Which API to use?                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Is the code path type-unspecified?                         │
│  (abstract fields without concrete type params,             │
│   runtime-determined function calls)                        │
│                                                             │
│     YES ──────────────> unsafe_acquire!()                   │
│      │                       │                              │
│      │                  Returns Array                       │
│      │                  (cached instance reused)            │
│      │                                                      │
│     NO ───────────────> acquire!()                          │
│                              │                              │
│                         Returns ReshapedArray               │
│                         (0 bytes creation)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Resolved Questions

1. **N-way cache retention level**: Configurable via `CACHE_WAYS` preference (default: 4-way). ✅
2. **nd_views field removal**: Removed. `acquire!` now returns `ReshapedArray` via `reshape()`. ✅
3. **Backward compatibility**: `acquire!` returns `ReshapedArray` (a type of `AbstractArray`), maintaining API compatibility. ✅

---

## References

- [nd_array_approach_comparison.md](./nd_array_approach_comparison.md) - Benchmark results and boxing analysis
