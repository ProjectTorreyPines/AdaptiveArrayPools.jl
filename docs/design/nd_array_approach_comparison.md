# N-D Array Approach Comparison: unsafe_wrap vs ReshapedArray

## Summary

This document analyzes two approaches for returning N-dimensional arrays from AdaptiveArrayPools:

1. **Current (v1.1.x)**: `unsafe_wrap(Array, pointer, dims)` with N-way cache
2. **Proposed (v1.0.2 style)**: `reshape(view(backing, 1:n), dims)` without cache

**Recommendation**: Switch back to ReshapedArray approach for `acquire!` N-D path.

---

## Key Finding: SubArray Wrapper Allocation

### The 48-byte Problem

When using `acquire!` vs `unsafe_acquire!` in real code:

```julia
# In flux_array! (TurbulentTransport)
xx = unsafe_acquire!(pool, T, size(x))  # 0 bytes - returns Array directly
xx = acquire!(pool, T, size(x))         # 48 bytes - SubArray wrapper!
```

**Root Cause**: `acquire!` returns `SubArray`, which allocates its wrapper struct (48 bytes) when it escapes the function scope.

### Allocation Breakdown

| API | Return Type | Allocation |
|-----|-------------|------------|
| `unsafe_acquire!(pool, T, m, n)` | `Matrix{T}` | 0 bytes (cache hit), 112 bytes (miss) |
| `acquire!(pool, T, m, n)` | `SubArray{..., Matrix}` | **48 bytes** (wrapper) + 112 bytes (miss) |

### Why SubArray Allocates

**Fundamental difference:**

```
unsafe_acquire! returns:
┌─────────────────────────────────────────────────────┐
│  Pool backing Vector                                │
│  [████████████████████████████████████]             │
│                ↓                                     │
│  Array header lives in pool cache (reused)          │
│  → Returns pointer to EXISTING object (0 alloc)     │
└─────────────────────────────────────────────────────┘

acquire! returns:
┌─────────────────────────────────────────────────────┐
│  Pool backing Vector                                │
│  [████████████████████████████████████]             │
│                ↓                                     │
│  Array in cache                                     │
│                ↓                                     │
│  NEW SubArray struct (parent, indices, stride...)   │
│  → Creates NEW wrapper object (48 bytes!)           │
└─────────────────────────────────────────────────────┘
```

SubArray is stack-allocated **only when**:
1. Used entirely within a single function
2. Compiler can prove it doesn't escape

In `flux_array!`, `xx` escapes because:
- Passed to `_pooled_chain(out_y, xx)`
- Used across multiple loop iterations
- Compiler can't optimize away the wrapper

### Root Cause: Type-Unspecified Path → Escape Optimization Failure

**Critical Finding**: The core reason for allocation is that **compiler escape optimization fails in type-unspecified call paths**.

> **Correction**: SubArray is a mutable struct (not immutable/isbits).
> The explanation that "dynamic dispatch causes boxing" is inaccurate.
> Precisely: wrapper objects are created at construction time, and the key factor is
> whether the compiler can optimize to stack allocation through escape analysis.

#### Why it happens in `flux_array!`:

```julia
# TGLFNNmodel struct (tglf_nn.jl)
struct TGLFNNmodel <: TGLFmodel
    fluxmodel::Flux.Chain
    # ...
    _pooled_chain::PooledChain  # ← No concrete type parameter!
end
```

The `_pooled_chain` field is declared without concrete type parameters,
so the call `fluxmodel._pooled_chain(out_y, xx)` is **not recompiled** and wrapper object optimization is not applied.

#### Escape Optimization Failure:

| Condition | Compiler Behavior | Result |
|-----------|-------------------|--------|
| Type specified + no escape | SROA/escape analysis applied | Stack allocation or elimination possible |
| Type specified + escape | Partial optimization possible | Depends on situation |
| **Type unspecified** | Optimization not applicable | **Wrapper object heap allocation** |

```
Type-specified path:
┌─────────────────────────────────────────────────────────────────┐
│  Compiler knows the type                                        │
│  → SROA/escape analysis can be applied                          │
│  → Wrapper object can be stack-allocated or completely removed  │
└─────────────────────────────────────────────────────────────────┘

Type-unspecified path (e.g., call through abstract field):
┌─────────────────────────────────────────────────────────────────┐
│  Compiler doesn't know concrete type                            │
│  → Escape analysis cannot be applied                            │
│  → Wrapper object is heap-allocated                             │
└─────────────────────────────────────────────────────────────────┘
```

#### Array vs View Types in Type-Unspecified Paths:

| Type | Characteristic | In Type-Unspecified Path |
|------|----------------|--------------------------|
| `Array` | Already heap-allocated object | Cached instance reuse → **No additional allocation** |
| `SubArray` | Requires wrapper object | Escape optimization failure → **Wrapper allocation** |
| `ReshapedArray` | Requires wrapper object | Escape optimization failure → **Wrapper allocation** |

**Key insight**: Array is an object that already exists on the heap, so returning the same instance from cache incurs no additional allocation.
In contrast, SubArray/ReshapedArray create new wrapper objects each time, and optimization is not applied when type is unspecified.

### Solutions for the Wrapper Allocation Problem

#### 1. Use `unsafe_acquire!` (Recommended for this case)

`unsafe_acquire!` returns `Array`, which is already a heap-allocated object (cached instance can be reused):

```julia
# flux_array! in tglf_nn.jl
xx = unsafe_acquire!(pool, T, size(x))  # Returns Matrix{T} → cache hit = 0 alloc
```

✅ Zero allocation on cache hit (cached Array instance reused)
✅ No code changes to TGLFNNmodel needed
✅ Safe since `xx` is only used as scratch memory

#### 2. Parameterize TGLFNNmodel (Fundamental fix)

```julia
struct TGLFNNmodel{M<:Flux.Chain, P<:PooledChain} <: TGLFmodel
    fluxmodel::M
    _pooled_chain::P  # Now compiler knows exact type
end
```

✅ Enables compiler escape optimization
✅ SubArray/ReshapedArray may become zero-alloc (depends on SROA/escape analysis)
❌ Requires significant code changes
❌ Changes serialization behavior

#### 3. Function Barrier

```julia
# Force type specialization
@inline function _call_pooled_chain(chain::PooledChain{M}, out, x) where M
    chain(out, x)
end
```

⚠️ May not help if extracting `_pooled_chain` from struct is also dynamic

### Implications for API Design

**Key Insight: Wrapper Types Depend on Compiler Optimization**

> **Note**: Whether wrapper types (SubArray, ReshapedArray) allocate depends on compiler SROA/escape analysis.
> The comparison below assumes **type-specified paths**. In type-unspecified paths, all wrapper types allocate.

| Approach | Return Type | Creation Cost | Type-Specified Path | Type-Unspecified Path |
|----------|-------------|---------------|---------------------|----------------------|
| `unsafe_acquire!` | `Array` | 112 bytes (miss) | cache hit: 0 | cache hit: 0 |
| `acquire!` (current) | `SubArray{Array}` | 112 bytes (miss) | Optimizable | Wrapper allocation |
| **`acquire!` (reshape)** | `ReshapedArray{View}` | **0 bytes** | **Optimizable** | Wrapper allocation |

**Advantages of ReshapedArray approach**:

1. ✅ No creation cost (no unsafe_wrap call)
2. ✅ Compiler optimization possible in type-specified paths
3. ✅ BLAS compatible (StridedArray)
4. ✅ Same operation performance (mul!, broadcast)

**`unsafe_acquire!` is better in type-unspecified paths**:
- Array already exists on heap → no additional allocation when reusing cached instance

---

## Benchmark Results

### Test Environment
- Benchmark file: `benchmark/nd_approach_comparison.jl`
- Tests 8-10 specifically compare N-way cache behavior

### Allocation Comparison

| Scenario | unsafe_wrap | reshape | Savings |
|----------|-------------|---------|---------|
| Single call (cache miss) | 112 bytes | **0 bytes** | 100% |
| 3-way cycling × 100 | 33,600 bytes | **0 bytes** | 100% |
| 5-way cycling × 100 | 56,000 bytes | **0 bytes** | 100% |
| With 4-way cache (3-way pattern) | 0 bytes | **0 bytes** | - |
| With 4-way cache (5-way pattern) | 56,000 bytes | **0 bytes** | 100% |

### Performance Comparison

| Operation | unsafe_wrap | reshape | Winner |
|-----------|-------------|---------|--------|
| mul! (BLAS) | 9.92 μs | 9.96 μs | Tie |
| Broadcast σ.(x) | 24.2 μs | 23.5 μs | Tie |
| Dense layer | 35.3 μs | 33.7 μs | Tie |
| 3-way cycling | 5.87 μs | **0.38 μs** | reshape (15x) |
| 5-way cycling | 10.38 μs | **0.59 μs** | reshape (18x) |

### Type Information

All three types are `StridedArray` (BLAS compatible):

```julia
# unsafe_wrap
Matrix{Float64}  # isa StridedArray ✓

# SubArray of unsafe_wrap
SubArray{Float64, 2, Matrix{Float64}, ...}  # isa StridedArray ✓

# ReshapedArray of 1D view
Base.ReshapedArray{Float64, 2, SubArray{Float64, 1, Vector{Float64}, ...}, ...}  # isa StridedArray ✓
```

### Memory Layout

All have identical column-major layout:
```
Strides: (1, 4)  # Same for all three
```

---

## Current Implementation (unsafe_wrap)

### Call Path
```
acquire!(pool, Float64, 64, 100)
  └─> get_nd_view!(tp, (64, 100))
        └─> get_nd_array!(tp, (64, 100))
              ├─> get_view!(tp, 6400)      # 1D view (0 alloc, cached)
              └─> unsafe_wrap(...)          # 112 bytes on cache miss!
```

### Code Location
`src/core.jl:129`:
```julia
arr = unsafe_wrap(Array{T, N}, pointer(flat_view), dims)
```

### N-way Cache Structure
```julia
# In TypedPool (src/types.jl)
nd_views::Vector{Any}      # Cached SubArray objects
nd_arrays::Vector{Any}     # Cached Array objects (from unsafe_wrap)
nd_dims::Vector{Any}       # Cached dimension tuples
nd_ptrs::Vector{UInt}      # Cached pointers for invalidation
```

---

## Proposed Implementation (ReshapedArray)

### Call Path
```
acquire!(pool, Float64, 64, 100)
  └─> get_nd_view!(tp, (64, 100))
        └─> get_view!(tp, 6400)      # 1D view (0 alloc, cached)
              └─> reshape(view, dims)  # 0 alloc always!
```

### Proposed Code Change
```julia
# Replace get_nd_view! in src/core.jl
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)  # 1D view (cached)
    return reshape(flat_view, dims)        # Zero-alloc ReshapedArray
end
```

---

## Pros and Cons

### ReshapedArray Approach (Proposed)

#### Pros
1. **Zero allocation always** - No 112-byte allocation regardless of cache hit/miss
2. **No N-D cache needed** - Simpler code, less memory overhead
3. **Faster** - No cache lookup overhead (0.38μs vs 5.87μs for cycling patterns)
4. **BLAS compatible** - ReshapedArray is StridedArray
5. **Same performance** - Identical mul!/broadcast speed
6. **No Bélády's Anomaly** - Works with any access pattern (5-way, 10-way, etc.)
7. **Simpler TypedPool** - Can remove nd_arrays, nd_dims, nd_ptrs fields

#### Cons
1. **Return type changes** - `SubArray{..., Array{...}}` → `ReshapedArray{..., SubArray{...}}`
2. **Some libraries might check `isa Array`** - Rare, but possible (not BLAS though)
3. **Slightly different printing** - Display shows as ReshapedArray

### unsafe_wrap Approach (Current)

#### Pros
1. **Returns actual Array** - Some code might expect `Matrix{Float64}`
2. **Cache hits are zero-alloc** - When pattern fits in N-way cache

#### Cons
1. **112 bytes per cache miss** - Adds up with varying batch sizes
2. **N-way cache complexity** - Extra fields, cache lookup logic
3. **Bélády's Anomaly** - 5+ patterns = 100% miss with 4-way cache
4. **Slower cycling** - Cache lookup overhead even on hits

---

## Impact Analysis

### TurbulentTransport Usage

In `src/pooled_layers.jl`:

```julia
# PooledDense (line 86)
out = acquire!(pool, Float64, size(d.weight, 1), size(xT, 2))
mul!(out, d.weight, xT)  # Works with ReshapedArray ✓

# PooledActivation (line 54)
out = acquire!(pool, Float64, size(x))
out .= pa.σ.(x)  # Works with ReshapedArray ✓
```

Both use cases are compatible with ReshapedArray:
- `mul!` accepts any `StridedMatrix`
- Broadcasting works on any `AbstractArray`

### Flux.NNlib.bias_act!

```julia
Flux.NNlib.bias_act!(d.σ, out, d.bias)
```

This function accepts `AbstractArray` - ReshapedArray is compatible.

### unsafe_acquire! Unchanged

For code that explicitly needs raw `Array` (FFI, specific BLAS paths):
```julia
unsafe_acquire!(pool, Float64, 64, 100)  # Still returns Matrix{Float64}
```

This API remains unchanged and still uses `unsafe_wrap` with caching.

---

## Migration Path

### Phase 1: Modify acquire! N-D path
```julia
# src/core.jl - Replace get_nd_view!
@inline function get_nd_view!(tp::TypedPool{T}, dims::NTuple{N, Int}) where {T, N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)
    return reshape(flat_view, dims)
end
```

### Phase 2: Simplify TypedPool (optional)
Remove N-D cache fields if `unsafe_acquire!` usage is rare:
- `nd_views`, `nd_arrays`, `nd_dims`, `nd_ptrs`

### Phase 3: Update documentation
- Note return type change in CHANGELOG
- Update docstrings for `acquire!`

---

## Conclusion

### Two Separate Problems, Two Solutions

This investigation revealed **two distinct allocation issues**:

#### Problem 1: N-D Array Creation (unsafe_wrap vs reshape)

| Metric | unsafe_wrap + cache | reshape |
|--------|---------------------|---------|
| Allocation (miss) | 112 bytes | **0 bytes** |
| Allocation (hit) | 0 bytes | **0 bytes** |
| Speed (cycling) | 5-10 μs | **0.3-0.6 μs** |
| BLAS compat | ✓ | ✓ |
| Code complexity | High (cache) | **Low** |
| Works with any pattern | ✗ (≤4 ways) | **✓ (any)** |

**Solution**: Switch `acquire!` N-D path to use `reshape(view, dims)` instead of `unsafe_wrap`.

#### Problem 2: Type-Unspecified Path Wrapper Allocation

> **Correction**: SubArray and ReshapedArray are both **mutable structs**.
> Whether allocation occurs depends on whether compiler escape optimization can be applied.

| Type | Type-Specified Path | Type-Unspecified Path |
|------|---------------------|----------------------|
| `Array` | cache hit: 0 bytes | cache hit: **0 bytes** ✓ |
| `SubArray` | Optimizable | **Wrapper allocation** ✗ |
| `ReshapedArray` | Optimizable | **Wrapper allocation** ✗ |

**Key difference**:
- `Array`: Object already exists on heap → no additional allocation when reusing cached instance
- Wrapper types: New object created each time → heap allocation when optimization fails in type-unspecified paths

**Solution for TurbulentTransport**: Use `unsafe_acquire!` which returns `Array` (cached instance reusable).

### Summary

| Context | Recommended API | Reason |
|---------|-----------------|--------|
| Type-specified path (general use) | `acquire!` → ReshapedArray | 0 bytes creation, compiler optimization possible |
| Type-unspecified path | `unsafe_acquire!` → Array | Cached instance can be reused |
| FFI / raw pointer needs | `unsafe_acquire!` → Array | Direct memory access |

**The N-way cache was solving the wrong problem** - caching `Array` objects when `reshape` is already zero-cost for creation.

For type-unspecified paths (like `TGLFNNmodel._pooled_chain` without concrete type parameters), `unsafe_acquire!` returning `Array` is the correct choice because cached Array instances can be reused without additional allocation.
