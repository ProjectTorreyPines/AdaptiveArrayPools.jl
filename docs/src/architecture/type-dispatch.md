# Type Dispatch & Caching

This page explains the internal mechanisms that enable zero-allocation performance.

## Fixed-Slot Type Dispatch

To achieve zero-lookup overhead, common types have dedicated struct fields:

```julia
struct AdaptiveArrayPool
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    complexf32::TypedPool{ComplexF32}
    bool::TypedPool{Bool}
    others::IdDict{DataType, Any}  # Fallback for rare types
end
```

When you call `acquire!(pool, Float64, n)`, the compiler inlines directly to `pool.float64` - no dictionary lookup, no type instability.

## N-D Wrapper Caching for `acquire!`

`acquire!` returns native `Array` types. The caching strategy depends on Julia version:

### Julia 1.11+: `setfield!`-based Wrapper Reuse

Julia 1.11 made `Array` a mutable struct, enabling in-place field mutation:

```julia
# acquire! wrapper reuse via setfield! (0-alloc)
setfield!(cached_arr, :ref, new_memory_ref)   # update backing memory
setfield!(cached_arr, :size, new_dims)         # update dimensions
```

Wrappers are stored in `nd_wrappers::Vector{Union{Nothing, Vector{Any}}}`, indexed directly by dimensionality N (~1ns lookup). `acquire!` uses these wrappers to return native `Array{T,N}` with **unlimited dimension patterns per slot, zero allocation after warmup.**

### Julia 1.10 / CUDA: N-Way Set Associative Cache

On Julia 1.10 (CPU) and CUDA, `Array`/`CuArray` fields cannot be mutated. These paths use a 4-way set-associative cache with round-robin eviction (`CACHE_WAYS = 4` default):

- **Cache hit** (≤4 dim patterns per slot): 0 bytes
- **Cache miss** (>4 patterns): ~80-144 bytes for Array header allocation

See [Configuration](../features/configuration.md) for `CACHE_WAYS` tuning.

---

## Array vs View: When to Use What?

| API | Return Type | Allocation (Julia 1.11+) | Allocation (1.10 / CUDA) | Recommended For |
|-----|-------------|--------------------------|--------------------------|-----------------|
| `acquire!` | `Vector{T}` / `Array{T,N}` | **0 bytes** (setfield! reuse) | 0-144 bytes (N-way cache) | 99% of cases |
| `acquire_view!` | `SubArray` / `ReshapedArray` | **Always 0 bytes** | **Always 0 bytes** | Lightweight view semantics |

### Why Array is the Default

1. **FFI/ccall compatible**: Native `Array` types provide `Ptr{T}` for C interop without conversion
2. **Zero-allocation on Julia 1.11+**: `setfield!`-based wrapper reuse achieves 0 bytes after warmup
3. **BLAS/LAPACK compatible**: `Array` is `StridedArray`, full compatibility with linear algebra routines

!!! note "`Bit` masks"
    For `T === Bit`, both APIs return native `BitVector`/`BitArray{N}` (not views) to preserve BitArray-specialized kernels (`count`, `any`, `all`, bitwise ops). Cache hit achieves 0 bytes allocation. These are not `StridedArray`.

### When to Use acquire_view!

1. **Lightweight view semantics**: When you prefer views and don't need a concrete `Array`

```julia
v = acquire_view!(pool, Float64, 100)
# v is a SubArray — always 0-alloc, compiler eliminates via SROA
```

2. **Guaranteed zero allocation**: Views are always 0-alloc regardless of Julia version

```julia
# Even on Julia 1.10 / CUDA, views never allocate
m = acquire_view!(pool, Float64, 10, 10)
# m is a ReshapedArray — 0 bytes guaranteed
```

3. **Compiler-friendly hot loops**: SROA (Scalar Replacement of Aggregates) eliminates view wrappers entirely

```julia
function inner_loop(pool)
    buf = acquire_view!(pool, Float64, 64)  # Compiler may eliminate wrapper
    # ... tight computation on buf
end
```

### Performance Comparison

| Operation | acquire! (Array) | acquire_view! (View) |
|-----------|------------------|----------------------|
| Allocation (Julia 1.11+) | 0 bytes (setfield! reuse) | 0 bytes |
| Allocation (Julia 1.10 / CUDA) | 0 bytes (hit) / 80-144 bytes (miss) | 0 bytes |
| BLAS operations | Identical | Identical |
| Type stability | Guaranteed | Guaranteed |
| FFI compatibility | Direct | Requires conversion |

### Header Size by Dimensionality (Julia 1.10 / CUDA only)

On Julia 1.11+ CPU, `acquire!` is always zero-allocation via `setfield!` reuse. On Julia 1.10 and CUDA, a cache miss allocates an `Array` header:

| Dimensions | Header Size |
|------------|-------------|
| 1D (Vector) | 80 bytes |
| 2D-3D | 112 bytes |
| 4D-5D | 144 bytes |

This is Julia's internal `Array` metadata; actual data memory is always reused from the pool.

---

## See Also

- [How It Works](how-it-works.md) - Checkpoint/Rewind mechanism
- [Design Documents](design-docs.md) - Detailed design analysis
- [Configuration](../features/configuration.md) - Cache tuning options
