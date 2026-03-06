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

## N-D Wrapper Caching for `unsafe_acquire!`

`unsafe_acquire!` returns native `Array` types. The caching strategy depends on Julia version:

### Julia 1.11+: `setfield!`-based Wrapper Reuse

Julia 1.11 made `Array` a mutable struct, enabling in-place field mutation:

```julia
# Cached wrapper reuse via setfield! (0-alloc)
setfield!(cached_arr, :ref, new_memory_ref)   # update backing memory
setfield!(cached_arr, :size, new_dims)         # update dimensions
```

Wrappers are stored in `nd_wrappers::Vector{Union{Nothing, Vector{Any}}}`, indexed directly by dimensionality N (~1ns lookup). **Unlimited dimension patterns per slot, zero allocation after warmup.**

### Julia 1.10 / CUDA: N-Way Set Associative Cache

On Julia 1.10 (CPU) and CUDA, `Array`/`CuArray` fields cannot be mutated. These paths use a 4-way set-associative cache with round-robin eviction (`CACHE_WAYS = 4` default):

- **Cache hit** (≤4 dim patterns per slot): 0 bytes
- **Cache miss** (>4 patterns): ~80-144 bytes for Array header allocation

See [Configuration](../features/configuration.md) for `CACHE_WAYS` tuning.

---

## View vs Array: When to Use What?

| API | Return Type | Allocation (Julia 1.11+) | Allocation (1.10 / CUDA) | Recommended For |
|-----|-------------|--------------------------|--------------------------|-----------------|
| `acquire!` | `SubArray` / `ReshapedArray` | **Always 0 bytes** | **Always 0 bytes** | 99% of cases |
| `unsafe_acquire!` | `Vector` / `Array` | **0 bytes** (setfield! reuse) | 0-144 bytes (N-way cache) | FFI, type constraints |

### Why View is the Default

1. **Zero-allocation guarantee**: Compiler eliminates view wrappers via SROA (Scalar Replacement of Aggregates)
2. **BLAS/LAPACK compatible**: Processed as `StridedArray`, no performance difference
3. **Type stable**: Always returns the same wrapper types

!!! note "`Bit` masks"
    For `T === Bit`, both APIs return native `BitVector`/`BitArray{N}` (not views) to preserve BitArray-specialized kernels (`count`, `any`, `all`, bitwise ops). Cache hit achieves 0 bytes allocation. These are not `StridedArray`.

### When to Use unsafe_acquire!

1. **C FFI**: When `ccall` requires `Ptr{T}` from contiguous memory

```julia
arr = unsafe_acquire!(pool, Float64, 100)
ccall(:c_function, Cvoid, (Ptr{Float64}, Cint), arr, 100)
```

2. **Type signature constraints**: Function explicitly requires `Array{T,N}`

```julia
function process(data::Array{Float64,2})
    # Only accepts Array, not AbstractArray
end

m = unsafe_acquire!(pool, Float64, 10, 10)
process(m)  # Works
```

3. **Runtime dispatch avoidance**: When types are determined at runtime

```julia
# Polymorphic code where type stability matters
function dispatch_heavy(pool, T)
    arr = unsafe_acquire!(pool, T, 100)  # Concrete Array type
    # ... operations that would trigger dispatch with views
end
```

### Performance Comparison

| Operation | acquire! (View) | unsafe_acquire! (Array) |
|-----------|-----------------|-------------------------|
| Allocation (Julia 1.11+) | 0 bytes | 0 bytes (setfield! reuse) |
| Allocation (Julia 1.10 / CUDA) | 0 bytes | 0 bytes (hit) / 80-144 bytes (miss) |
| BLAS operations | Identical | Identical |
| Type stability | Guaranteed | Guaranteed |
| FFI compatibility | Requires conversion | Direct |

### Header Size by Dimensionality (Julia 1.10 / CUDA only)

On Julia 1.11+ CPU, `unsafe_acquire!` is always zero-allocation via `setfield!` reuse. On Julia 1.10 and CUDA, a cache miss allocates an `Array` header:

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
