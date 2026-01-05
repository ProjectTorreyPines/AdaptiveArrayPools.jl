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

## N-Way Set Associative Cache

For `unsafe_acquire!` (which returns native `Array` types), we use an N-way cache to reduce header allocation:

```
                    CACHE_WAYS = 4 (default)
                    +----+----+----+----+
Slot 0 (Float64):   |way0|way1|way2|way3|  <-- round-robin eviction
                    +----+----+----+----+
                    +----+----+----+----+
Slot 1 (Float32):   |way0|way1|way2|way3|
                    +----+----+----+----+
                    ...
```

### Cache Lookup Logic

```julia
function unsafe_acquire!(pool, T, dims...)
    typed_pool = get_typed_pool!(pool, T)
    slot = n_active + 1
    base = (slot - 1) * CACHE_WAYS

    # Search all ways for matching dimensions
    for k in 1:CACHE_WAYS
        idx = base + k
        if dims == typed_pool.nd_dims[idx]
            # Cache hit! Check if underlying vector was resized
            if pointer matches
                return typed_pool.nd_arrays[idx]
            end
        end
    end

    # Cache miss: create new Array header, store in next way (round-robin)
    way = typed_pool.nd_next_way[slot]
    typed_pool.nd_next_way[slot] = (way % CACHE_WAYS) + 1
    # ... create and cache Array ...
end
```

**Key insight**: Even on cache miss, only the `Array` header (~80-144 bytes) is allocated. The actual data memory is always reused from the pool.

---

## View vs Array: When to Use What?

| API | Return Type | Allocation | Recommended For |
|-----|-------------|------------|-----------------|
| `acquire!` | `SubArray` / `ReshapedArray` | **Always 0 bytes** | 99% of cases |
| `unsafe_acquire!` | `Vector` / `Array` | 0-144 bytes | FFI, type constraints |

### Why View is the Default

1. **Zero-allocation guarantee**: Compiler eliminates view wrappers via SROA (Scalar Replacement of Aggregates)
2. **BLAS/LAPACK compatible**: Processed as `StridedArray`, no performance difference
3. **Type stable**: Always returns the same wrapper types

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
| Allocation (cached) | 0 bytes | 0 bytes |
| Allocation (miss) | 0 bytes | 80-144 bytes |
| BLAS operations | Identical | Identical |
| Type stability | Guaranteed | Guaranteed |
| FFI compatibility | Requires conversion | Direct |

### Header Size by Dimensionality

When `unsafe_acquire!` has a cache miss:

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
