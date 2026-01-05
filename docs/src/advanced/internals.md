# Internals

This page provides an overview of the internal architecture of AdaptiveArrayPools.jl. For detailed design documents, see the [`docs/design/`](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/tree/master/docs/design) folder in the repository.

## Checkpoint/Rewind Lifecycle

The core mechanism that enables zero-allocation reuse:

```
@with_pool pool function foo()
    │
    ├─► checkpoint!(pool)     # Save current state (n_active counters)
    │
    │   A = acquire!(pool, ...)  # n_active += 1
    │   B = acquire!(pool, ...)  # n_active += 1
    │   C = acquire!(pool, ...)  # n_active += 1
    │   ... compute ...
    │
    └─► rewind!(pool)         # Restore n_active → all arrays recycled
end
```

On repeated calls, the same memory is reused without any allocation.

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

When you call `acquire!(pool, Float64, n)`, the compiler inlines directly to `pool.float64` — no dictionary lookup, no type instability.

## N-Way Set Associative Cache

For `unsafe_acquire!` (which returns native `Array` types), we use an N-way cache to reduce header allocation:

```
                    CACHE_WAYS = 4 (default)
                    ┌────┬────┬────┬────┐
Slot 0 (Float64):   │way0│way1│way2│way3│  ← round-robin eviction
                    └────┴────┴────┴────┘
                    ┌────┬────┬────┬────┐
Slot 1 (Float32):   │way0│way1│way2│way3│
                    └────┴────┴────┴────┘
                    ...
```

### Cache Lookup Pseudocode

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
    typed_pool.nd_next_way[slot] = (way + 1) % CACHE_WAYS
    # ... create and cache Array ...
end
```

**Key insight**: Even on cache miss, only the `Array` header (~80-144 bytes) is allocated. The actual data memory is always reused from the pool.

## View vs Array Return Types

Type stability is critical for performance. AdaptiveArrayPools provides two APIs:

| API | 1D Return | N-D Return | Allocation |
|-----|-----------|------------|------------|
| `acquire!` | `SubArray{T,1}` | `ReshapedArray{T,N}` | Always 0 bytes |
| `unsafe_acquire!` | `Vector{T}` | `Array{T,N}` | 0 bytes (hit) / ~100 bytes (miss) |

### Why Two APIs?

**`acquire!` (views)** — The compiler can eliminate view wrappers entirely through SROA (Scalar Replacement of Aggregates) and escape analysis. This is why 1D `SubArray` and N-D `ReshapedArray` achieve true zero allocation.

**`unsafe_acquire!` (arrays)** — Sometimes you need a concrete `Array` type:
- FFI/C interop requiring `Ptr{T}` from contiguous memory
- Type signatures that explicitly require `Array{T,N}`
- Avoiding runtime dispatch in polymorphic code

## Typed Checkpoint/Rewind Optimization

When the `@with_pool` macro can statically determine which types are used, it generates optimized code:

```julia
# If only Float64 is used in the block:
checkpoint!(pool, Float64)  # ~77% faster than full checkpoint
# ... compute ...
rewind!(pool, Float64)
```

This avoids iterating over all type slots and the `others` IdDict.

## 1-Based Sentinel Pattern

Internal state vectors use a sentinel at index 0 to eliminate `isempty()` checks:

```julia
_checkpoint_n_active = [0]  # Sentinel at depth=0
_checkpoint_depths = [0]    # Global scope marker
```

This pattern reduces branching in hot paths where every nanosecond counts.

## Further Reading

For detailed design documents:

- [`hybrid_api_design.md`](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/hybrid_api_design.md) — Two-API strategy (`acquire!` vs `unsafe_acquire!`) and type stability analysis
- [`nd_array_approach_comparison.md`](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/nd_array_approach_comparison.md) — N-way cache design, boxing analysis, and ReshapedArray benchmarks
- [`untracked_acquire_design.md`](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/untracked_acquire_design.md) — Macro-based untracked acquire detection and 1-based sentinel pattern
- [`fixed_slots_codegen_design.md`](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/fixed_slots_codegen_design.md) — Zero-allocation iteration via `@generated` functions
- [`cuda_extension_design.md`](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/cuda_extension_design.md) — CUDA backend architecture and extension loading
