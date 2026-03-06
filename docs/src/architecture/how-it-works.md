# How It Works

This page explains the core mechanisms that enable zero-allocation array reuse.

## The Zero-Allocation Promise

```
+-------------------------------------------------------------+
|  Call 1 (warmup):                                           |
|    checkpoint! --> acquire! x 3 --> rewind!                 |
|         |                                                   |
|         +-- backing memory allocated                        |
|                                                             |
|  Call 2+ (zero-alloc):                                      |
|    checkpoint! --> acquire! x 3 --> rewind!                 |
|         |                                                   |
|         +-- same memory reused, 0 bytes allocated           |
+-------------------------------------------------------------+
```

## Checkpoint/Rewind Lifecycle

The core mechanism that enables memory reuse:

```
@with_pool pool function foo()
    |
    +---> checkpoint!(pool)     # Save current state (n_active counters)
    |
    |     A = acquire!(pool, ...)  # n_active += 1
    |     B = acquire!(pool, ...)  # n_active += 1
    |     C = acquire!(pool, ...)  # n_active += 1
    |     ... compute ...
    |
    +---> rewind!(pool)         # Restore n_active, arrays available for reuse
end
```

On repeated calls, the same memory is reused without any allocation.

## Exception Safety: try...finally

The `@with_pool` macro generates code with exception-safe cleanup:

```julia
# What you write:
@with_pool pool begin
    A = acquire!(pool, Float64, 100)
    result = compute(A)
end

# What the macro generates:
let pool = get_task_local_pool()
    checkpoint!(pool)
    try
        A = acquire!(pool, Float64, 100)
        result = compute(A)
    finally
        rewind!(pool)  # Always executes, even on exception
    end
end
```

**Key guarantee**: The `finally` block ensures `rewind!` is called even if an exception occurs, preventing memory leaks and state corruption.

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

## N-D Wrapper Reuse (CPU)

For `unsafe_acquire!` (which returns native `Array` types), the caching strategy depends on the Julia version:

### Julia 1.11+: `setfield!`-based Wrapper Reuse (Zero-Allocation)

Julia 1.11 changed `Array` from an opaque C struct to a mutable Julia struct with `ref::MemoryRef{T}` and `size::NTuple{N,Int}` fields. This enables in-place mutation of cached `Array` wrappers via `setfield!`:

```
nd_wrappers[N][slot] → cached Array{T,N}
    │
    ├─ setfield!(:ref, new_memory_ref)   ← update backing memory (0-alloc)
    └─ setfield!(:size, new_dims)        ← update dimensions (0-alloc)
```

**Result**: Unlimited dimension patterns per slot with **zero allocation** after warmup. No eviction, no round-robin, no `CACHE_WAYS` limit.

```julia
# Pseudocode for Julia 1.11+ path
function unsafe_acquire!(pool, T, dims...)
    typed_pool = get_typed_pool!(pool, T)
    flat_view = get_view!(typed_pool, prod(dims))
    slot = typed_pool.n_active

    # Direct index lookup by dimensionality N (~1ns)
    wrapper = typed_pool.nd_wrappers[N][slot]
    if wrapper !== nothing
        setfield!(wrapper, :ref, getfield(vec, :ref))  # 0-alloc
        setfield!(wrapper, :size, dims)                 # 0-alloc
        return wrapper
    end

    # First call for this (slot, N): unsafe_wrap once, cached forever
    arr = wrap_array(typed_pool, flat_view, dims)
    store_wrapper!(typed_pool, N, slot, arr)
    return arr
end
```

### Julia 1.10 (Legacy): N-Way Set Associative Cache

On Julia 1.10, `Array` fields cannot be mutated, so the legacy path uses a 4-way set-associative cache with round-robin eviction:

- Cache hit (≤`CACHE_WAYS` dimension patterns per slot): **0 bytes**
- Cache miss (>`CACHE_WAYS` patterns): **~80-144 bytes** per `unsafe_wrap` call

See [Configuration](../features/configuration.md) for `CACHE_WAYS` tuning (Julia 1.10 / CUDA only).

### CUDA: N-Way Cache

The CUDA backend still uses the N-way set-associative cache (same as Julia 1.10 legacy), since `CuArray` does not support `setfield!`-based mutation.

## View vs Array Return Types

Type stability is critical for performance. AdaptiveArrayPools provides two APIs:

| API | 1D Return | N-D Return | Allocation (Julia 1.11+) | Allocation (Julia 1.10 / CUDA) |
|-----|-----------|------------|--------------------------|-------------------------------|
| `acquire!` | `SubArray{T,1}` | `ReshapedArray{T,N}` | Always 0 bytes | Always 0 bytes |
| `unsafe_acquire!` | `Vector{T}` | `Array{T,N}` | 0 bytes (setfield! reuse) | 0 bytes (hit) / ~100 bytes (miss) |

!!! note "`Bit` type behavior"
    For `T === Bit`, both `acquire!` and `unsafe_acquire!` return native `BitVector` / `BitArray{N}` (not views). Cache hit achieves 0 bytes allocation.

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
