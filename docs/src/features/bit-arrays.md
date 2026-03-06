# BitArray Support

AdaptiveArrayPools.jl includes specialized support for `BitArray` (including `BitVector` and N-dimensional `BitArray{N}`), enabling **~8x memory savings** for boolean arrays compared to standard `Vector{Bool}`.

## The `Bit` Sentinel Type

To distinguish between standard boolean arrays (`Vector{Bool}`, 1 byte/element) and bit-packed arrays (`BitVector`, 1 bit/element), use the `Bit` sentinel type.

| Call | Result | Memory |
|------|--------|--------|
| `acquire!(pool, Bool, 1000)` | `Vector{Bool}` | 1000 bytes |
| `acquire!(pool, Bit, 1000)` | `BitVector` | ~125 bytes |

## Usage

### 1D Arrays (BitVector)
For 1D arrays, `acquire!` returns a native `BitVector` backed by pooled storage. This design enables full SIMD optimization, making operations like `count`, `any`, and bitwise ops significantly faster than views.

```julia
@with_pool pool begin
    # Acquire a BitVector of length 1000
    bv = acquire!(pool, Bit, 1000)

    # Use like normal
    bv .= true
    bv[1] = false

    # Supports standard operations with full SIMD acceleration
    count(bv)
end
```

### N-D Arrays (BitArray)
For multi-dimensional arrays, `acquire!` returns a native `BitArray{N}` (e.g., `BitMatrix` for 2D) that shares chunks with pooled storage. This preserves the packed memory layout and SIMD benefits while providing N-D indexing.

```julia
@with_pool pool begin
    # 100x100 bit matrix (returns BitMatrix)
    mask = zeros!(pool, Bit, 100, 100)

    mask[5, 5] = true

    # 3D BitArray
    volume = acquire!(pool, Bit, 10, 10, 10)
end
```

### Convenience Functions

For specific `BitVector` operations, prefer `trues!` and `falses!` which mirror Julia's standard functions:

```julia
@with_pool pool begin
    # Filled with false (equivalent to `falses(256)`)
    mask = falses!(pool, 256)

    # Filled with true (equivalent to `trues(256)`)
    flags = trues!(pool, 256)

    # Multidimensional
    grid = trues!(pool, 100, 100)

    # Similar to existing BitArray
    A = BitVector(undef, 50)
    B = similar!(pool, A)  # Reuses eltype(A) -> Bool

    # To explicit get Bit-packed from pool irrespective of source
    C = similar!(pool, A, Bit)
end
```

Note: `zeros!(pool, Bit, ...)` and `ones!(pool, Bit, ...)` are also supported (aliased to `falses!` and `trues!`).

## Performance & Safety

### Why Native BitArray?
The pool returns native `BitVector`/`BitArray` types instead of `SubArray` views for **performance**.
Operations like `count()`, `sum()`, and bitwise broadcasting are **10x~100x faster** on native bit arrays because they utilize SIMD instructions on packed 64-bit chunks.

### N-D Caching & Zero Allocation

The pool reuses `BitArray{N}` wrapper instances via `setfield!`-based in-place mutation (Julia 1.11+) or N-way cache (Julia 1.10 / CUDA):

| Scenario | Julia 1.11+ | Julia 1.10 / CUDA |
|----------|-------------|-------------------|
| First call with new (slot, N) | ~944 bytes (new `BitArray{N}`) | ~944 bytes |
| Subsequent call, any dims | **0 bytes** (setfield! reuse) | **0 bytes** (same ndims) / ~944 bytes (different ndims) |

On Julia 1.11+, `BitArray` fields (`len`, `dims`, `chunks`) are mutated in-place via `setfield!`, achieving **zero allocation** on all repeated calls regardless of dimension pattern.

```julia
@with_pool pool begin
    # First call: allocates BitMatrix wrapper (~944 bytes)
    m1 = acquire!(pool, Bit, 100, 100)

    # Rewind to reuse the same slot
    rewind!(pool)

    # Same dims: 0 allocation (cached wrapper reused)
    m2 = acquire!(pool, Bit, 100, 100)

    rewind!(pool)

    # Different dims but same ndims: 0 allocation (fields updated in-place)
    m3 = acquire!(pool, Bit, 50, 200)
end
```

### ⚠️ Important: Do Not Resize

While the returned arrays are standard `BitVector` types, they share their underlying memory chunks with the pool.

!!! warning "Do Not Resize"
    **NEVER** resize (`push!`, `pop!`, `resize!`) a pooled `BitVector` or `BitArray`.

    The underlying memory is owned and managed by the pool. Resizing it will detach it from the pool or potentially corrupt the shared state. Treat these arrays as **fixed-size** scratch buffers only.
