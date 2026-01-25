# BitVector Support

AdaptiveArrayPools.jl includes specialized support for `BitArray` (specifically `BitVector`), enabling **~8x memory savings** for boolean arrays compared to standard `Vector{Bool}`.

## The `Bit` Sentinel Type

To distinguish between standard boolean arrays (`Vector{Bool}`, 1 byte/element) and bit-packed arrays (`BitVector`, 1 bit/element), use the `Bit` sentinel type.

| Call | Result | Memory |
|------|--------|--------|
| `acquire!(pool, Bool, 1000)` | `Vector{Bool}` | 1000 bytes |
| `acquire!(pool, Bit, 1000)` | `BitVector` | ~125 bytes |

## Usage

### 1D Arrays (BitVector)
For 1D arrays, `acquire!` returns a view into a pooled `BitVector`.

```julia
@with_pool pool begin
    # Acquire a BitVector of length 1000
    bv = acquire!(pool, Bit, 1000)
    
    # Use like normal
    bv .= true
    bv[1] = false
    
    # Supports standard operations
    count(bv)
end
```

### N-D Arrays (BitArray / Reshaped)
For multi-dimensional arrays, `acquire!` returns a `ReshapedArray` wrapper around the linear `BitVector`. This maintains zero-allocation efficiency while providing N-D indexing.

```julia
@with_pool pool begin
    # 100x100 bit matrix
    mask = zeros!(pool, Bit, 100, 100)
    
    mask[5, 5] = true
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

Note: `zeros!(pool, Bit, ...)` and `ones!(pool, Bit, ...)` are also supported (aliased to `falses!` and `trues!`).
```

## How It Works

The pool maintains a separate `BitTypedPool` specifically for `BitVector` storage.
- **Sentinel**: `acquire!(..., Bit, ...)` dispatches to this special pool.
- **Views**: 1D returns `SubArray{Bool, 1, BitVector, ...}`.
- **Reshaping**: N-D returns `ReshapedArray{Bool, N, SubArray{...}}`.

This ensures that even for complex shapes, the underlying storage is always a compact `BitVector` reused from the pool.
