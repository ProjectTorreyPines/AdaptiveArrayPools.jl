# Essential API

This page covers the core functions you'll use 99% of the time. For the complete API reference, see [Full API](../reference/api.md).

## Array Acquisition

### `acquire!(pool, T, dims...)`

The primary function. Returns `Array{T,N}` for most `T`. For `T === Bit`, returns a native `BitVector`/`BitArray{N}`.

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)        # 1D: Array{Float64,1} (Vector)
    m = acquire!(pool, Float64, 10, 10)     # 2D: Array{Float64,2} (Matrix)
    t = acquire!(pool, Int64, 2, 3, 4)      # 3D: Array{Int64,3}
    mask = acquire!(pool, Bit, 1000)        # BitVector (bit-packed)
end
```

**Always use `acquire!` by default.** Returns native `Array{T,N}` that works seamlessly with BLAS/LAPACK, FFI/ccall, and any function expecting `Array`; `Bit` masks are bit-packed and optimized for boolean-kernel operations (`count`, `any`, `all`, etc.). `acquire_array!` is an alias for `acquire!`.

### `acquire_view!(pool, T, dims...)`

Returns a lightweight view into pool storage (`SubArray` for 1D, `ReshapedArray` for N-D). This is the old `acquire!` behavior from v0.2.x. Use when you want zero-allocation views and don't need a native `Array`:

```julia
@with_pool pool begin
    v = acquire_view!(pool, Float64, 100)      # SubArray{Float64,1}
    m = acquire_view!(pool, Float64, 10, 10)   # ReshapedArray{Float64,2}
end
```

!!! note "`Bit` behavior"
    For `T === Bit`, `acquire_view!` returns native `BitVector`/`BitArray{N}` (same as `acquire!`).

## Convenience Functions

Zero-initialized arrays:

```julia
@with_pool pool begin
    z = zeros!(pool, Float64, 10, 10)   # All zeros
    o = ones!(pool, Float64, 100)       # All ones
end
```

Match existing array properties:

```julia
@with_pool pool begin
    A = acquire!(pool, Float64, 50, 50)
    B = similar!(pool, A)                # Same type and size as A
    C = similar!(pool, A, ComplexF64)    # Same size, different type
end
```

### Reshaping with `reshape!`

Reshape an existing array using the pool's wrapper cache. The result shares memory with the original — mutations are visible in both:

```julia
@with_pool pool function process_grid(data, nx, ny)
    M = reshape!(pool, data, nx, ny)   # 1D → 2D, shares memory with data
    col_sums = zeros!(pool, Float64, ny)
    for j in 1:ny, i in 1:nx
        col_sums[j] += M[i, j]
    end
    return sum(col_sums)
end
```

On Julia 1.11+, cross-dimensional reshapes are **zero-allocation** after warmup via `setfield!`-based wrapper reuse. On Julia 1.10, falls back to `Base.reshape`.

!!! warning "DimensionMismatch"
    `prod(dims)` must equal `length(A)`, otherwise a `DimensionMismatch` is thrown.

### Custom Initialization with `fill!`

For values other than 0 or 1, use Julia's built-in `fill!`:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    fill!(v, 3.14)              # Fill with pi

    m = acquire!(pool, Int64, 10, 10)
    fill!(m, -1)                # Fill with sentinel value
end
```

This pattern works because pool arrays share underlying storage with the pool.

## Pool Management

### `reset!(pool)`

Releases all memory held by the pool. Useful for long-running processes:

```julia
# After processing a large batch
@with_pool pool begin
    # ... large computation ...
end

# Optionally release memory if pool grew too large
reset!(get_task_local_pool())
```

### `pooling_enabled(pool)`

Check if pooling is active (returns `false` for `DisabledPool`):

```julia
@maybe_with_pool pool begin
    if pooling_enabled(pool)
        println("Using pool")
    else
        println("Pooling disabled")
    end
end
```

## Quick Reference

| Function | Returns | Allocation | Use Case |
|----------|---------|------------|----------|
| `acquire!(pool, T, dims...)` | `Array{T,N}` | 0 bytes | Default choice |
| `acquire_array!(pool, T, dims...)` | `Array{T,N}` | 0 bytes | Alias for `acquire!` |
| `acquire_view!(pool, T, dims...)` | View type | 0 bytes | Lightweight views |
| `zeros!(pool, [T,] dims...)` | `Array{T,N}` | 0 bytes | Zero-initialized |
| `ones!(pool, [T,] dims...)` | `Array{T,N}` | 0 bytes | One-initialized |
| `similar!(pool, A)` | `Array{T,N}` | 0 bytes | Match existing array |
| `reshape!(pool, A, dims...)` | Reshaped array | 0 bytes (1.11+) | Reshape sharing memory |
| `reset!(pool)` | `nothing` | - | Release all memory |
| `pooling_enabled(pool)` | `Bool` | - | Check pool status |

## See Also

- [Full API Reference](../reference/api.md) - Complete function list
- [@with_pool Patterns](with-pool-patterns.md) - Usage patterns
- [Safety Rules](safety-rules.md) - Scope rules
