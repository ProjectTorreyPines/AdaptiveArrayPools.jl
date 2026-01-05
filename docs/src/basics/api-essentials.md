# Essential API

This page covers the core functions you'll use 99% of the time. For the complete API reference, see [Full API](../reference/api.md).

## Array Acquisition

### `acquire!(pool, T, dims...)`

The primary function. Returns a view (`SubArray` for 1D, `ReshapedArray` for N-D).

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)        # 1D: SubArray{Float64,1}
    m = acquire!(pool, Float64, 10, 10)     # 2D: ReshapedArray{Float64,2}
    t = acquire!(pool, Int64, 2, 3, 4)      # 3D: ReshapedArray{Int64,3}
end
```

**Always use `acquire!` by default.** Views are zero-allocation and work with all BLAS/LAPACK operations.

### `unsafe_acquire!(pool, T, dims...)`

Returns a native `Array` type. **Zero-allocation on cache hit**—only allocates a small header (~80-144 bytes) on cache miss. Use when you specifically need `Array{T,N}`:

```julia
@with_pool pool begin
    # Use when you need Array for:
    arr = unsafe_acquire!(pool, Float64, 100)

    # - FFI/ccall requiring Ptr{T}
    ccall(:some_c_function, Cvoid, (Ptr{Float64}, Cint), arr, length(arr))

    # - Functions with strict Array{T,N} type signatures
end
```

!!! tip "Cache behavior"
    Same dimension pattern → **0 bytes**. Different pattern → 80-144 bytes header only (data memory always reused). See [N-Way Cache](../architecture/type-dispatch.md#n-way-set-associative-cache) for details.

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

This pattern works because pool arrays are mutable views into the underlying storage.

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
| `acquire!(pool, T, dims...)` | View type | 0 bytes | Default choice |
| `unsafe_acquire!(pool, T, dims...)` | `Array{T,N}` | 0 (hit) / 80-144 (miss) | FFI, type constraints |
| `zeros!(pool, [T,] dims...)` | View type | 0 bytes | Zero-initialized |
| `ones!(pool, [T,] dims...)` | View type | 0 bytes | One-initialized |
| `similar!(pool, A)` | View type | 0 bytes | Match existing array |
| `reset!(pool)` | `nothing` | - | Release all memory |
| `pooling_enabled(pool)` | `Bool` | - | Check pool status |

## See Also

- [Full API Reference](../reference/api.md) - Complete function list
- [@with_pool Patterns](with-pool-patterns.md) - Usage patterns
- [Safety Rules](safety-rules.md) - Scope rules
