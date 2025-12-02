# PR: Add `unsafe_acquire!` API and Optimize N-D Array Performance

## Summary

- **`unsafe_acquire!`**: New API returning raw `Array{T,N}` for BLAS/FFI/hot-path dispatch
- **N-D `acquire!` optimization**: Returns `SubArray{T,N,Array{T,N}}` instead of `ReshapedArray` (better dispatch performance)
- **Similar-style convenience**: `acquire!(pool, x)` and `unsafe_acquire!(pool, x)` matching `eltype(x)` and `size(x)`

## API Comparison

| Function | Return Type | Use Case |
|----------|-------------|----------|
| `acquire!(pool, T, dims...)` | `SubArray{T,N}` | Safe default, prevents `resize!` |
| `unsafe_acquire!(pool, T, dims...)` | `Array{T,N}` | BLAS, FFI, hot-path dispatch |

## Changes

**Core (`src/core.jl`)**
- N-D `acquire!`: `reshape()` → `unsafe_wrap() + view()`
- New `unsafe_acquire!` API (1D, N-D, Tuple, Nothing fallback)
- Similar-style convenience methods

**Validation (`src/utils.jl`)**
- Unified pointer overlap check for ALL Array parents (Vector and N-D)
- `POOL_DEBUG` now catches both `SubArray` and raw `Array` escapes
- Properly detects `view(unsafe_acquire!(...), :)` escape patterns

**Docs**
- Updated Quick Start with both APIs
- New `acquire!` vs `unsafe_acquire!` section
- Updated API reference

## Breaking Changes

| Before | After |
|--------|-------|
| `acquire!(pool, T, n, m)` returns `ReshapedArray` | Returns `SubArray{T,2,Matrix{T}}` |

Most code unaffected (`<: AbstractArray`). Only explicit `isa ReshapedArray` checks break.
