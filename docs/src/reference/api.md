# API Reference

## API Summary

### Macros

| Macro | Description |
|-------|-------------|
| `@with_pool name expr` | **Recommended.** Injects a global, task-local pool named `name`. Automatically checkpoints and rewinds. |
| `@maybe_with_pool name expr` | Same as `@with_pool`, but can be toggled on/off at runtime via `MAYBE_POOLING[]`. |

### Functions

| Function | Description |
|----------|-------------|
| `acquire!(pool, T, dims...)` | Returns `Array{T,N}`: `Vector{T}` for 1D, `Array{T,N}` for N-D. For `T === Bit`, returns native `BitVector`/`BitArray{N}`. Cache hit 0 bytes. |
| `acquire!(pool, T, dims::Tuple)` | Tuple overload for `acquire!` (e.g., `acquire!(pool, T, size(x))`). |
| `acquire!(pool, x::AbstractArray)` | Similar-style: acquires array matching `eltype(x)` and `size(x)`. |
| `acquire_view!(pool, T, dims...)` | Returns a view: `SubArray{T,1}` for 1D, `ReshapedArray{T,N}` for N-D. For `T === Bit`, returns `BitVector`/`BitArray{N}` (same as `acquire!`). Cache hit 0 bytes. |
| `acquire_view!(pool, x::AbstractArray)` | Similar-style: acquires view matching `eltype(x)` and `size(x)`. |
| `acquire_array!(pool, T, dims...)` | Alias for `acquire!`. Explicit name for symmetric naming with `acquire_view!`. |
| `checkpoint!(pool)` | Saves the current pool state (stack pointer). |
| `rewind!(pool)` | Restores the pool to the last checkpoint, freeing all arrays acquired since then. |
| `pool_stats(pool)` | Prints detailed statistics about pool usage. |
| `get_task_local_pool()` | Returns the task-local pool instance. |
| `reset!(pool)` | Resets active-slot counters to 0 but **keeps** all buffers for reuse. |
| `trim!(pool; force_gc=false)` | Releases **inactive** retained buffers, keeping active arrays. Returns `(; slots_released, wrappers_released, estimated_bytes_released, gc_triggered)`. Works on CPU, CUDA, and Metal pools. Reclaims on Julia 1.12+; defined no-op (zero summary + one-time warning) on older Julia. |
| `compact!(pool; factor=10, shrink_to=1.5, min_bytes=2^20, active=true, force_gc=false)` | Shrinks over-allocated backing buffers **in place** (held arrays follow). Returns `(; slots_compacted, bytes_reclaimed, gc_triggered)`. Usually run automatically — see [Automatic Memory Management](@ref). |
| `empty!(pool)` | Clears all internal storage, releasing **all** memory. |

### Convenience Functions

Default element type is `Float64` (CPU) or `Float32` (CUDA).

| Function | Description |
|----------|-------------|
| `zeros!(pool, [T,] dims...)` | Zero-initialized array. Equivalent to `acquire!` + `fill!(0)`. |
| `ones!(pool, [T,] dims...)` | One-initialized array. Equivalent to `acquire!` + `fill!(1)`. |
| `trues!(pool, dims...)` | Bit-packed `BitVector` / `BitArray{N}` filled with `true`. |
| `falses!(pool, dims...)` | Bit-packed `BitVector` / `BitArray{N}` filled with `false`. |
| `similar!(pool, A)` | Array matching `eltype(A)` and `size(A)`. |
| `reshape!(pool, A, dims...)` | Reshape `A` to `dims`, sharing memory. Zero-alloc on Julia 1.12+. |

### Types

| Type | Description |
|------|-------------|
| `AdaptiveArrayPool` | The main pool type. Create with `AdaptiveArrayPool()`. |
| `Bit` | Sentinel type to request packed `BitVector` storage (1 bit/element). |
| `DisabledPool{Backend}` | Sentinel type when pooling is disabled. |

### Configuration & Utilities

| Symbol | Description |
|--------|-------------|
| `STATIC_POOLING` | Compile-time constant to disable all pooling. (alias: `USE_POOLING`) |
| `MAYBE_POOLING` | Runtime `Ref{Bool}` for `@maybe_with_pool`. (alias: `MAYBE_POOLING_ENABLED`) |
| `RUNTIME_CHECK` | Compile-time `Int` constant (0=off, 1=on). Set via `runtime_check` preference. Restart required. |
| `set_cache_ways!(n)` | Set N-way cache size (≤1.11 / CUDA only; no effect on Julia 1.12+ CPU). |
| `enable_auto_manage!(; …)` | (Re)start the background auto-compact/auto-trim timer. See [Automatic Memory Management](@ref). |
| `disable_auto_manage!()` | Stop the background timer for this session. |
| `auto_manage_enabled()` | `Bool`: is the background timer running? |
| `AUTO_MANAGE` | Compile-time `Bool` constant. Set via `auto_manage` preference (default `true`); `false` compiles the feature out. |

---

## Detailed Reference

```@autodocs
Modules = [AdaptiveArrayPools]
Order = [:macro, :function, :type, :constant]
```
