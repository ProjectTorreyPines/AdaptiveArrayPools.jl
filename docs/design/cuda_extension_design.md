# AdaptiveArrayPools.jl CUDA Extension Design

> **Status**: Draft v0.6 (Post-Review Revision)
> **Version**: 0.6
> **Date**: 2024-12-14
> **Authors**: Design discussion with AI assistance

## 1. Executive Summary

This document outlines the design for extending AdaptiveArrayPools.jl to support GPU arrays via CUDA.jl. The design prioritizes:

1. **Zero overhead when CUDA not loaded** - Pure CPU code path unchanged
2. **Maximum code reuse** - Generic functions with minimal dispatch points
3. **Extensibility** - Abstract type hierarchy for future GPU backends
4. **Separate namespaces** - Independent CPU and GPU pools per task

### Key Design Decision: Parametric Abstract Types

Instead of duplicating code in the extension, we use parametric abstract types and generic functions. The extension only needs to define:
- Type definitions (~50 lines)
- One allocation method (~3 lines)
- Task-local getter (~15 lines)

**Total extension code: ~70 lines** (vs ~300 lines with full duplication)

---

## 2. Current Architecture Analysis

### 2.1 Core Type Structure

```julia
# Current: Concrete types only
mutable struct TypedPool{T}
    vectors::Vector{Vector{T}}
    views::Vector{SubArray{T,1,Vector{T},...}}
    view_lengths::Vector{Int}
    nd_arrays::Vector{Any}
    nd_dims::Vector{Any}
    nd_ptrs::Vector{UInt}
    nd_next_way::Vector{Int}
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end
```

### 2.2 Code Reuse Analysis

| Component | Operates On | GPU-Specific? |
|-----------|-------------|---------------|
| `get_view!` | vectors, n_active, cache | Only allocation |
| `get_nd_view!` | calls get_view!, reshape | **No** |
| `get_nd_array!` | calls get_view!, unsafe_wrap | Only wrap call |
| `checkpoint!` | n_active, checkpoint stacks | **No** |
| `rewind!` | n_active, checkpoint stacks | **No** |
| `reset!` | n_active, checkpoint stacks | **No** |
| `empty!` | all fields | Clear vectors only |

**Key insight**: 95%+ of logic is type-agnostic. Only allocation/wrapping differs.

---

## 3. Proposed Architecture

### 3.1 Type Hierarchy

```
              AbstractTypedPool{T, V<:AbstractVector{T}}
                              │
               ┌──────────────┼──────────────┐
               │              │              │
   TypedPool{T}        CuTypedPool{T}   [Future: ROCTypedPool{T}]
   V = Vector{T}       V = CuVector{T}   V = ROCArray{T,1}


                      AbstractArrayPool
                              │
               ┌──────────────┼──────────────┐
               │              │              │
   AdaptiveArrayPool   CuAdaptiveArrayPool   [Future: ROCArrayPool]
```

### 3.2 Abstract Type Definitions

```julia
# src/types.jl - New additions

"""
    AbstractTypedPool{T, V<:AbstractVector{T}}

Abstract base for type-specific memory pools.
`T` is the element type, `V` is the storage vector type.

Concrete subtypes must have these fields:
- vectors::Vector{V}
- views, view_lengths, nd_* fields
- n_active::Int
- _checkpoint_n_active::Vector{Int}
- _checkpoint_depths::Vector{Int}
"""
abstract type AbstractTypedPool{T, V<:AbstractVector{T}} end

"""
    AbstractArrayPool

Abstract base for multi-type array pools.

Concrete subtypes must have these fields:
- Fixed slot fields (type-specific TypedPools)
- others::IdDict{DataType,Any}
- _current_depth::Int
- _untracked_flags::Vector{Bool}
"""
abstract type AbstractArrayPool end

# Storage type accessor (for generic code)
storage_type(::AbstractTypedPool{T,V}) where {T,V} = V
storage_type(::Type{<:AbstractTypedPool{T,V}}) where {T,V} = V
```

### 3.3 Concrete Types

#### CPU (existing, modified to inherit)

```julia
# src/types.jl

mutable struct TypedPool{T} <: AbstractTypedPool{T, Vector{T}}
    # Storage
    vectors::Vector{Vector{T}}
    views::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}
    view_lengths::Vector{Int}

    # N-D cache
    nd_arrays::Vector{Any}
    nd_dims::Vector{Any}
    nd_ptrs::Vector{UInt}
    nd_next_way::Vector{Int}

    # State
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

mutable struct AdaptiveArrayPool <: AbstractArrayPool
    # Fixed slots (CPU types)
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    complexf32::TypedPool{ComplexF32}
    bool::TypedPool{Bool}

    others::IdDict{DataType,Any}
    _current_depth::Int
    _untracked_flags::Vector{Bool}
end
```

#### GPU (extension - minimal definitions)

> **[AI Review: Float16 & Device Safety]**
> 1. **Float16 Support**: Added `Float16` to fixed slots. This is critical for modern AI/ML workloads on GPU.
> 2. **Device Awareness**: Added `device_id::Int` to `CuAdaptiveArrayPool`. This is crucial for multi-GPU setups. A pool created on Device 0 cannot be safely used on Device 1. We must track which device owns the memory.

> **[Post-Review v0.6: Critical Type Correction]**
> **`view(CuVector, 1:n)` returns `CuVector`, NOT `SubArray`!**
>
> GPUArrays.jl handles contiguous views via `derive()` which returns a new GPU array
> sharing the same memory buffer (see `~/.julia/packages/GPUArrays/.../src/host/base.jl:302`).
> This is fundamentally different from CPU where `view()` returns `SubArray`.
>
> **Implications for pool design**:
> 1. We **cannot cache views separately** from backing vectors on GPU
> 2. Instead, we store `CuVector{T}` directly and return slices via `view()` on each call
> 3. View creation is cheap (no allocation, just metadata), so no caching benefit
> 4. This simplifies the GPU pool: no `views` or `view_lengths` fields needed

```julia
# ext/AdaptiveArrayPoolsCUDAExt/types.jl

using CUDA

# IMPORTANT: Unlike CPU, GPU views are derived CuArrays, not SubArrays.
# view(::CuVector{T}, ::UnitRange) -> CuVector{T} (shared memory, different offset/length)
# This means:
# 1. "views" vector would just hold more CuVectors (no savings)
# 2. We skip view caching entirely - just return view(vec, 1:n) each time
# 3. View creation is O(1) metadata operation, no GPU memory allocation

mutable struct CuTypedPool{T} <: AbstractTypedPool{T, CuVector{T}}
    # Storage (GPU vectors)
    vectors::Vector{CuVector{T}}

    # View length cache (for resize decision, but no view object cache)
    # The actual view is created fresh each time since it's just metadata
    view_lengths::Vector{Int}

    # N-D cache (same structure as CPU)
    nd_arrays::Vector{Any}
    nd_dims::Vector{Any}
    nd_ptrs::Vector{UInt}
    nd_next_way::Vector{Int}

    # State (identical to CPU)
    n_active::Int
    _checkpoint_n_active::Vector{Int}
    _checkpoint_depths::Vector{Int}
end

# Constructor with sentinel pattern
function CuTypedPool{T}() where T
    CuTypedPool{T}(
        CuVector{T}[], Int[],           # No views vector!
        Any[], Any[], UInt[], Int[],
        0, [0], [0]
    )
end

# GPU-optimized fixed slots (different from CPU!)
const GPU_FIXED_SLOT_FIELDS = (
    :float32,      # Primary (GPU-optimized)
    :float64,      # Precision when needed
    :float16,      # ML inference (added per AI review)
    :int32,        # Indexing (GPU-preferred)
    :int64,        # Large indices
    :complexf32,   # FFT, signal processing
    :complexf64,   # High-precision complex
    :bool,         # Masks
)

mutable struct CuAdaptiveArrayPool <: AbstractArrayPool
    # Fixed slots (GPU-optimized order: Float32 first)
    float32::CuTypedPool{Float32}
    float64::CuTypedPool{Float64}
    float16::CuTypedPool{Float16}  # Added per AI review
    int32::CuTypedPool{Int32}
    int64::CuTypedPool{Int64}
    complexf32::CuTypedPool{ComplexF32}
    complexf64::CuTypedPool{ComplexF64}
    bool::CuTypedPool{Bool}

    others::IdDict{DataType,Any}
    _current_depth::Int
    _untracked_flags::Vector{Bool}

    # Safety: Track which device this pool belongs to (use public API!)
    device_id::Int
end

function CuAdaptiveArrayPool()
    dev = CUDA.device()
    CuAdaptiveArrayPool(
        CuTypedPool{Float32}(), CuTypedPool{Float64}(), CuTypedPool{Float16}(),
        CuTypedPool{Int32}(), CuTypedPool{Int64}(),
        CuTypedPool{ComplexF32}(), CuTypedPool{ComplexF64}(),
        CuTypedPool{Bool}(),
        IdDict{DataType,Any}(), 1, [false],
        CUDA.deviceid(dev)  # Use public API, not internal .handle
    )
end
```

---

## 4. Generic Functions with Minimal Dispatch

### 4.1 Allocation Dispatch Point

The **only** type-specific function needed:

```julia
# src/acquire.jl - CPU default
"""
    allocate_vector(tp::AbstractTypedPool{T}, n::Int) -> V

Allocate a new vector of type V with n elements.
This is the single dispatch point for storage-specific allocation.
"""
@inline allocate_vector(::AbstractTypedPool{T,Vector{T}}, n::Int) where T =
    Vector{T}(undef, n)

# ext/ - GPU override (THE ONLY METHOD EXTENSION NEEDS TO ADD!)
@inline allocate_vector(::AbstractTypedPool{T,CuVector{T}}, n::Int) where T =
    CuVector{T}(undef, n)
```

> **[AI Review: Interaction with CUDA.jl Allocator]**
> It is important to note that `CuVector{T}(undef, n)` uses `CUDA.jl`'s own internal memory pool.
> **Why do we need another pool?**
> 1. **Overhead Reduction**: Even cached CUDA allocations have Julia-side overhead (struct creation, finalizer registration). `AdaptiveArrayPools` reuses the *Julia objects* (`CuArray` structs) and views, reducing GC pressure and allocation latency further.
> 2. **Logical Grouping**: It allows "rewinding" a whole block of temporary allocations in one go, which `CUDA.jl`'s allocator doesn't support (it's `malloc`/`free` style).

### 4.2 get_view! Implementation

> **[Post-Review v0.6: CPU vs GPU Differences]**
> Due to type differences (`view(Vector, 1:n) → SubArray` vs `view(CuVector, 1:n) → CuVector`),
> the CPU and GPU implementations differ slightly. CPU caches view objects; GPU creates them fresh.

#### CPU Version (existing, unchanged)

```julia
# src/acquire.jl - CPU implementation (caches SubArray views)

function get_view!(tp::AbstractTypedPool{T,Vector{T}}, n::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active

    # 1. Expand pool if needed
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, n))
        new_view = view(tp.vectors[idx], 1:n)
        push!(tp.views, new_view)         # Cache the SubArray
        push!(tp.view_lengths, n)
        # ... growth warning ...
        return new_view
    end

    # 2. Cache hit (return cached SubArray - ZERO ALLOC)
    @inbounds cached_len = tp.view_lengths[idx]
    if cached_len == n
        return @inbounds tp.views[idx]
    end

    # 3. Cache miss - resize and update cached view
    @inbounds vec = tp.vectors[idx]
    if length(vec) < n
        resize!(vec, n)
    end
    new_view = view(vec, 1:n)
    @inbounds tp.views[idx] = new_view
    @inbounds tp.view_lengths[idx] = n
    return new_view
end
```

#### GPU Version (extension)

> **[Post-Review v0.6: resize! Cost Warning]**
> `resize!(::CuVector, n)` with capacity increase triggers:
> 1. New GPU buffer allocation
> 2. Async copy of existing elements (even if we don't need them!)
>
> For pools, we typically don't need old data. Consider using `CUDA.unsafe_free!` + fresh
> allocation instead, or just allocating oversized initially. This is a **performance
> optimization opportunity** for v1.1+.

```julia
# ext/AdaptiveArrayPoolsCUDAExt/acquire.jl

# GPU version: no view caching (view() returns CuVector, not SubArray)
function AdaptiveArrayPools.get_view!(tp::CuTypedPool{T}, n::Int) where {T}
    tp.n_active += 1
    idx = tp.n_active

    # 1. Expand pool if needed
    if idx > length(tp.vectors)
        push!(tp.vectors, allocate_vector(tp, n))
        push!(tp.view_lengths, n)
        # Return fresh view (no caching - view creates CuVector metadata)
        return view(tp.vectors[idx], 1:n)
    end

    # 2. Check if resize needed
    @inbounds cached_len = tp.view_lengths[idx]
    @inbounds vec = tp.vectors[idx]

    if length(vec) < n
        # WARNING: resize! on CuVector copies old data (wasteful for pools)
        # TODO v1.1: Consider CUDA.unsafe_free! + fresh alloc instead
        resize!(vec, n)
    end

    @inbounds tp.view_lengths[idx] = n

    # Always create fresh view (O(1) metadata, no GPU allocation)
    return view(vec, 1:n)
end
```

### 4.3 get_nd_view! Implementation

> **[Post-Review v0.6: reshape Behavior on GPU]**
> `reshape(::CuVector, dims)` also uses GPUArrays' `derive()` mechanism, returning a
> `CuArray{T,N}` (not `ReshapedArray`). This is actually simpler - we get a proper
> GPU array that CUDA kernels can use directly.

```julia
# src/acquire.jl - Works for both, but return types differ:
# - CPU: ReshapedArray{T,N,SubArray{...}}
# - GPU: CuArray{T,N} (via derive)

@inline function get_nd_view!(tp::AbstractTypedPool{T}, dims::NTuple{N,Int}) where {T,N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)
    return reshape(flat_view, dims)  # CPU: ReshapedArray, GPU: CuArray
end
```

### 4.4 Generic get_nd_array! (minimal dispatch)

```julia
# src/acquire.jl

# CPU version uses unsafe_wrap
@inline function wrap_array(::AbstractTypedPool{T,Vector{T}},
                            flat_view, dims::NTuple{N,Int}) where {T,N}
    unsafe_wrap(Array{T,N}, pointer(flat_view), dims)
end

# ext/ - GPU version
@inline function wrap_array(::AbstractTypedPool{T,CuVector{T}},
                            flat_view, dims::NTuple{N,Int}) where {T,N}
    # Use reshape - returns CuArray{T,N} via GPUArrays derive()
    reshape(flat_view, dims)
end

# Generic implementation
@inline function get_nd_array!(tp::AbstractTypedPool{T}, dims::NTuple{N,Int}) where {T,N}
    total_len = safe_prod(dims)
    flat_view = get_view!(tp, total_len)
    slot = tp.n_active

    # ... cache lookup logic (identical) ...

    # DISPATCH POINT for array wrapping
    arr = wrap_array(tp, flat_view, dims)

    # ... cache update logic (identical) ...

    return arr
end
```

> **[Post-Review v0.6: GPU reshape Clarification]**
> `reshape(::CuArray, dims)` returns a `CuArray{T,N}` (via GPUArrays `derive()`), **NOT**
> `ReshapedArray`. This is actually better for GPU kernels - they work directly with
> `CuArray` without any wrapper overhead. The `derive()` mechanism shares the underlying
> GPU memory buffer with different offset/strides metadata.

---

## 5. State Management (100% Reusable)

### 5.1 Generic State Functions

All state functions operate only on `n_active` and checkpoint vectors - pure CPU operations.

```julia
# src/state.jl - These work for ANY AbstractTypedPool!

@inline function _checkpoint_typed_pool!(tp::AbstractTypedPool, depth::Int)
    push!(tp._checkpoint_n_active, tp.n_active)
    push!(tp._checkpoint_depths, depth)
    nothing
end

@inline function _rewind_typed_pool!(tp::AbstractTypedPool, current_depth::Int)
    # Orphan cleanup
    while @inbounds tp._checkpoint_depths[end] > current_depth
        pop!(tp._checkpoint_depths)
        pop!(tp._checkpoint_n_active)
    end

    # Restore
    if @inbounds tp._checkpoint_depths[end] == current_depth
        pop!(tp._checkpoint_depths)
        tp.n_active = pop!(tp._checkpoint_n_active)
    else
        tp.n_active = @inbounds tp._checkpoint_n_active[end]
    end
    nothing
end

function _reset_typed_pool!(tp::AbstractTypedPool)
    tp.n_active = 0
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)
    tp
end

# Concrete dispatches (trivial wrappers)
reset!(tp::TypedPool) = _reset_typed_pool!(tp)
reset!(tp::CuTypedPool) = _reset_typed_pool!(tp)  # ext/ adds this
```

### 5.2 empty! (Type-Specific)

`empty!` needs to clear storage, but the logic is identical:

```julia
# src/state.jl - Generic implementation

function Base.empty!(tp::AbstractTypedPool)
    empty!(tp.vectors)
    empty!(tp.views)          # CPU only (GPU CuTypedPool has no views field)
    empty!(tp.view_lengths)
    empty!(tp.nd_arrays)
    empty!(tp.nd_dims)
    empty!(tp.nd_ptrs)
    empty!(tp.nd_next_way)
    _reset_typed_pool!(tp)
    tp
end

# GPU-specific version (no views field)
function Base.empty!(tp::CuTypedPool)
    empty!(tp.vectors)
    empty!(tp.view_lengths)
    empty!(tp.nd_arrays)
    empty!(tp.nd_dims)
    empty!(tp.nd_ptrs)
    empty!(tp.nd_next_way)
    _reset_typed_pool!(tp)
    tp
end
```

> **[Post-Review v0.6: GPU Memory Release Clarification]**
> `empty!(tp.vectors)` **removes Julia references** to `CuVector` objects. This does NOT
> guarantee immediate VRAM release! The actual GPU memory lifecycle is:
>
> 1. **Reference removed** → CuArray becomes GC-eligible
> 2. **GC runs** → CuArray finalizer queued
> 3. **Finalizer runs** → Returns memory to CUDA.jl's internal pool
> 4. **CUDA.jl pool decision** → May or may not release to driver
>
> For **immediate VRAM release**, use `CUDA.reclaim()` after `empty!()`:
> ```julia
> empty!(get_task_local_cuda_pool())
> GC.gc()           # Force finalizers to run
> CUDA.reclaim()    # Request CUDA.jl to release cached memory
> ```

---

## 6. Task-Local Pool Design

> **[AI Review: Multi-Device Safety]**
> The original design for `get_task_local_cuda_pool` was unsafe for multi-GPU workflows. If a task switches devices (e.g., `CUDA.device!(1)`), it must not use the pool created for Device 0.
> **Revised Design**: We use a `Dict{Int, CuAdaptiveArrayPool}` in task local storage to manage one pool per device per task.

### 6.1 Separate Keys & Device Awareness

```julia
# src/task_local_pool.jl
const _POOL_KEY = :ADAPTIVE_ARRAY_POOL

@inline function get_task_local_pool()
    pool = get(task_local_storage(), _POOL_KEY, nothing)
    if pool === nothing
        pool = AdaptiveArrayPool()
        task_local_storage(_POOL_KEY, pool)
    end
    return pool::AdaptiveArrayPool
end

# ext/AdaptiveArrayPoolsCUDAExt/task_local_pool.jl
const _CU_POOL_KEY = :ADAPTIVE_ARRAY_POOL_CUDA

@inline function get_task_local_cuda_pool()
    # Get the dictionary of pools (one per device)
    pools = get(task_local_storage(), _CU_POOL_KEY, nothing)
    if pools === nothing
        pools = Dict{Int, CuAdaptiveArrayPool}()
        task_local_storage(_CU_POOL_KEY, pools)
    end

    # Get current device ID using public API
    dev_id = CUDA.deviceid(CUDA.device())

    # Get or create pool for this device
    if !haskey(pools, dev_id)
        pools[dev_id] = CuAdaptiveArrayPool() # Constructor captures device_id
    end

    return pools[dev_id]
end
```

> **[Post-Review v0.6: Public API for Device ID]**
> Always use `CUDA.deviceid(dev)` instead of `dev.handle`. The `.handle` field is internal
> and may change between CUDA.jl versions. `deviceid()` is the stable public API.

### 6.2 Rationale for Separation

| Scenario | Benefit |
|----------|---------|
| Mixed CPU/GPU workflow | Use both pools independently |
| GPU memory pressure | `empty!(cuda_pool)` without affecting CPU |
| Different lifecycles | CPU warm, GPU cleared per batch |
| **Multi-GPU** | **Safety**: Prevents cross-device access errors |
| Debugging | Clear distinction in profiling |

---

## 7. Macro Design

### 7.1 Recommended: Unified Macro with Backend Symbol

```julia
# Unified API - single macro with optional backend symbol
@with_pool pool begin ... end              # CPU (default, :cpu implied)
@with_pool :cuda pool begin ... end        # GPU via CUDA
@with_pool :metal pool begin ... end       # GPU via Metal (future)
@with_pool :cpu pool begin ... end         # Explicit CPU

# Without pool name (auto-generated)
@with_pool begin ... end                   # CPU default
@with_pool :cuda begin ... end             # GPU
```

**Advantages:**
- Single macro to learn
- Easy backend switching (`:cuda` → `:metal`)
- Future-proof (just add new symbols in extensions)
- Clean, consistent API

### 7.2 Implementation

> **[Post-Review v0.6: Zero-Overhead Backend Selection]**
> The original `Dict{Symbol, Function}` registry has a critical flaw: runtime dictionary
> lookup weakens type inference, preventing the compiler from inlining the pool getter.
> This conflicts with our "zero overhead for CPU path" goal.
>
> **Solution**: Use `Val{:backend}` dispatch instead. Extensions add methods at load time,
> and the compiler can fully inline the call chain.

```julia
# src/macros.jl - Val-based dispatch for zero overhead

"""
    _get_pool_for_backend(::Val{:cpu}) -> AdaptiveArrayPool

Get task-local pool for the specified backend. Extensions add methods for their backends.
Using Val{Symbol} enables compile-time dispatch and full inlining.
"""
@inline _get_pool_for_backend(::Val{:cpu}) = get_task_local_pool()

# Fallback with helpful error message
@noinline function _get_pool_for_backend(::Val{B}) where B
    error("Pool backend :$B not found. Did you forget to load the extension (e.g., `using CUDA`)?")
end

# Macro signatures
macro with_pool(backend::QuoteNode, pool_name, expr)
    _generate_pool_code_with_backend(backend.value, pool_name, expr)
end

macro with_pool(backend::QuoteNode, expr)
    # Backend symbol without pool name
    pool_name = gensym(:pool)
    _generate_pool_code_with_backend(backend.value, pool_name, expr)
end

macro with_pool(pool_name, expr)
    # No backend = CPU default
    _generate_pool_code_with_backend(:cpu, pool_name, expr)
end

macro with_pool(expr)
    # No backend, no pool name
    pool_name = gensym(:pool)
    _generate_pool_code_with_backend(:cpu, pool_name, expr)
end

function _generate_pool_code_with_backend(backend::Symbol, pool_name, expr)
    transformed_expr = _transform_acquire_calls(expr, pool_name)

    # Use Val{backend} for compile-time dispatch - fully inlinable!
    quote
        local $(esc(pool_name)) = $_get_pool_for_backend($(Val{backend}()))
        checkpoint!($(esc(pool_name)))
        try
            $(esc(transformed_expr))
        finally
            rewind!($(esc(pool_name)))
        end
    end
end
```

> **Why Val{:backend} instead of Dict?**
>
> | Approach | Lookup Cost | Type Inference | Inlining |
> |----------|-------------|----------------|----------|
> | `Dict{Symbol,Function}` | O(1) hash | ❌ Returns `Function` | ❌ Dynamic call |
> | `Val{:cpu}` dispatch | O(0) compiled | ✅ Concrete type | ✅ Full inlining |
>
> With Val dispatch, `@with_pool :cpu` compiles to exactly the same code as the
> original non-backend version—zero overhead.

### 7.3 Extension Registration

```julia
# ext/AdaptiveArrayPoolsCUDAExt/macros.jl

# Add method for :cuda backend via Val dispatch (no __init__ needed!)
@inline AdaptiveArrayPools._get_pool_for_backend(::Val{:cuda}) = get_task_local_cuda_pool()

# Optional: Explicit macro alias for users who prefer it
macro with_cuda_pool(pool_name, expr)
    esc(:(@with_pool :cuda $pool_name $expr))
end

macro with_cuda_pool(expr)
    esc(:(@with_pool :cuda $expr))
end

export @with_cuda_pool  # Optional explicit alias
```

> **Note**: With Val dispatch, no `__init__` registration is needed. The method is added
> when the extension module loads, and Julia's method dispatch handles the rest.

### 7.4 Design Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Unified** (`@with_pool :cuda`) | Single API, easy switching, extensible | Symbol must be literal |
| **Explicit** (`@with_cuda_pool`) | Clear intent, better autocomplete | Multiple macros to learn |
| **Hybrid** (both available) | User choice | Slight API redundancy |

**Recommendation: Hybrid approach** - unified macro as primary API, explicit aliases optional.

---

## 8. Package Extension Structure

### 8.1 Project.toml Changes

```toml
[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
AdaptiveArrayPoolsCUDAExt = "CUDA"
```

### 8.2 File Structure

```
AdaptiveArrayPools/
├── src/
│   ├── AdaptiveArrayPools.jl
│   ├── types.jl              # + AbstractTypedPool{T,V}, AbstractArrayPool
│   ├── acquire.jl            # + allocate_vector, wrap_array dispatch points
│   ├── state.jl              # Generic _checkpoint/_rewind/_reset (unchanged logic)
│   ├── task_local_pool.jl    # (unchanged)
│   ├── macros.jl             # + _get_pool_for_backend(::Val{:cpu}) dispatch
│   └── utils.jl              # (unchanged)
└── ext/
    └── AdaptiveArrayPoolsCUDAExt/
        ├── AdaptiveArrayPoolsCUDAExt.jl  # ~25 lines
        ├── types.jl                       # ~50 lines (no views field!)
        ├── acquire.jl                     # ~30 lines (GPU-specific get_view!)
        ├── dispatch.jl                    # ~35 lines (+ checkpoint correction)
        ├── task_local_pool.jl             # ~25 lines (multi-device, public API)
        └── macros.jl                      # ~15 lines (@with_cuda_pool)
```

**Total extension: ~180 lines** (slightly more due to GPU-specific get_view!)

### 8.3 Extension Entry Point

```julia
# ext/AdaptiveArrayPoolsCUDAExt/AdaptiveArrayPoolsCUDAExt.jl

module AdaptiveArrayPoolsCUDAExt

using AdaptiveArrayPools
using AdaptiveArrayPools: AbstractTypedPool, AbstractArrayPool,
                          allocate_vector, wrap_array, get_view!,
                          _checkpoint_typed_pool!, _rewind_typed_pool!,
                          _reset_typed_pool!, _get_pool_for_backend,
                          CACHE_WAYS, checkpoint!, rewind!, reset!
using CUDA

include("types.jl")
include("acquire.jl")      # GPU-specific get_view!
include("dispatch.jl")
include("task_local_pool.jl")
include("macros.jl")

# Exports
export CuAdaptiveArrayPool, CuTypedPool
export get_task_local_cuda_pool
export @with_cuda_pool

end # module
```

### 8.4 dispatch.jl

```julia
# ext/AdaptiveArrayPoolsCUDAExt/dispatch.jl

# THE KEY DISPATCH METHODS

@inline AdaptiveArrayPools.allocate_vector(
    ::AbstractTypedPool{T,CuVector{T}}, n::Int
) where T = CuVector{T}(undef, n)

@inline AdaptiveArrayPools.wrap_array(
    ::AbstractTypedPool{T,CuVector{T}}, flat_view, dims::NTuple{N,Int}
) where {T,N} = reshape(flat_view, dims)

# get_typed_pool! dispatches for CuAdaptiveArrayPool
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Float32}) = p.float32
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Float64}) = p.float64
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Float16}) = p.float16
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Int32}) = p.int32
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Int64}) = p.int64
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{ComplexF32}) = p.complexf32
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{ComplexF64}) = p.complexf64
@inline AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{Bool}) = p.bool

# Fallback for other types (with checkpoint correction!)
@inline function AdaptiveArrayPools.get_typed_pool!(p::CuAdaptiveArrayPool, ::Type{T}) where T
    get!(p.others, T) do
        tp = CuTypedPool{T}()
        # CRITICAL: Match CPU behavior - auto-checkpoint new pool if inside @with_pool scope
        # Without this, rewind! would corrupt state for dynamically-created pools
        if p._current_depth > 1
            push!(tp._checkpoint_n_active, 0)  # n_active starts at 0
            push!(tp._checkpoint_depths, p._current_depth)
        end
        tp
    end::CuTypedPool{T}
end
```

> **[Post-Review v0.6: Checkpoint Correction for Dynamic Pools]**
> When a new `CuTypedPool{T}` is created inside a `@with_pool` scope (i.e., when
> `_current_depth > 1`), we must initialize its checkpoint state to match the current
> depth. Otherwise, `rewind!` would pop from an incorrect checkpoint stack state.
>
> This mirrors the CPU implementation in `src/types.jl:230-238`.

---

## 9. Memory Layout Clarification

### 9.1 Why `Vector{CuVector{T}}` (not `CuVector{CuVector{T}}`)

```
✅ Correct: Vector{CuVector{T}}

   CPU RAM                      GPU VRAM
   ┌─────────────────┐          ┌─────────────────┐
   │ Vector          │          │                 │
   │ ├─ CuVec meta1 ─┼──────────┼─► data1 [...]   │
   │ ├─ CuVec meta2 ─┼──────────┼─► data2 [...]   │
   │ └─ CuVec meta3 ─┼──────────┼─► data3 [...]   │
   └─────────────────┘          └─────────────────┘

   Pool management: CPU         Computation: GPU
```

### 9.2 What Lives Where

| Component | Location | Reason |
|-----------|----------|--------|
| Pool struct | CPU | Julia runtime |
| `vectors::Vector{...}` | CPU | Pool indexing |
| CuVector metadata | CPU | Julia object wrapper |
| CuVector data | **GPU** | Actual computation |
| n_active, checkpoints | CPU | State management |

---

## 10. Migration Path

### 10.1 Phase 1: Abstract Types (Non-Breaking)

**Changes to src/:**
```julia
# types.jl
+ abstract type AbstractTypedPool{T, V<:AbstractVector{T}} end
+ abstract type AbstractArrayPool end
- mutable struct TypedPool{T}
+ mutable struct TypedPool{T} <: AbstractTypedPool{T, Vector{T}}
- mutable struct AdaptiveArrayPool
+ mutable struct AdaptiveArrayPool <: AbstractArrayPool

# acquire.jl
+ allocate_vector(::AbstractTypedPool{T,Vector{T}}, n) where T = Vector{T}(undef, n)
+ wrap_array(::AbstractTypedPool{T,Vector{T}}, view, dims) where {T,N} = unsafe_wrap(...)
# Change get_view!, get_nd_array! signatures to use AbstractTypedPool

# state.jl
# Change _checkpoint_typed_pool!, _rewind_typed_pool! to use AbstractTypedPool
```

**Breaking potential**: None (only adding supertypes and using more general signatures)

### 10.2 Phase 2: CUDA Extension

**New files in ext/:**
- Minimal implementation as described above

**Breaking potential**: None (purely additive)

### 10.3 Phase 3: Macro Enhancement (Optional)

- Consider Option B unified macro
- Add `@with_cuda_pool` first, evaluate need for unification

---

## 11. Example Usage (Target API)

### 11.1 Basic Usage - Unified Macro

```julia
using AdaptiveArrayPools
using CUDA  # Triggers extension loading, registers :cuda backend

# CPU workflow (default, unchanged)
function cpu_compute(data)
    @with_pool pool begin
        tmp = acquire!(pool, Float64, length(data))
        tmp .= data
        sum(tmp)
    end
end

# GPU workflow - using :cuda backend symbol
function gpu_compute(data::CuVector)
    @with_pool :cuda pool begin
        A = acquire!(pool, Float32, 1000, 1000)  # Returns CuMatrix{Float32}
        B = acquire!(pool, Float32, 1000, 1000)

        A .= CUDA.rand(1000, 1000)
        B .= A .* 2

        sum(B)
    end
end

# Explicit CPU backend (equivalent to default)
function explicit_cpu_compute(data)
    @with_pool :cpu pool begin
        tmp = acquire!(pool, Float64, length(data))
        tmp .= data
        sum(tmp)
    end
end
```

### 11.2 Mixed CPU/GPU Workflow

```julia
function mixed_compute(host_data::Vector{Float32})
    # CPU pool for staging
    @with_pool cpu_pool begin
        staging = acquire!(cpu_pool, Float32, length(host_data))
        staging .= host_data

        # Nested GPU pool
        @with_pool :cuda gpu_pool begin
            device_data = acquire!(gpu_pool, Float32, length(staging))
            copyto!(device_data, staging)  # CPU → GPU
            device_data .= device_data .^ 2
            copyto!(staging, device_data)  # GPU → CPU
        end  # GPU pool rewinds here

        sum(staging)
    end  # CPU pool rewinds here
end
```

### 11.3 Without Pool Name (Auto-generated)

```julia
# When you don't need to reference the pool directly
function simple_gpu_compute()
    @with_pool :cuda begin
        # pool name auto-generated, use get_task_local_cuda_pool() if needed
        A = acquire!(get_task_local_cuda_pool(), Float32, 100, 100)
        sum(A)
    end
end

# Or use the explicit getter within the block
function gpu_with_getter()
    @with_pool :cuda begin
        pool = get_task_local_cuda_pool()
        A = acquire!(pool, Float32, 100, 100)
        B = acquire!(pool, Float32, 100, 100)
        A .+ B
    end
end
```

### 11.4 Backend Switching (Same Code, Different Backend)

```julia
# Parameterized backend - useful for testing/benchmarking
function compute_on_backend(data, backend::Symbol)
    if backend == :cpu
        @with_pool pool begin
            tmp = acquire!(pool, Float32, length(data))
            tmp .= data
            sum(tmp)
        end
    elseif backend == :cuda
        @with_pool :cuda pool begin
            tmp = acquire!(pool, Float32, length(data))
            tmp .= data
            sum(tmp)
        end
    end
end

# Note: Backend symbol must be literal in macro (compile-time)
# For runtime dispatch, use explicit pool getters:
function runtime_backend_dispatch(data, use_gpu::Bool)
    pool = use_gpu ? get_task_local_cuda_pool() : get_task_local_pool()
    checkpoint!(pool)
    try
        tmp = acquire!(pool, Float32, length(data))
        tmp .= data
        sum(tmp)
    finally
        rewind!(pool)
    end
end
```

### 11.5 Explicit Pool Management (Advanced)

```julia
# Manual checkpoint/rewind for fine-grained control
function explicit_pool_management()
    cpu = get_task_local_pool()
    gpu = get_task_local_cuda_pool()

    # Checkpoint both pools
    checkpoint!(cpu)
    checkpoint!(gpu)
    try
        cpu_buf = acquire!(cpu, Float64, 1000)
        gpu_buf = acquire!(gpu, Float32, 1000)

        # ... computation ...

    finally
        # Rewind in reverse order (LIFO)
        rewind!(gpu)
        rewind!(cpu)
    end
end

# Clear GPU memory when under pressure
function memory_sensitive_workflow()
    @with_pool :cuda pool begin
        # Heavy GPU computation
        A = acquire!(pool, Float32, 10000, 10000)
        # ...
    end

    # Explicitly free GPU memory if needed
    empty!(get_task_local_cuda_pool())

    # Continue with CPU work
    @with_pool pool begin
        # CPU pool unaffected
    end
end
```

### 11.6 Future: Multiple GPU Backends

```julia
# When Metal.jl extension is added (future)
using Metal  # Registers :metal backend

function apple_silicon_compute()
    @with_pool :metal pool begin
        A = acquire!(pool, Float32, 1000, 1000)  # MtlMatrix{Float32}
        # Metal-specific computation
    end
end
```

> **[Post-Review v0.6: Backend Symbol Must Be Literal]**
> The macro `@with_pool :backend` requires a **literal symbol** (`:cuda`, `:metal`),
> not a variable containing a symbol. This is a Julia macro limitation—the backend
> is resolved at macro expansion time (compile time), not runtime.
>
> **This does NOT work:**
> ```julia
> const GPU_BACKEND = Sys.isapple() ? :metal : :cuda
> @with_pool GPU_BACKEND pool begin ... end  # ERROR: GPU_BACKEND is not a QuoteNode
> ```
>
> **For runtime backend selection, use explicit pool getters:**
> ```julia
> function portable_gpu_compute(use_metal::Bool)
>     pool = use_metal ? get_task_local_metal_pool() : get_task_local_cuda_pool()
>     checkpoint!(pool)
>     try
>         A = acquire!(pool, Float32, 1000, 1000)
>         # ... computation ...
>     finally
>         rewind!(pool)
>     end
> end
> ```
>
> **Or use `@static` for compile-time platform selection:**
> ```julia
> function portable_gpu_compute()
>     @static if Sys.isapple()
>         @with_pool :metal pool begin
>             # Metal path
>         end
>     else
>         @with_pool :cuda pool begin
>             # CUDA path
>         end
>     end
> end
> ```

---

## 12. Open Questions

### 12.1 Resolved

1. **Code duplication in extension** → Solved with parametric abstract types
2. **Macro approach** → Hybrid: unified `@with_pool :cuda` + optional `@with_cuda_pool`
3. **Memory layout** → `Vector{CuVector{T}}` is correct
4. **Float16 support** → **Added** to GPU fixed slots (per AI review)
5. **Multi-Device Safety** → **Solved** with `Dict{Int, Pool}` in task local storage (per AI review)
6. **unsafe_wrap for GPU** → Use `reshape` instead (per AI review)
7. **[v0.6] GPU view type** → `view(CuVector, 1:n)` returns `CuVector`, not `SubArray`. Pool design simplified.
8. **[v0.6] Zero-overhead backend selection** → `Val{:backend}` dispatch instead of Dict registry
9. **[v0.6] GPU checkpoint correction** → Added to `get_typed_pool!` fallback for `others` dict
10. **[v0.6] Device ID API** → Use `CUDA.deviceid(dev)` instead of internal `.handle`
11. **[v0.6] Backend symbol literal requirement** → Documented; `@static if` for platform selection

### 12.2 Stream Synchronization (Critical Safety Documentation)

> **[Post-Review v0.6: Expanded Safety Documentation]**

**The Problem**: `rewind!` logically "frees" pooled memory. If a GPU kernel is still
running asynchronously using that memory, and the pool re-issues it for a new allocation,
**data corruption** or **use-after-free** occurs.

**When It's Safe** (no synchronization needed):
- Single Task, default stream: Julia tasks typically use CUDA's default stream, which
  serializes operations. `rewind!` happens after all prior operations complete.
- `CUDA.@sync` inside the block: Explicit synchronization before rewind.

**When It's DANGEROUS** (must synchronize):

1. **Passing arrays to other Tasks**:
   ```julia
   @with_pool :cuda pool begin
       A = acquire!(pool, Float32, 1000)
       @spawn begin
           # DANGER: This task may still be using A after rewind!
           expensive_computation!(A)
       end
   end  # rewind! happens here - A is now invalid!
   ```
   **Fix**: Wait for spawned task before exiting scope.

2. **Explicit async streams**:
   ```julia
   @with_pool :cuda pool begin
       A = acquire!(pool, Float32, 1000)
       stream = CUDA.stream()
       CUDA.@sync stream begin
           # Kernel launched on non-default stream
           my_kernel!(A; stream)
       end
       # If no @sync: kernel may still be running when rewind! executes
   end
   ```
   **Fix**: `CUDA.synchronize(stream)` or use `CUDA.@sync` before scope ends.

3. **Kernel launch then immediate exit**:
   ```julia
   @with_pool :cuda pool begin
       A = acquire!(pool, Float32, 1000)
       @cuda threads=1024 my_kernel!(A)
       # Kernel is async! May still be running...
   end  # rewind! immediately follows!
   ```
   **Fix**: `CUDA.synchronize()` or `CUDA.@sync @cuda ...`

**Recommendation for Documentation**:
```julia
# GPU POOLING SAFETY RULES
#
# 1. DO NOT pass pooled arrays to other Tasks without synchronization
# 2. DO synchronize before @with_pool block ends if using async streams
# 3. PREFER `CUDA.@sync` around kernel launches in pooled scopes
# 4. WHEN IN DOUBT: `CUDA.synchronize()` before the block ends
```

### 12.3 Still Open

1. **Typed checkpoint for GPU**: Reuse existing macro logic?
   - Should work with minimal changes
   - Need to export `_transform_acquire_calls` etc.

2. **resize! optimization for GPU** (v1.1+):
   - Current: `resize!(CuVector, n)` copies old data (wasteful for pools)
   - Consider: `CUDA.unsafe_free!` + fresh allocation, or pre-allocate oversized

3. **Multi-backend single macro**: Support multiple pools in one call?
   - Tuple syntax: `@with_pool (:cpu, cpu_pool) (:cuda, cuda_pool) begin ... end`
   - Pro: Cleaner for mixed workflows, guaranteed proper rewind order
   - Con: More complex macro implementation, less common use case
   - Alternative: Nested `@with_pool` blocks (current approach)
   - > **[AI Review]**: The tuple syntax is elegant but maybe over-engineering for V1.

---

## 13. Summary: What Changes Where

### src/ Changes (Phase 1)

| File | Changes |
|------|---------|
| types.jl | Add abstract types, inherit from them |
| acquire.jl | Add `allocate_vector`, `wrap_array` dispatch points; generalize signatures |
| state.jl | Generalize to `AbstractTypedPool` |
| macros.jl | Add `_get_pool_for_backend(::Val{:cpu})` dispatch (NOT Dict registry) |
| Others | No changes |

### ext/ New Files (Phase 2)

| File | Lines | Content |
|------|-------|---------|
| AdaptiveArrayPoolsCUDAExt.jl | ~20 | Module, imports, exports |
| types.jl | ~50 | CuTypedPool (no views field!), CuAdaptiveArrayPool (+ Float16, device_id) |
| acquire.jl | ~30 | GPU-specific `get_view!` (no view caching) |
| dispatch.jl | ~35 | allocate_vector, wrap_array, get_typed_pool! (with checkpoint correction) |
| task_local_pool.jl | ~25 | get_task_local_cuda_pool (multi-device aware, public API) |
| macros.jl | ~25 | @with_cuda_pool |
| **Total** | **~155** | |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2024-12-10 | Initial draft |
| 0.2 | 2024-12-10 | Redesigned with parametric abstract types for maximum code reuse |
| 0.3 | 2024-12-10 | Unified macro design (`@with_pool :cuda`), comprehensive usage examples |
| 0.3.1 | 2024-12-10 | Added open questions: macro style preference, multi-backend single macro |
| 0.4 | 2024-12-10 | AI Review: Added Float16, device_id, multi-device pool getter, stream sync warning |
| 0.5 | 2024-12-10 | Merged AI feedback with restored full documentation |
| 0.6 | 2024-12-14 | **Post-Review Revision**: (1) Fixed GPU view type—`view(CuVector,1:n)` returns `CuVector` via GPUArrays `derive()`, not `SubArray`; simplified pool design by removing view caching. (2) Replaced Dict registry with `Val{:backend}` dispatch for zero-overhead backend selection. (3) Added checkpoint correction to GPU `get_typed_pool!` fallback. (4) Fixed `device_id` to use public API `CUDA.deviceid()`. (5) Clarified `empty!` semantics (reference removal ≠ VRAM release). (6) Documented `resize!` cost on GPU. (7) Expanded stream synchronization safety documentation. (8) Fixed backend symbol literal requirement (removed invalid `GPU_BACKEND` variable example). |
