# Vector Resize Memory Behavior: CPU vs GPU

## Context
AdaptiveArrayPools uses backing vectors that may need to grow when larger arrays are requested.
Current implementation only grows vectors, never shrinks them.

**Question**: Should we shrink vectors when smaller sizes are requested? What are the memory implications?

---

## CPU Julia Vector Behavior

```julia
v = Vector{Float64}(undef, 1000)
resize!(v, 100)   # Shrink to 100 elements
resize!(v, 500)   # Grow back to 500
```

### Key Facts (needs verification):
1. **Capacity vs Length**: Does Julia Vector maintain separate capacity?
2. **Shrink behavior**: Does `resize!(v, smaller)` release memory immediately?
3. **Regrow cost**: If we shrink then grow again, is there reallocation?

### My Understanding:
- Julia's `Vector` uses a growth strategy (typically 2x)
- `resize!` to smaller size may NOT release memory (keeps capacity)
- Growing back within capacity is O(1), no allocation
- Memory is only released when Vector is GC'd

**Question for review**: Is this accurate? Does Julia guarantee capacity preservation on shrink?

---

## GPU CuVector Behavior

```julia
using CUDA
v = CUDA.zeros(Float64, 1000)
resize!(v, 100)   # Shrink - what happens to GPU memory?
resize!(v, 500)   # Grow back - allocation?
```

### VERIFIED: CUDA.jl resize! Implementation (src/array.jl:889)

**CuVector has capacity tracking via `A.maxsize` field.**

```julia
# CUDA.jl constants
const RESIZE_THRESHOLD = 100 * 1024^2     # 100 MiB
const RESIZE_INCREMENT = 32  * 1024^2     # 32  MiB

function Base.resize!(A::CuVector{T}, n::Integer) where T
  n == length(A) && return A

  # only resize when the new length exceeds the capacity or is much smaller
  cap = A.maxsize ÷ aligned_sizeof(T)
  if n > cap || n < cap ÷ 4    # ← SHRINK THRESHOLD: 25%
    len = if n < cap
      # shrink to fit (allocates EXACT new size, no over-allocation)
      n
    elseif A.maxsize > RESIZE_THRESHOLD
      # large arrays (>100MB): grow by fixed +32 MiB increments
      max(n, cap + RESIZE_INCREMENT ÷ aligned_sizeof(T))
    else
      # small arrays (<100MB): double in size
      max(n, 2 * length(A))
    end
    # ... allocates new buffer, copies data ...
  end
  # If within capacity: just update length, no reallocation
end
```

### Key Findings:

| Aspect | CUDA.jl CuVector |
|--------|------------------|
| **Capacity tracking** | Yes, via `A.maxsize` |
| **Shrink threshold** | `n < cap ÷ 4` (25%) |
| **Shrink behavior** | Reallocates to EXACT new size |
| **Growth (small <100MB)** | 2x doubling |
| **Growth (large ≥100MB)** | +32 MiB increments |

### CUDA.jl Memory Management:
- CUDA.jl uses a memory pool (stream-ordered or binned allocator)
- Released memory goes back to pool, not immediately to OS/driver
- `CUDA.reclaim()` forces return to driver
- Pool may return same block on regrow (observed in verification tests)

---

## Current Pool Design Trade-offs

### Current Approach: Never Shrink
```julia
# In get_view!:
if length(vec) < total_len
    resize!(vec, total_len)  # Only grow, never shrink
end
new_view = view(vec, 1:total_len)  # View handles size
```

**Pros**:
- Simple implementation
- Avoids any potential reallocation costs
- Views already handle returning correct size

**Cons**:
- One large allocation permanently increases memory footprint
- GPU memory is precious and limited
- No way to recover memory without `empty!(pool)`

### Alternative: Shrink When Significantly Smaller
```julia
if length(vec) < total_len
    resize!(vec, total_len)
elseif length(vec) > total_len * 4  # Example: 4x threshold
    resize!(vec, total_len)  # Shrink to save memory
end
```

**Pros**:
- Recovers memory from outlier large allocations
- Better memory efficiency over time

**Cons**:
- May cause reallocations
- Added complexity
- Need to invalidate cached views on shrink too

---

## Specific Questions for Review

1. **Julia Vector capacity**:
   - Does `resize!(v, smaller)` preserve capacity?
   - Is this behavior documented/guaranteed?
   - Is there a way to query capacity vs length?

2. **CuVector resize behavior**:
   - Does CUDA.jl's CuVector follow same capacity model?
   - What happens to GPU memory on shrink?
   - Does CUDA memory pool make shrink "free" anyway?

3. **Design recommendation**:
   - Should pools shrink vectors at some threshold?
   - What threshold makes sense? (2x? 4x? 10x?)
   - Should CPU and GPU have different policies?

4. **Memory pressure handling**:
   - Should pool respond to memory pressure signals?
   - Is there a way to detect "memory is tight"?

---

## Test Code to Verify Behavior

```julia
# CPU Test
function test_cpu_resize_behavior()
    v = Vector{Float64}(undef, 10_000_000)  # ~80MB
    @show Base.summarysize(v)

    resize!(v, 100)
    @show Base.summarysize(v)  # Does this shrink?

    resize!(v, 5_000_000)
    @show Base.summarysize(v)  # Reallocation needed?

    # Is there a way to check capacity?
end

# GPU Test
function test_gpu_resize_behavior()
    CUDA.reclaim()  # Start clean

    v = CUDA.zeros(Float64, 10_000_000)  # ~80MB GPU
    @show CUDA.memory_status()

    resize!(v, 100)
    @show CUDA.memory_status()  # Memory returned to pool?

    resize!(v, 5_000_000)
    @show CUDA.memory_status()  # New allocation?
end
```

---

## Related: View Cache Invalidation

Currently, when `resize!` is called (grow only), we invalidate all cached views:

```julia
if length(vec) < total_len
    resize!(vec, total_len)
    # Invalidate all N-way cache entries for this slot
    for k in 1:CUDA_CACHE_WAYS
        @inbounds tp.views[base + k] = nothing
        @inbounds tp.view_dims[base + k] = nothing
    end
end
```

If we add shrinking, same invalidation would be needed since shrink can also reallocate.

---

## Summary

### VERIFIED Results

| Aspect | CPU Vector | GPU CuVector |
|--------|------------|--------------|
| **Capacity tracking** | Yes (implicit) | Yes (`A.maxsize`) |
| **Capacity preservation on shrink** | Yes (pointer unchanged) | No (reallocates at 25%) |
| **Memory returned on shrink** | No (until GC) | To pool (can be reclaimed) |
| **Regrow cost after shrink** | O(1) within capacity | May realloc (pool often returns same block) |
| **CUDA.jl shrink threshold** | N/A | `n < cap ÷ 4` (25%) |

### Design Recommendation for AdaptiveArrayPools

**Current "never shrink" is suboptimal for GPU.** CUDA.jl already implements a 25% threshold, meaning:

1. **Our explicit `resize!(vec, smaller)` calls would trigger CUDA.jl's internal shrink anyway** if below 25%
2. **We're just deferring the inevitable reallocation** when usage drops significantly
3. **GPU memory is precious** - holding 4x+ more than needed is wasteful

**Recommendation**: Add lazy shrink for GPU at 25% threshold (matching CUDA.jl):

```julia
# In get_view! for CuTypedPool:
cap = length(vec)
if total_len > cap
    resize!(vec, total_len)  # Grow
    # invalidate cache...
elseif total_len < cap ÷ 4
    resize!(vec, total_len)  # Shrink when using <25% capacity
    # invalidate cache...
end
```

**Why 25%?**
- Matches CUDA.jl's internal threshold
- Consistent behavior - calling resize! directly would shrink at same point
- Allows 4x variation without reallocation (handles typical size fluctuations)
- Recovers memory from outlier large allocations
