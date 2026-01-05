# Fixed Slots Iteration Automation Design Document

## 1. Problem Definition

### Current State
Iteration over fixed slot types was **manually repeated** across multiple functions:

```julia
# Inside checkpoint!
_checkpoint_typed_pool!(pool.float64, depth)
_checkpoint_typed_pool!(pool.float32, depth)
_checkpoint_typed_pool!(pool.int64, depth)
_checkpoint_typed_pool!(pool.int32, depth)
_checkpoint_typed_pool!(pool.complexf64, depth)
_checkpoint_typed_pool!(pool.complexf32, depth)
_checkpoint_typed_pool!(pool.bool, depth)

# Inside rewind! - same pattern repeated
_rewind_typed_pool!(pool.float64, depth)
_rewind_typed_pool!(pool.float32, depth)
...

# Inside empty! - repeated again
empty!(pool.float64)
empty!(pool.float32)
...
```

### Improvement Goals
- **Centralized iteration logic**: Define once, use everywhere
- **Zero allocation**: No runtime overhead
- **IDE support preserved**: Keep struct definitions explicit

---

## 2. Design Decision

### Option B Adopted: const tuple + @generated (automate iteration only)

**Core principle**: Keep struct definition manual, automate only iteration

```julia
# 1. Keep struct explicitly defined (full IDE support)
mutable struct AdaptiveArrayPool
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    complexf32::TypedPool{ComplexF32}
    bool::TypedPool{Bool}
    others::IdDict{DataType, Any}
    _current_depth::Int
    _untracked_flags::Vector{Bool}
end

# 2. Define field names as const tuple
const FIXED_SLOT_FIELDS = (:float64, :float32, :int64, :int32, :complexf64, :complexf32, :bool)

# 3. Use @generated for zero-allocation iteration
@generated function foreach_fixed_slot(f::F, pool::AdaptiveArrayPool) where {F}
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in FIXED_SLOT_FIELDS]
    quote
        $(exprs...)
        nothing
    end
end
```

---

## 3. Detailed Implementation

### 3.1 types.jl Changes

```julia
# ==============================================================================
# Fixed Slot Configuration
# ==============================================================================

"""
    FIXED_SLOT_FIELDS

Fixed slot field names for iteration. Used by `foreach_fixed_slot`.

Note: When adding/removing fixed slots, update BOTH:
1. This tuple
2. The AdaptiveArrayPool struct definition below
"""
const FIXED_SLOT_FIELDS = (:float64, :float32, :int64, :int32, :complexf64, :complexf32, :bool)

# ==============================================================================
# AdaptiveArrayPool (explicit definition - full IDE support)
# ==============================================================================

mutable struct AdaptiveArrayPool
    # Fixed Slots: common types with zero lookup overhead
    # NOTE: Keep in sync with FIXED_SLOT_FIELDS above
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    complexf32::TypedPool{ComplexF32}
    bool::TypedPool{Bool}

    # Fallback: rare types
    others::IdDict{DataType, Any}

    # Untracked acquire detection
    _current_depth::Int
    _untracked_flags::Vector{Bool}
end

# ... constructor, get_typed_pool! etc. remain unchanged ...

# ==============================================================================
# Zero-Allocation Iteration
# ==============================================================================

"""
    foreach_fixed_slot(f, pool::AdaptiveArrayPool)

Apply function `f` to each fixed slot TypedPool.
Zero allocation via compile-time unrolling.

## Example
```julia
foreach_fixed_slot(pool) do tp
    _checkpoint_typed_pool!(tp, depth)
end
```
"""
@generated function foreach_fixed_slot(f::F, pool::AdaptiveArrayPool) where {F}
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in FIXED_SLOT_FIELDS]
    quote
        $(exprs...)
        nothing
    end
end
```

### 3.2 state.jl Changes

```julia
function checkpoint!(pool::AdaptiveArrayPool)
    pool._current_depth += 1
    push!(pool._untracked_flags, false)
    depth = pool._current_depth

    # Fixed slots - zero allocation via @generated
    foreach_fixed_slot(pool) do tp
        _checkpoint_typed_pool!(tp, depth)
    end

    # Others - fallback types
    for p in values(pool.others)
        _checkpoint_typed_pool!(p, depth)
    end
    nothing
end

function rewind!(pool::AdaptiveArrayPool)
    depth = pool._current_depth

    # Fixed slots - zero allocation
    foreach_fixed_slot(pool) do tp
        _rewind_typed_pool!(tp, depth)
    end

    # Others
    for tp in values(pool.others)
        _rewind_typed_pool!(tp, depth)
    end

    pop!(pool._untracked_flags)
    pool._current_depth -= 1
    nothing
end

function Base.empty!(pool::AdaptiveArrayPool)
    # Fixed slots
    foreach_fixed_slot(empty!, pool)

    # Others
    for tp in values(pool.others)
        empty!(tp)
    end
    empty!(pool.others)

    # Reset untracked detection state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._untracked_flags)
    push!(pool._untracked_flags, false)       # Sentinel: global scope starts with false
    pool
end
```

---

## 4. Type Add/Remove Procedure

### Adding UInt8

**Locations requiring manual update (2 places)**:

```julia
# 1. Update FIXED_SLOT_FIELDS
const FIXED_SLOT_FIELDS = (:float64, :float32, :int64, :int32, :complexf64, :complexf32, :bool, :uint8)

# 2. Update AdaptiveArrayPool struct
mutable struct AdaptiveArrayPool
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    complexf32::TypedPool{ComplexF32}
    bool::TypedPool{Bool}
    uint8::TypedPool{UInt8}      # ← Added
    ...
end

# 3. Update constructor
function AdaptiveArrayPool()
    AdaptiveArrayPool(
        TypedPool{Float64}(),
        ...
        TypedPool{UInt8}(),      # ← Added
        ...
    )
end

# 4. Add get_typed_pool! dispatch
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{UInt8}) = p.uint8
```

**Automatically updated**:
- `checkpoint!` internal iteration
- `rewind!` internal iteration
- `empty!` internal iteration
- All code using `foreach_fixed_slot`

---

## 5. Testing Strategy

```julia
@testset "Fixed Slot Iteration" begin
    pool = AdaptiveArrayPool()

    # Verify FIXED_SLOT_FIELDS and struct synchronization
    for field in FIXED_SLOT_FIELDS
        @test hasfield(AdaptiveArrayPool, field)
        @test getfield(pool, field) isa TypedPool
    end

    # Verify foreach_fixed_slot visits all slots
    count = Ref(0)
    foreach_fixed_slot(pool) do tp
        count[] += 1
    end
    @test count[] == length(FIXED_SLOT_FIELDS)

    # Zero allocation verification
    pool2 = AdaptiveArrayPool()
    foreach_fixed_slot(identity, pool2)  # warmup
    allocs = @allocated foreach_fixed_slot(identity, pool2)
    @test allocs == 0
end
```

---

## 6. Benefits

### 6.1 Full IDE Support
- Explicit struct definition → autocomplete, Go to Definition work correctly
- Perfect type inference
- LSP/Language Server compatible

### 6.2 Simple Implementation
- Single `@generated` function automates iteration
- No `@eval` needed → no precompilation concerns
- Most existing code preserved

### 6.3 Easy Debugging
- Clear struct definition allows field inspection in debugger
- Compatible with tools like `@infiltrate`

### 6.4 Zero Runtime Overhead
```julia
# @generated unrolls at compile time:
# foreach_fixed_slot(f, pool) is equivalent to:
f(pool.float64)
f(pool.float32)
f(pool.int64)
f(pool.int32)
f(pool.complexf64)
f(pool.complexf32)
f(pool.bool)
```

---

## 7. Drawbacks and Considerations

### 7.1 Synchronization Required (2 places)
```julia
# These two locations must always be in sync:
const FIXED_SLOT_FIELDS = (:float64, :float32, ...)  # 1
mutable struct AdaptiveArrayPool                      # 2
    float64::TypedPool{Float64}
    ...
end
```

**Mitigation**: Explicit comments + test verification

### 7.2 @generated First-Call Cost
```julia
# Recompiles for different closures
foreach_fixed_slot(x -> checkpoint!(x, 1), pool)  # Compiles
foreach_fixed_slot(x -> rewind!(x, 1), pool)      # Compiles again
```

**Impact**: Slight effect on TTFX (Time To First X)
**Mitigation**: Warmup (precompile) at package load

### 7.3 Metaprogramming Knowledge Required
```julia
@generated function foreach_fixed_slot(f::F, pool) where {F}
    # Understanding this code requires @generated knowledge
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in FIXED_SLOT_FIELDS]
    ...
end
```

**Mitigation**: Thorough comments and docstrings

---

## 8. Option Comparison Summary

| Aspect | Current (Manual) | Option B (Adopted) | Option C (@eval) |
|--------|------------------|-------------------|------------------|
| Modification locations | 6+ places | 2 places + α | 1 place |
| IDE support | Perfect | Perfect | Partial |
| Complexity | Low | Low | High |
| Debugging | Easy | Easy | Difficult |
| Type addition safety | May miss | Test-verified | Automatic |

---

## 9. Conclusion

**Reasons for adopting Option B**:

1. **Practical balance**: Removes repetitive code without the complexity of full automation (Option C)
2. **IDE support preserved**: Maintains the most important developer experience
3. **Low risk**: Uses only `@generated` without `@eval`, ensuring precompilation stability
4. **Incremental improvement**: Improves only iteration while preserving most existing code

Since type changes are rare (1-2 times during package lifetime), and struct definition synchronization across 2 locations can be sufficiently verified by tests, Option B is the optimal choice.
