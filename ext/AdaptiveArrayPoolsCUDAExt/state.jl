# ==============================================================================
# State Management for CUDA Pools
# ==============================================================================
# checkpoint!, rewind!, reset!, empty! implementations for CuAdaptiveArrayPool.
# Note: _checkpoint_typed_pool! and _rewind_typed_pool! already work with
# AbstractTypedPool, so they work for CuTypedPool automatically.

using AdaptiveArrayPools: checkpoint!, rewind!, reset!,
                          _checkpoint_typed_pool!, _rewind_typed_pool!

# ==============================================================================
# GPU Fixed Slot Iteration
# ==============================================================================

"""
    foreach_fixed_slot(f, pool::CuAdaptiveArrayPool)

Apply `f` to each fixed slot CuTypedPool. Zero allocation via compile-time unrolling.
"""
@generated function AdaptiveArrayPools.foreach_fixed_slot(f::F, pool::CuAdaptiveArrayPool) where {F}
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in GPU_FIXED_SLOT_FIELDS]
    quote
        Base.@_inline_meta
        $(exprs...)
        nothing
    end
end

# ==============================================================================
# checkpoint! for CuAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.checkpoint!(pool::CuAdaptiveArrayPool)
    # Increment depth and initialize untracked bitmask state
    pool._current_depth += 1
    push!(pool._untracked_fixed_masks, UInt16(0))
    push!(pool._untracked_has_others, false)
    depth = pool._current_depth

    # Fixed slots - zero allocation via @generated iteration
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _checkpoint_typed_pool!(tp, depth)
    end

    # Others - iterate without allocation
    for p in values(pool.others)
        _checkpoint_typed_pool!(p, depth)
    end

    return nothing
end

# Type-specific checkpoint (single type)
@inline function AdaptiveArrayPools.checkpoint!(pool::CuAdaptiveArrayPool, ::Type{T}) where {T}
    pool._current_depth += 1
    push!(pool._untracked_fixed_masks, UInt16(0))
    push!(pool._untracked_has_others, false)
    _checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth)
    nothing
end

# Type-specific checkpoint (multiple types)
@generated function AdaptiveArrayPools.checkpoint!(pool::CuAdaptiveArrayPool, types::Type...)
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    checkpoint_exprs = [:(_checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in unique_indices]
    quote
        pool._current_depth += 1
        push!(pool._untracked_fixed_masks, UInt16(0))
        push!(pool._untracked_has_others, false)
        $(checkpoint_exprs...)
        nothing
    end
end

# ==============================================================================
# rewind! for CuAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool)
    cur_depth = pool._current_depth

    # Safety guard: at global scope (depth=1), delegate to reset!
    if cur_depth == 1
        reset!(pool)
        return nothing
    end

    # Fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _rewind_typed_pool!(tp, cur_depth)
    end

    # Others
    for tp in values(pool.others)
        _rewind_typed_pool!(tp, cur_depth)
    end

    pop!(pool._untracked_fixed_masks)
    pop!(pool._untracked_has_others)
    pool._current_depth -= 1

    return nothing
end

# Type-specific rewind (single type)
@inline function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool, ::Type{T}) where {T}
    if pool._current_depth == 1
        reset!(AdaptiveArrayPools.get_typed_pool!(pool, T))
        return nothing
    end
    _rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth)
    pop!(pool._untracked_fixed_masks)
    pop!(pool._untracked_has_others)
    pool._current_depth -= 1
    nothing
end

# Type-specific rewind (multiple types)
@generated function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool, types::Type...)
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    rewind_exprs = [:(_rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in reverse(unique_indices)]
    reset_exprs = [:(reset!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]))) for i in unique_indices]
    quote
        if pool._current_depth == 1
            $(reset_exprs...)
            return nothing
        end
        $(rewind_exprs...)
        pop!(pool._untracked_fixed_masks)
        pop!(pool._untracked_has_others)
        pool._current_depth -= 1
        nothing
    end
end

# ==============================================================================
# reset! for CuAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.reset!(pool::CuAdaptiveArrayPool)
    # Fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        reset!(tp)
    end

    # Others
    for tp in values(pool.others)
        reset!(tp)
    end

    # Reset depth and bitmask sentinel state
    pool._current_depth = 1
    empty!(pool._untracked_fixed_masks)
    push!(pool._untracked_fixed_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._untracked_has_others)
    push!(pool._untracked_has_others, false)         # Sentinel: no others

    return pool
end

# Type-specific reset
@inline function AdaptiveArrayPools.reset!(pool::CuAdaptiveArrayPool, ::Type{T}) where {T}
    reset!(AdaptiveArrayPools.get_typed_pool!(pool, T))
    pool
end

# ==============================================================================
# empty! for CuTypedPool and CuAdaptiveArrayPool
# ==============================================================================

"""
    empty!(tp::CuTypedPool)

Clear all GPU storage. Note: This removes Julia references to CuVectors.
Actual VRAM release depends on GC + CUDA.jl's memory pool.

For immediate VRAM release:
```julia
empty!(pool)
GC.gc()
CUDA.reclaim()
```
"""
function Base.empty!(tp::CuTypedPool)
    empty!(tp.vectors)
    empty!(tp.views)
    empty!(tp.view_dims)
    empty!(tp.next_way)
    tp.n_active = 0
    # Restore sentinel values
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)
    return tp
end

function Base.empty!(pool::CuAdaptiveArrayPool)
    # Fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        empty!(tp)
    end

    # Others - clear all then the IdDict
    for tp in values(pool.others)
        empty!(tp)
    end
    empty!(pool.others)

    # Reset depth and bitmask sentinel state
    pool._current_depth = 1
    empty!(pool._untracked_fixed_masks)
    push!(pool._untracked_fixed_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._untracked_has_others)
    push!(pool._untracked_has_others, false)         # Sentinel: no others

    return pool
end
