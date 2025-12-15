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
    # Increment depth and initialize untracked flag
    pool._current_depth += 1
    push!(pool._untracked_flags, false)
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
    push!(pool._untracked_flags, false)
    _checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth)
    nothing
end

# Type-specific checkpoint (multiple types)
@generated function AdaptiveArrayPools.checkpoint!(pool::CuAdaptiveArrayPool, types::Type...)
    checkpoint_exprs = [:(_checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in 1:length(types)]
    quote
        pool._current_depth += 1
        push!(pool._untracked_flags, false)
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

    pop!(pool._untracked_flags)
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
    pop!(pool._untracked_flags)
    pool._current_depth -= 1
    nothing
end

# Type-specific rewind (multiple types)
@generated function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool, types::Type...)
    rewind_exprs = [:(_rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in length(types):-1:1]
    reset_exprs = [:(reset!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]))) for i in 1:length(types)]
    quote
        if pool._current_depth == 1
            $(reset_exprs...)
            return nothing
        end
        $(rewind_exprs...)
        pop!(pool._untracked_flags)
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

    # Reset untracked detection state
    pool._current_depth = 1
    empty!(pool._untracked_flags)
    push!(pool._untracked_flags, false)

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

    # Reset state
    pool._current_depth = 1
    empty!(pool._untracked_flags)
    push!(pool._untracked_flags, false)

    return pool
end
