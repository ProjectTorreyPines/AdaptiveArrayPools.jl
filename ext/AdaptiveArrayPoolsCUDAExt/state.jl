# ==============================================================================
# State Management for CUDA Pools
# ==============================================================================
# checkpoint!, rewind!, reset!, empty! implementations for CuAdaptiveArrayPool.
# Note: _checkpoint_typed_pool! and _rewind_typed_pool! already work with
# AbstractTypedPool, so they work for CuTypedPool automatically.

using AdaptiveArrayPools: checkpoint!, rewind!, reset!,
                          _checkpoint_typed_pool!, _rewind_typed_pool!, _has_bit,
                          _LAZY_MODE_BIT, _TYPED_LAZY_BIT, _TYPE_BITS_MASK

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
    # Increment depth and initialize type-touch tracking state
    pool._current_depth += 1
    push!(pool._touched_type_masks, UInt16(0))
    push!(pool._touched_has_others, false)
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
    push!(pool._touched_type_masks, UInt16(0))
    push!(pool._touched_has_others, AdaptiveArrayPools._fixed_slot_bit(T) == UInt16(0))
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
    has_any_fallback = any(i -> AdaptiveArrayPools._fixed_slot_bit(types[i].parameters[1]) == UInt16(0), unique_indices)
    checkpoint_exprs = [:(_checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in unique_indices]
    quote
        pool._current_depth += 1
        push!(pool._touched_type_masks, UInt16(0))
        push!(pool._touched_has_others, $has_any_fallback)
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

    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
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
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
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
        pop!(pool._touched_type_masks)
        pop!(pool._touched_has_others)
        pool._current_depth -= 1
        nothing
    end
end

# ==============================================================================
# Lazy Mode for CuAdaptiveArrayPool (use_typed=false path)
# ==============================================================================
# Mirrors CPU _lazy_checkpoint! / _lazy_rewind! in src/state.jl.
#
# Float16 on CUDA: direct struct field (not in pool.others dict), but _fixed_slot_bit(Float16)=0.
# We reassign Float16 to bit 7 (unused on CUDA; CPU uses bit 7 for Bit type which has no GPU equivalent).
# This gives Float16 the same lazy-first-touch checkpoint treatment as other fixed-slot types,
# avoiding the unsafe unconditional-rewind issue (Option B) and the has_others confusion.

# Bit 7 on CUDA is reserved for Float16 (CPU uses it for Bit; Bit type does not exist on GPU).
@inline _cuda_float16_bit() = UInt16(1) << 7

@inline function AdaptiveArrayPools._lazy_checkpoint!(pool::CuAdaptiveArrayPool)
    pool._current_depth += 1
    push!(pool._touched_type_masks, _LAZY_MODE_BIT)  # lazy mode flag
    push!(pool._touched_has_others, false)
    depth = pool._current_depth
    # Eagerly checkpoint pre-existing others entries — same as CPU _lazy_checkpoint!.
    # New types created during the scope start at n_active=0 (sentinel covers them, Case B safe).
    # Pre-existing types need their count saved now so Case A fires correctly at rewind.
    for p in values(pool.others)
        _checkpoint_typed_pool!(p, depth)
        @inbounds pool._touched_has_others[depth] = true
    end
    # Float16 uses lazy first-touch via bit 7 in _record_type_touch! — no eager checkpoint needed.
    nothing
end

@inline function AdaptiveArrayPools._lazy_rewind!(pool::CuAdaptiveArrayPool)
    d = pool._current_depth
    mask = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    _has_bit(mask, Float64)    && _rewind_typed_pool!(pool.float64,    d)
    _has_bit(mask, Float32)    && _rewind_typed_pool!(pool.float32,    d)
    _has_bit(mask, Int64)      && _rewind_typed_pool!(pool.int64,      d)
    _has_bit(mask, Int32)      && _rewind_typed_pool!(pool.int32,      d)
    _has_bit(mask, ComplexF64) && _rewind_typed_pool!(pool.complexf64, d)
    _has_bit(mask, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d)
    _has_bit(mask, Bool)       && _rewind_typed_pool!(pool.bool,       d)
    # Bit 7: Float16 (CUDA reassignment — _fixed_slot_bit(Float16)==0, must use explicit bit check)
    mask & _cuda_float16_bit() != 0 && _rewind_typed_pool!(pool.float16, d)
    if @inbounds(pool._touched_has_others[d])
        for tp in values(pool.others)
            _rewind_typed_pool!(tp, d)
        end
    end
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    nothing
end

# ==============================================================================
# Typed-Fallback Helpers for CuAdaptiveArrayPool (Phase 5 parity)
# ==============================================================================

# _typed_lazy_checkpoint!: typed checkpoint + set bit 14 for lazy extra-type tracking.
# Also eagerly snapshots pre-existing others entries (mirrors CPU fix for Issue #3).
@inline function AdaptiveArrayPools._typed_lazy_checkpoint!(pool::CuAdaptiveArrayPool, types::Type...)
    checkpoint!(pool, types...)
    d = pool._current_depth
    @inbounds pool._touched_type_masks[d] |= _TYPED_LAZY_BIT
    # Eagerly snapshot pre-existing others entries — same reasoning as _lazy_checkpoint!.
    # Skip re-snapshot for entries already checkpointed at d by checkpoint!(pool, types...)
    # (e.g. Float16 in types... was just checkpointed above — avoid double-push).
    for p in values(pool.others)
        if @inbounds(p._checkpoint_depths[end]) != d
            _checkpoint_typed_pool!(p, d)
        end
        @inbounds pool._touched_has_others[d] = true
    end
    # Float16 uses lazy first-touch via bit 7 in _record_type_touch! — no eager checkpoint needed.
    nothing
end

# _typed_lazy_rewind!: selective rewind of (tracked | touched) mask.
# Uses direct field access with bit checks — foreach_fixed_slot is single-argument (no bit yield).
# Bit 7: Float16 (CUDA-specific; lazy-checkpointed on first touch by _record_type_touch!).
# has_others: genuine others types (UInt8, Int8, etc.) — eagerly checkpointed at scope entry.
@inline function AdaptiveArrayPools._typed_lazy_rewind!(pool::CuAdaptiveArrayPool, tracked_mask::UInt16)
    d = pool._current_depth
    touched = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    combined = tracked_mask | touched
    _has_bit(combined, Float64)    && _rewind_typed_pool!(pool.float64,    d)
    _has_bit(combined, Float32)    && _rewind_typed_pool!(pool.float32,    d)
    _has_bit(combined, Int64)      && _rewind_typed_pool!(pool.int64,      d)
    _has_bit(combined, Int32)      && _rewind_typed_pool!(pool.int32,      d)
    _has_bit(combined, ComplexF64) && _rewind_typed_pool!(pool.complexf64, d)
    _has_bit(combined, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d)
    _has_bit(combined, Bool)       && _rewind_typed_pool!(pool.bool,       d)
    # Float16: bit 7 is set by _record_type_touch! on first touch (lazy first-touch).
    # Also rewind when Float16 was a *tracked* type in the macro: _typed_lazy_checkpoint!
    # calls checkpoint!(pool, Float16) which pushes a checkpoint at depth d, but _acquire_impl!
    # (macro transform) bypasses _record_type_touch!, leaving bit 7 = 0.
    # _tracked_mask_for_types(Float16) == 0 (since _fixed_slot_bit(Float16) == 0), so
    # tracked_mask carries no bit for Float16 either.
    # Solution: check _checkpoint_depths to detect "Float16 was checkpointed at this depth".
    if combined & _cuda_float16_bit() != 0 || @inbounds(pool.float16._checkpoint_depths[end]) == d
        _rewind_typed_pool!(pool.float16, d)
    end
    if @inbounds(pool._touched_has_others[d])
        for tp in values(pool.others)
            _rewind_typed_pool!(tp, d)
        end
    end
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    nothing
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
    empty!(pool._touched_type_masks)
    push!(pool._touched_type_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._touched_has_others)
    push!(pool._touched_has_others, false)         # Sentinel: no others

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
    empty!(pool._touched_type_masks)
    push!(pool._touched_type_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._touched_has_others)
    push!(pool._touched_has_others, false)         # Sentinel: no others

    return pool
end
