# ==============================================================================
# State Management for CUDA Pools
# ==============================================================================
# checkpoint!, rewind!, reset!, empty! implementations for CuAdaptiveArrayPool{S}.
# Note: _checkpoint_typed_pool! and _rewind_typed_pool! already work with
# AbstractTypedPool, so they work for CuTypedPool automatically.
# S parameter is threaded through rewind paths for compile-time safety dispatch.

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
@generated function AdaptiveArrayPools.foreach_fixed_slot(f::F, pool::CuAdaptiveArrayPool{S}) where {F, S}
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in GPU_FIXED_SLOT_FIELDS]
    return quote
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
    return nothing
end

# Type-specific checkpoint (multiple types)
@generated function AdaptiveArrayPools.checkpoint!(pool::CuAdaptiveArrayPool{S}, types::Type...) where {S}
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
    return quote
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

function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool{S}) where {S}
    cur_depth = pool._current_depth

    # Safety guard: at global scope (depth=1), delegate to reset!
    if cur_depth == 1
        reset!(pool)
        return nothing
    end

    # Fixed slots — pass S for compile-time safety dispatch
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _rewind_typed_pool!(tp, cur_depth, S)
    end

    # Others
    for tp in values(pool.others)
        _rewind_typed_pool!(tp, cur_depth, S)
    end

    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1

    return nothing
end

# Type-specific rewind (single type)
@inline function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool{S}, ::Type{T}) where {S, T}
    if pool._current_depth == 1
        reset!(AdaptiveArrayPools.get_typed_pool!(pool, T), S)
        return nothing
    end
    _rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth, S)
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

# Type-specific rewind (multiple types)
@generated function AdaptiveArrayPools.rewind!(pool::CuAdaptiveArrayPool{S}, types::Type...) where {S}
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    rewind_exprs = [:(_rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth, S)) for i in reverse(unique_indices)]
    reset_exprs = [:(reset!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), S)) for i in unique_indices]
    return quote
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
    return nothing
end

@inline function AdaptiveArrayPools._lazy_rewind!(pool::CuAdaptiveArrayPool{S}) where {S}
    d = pool._current_depth
    mask = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    _has_bit(mask, Float64)    && _rewind_typed_pool!(pool.float64, d, S)
    _has_bit(mask, Float32)    && _rewind_typed_pool!(pool.float32, d, S)
    _has_bit(mask, Int64)      && _rewind_typed_pool!(pool.int64, d, S)
    _has_bit(mask, Int32)      && _rewind_typed_pool!(pool.int32, d, S)
    _has_bit(mask, ComplexF64) && _rewind_typed_pool!(pool.complexf64, d, S)
    _has_bit(mask, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d, S)
    _has_bit(mask, Bool)       && _rewind_typed_pool!(pool.bool, d, S)
    # Bit 7: Float16 (CUDA reassignment — _fixed_slot_bit(Float16)==0, must use explicit bit check)
    mask & _cuda_float16_bit() != 0 && _rewind_typed_pool!(pool.float16, d, S)
    if @inbounds(pool._touched_has_others[d])
        for tp in values(pool.others)
            _rewind_typed_pool!(tp, d, S)
        end
    end
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
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
    return nothing
end

# _typed_lazy_rewind!: selective rewind of (tracked | touched) mask.
# Uses direct field access with bit checks — foreach_fixed_slot is single-argument (no bit yield).
# Bit 7: Float16 (CUDA-specific; lazy-checkpointed on first touch by _record_type_touch!).
# has_others: genuine others types (UInt8, Int8, etc.) — eagerly checkpointed at scope entry.
@inline function AdaptiveArrayPools._typed_lazy_rewind!(pool::CuAdaptiveArrayPool{S}, tracked_mask::UInt16) where {S}
    d = pool._current_depth
    touched = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    combined = tracked_mask | touched
    _has_bit(combined, Float64)    && _rewind_typed_pool!(pool.float64, d, S)
    _has_bit(combined, Float32)    && _rewind_typed_pool!(pool.float32, d, S)
    _has_bit(combined, Int64)      && _rewind_typed_pool!(pool.int64, d, S)
    _has_bit(combined, Int32)      && _rewind_typed_pool!(pool.int32, d, S)
    _has_bit(combined, ComplexF64) && _rewind_typed_pool!(pool.complexf64, d, S)
    _has_bit(combined, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d, S)
    _has_bit(combined, Bool)       && _rewind_typed_pool!(pool.bool, d, S)
    # Float16: bit 7 is set by _record_type_touch! on first touch (lazy first-touch).
    # Also rewind when Float16 was a *tracked* type in the macro: _typed_lazy_checkpoint!
    # calls checkpoint!(pool, Float16) which pushes a checkpoint at depth d, but _acquire_impl!
    # (macro transform) bypasses _record_type_touch!, leaving bit 7 = 0.
    # _tracked_mask_for_types(Float16) == 0 (since _fixed_slot_bit(Float16) == 0), so
    # tracked_mask carries no bit for Float16 either.
    # Solution: check _checkpoint_depths to detect "Float16 was checkpointed at this depth".
    if combined & _cuda_float16_bit() != 0 || @inbounds(pool.float16._checkpoint_depths[end]) == d
        _rewind_typed_pool!(pool.float16, d, S)
    end
    if @inbounds(pool._touched_has_others[d])
        for tp in values(pool.others)
            _rewind_typed_pool!(tp, d, S)
        end
    end
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

# ==============================================================================
# reset! for CuAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.reset!(pool::CuAdaptiveArrayPool{S}) where {S}
    # Fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        reset!(tp, S)
    end

    # Others
    for tp in values(pool.others)
        reset!(tp, S)
    end

    # Reset depth and bitmask sentinel state
    pool._current_depth = 1
    empty!(pool._touched_type_masks)
    push!(pool._touched_type_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._touched_has_others)
    push!(pool._touched_has_others, false)         # Sentinel: no others

    # Reset borrow tracking state
    pool._pending_callsite = ""
    pool._pending_return_site = ""
    pool._borrow_log = nothing

    return pool
end

# Type-specific reset
@inline function AdaptiveArrayPools.reset!(pool::CuAdaptiveArrayPool{S}, ::Type{T}) where {S, T}
    reset!(AdaptiveArrayPools.get_typed_pool!(pool, T), S)
    return pool
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
    empty!(tp.arr_wrappers)
    empty!(tp.slot_extents)
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

    # Reset borrow tracking state
    pool._pending_callsite = ""
    pool._pending_return_site = ""
    pool._borrow_log = nothing

    return pool
end

# ==============================================================================
# trim! for CuAdaptiveArrayPool (parity with CPU/Metal)
# ==============================================================================

# CUDA backing storage lives on the GPU. Use `maxsize` (allocated device-buffer
# bytes), NOT `sizeof`: in runtime-check mode (S>=1), rewind shrinks a released
# slot's logical dims to 0 via `_resize_to_fit!` while PRESERVING `maxsize`, so
# `sizeof` (logical length) would report 0 for a buffer that still holds device
# capacity. `Base.summarysize` only sees the small CPU-side handle.
function AdaptiveArrayPools._inactive_storage_bytes(tp::CuTypedPool, first::Int, last::Int)
    total = 0
    for i in first:last
        @inbounds total += getfield(tp.vectors[i], :maxsize)
    end
    return total
end

function AdaptiveArrayPools.trim!(pool::CuAdaptiveArrayPool; force_gc::Bool = false)
    # Mirror the CPU type-stable accumulation, but reuse the GPU-specific
    # `foreach_fixed_slot` (different fixed-slot field set: float16, no Bit).
    # `Ref` accumulators keep the closure box-free, so the summary stays concrete.
    slots = Ref(0)
    wrappers = Ref(0)
    bytes = Ref(0)
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        c = AdaptiveArrayPools._trim_counts!(tp)
        slots[] += c[1]
        wrappers[] += c[2]
        bytes[] += c[3]
    end
    oc = AdaptiveArrayPools._trim_others_counts!(values(pool.others))
    # force_gc on CUDA also returns CUDA.jl's pooled blocks to the driver
    # (CUDA.reclaim()), making freed VRAM available for reallocation. As the main
    # docstring notes, immediate OS/VRAM return is still not guaranteed — the
    # driver may keep the blocks in its own pool.
    force_gc && (GC.gc(); CUDA.reclaim())
    return AdaptiveArrayPools._trim_summary(
        (slots[] + oc[1], wrappers[] + oc[2], bytes[] + oc[3]), force_gc,
    )
end

# Guard + trim for one element type on CUDA. Mirrors the CPU `_trim_one_counts!`
# but uses the CUDA fixed-type set; shared by the single-type and varargs forms.
# Never creates a pool for a never-used fallback type (get_typed_pool! would
# register a new fallback pool on a miss).
@inline function AdaptiveArrayPools._trim_one_counts!(pool::CuAdaptiveArrayPool, ::Type{T})::NTuple{3, Int} where {T}
    (!(T <: _CUDA_FIXED_TYPES) && !haskey(pool.others, T)) && return (0, 0, 0)
    return AdaptiveArrayPools._trim_counts!(AdaptiveArrayPools.get_typed_pool!(pool, T))
end

@inline function AdaptiveArrayPools.trim!(pool::CuAdaptiveArrayPool, ::Type{T}; force_gc::Bool = false) where {T}
    counts = AdaptiveArrayPools._trim_one_counts!(pool, T)
    # force_gc on CUDA also returns CUDA.jl's pooled blocks to the driver
    # (CUDA.reclaim()), making freed VRAM available for reallocation. As the main
    # docstring notes, immediate OS/VRAM return is still not guaranteed — the
    # driver may keep the blocks in its own pool.
    force_gc && (GC.gc(); CUDA.reclaim())
    return AdaptiveArrayPools._trim_summary(counts, force_gc)
end

# Varargs form — parity with `checkpoint!`/`rewind!`, which also expose a `Type...`
# overload on CUDA. Unrolled at compile time; one GC.gc()/CUDA.reclaim() after all
# listed types are detached. Stays type-stable via `_trim_one_counts!`'s annotation.
@generated function AdaptiveArrayPools.trim!(pool::CuAdaptiveArrayPool, types::Type...; force_gc::Bool = false)
    n = length(types)
    syms = [Symbol(:c, i) for i in 1:n]
    assigns = [:($(syms[i]) = AdaptiveArrayPools._trim_one_counts!(pool, types[$i])) for i in 1:n]
    counts_expr = n == 0 ? :((0, 0, 0)) :
        Expr(:tuple, (Expr(:call, :+, (:($(s)[$j]) for s in syms)...) for j in 1:3)...)
    return quote
        $(assigns...)
        counts = $counts_expr
        force_gc && (GC.gc(); CUDA.reclaim())
        return AdaptiveArrayPools._trim_summary(counts, force_gc)
    end
end

# ==============================================================================
# compact! for CuAdaptiveArrayPool (parity with CPU / Metal)
# ==============================================================================
#
# Shrinks an over-allocated device buffer in place. Like Metal (and unlike CPU),
# CUDA has no uncached views — acquire_view! == acquire! → cached CuArray wrapper —
# so a slot's live references are exactly its cached wrappers; compaction swaps the
# backing buffer's DataRef and re-syncs those wrappers via the same refcount helper
# used on grow (`_update_cuda_wrapper_data!`). `slot_extents` (set in
# `_cuda_claim_slot!`) gives the live extent, matching the CPU `_slot_used` model.
# Device capacity uses `_aligned_sizeof(T)` (CUDA buffer alignment), like `_resize_to_fit!`.

# Device capacity (elements) of a CUDA backing buffer.
@inline AdaptiveArrayPools._slot_capacity(v::CuVector{T}) where {T} =
    Int(getfield(v, :maxsize) ÷ _aligned_sizeof(T))

# Current logical extent of a slot (records both acquire!/acquire_view! since CUDA
# routes both through `_cuda_claim_slot!`). `::Int` keeps the gate allocation-free.
@inline function AdaptiveArrayPools._slot_used(tp::CuTypedPool, slot::Int)::Int
    ext = tp.slot_extents
    return slot <= length(ext) ? (@inbounds ext[slot]) : 0
end

# Keep `slot_extents` parallel to `vectors` when `trim!` truncates a CUDA pool.
@inline function AdaptiveArrayPools._trim_slot_extents!(tp::CuTypedPool, keep::Int)
    length(tp.slot_extents) > keep && resize!(tp.slot_extents, keep)
    return nothing
end

# Shrink one slot's device buffer to capacity `target` in place: allocate a smaller
# CuVector, copy the live `used` elements, swap the backing's DataRef (keeping the
# backing object's identity), re-sync cached wrappers, and drop the temporary's own
# reference. All buffer ownership goes through `_update_cuda_wrapper_data!`
# (unsafe_free! old + copy/retain new), so the old device buffer's refcount reaches
# zero and CUDA frees it.
# `copy_live` (parity with CPU `_compact_slot!`): copy the `used` live elements into the
# new device buffer. TRUE for ACTIVE slots — still held, must survive the shrink. FALSE
# for INACTIVE slots: their contents are dead (the next `acquire!` returns uninitialized
# device memory the caller fills), so the GPU→GPU `copyto!` is pure waste; skipping it
# makes inactive compaction allocate-only (no device copy / kernel launch).
function AdaptiveArrayPools._compact_slot!(tp::CuTypedPool{T}, slot::Int, target::Int, used::Int, copy_live::Bool) where {T}
    v = @inbounds tp.vectors[slot]
    nv = CuVector{T}(undef, target)
    copy_live && copyto!(nv, 1, v, 1, used)
    _update_cuda_wrapper_data!(v, nv)           # backing takes nv's buffer (refcounted)
    setfield!(v, :dims, (target,))
    for wrappers in tp.arr_wrappers
        wrappers === nothing && continue
        slot <= length(wrappers) || continue
        w = @inbounds wrappers[slot]
        w === nothing && continue
        _update_cuda_wrapper_data!(w, v)        # held wrapper re-points to new buffer (dims kept)
    end
    unsafe_free!(getfield(nv, :data))           # drop the temporary's own reference
    return nothing
end

# Gate + compact one CUDA slot. Returns reclaimed device bytes (0 if skipped).
function AdaptiveArrayPools._maybe_compact_slot!(tp::CuTypedPool{T}, slot::Int, factor::Real, shrink_to::Real, min_bytes::Int) where {T}
    used = AdaptiveArrayPools._slot_used(tp, slot)
    used == 0 && return 0
    # RUNTIME_CHECK (S=1) invalidation poisons a released slot AND shrinks its logical length
    # to 0 (device buffer retained) so an escaped view reads NaN/typemax. Compacting it would
    # swap in fresh device storage and UNDO the poison — skip it (compacts once re-acquired).
    # Parity with CPU `_maybe_compact_slot!`; `length == 0 && used > 0` ⟺ poisoned.
    @inbounds(length(tp.vectors[slot])) == 0 && return 0
    cap = AdaptiveArrayPools._slot_capacity(@inbounds tp.vectors[slot])
    cap >= factor * used || return 0
    target = max(used, ceil(Int, min(Float64(cap), shrink_to * used)))   # clamp to [used, cap]
    reclaim = (cap - target) * _aligned_sizeof(T)
    reclaim >= min_bytes || return 0
    # Copy live data only for ACTIVE slots (1:n_active); inactive slots hold dead data.
    AdaptiveArrayPools._compact_slot!(tp, slot, target, used, slot <= tp.n_active)
    return reclaim
end

# Guard + compact for one element type on CUDA (mirrors `_trim_one_counts!`): never
# creates a pool for a never-used fallback type.
@inline function AdaptiveArrayPools._compact_one_counts!(pool::CuAdaptiveArrayPool, ::Type{T}, factor::Real, shrink_to::Real, min_bytes::Int, active::Bool)::NTuple{2, Int} where {T}
    (!(T <: _CUDA_FIXED_TYPES) && !haskey(pool.others, T)) && return (0, 0)
    return AdaptiveArrayPools._compact_counts!(AdaptiveArrayPools.get_typed_pool!(pool, T), factor, shrink_to, min_bytes, active)
end

function AdaptiveArrayPools.compact!(
        pool::CuAdaptiveArrayPool;
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = true, force_gc::Bool = false,
    )
    # Mirror the CUDA `trim!`: Ref accumulators keep the `foreach_fixed_slot` closure
    # box-free over the CUDA fixed-slot set, so the returned summary stays concrete.
    slots = Ref(0)
    bytes = Ref(0)
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        c = AdaptiveArrayPools._compact_counts!(tp, factor, shrink_to, min_bytes, active)
        slots[] += c[1]
        bytes[] += c[2]
    end
    oc = AdaptiveArrayPools._compact_others_counts!(values(pool.others), factor, shrink_to, min_bytes, active)
    force_gc && (GC.gc(); CUDA.reclaim())
    return AdaptiveArrayPools._compact_summary((slots[] + oc[1], bytes[] + oc[2]), force_gc)
end

@inline function AdaptiveArrayPools.compact!(
        pool::CuAdaptiveArrayPool, ::Type{T};
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = true, force_gc::Bool = false,
    ) where {T}
    counts = AdaptiveArrayPools._compact_one_counts!(pool, T, factor, shrink_to, min_bytes, active)
    force_gc && (GC.gc(); CUDA.reclaim())
    return AdaptiveArrayPools._compact_summary(counts, force_gc)
end

@generated function AdaptiveArrayPools.compact!(
        pool::CuAdaptiveArrayPool, types::Type...;
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = true, force_gc::Bool = false,
    )
    n = length(types)
    syms = [Symbol(:c, i) for i in 1:n]
    assigns = [:($(syms[i]) = AdaptiveArrayPools._compact_one_counts!(pool, types[$i], factor, shrink_to, min_bytes, active)) for i in 1:n]
    counts_expr = n == 0 ? :((0, 0)) :
        Expr(:tuple, (Expr(:call, :+, (:($(s)[$j]) for s in syms)...) for j in 1:2)...)
    return quote
        $(assigns...)
        counts = $counts_expr
        force_gc && (GC.gc(); CUDA.reclaim())
        return AdaptiveArrayPools._compact_summary(counts, force_gc)
    end
end
