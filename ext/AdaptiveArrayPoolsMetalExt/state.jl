# ==============================================================================
# State Management for Metal Pools
# ==============================================================================
# checkpoint!, rewind!, reset!, empty! implementations for MetalAdaptiveArrayPool{R,S}.
# Note: _checkpoint_typed_pool! and _rewind_typed_pool! already work with
# AbstractTypedPool, so they work for MetalTypedPool automatically.
# R parameter is threaded through rewind paths for compile-time safety dispatch.

using AdaptiveArrayPools: checkpoint!, rewind!, reset!,
    _checkpoint_typed_pool!, _rewind_typed_pool!, _has_bit,
    _LAZY_MODE_BIT, _TYPED_LAZY_BIT, _TYPE_BITS_MASK,
    _touch_fallback_pool!, _drain_touched_others!, _truncate_touched_others!

# Genuine fallback = lives in pool.others (stack-managed). NOT equivalent to
# _fixed_slot_bit(T) == 0: Float16 has bit 0 (bit-7 reassignment) but is a fixed
# struct field — routing it through the touched-others stack would double-rewind
# it against the lazy rewinds' Float16 special case (Case A then Case B).
@inline _metal_is_fallback_type(::Type{T}) where {T} = !(T <: _METAL_FIXED_TYPES)

# ==============================================================================
# Metal Fixed Slot Iteration
# ==============================================================================

"""
    foreach_fixed_slot(f, pool::MetalAdaptiveArrayPool)

Apply `f` to each fixed slot MetalTypedPool. Zero allocation via compile-time unrolling.
"""
@generated function AdaptiveArrayPools.foreach_fixed_slot(f::F, pool::MetalAdaptiveArrayPool{R, S}) where {F, R, S}
    exprs = [:(f(getfield(pool, $(QuoteNode(field))))) for field in METAL_FIXED_SLOT_FIELDS]
    return quote
        Base.@_inline_meta
        $(exprs...)
        nothing
    end
end

# ==============================================================================
# checkpoint! for MetalAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.checkpoint!(pool::MetalAdaptiveArrayPool)
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
@inline function AdaptiveArrayPools.checkpoint!(pool::MetalAdaptiveArrayPool, ::Type{T}) where {T}
    pool._current_depth += 1
    push!(pool._touched_type_masks, UInt16(0))
    # Flag push stays bit-based (feeds _can_use_typed_path/R>=1 validation only) —
    # Float16 has bit 0 here even though it is routed as a fixed slot below.
    push!(pool._touched_has_others, AdaptiveArrayPools._fixed_slot_bit(T) == UInt16(0))
    if _metal_is_fallback_type(T)
        _touch_fallback_pool!(pool, AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth)
    else
        # Fixed slots INCLUDING Float16: direct checkpoint, never stack-managed.
        _checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth)
    end
    return nothing
end

# Type-specific checkpoint (multiple types)
@generated function AdaptiveArrayPools.checkpoint!(pool::MetalAdaptiveArrayPool{R, S}, types::Type...) where {R, S}
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    # has_any_fallback keeps its current bit-based computation (flag semantics
    # unchanged — Float16 contributes true here even though it is routed as a
    # fixed slot below via _metal_is_fallback_type).
    has_any_fallback = any(i -> AdaptiveArrayPools._fixed_slot_bit(types[i].parameters[1]) == UInt16(0), unique_indices)
    checkpoint_exprs = map(unique_indices) do i
        if !(types[i].parameters[1] <: _METAL_FIXED_TYPES)
            :(_touch_fallback_pool!(pool, AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth))
        else
            :(_checkpoint_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth))
        end
    end
    return quote
        pool._current_depth += 1
        push!(pool._touched_type_masks, UInt16(0))
        push!(pool._touched_has_others, $has_any_fallback)
        $(checkpoint_exprs...)
        nothing
    end
end

# ==============================================================================
# rewind! for MetalAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.rewind!(pool::MetalAdaptiveArrayPool{R, S}) where {R, S}
    cur_depth = pool._current_depth

    # Safety guard: at global scope (depth=1), delegate to reset!
    if cur_depth == 1
        reset!(pool)
        return nothing
    end

    # Fixed slots — pass R for compile-time safety dispatch
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _rewind_typed_pool!(tp, cur_depth, R)
    end

    # Others
    for tp in values(pool.others)
        _rewind_typed_pool!(tp, cur_depth, R)
    end
    # Full sweep above already rewound every fallback pool — truncate-only (no
    # re-rewind) to avoid double-popping the touched-others stack.
    _truncate_touched_others!(pool, cur_depth)

    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1

    return nothing
end

# Type-specific rewind (single type)
@inline function AdaptiveArrayPools.rewind!(pool::MetalAdaptiveArrayPool{R, S}, ::Type{T}) where {R, S, T}
    if pool._current_depth == 1
        reset!(AdaptiveArrayPools.get_typed_pool!(pool, T), R)
        return nothing
    end
    # Fixed slots (INCLUDING Float16) rewind directly; genuine-fallback T was
    # pushed onto the touched-others stack by checkpoint!(pool, T) and is
    # covered by the drain below.
    if !_metal_is_fallback_type(T)
        _rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, T), pool._current_depth, R)
    end
    _drain_touched_others!(pool, pool._current_depth)
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

# Type-specific rewind (multiple types)
@generated function AdaptiveArrayPools.rewind!(pool::MetalAdaptiveArrayPool{R, S}, types::Type...) where {R, S}
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    # Fixed slots INCLUDING Float16 rewind directly; genuine-fallback types were
    # pushed onto the touched-others stack by checkpoint!(pool, types...) and
    # are covered by the drain below.
    fixed_indices = [i for i in unique_indices if types[i].parameters[1] <: _METAL_FIXED_TYPES]
    rewind_exprs = [:(_rewind_typed_pool!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), pool._current_depth, R)) for i in reverse(fixed_indices)]
    reset_exprs = [:(reset!(AdaptiveArrayPools.get_typed_pool!(pool, types[$i]), R)) for i in unique_indices]
    return quote
        if pool._current_depth == 1
            $(reset_exprs...)
            return nothing
        end
        $(rewind_exprs...)
        _drain_touched_others!(pool, pool._current_depth)
        pop!(pool._touched_type_masks)
        pop!(pool._touched_has_others)
        pool._current_depth -= 1
        nothing
    end
end

# ==============================================================================
# Lazy Mode for MetalAdaptiveArrayPool (use_typed=false path)
# ==============================================================================
# Mirrors CPU _lazy_checkpoint! / _lazy_rewind! in src/state.jl.
#
# Float16 on Metal: direct struct field (not in pool.others dict), but _fixed_slot_bit(Float16)=0.
# We reassign Float16 to bit 7 (unused on Metal; CPU uses bit 7 for Bit type which has no GPU equivalent).
# This gives Float16 the same lazy-first-touch checkpoint treatment as other fixed-slot types.

# Bit 7 on Metal is reserved for Float16 (CPU uses it for Bit; Bit type does not exist on GPU).
@inline _metal_float16_bit() = UInt16(1) << 7

@inline function AdaptiveArrayPools._lazy_checkpoint!(pool::MetalAdaptiveArrayPool)
    pool._current_depth += 1
    push!(pool._touched_type_masks, _LAZY_MODE_BIT)  # lazy mode flag
    push!(pool._touched_has_others, false)
    # Fallback (non-fixed-slot) pools are NOT eagerly checkpointed here: they are
    # first-touch checkpointed via _touch_fallback_pool! (from _record_type_touch!
    # or get_typed_pool!) and drained selectively at rewind via
    # _drain_touched_others!, so only the fallback pools this scope actually
    # touches pay any cost. Float16 uses its own lazy first-touch via bit 7.
    return nothing
end

@inline function AdaptiveArrayPools._lazy_rewind!(pool::MetalAdaptiveArrayPool{R, S}) where {R, S}
    d = pool._current_depth
    mask = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    _has_bit(mask, Float32)    && _rewind_typed_pool!(pool.float32, d, R)
    _has_bit(mask, Int64)      && _rewind_typed_pool!(pool.int64, d, R)
    _has_bit(mask, Int32)      && _rewind_typed_pool!(pool.int32, d, R)
    _has_bit(mask, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d, R)
    _has_bit(mask, Bool)       && _rewind_typed_pool!(pool.bool, d, R)
    # Bit 7: Float16 (Metal reassignment — _fixed_slot_bit(Float16)==0, must use explicit bit check)
    mask & _metal_float16_bit() != 0 && _rewind_typed_pool!(pool.float16, d, R)
    _drain_touched_others!(pool, d)
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

# ==============================================================================
# Typed-Fallback Helpers for MetalAdaptiveArrayPool (Phase 5 parity)
# ==============================================================================

# _typed_lazy_checkpoint!: typed checkpoint + set bit 14 for lazy extra-type tracking.
# checkpoint!(pool, types...) already routes fallback types among `types` through
# _touch_fallback_pool! (one depth-tagged stack entry each); extra fallback types
# touched by helpers are first-touch checkpointed and stacked by
# _record_type_touch!'s genuine-fallback branch. Float16 uses lazy first-touch via
# bit 7 in _record_type_touch! — no eager checkpoint needed.
@inline function AdaptiveArrayPools._typed_lazy_checkpoint!(pool::MetalAdaptiveArrayPool, types::Type...)
    checkpoint!(pool, types...)
    d = pool._current_depth
    @inbounds pool._touched_type_masks[d] |= _TYPED_LAZY_BIT
    return nothing
end

# _typed_lazy_rewind!: selective rewind of (tracked | touched) mask.
# Uses direct field access with bit checks — foreach_fixed_slot is single-argument (no bit yield).
# Bit 7: Float16 (Metal-specific; lazy-checkpointed on first touch by _record_type_touch!).
# Genuine fallback types (UInt8, Int8, etc.) are drained selectively via
# _drain_touched_others! — the ONLY rewinder for typed-Float16 scopes stays the
# direct _checkpoint_depths[end] == d special case below (Float16 never gets a
# stack entry).
@inline function AdaptiveArrayPools._typed_lazy_rewind!(pool::MetalAdaptiveArrayPool{R, S}, tracked_mask::UInt16) where {R, S}
    d = pool._current_depth
    touched = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    combined = tracked_mask | touched
    _has_bit(combined, Float32)    && _rewind_typed_pool!(pool.float32, d, R)
    _has_bit(combined, Int64)      && _rewind_typed_pool!(pool.int64, d, R)
    _has_bit(combined, Int32)      && _rewind_typed_pool!(pool.int32, d, R)
    _has_bit(combined, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d, R)
    _has_bit(combined, Bool)       && _rewind_typed_pool!(pool.bool, d, R)
    # Float16: bit 7 is set by _record_type_touch! on first touch (lazy first-touch).
    # Also rewind when Float16 was a *tracked* type in the macro.
    if combined & _metal_float16_bit() != 0 || @inbounds(pool.float16._checkpoint_depths[end]) == d
        _rewind_typed_pool!(pool.float16, d, R)
    end
    _drain_touched_others!(pool, d)
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

# ==============================================================================
# reset! for MetalAdaptiveArrayPool
# ==============================================================================

function AdaptiveArrayPools.reset!(pool::MetalAdaptiveArrayPool{R, S}) where {R, S}
    # Fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        reset!(tp, R)
    end

    # Others
    for tp in values(pool.others)
        reset!(tp, R)
    end

    # Reset touched-others tracking (transient scope state; memo intentionally
    # survives — registered fallback identities are preserved by reset!).
    empty!(pool._touched_others_states)
    empty!(pool._touched_others_depths)
    empty!(pool._touched_others_pools)

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
@inline function AdaptiveArrayPools.reset!(pool::MetalAdaptiveArrayPool{R, S}, ::Type{T}) where {R, S, T}
    reset!(AdaptiveArrayPools.get_typed_pool!(pool, T), R)
    return pool
end

# ==============================================================================
# empty! for MetalTypedPool and MetalAdaptiveArrayPool
# ==============================================================================

"""
    empty!(tp::MetalTypedPool)

Clear all GPU storage. Note: This removes Julia references to MtlArrays.
Actual VRAM release depends on GC + Metal.jl's memory pool.
"""
function Base.empty!(tp::MetalTypedPool)
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

function Base.empty!(pool::MetalAdaptiveArrayPool)
    # Fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        empty!(tp)
    end

    # Others - clear all then the IdDict
    for tp in values(pool.others)
        empty!(tp)
    end
    empty!(pool.others)

    # Memo points into the registry being cleared — drop it with the registry.
    pool._lookup_memo_type = nothing
    pool._lookup_memo_tp = nothing
    empty!(pool._touched_others_states)
    empty!(pool._touched_others_depths)
    empty!(pool._touched_others_pools)

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
# trim! for MetalAdaptiveArrayPool (parity with CPU)
# ==============================================================================

# Metal backing storage lives on the GPU. Use `maxsize` (allocated device-buffer
# bytes), NOT `sizeof`: in runtime-check mode (R>=1), rewind shrinks a released
# slot's logical dims to 0 via `_resize_to_fit!` while PRESERVING `maxsize`, so
# `sizeof` (logical length) would report 0 for a buffer that still holds device
# capacity. `Base.summarysize` only sees the small CPU-side handle.
function AdaptiveArrayPools._inactive_storage_bytes(tp::MetalTypedPool, first::Int, last::Int)
    total = 0
    for i in first:last
        @inbounds total += getfield(tp.vectors[i], :maxsize)
    end
    return total
end

function AdaptiveArrayPools.trim!(pool::MetalAdaptiveArrayPool; force_gc::Bool = false)
    # Mirror the CPU type-stable accumulation, but reuse the Metal-specific
    # `foreach_fixed_slot` (different fixed-slot field set). `Ref` accumulators keep
    # the closure box-free, so the returned summary stays a concrete NamedTuple.
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
    force_gc && GC.gc()
    return AdaptiveArrayPools._trim_summary(
        (slots[] + oc[1], wrappers[] + oc[2], bytes[] + oc[3]), force_gc,
    )
end

# Guard + trim for one element type on Metal. Mirrors the CPU `_trim_one_counts!`
# but uses the Metal fixed-type set; shared by the single-type and varargs forms.
# Never creates a pool for a never-used fallback type (get_typed_pool! would
# register a new fallback pool on a miss).
@inline function AdaptiveArrayPools._trim_one_counts!(pool::MetalAdaptiveArrayPool, ::Type{T})::NTuple{3, Int} where {T}
    (!(T <: _METAL_FIXED_TYPES) && !haskey(pool.others, T)) && return (0, 0, 0)
    return AdaptiveArrayPools._trim_counts!(AdaptiveArrayPools.get_typed_pool!(pool, T))
end

@inline function AdaptiveArrayPools.trim!(pool::MetalAdaptiveArrayPool, ::Type{T}; force_gc::Bool = false) where {T}
    counts = AdaptiveArrayPools._trim_one_counts!(pool, T)
    force_gc && GC.gc()
    return AdaptiveArrayPools._trim_summary(counts, force_gc)
end

# Varargs form — parity with `checkpoint!`/`rewind!`, which also expose a `Type...`
# overload on Metal. Unrolled at compile time; one GC.gc() after all listed types
# are detached. Stays type-stable via `_trim_one_counts!`'s return annotation.
@generated function AdaptiveArrayPools.trim!(pool::MetalAdaptiveArrayPool, types::Type...; force_gc::Bool = false)
    n = length(types)
    syms = [Symbol(:c, i) for i in 1:n]
    assigns = [:($(syms[i]) = AdaptiveArrayPools._trim_one_counts!(pool, types[$i])) for i in 1:n]
    counts_expr = n == 0 ? :((0, 0, 0)) :
        Expr(:tuple, (Expr(:call, :+, (:($(s)[$j]) for s in syms)...) for j in 1:3)...)
    return quote
        $(assigns...)
        counts = $counts_expr
        force_gc && GC.gc()
        return AdaptiveArrayPools._trim_summary(counts, force_gc)
    end
end

# ==============================================================================
# compact! for MetalAdaptiveArrayPool (parity with CPU)
# ==============================================================================
#
# Shrinks an over-allocated device buffer in place. Unlike CPU, Metal has no
# uncached views (`acquire_view!` == `acquire!` → cached `MtlArray` wrapper), so a
# slot's live references are exactly its cached wrappers; compaction swaps the
# backing buffer's `DataRef` and re-syncs those wrappers via the same refcount
# helper used on grow (`_update_metal_wrapper_data!`). `slot_extents` (set in
# `_metal_claim_slot!`) gives the live extent, matching the CPU `_slot_used` model.

# Device capacity (elements) of a Metal backing buffer.
@inline AdaptiveArrayPools._slot_capacity(v::MtlArray{T}) where {T} =
    Int(getfield(v, :maxsize) ÷ sizeof(T))

# Current logical extent of a slot (records both acquire!/acquire_view! since Metal
# routes both through `_metal_claim_slot!`). `::Int` keeps the gate allocation-free.
@inline function AdaptiveArrayPools._slot_used(tp::MetalTypedPool, slot::Int)::Int
    ext = tp.slot_extents
    return slot <= length(ext) ? (@inbounds ext[slot]) : 0
end

# Keep `slot_extents` parallel to `vectors` when `trim!` truncates a Metal pool.
@inline function AdaptiveArrayPools._trim_slot_extents!(tp::MetalTypedPool, keep::Int)
    length(tp.slot_extents) > keep && resize!(tp.slot_extents, keep)
    return nothing
end

# Shrink one slot's device buffer to capacity `target` in place: allocate a smaller
# MtlArray, copy the live `used` elements, swap the backing's `DataRef` (keeping the
# backing object's identity), re-sync cached wrappers, and drop the temporary's own
# reference. All buffer ownership goes through `_update_metal_wrapper_data!`
# (unsafe_free! old + copy/retain new), so the old device buffer's refcount reaches
# zero and Metal frees it.
# `copy_live` (parity with CPU `_compact_slot!`): copy the `used` live elements into the
# new device buffer. TRUE for ACTIVE slots — still held, must survive the shrink. FALSE
# for INACTIVE slots: their contents are dead (the next `acquire!` returns uninitialized
# device memory the caller fills), so the GPU→GPU `copyto!` is pure waste; skipping it
# makes inactive compaction allocate-only (no device copy / kernel launch).
function AdaptiveArrayPools._compact_slot!(tp::MetalTypedPool{T, S}, slot::Int, target::Int, used::Int, copy_live::Bool) where {T, S}
    v = @inbounds tp.vectors[slot]
    old_rc = getfield(getfield(v, :data), :rc)   # OLD device-buffer identity, captured BEFORE the swap
    nv = MtlArray{T, 1, S}(undef, target)
    copy_live && copyto!(nv, 1, v, 1, used)
    _update_metal_wrapper_data!(v, nv)          # backing takes nv's buffer (refcounted)
    setfield!(v, :dims, (target,))
    # Re-point EVERY cached wrapper sharing this slot's OLD device buffer — not only the wrappers
    # cached AT `slot`. A `reshape!` alias lives in a SEPARATE placeholder slot while sharing this
    # buffer's DataRef; the per-slot scan missed it, stranding it on the freed buffer after an
    # `active=true` compact!. Match by refcount identity (the same `.data.rc` test `reshape!` uses)
    # and migrate each alias onto the new buffer via the refcount-balanced helper (dims kept).
    for wrappers in tp.arr_wrappers
        wrappers === nothing && continue
        for w in wrappers
            w === nothing && continue
            getfield(getfield(w, :data), :rc) === old_rc && _update_metal_wrapper_data!(w, v)
        end
    end
    unsafe_free!(getfield(nv, :data))           # drop the temporary's own reference
    return nothing
end

# Gate + compact one Metal slot. Returns reclaimed device bytes (0 if skipped).
function AdaptiveArrayPools._maybe_compact_slot!(tp::MetalTypedPool{T, S}, slot::Int, factor::Real, shrink_to::Real, min_bytes::Int) where {T, S}
    used = AdaptiveArrayPools._slot_used(tp, slot)
    used == 0 && return 0
    # RUNTIME_CHECK (R=1) invalidation poisons a released slot AND shrinks its logical length
    # to 0 (device buffer retained) so an escaped view reads NaN/typemax. Compacting it would
    # swap in fresh device storage and UNDO the poison — skip it (compacts once re-acquired).
    # Parity with CPU `_maybe_compact_slot!`; `length == 0 && used > 0` ⟺ poisoned.
    @inbounds(length(tp.vectors[slot])) == 0 && return 0
    cap = AdaptiveArrayPools._slot_capacity(@inbounds tp.vectors[slot])
    cap >= factor * used || return 0
    target = max(used, ceil(Int, min(Float64(cap), shrink_to * used)))   # clamp to [used, cap]
    reclaim = (cap - target) * sizeof(T)
    reclaim >= min_bytes || return 0
    # Copy live data only for ACTIVE slots (1:n_active); inactive slots hold dead data.
    AdaptiveArrayPools._compact_slot!(tp, slot, target, used, slot <= tp.n_active)
    return reclaim
end

# Guard + compact for one element type on Metal (mirrors `_trim_one_counts!`): never
# creates a pool for a never-used fallback type.
@inline function AdaptiveArrayPools._compact_one_counts!(pool::MetalAdaptiveArrayPool, ::Type{T}, factor::Real, shrink_to::Real, min_bytes::Int, active::Bool)::NTuple{2, Int} where {T}
    (!(T <: _METAL_FIXED_TYPES) && !haskey(pool.others, T)) && return (0, 0)
    return AdaptiveArrayPools._compact_counts!(AdaptiveArrayPools.get_typed_pool!(pool, T), factor, shrink_to, min_bytes, active)
end

function AdaptiveArrayPools.compact!(
        pool::MetalAdaptiveArrayPool;
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = true, force_gc::Bool = false,
    )
    # Mirror the Metal `trim!`: Ref accumulators keep the `foreach_fixed_slot`
    # closure box-free over the Metal-specific fixed-slot set, so the returned
    # summary stays a concrete NamedTuple.
    slots = Ref(0)
    bytes = Ref(0)
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        c = AdaptiveArrayPools._compact_counts!(tp, factor, shrink_to, min_bytes, active)
        slots[] += c[1]
        bytes[] += c[2]
    end
    oc = AdaptiveArrayPools._compact_others_counts!(values(pool.others), factor, shrink_to, min_bytes, active)
    force_gc && GC.gc()
    return AdaptiveArrayPools._compact_summary((slots[] + oc[1], bytes[] + oc[2]), force_gc)
end

@inline function AdaptiveArrayPools.compact!(
        pool::MetalAdaptiveArrayPool, ::Type{T};
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = true, force_gc::Bool = false,
    ) where {T}
    counts = AdaptiveArrayPools._compact_one_counts!(pool, T, factor, shrink_to, min_bytes, active)
    force_gc && GC.gc()
    return AdaptiveArrayPools._compact_summary(counts, force_gc)
end

@generated function AdaptiveArrayPools.compact!(
        pool::MetalAdaptiveArrayPool, types::Type...;
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
        force_gc && GC.gc()
        return AdaptiveArrayPools._compact_summary(counts, force_gc)
    end
end
