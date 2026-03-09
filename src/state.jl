# ==============================================================================
# State Management - checkpoint!
# ==============================================================================

"""
    checkpoint!(pool::AdaptiveArrayPool)

Save the current pool state (n_active counters) to internal stacks.

This is called automatically by `@with_pool` and related macros.
After warmup, this function has **zero allocation**.

See also: [`rewind!`](@ref), [`@with_pool`](@ref)
"""
function checkpoint!(pool::AdaptiveArrayPool)

    # Increment depth and initialize type touch tracking state
    pool._current_depth += 1
    push!(pool._touched_type_masks, UInt16(0))
    push!(pool._touched_has_others, false)
    depth = pool._current_depth

    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        _checkpoint_typed_pool!(tp, depth)
    end

    # Others - iterate without allocation (values() returns iterator)
    for p in values(pool.others)
        _checkpoint_typed_pool!(p, depth)
    end

    return nothing
end

"""
    checkpoint!(pool::AdaptiveArrayPool, ::Type{T})

Save state for a specific type only. Used by optimized macros that know
which types will be used at compile time.

Also updates _current_depth and bitmask state for type touch tracking.

~77% faster than full checkpoint! when only one type is used.
"""
@inline function checkpoint!(pool::AdaptiveArrayPool, ::Type{T}) where {T}

    pool._current_depth += 1
    push!(pool._touched_type_masks, UInt16(0))
    # Push true when T is a fallback type (non-fixed-slot) so that
    # _typed_lazy_rewind! iterates pool.others even if _acquire_impl!
    # (which bypasses _record_type_touch!) is the only acquire path.
    push!(pool._touched_has_others, _fixed_slot_bit(T) == UInt16(0))
    _checkpoint_typed_pool!(get_typed_pool!(pool, T), pool._current_depth)
    return nothing
end

"""
    checkpoint!(pool::AdaptiveArrayPool, types::Type...)

Save state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling. Increments _current_depth once for all types.
"""
@generated function checkpoint!(pool::AdaptiveArrayPool, types::Type...)

    # Deduplicate types at compile time (e.g., Float64, Float64 → Float64)
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    # Check at compile time if any type is a fallback (non-fixed-slot).
    # If so, push has_others=true so _typed_lazy_rewind! iterates pool.others
    # even when _acquire_impl! (bypassing _record_type_touch!) is used.
    has_any_fallback = any(i -> _fixed_slot_bit(types[i].parameters[1]) == UInt16(0), unique_indices)
    checkpoint_exprs = [:(_checkpoint_typed_pool!(get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in unique_indices]
    return quote
        pool._current_depth += 1
        push!(pool._touched_type_masks, UInt16(0))
        push!(pool._touched_has_others, $has_any_fallback)
        $(checkpoint_exprs...)
        nothing
    end
end

# Internal helper for checkpoint (works for any AbstractTypedPool)
@inline function _checkpoint_typed_pool!(tp::AbstractTypedPool, depth::Int)

    # Guard: skip if already checkpointed at this depth (prevents double-push
    # when get_typed_pool! auto-checkpoints a new fallback type and then
    # checkpoint!(pool, types...) calls _checkpoint_typed_pool! for the same type).
    if @inbounds(tp._checkpoint_depths[end]) != depth
        push!(tp._checkpoint_n_active, tp.n_active)
        push!(tp._checkpoint_depths, depth)
    end
    return nothing
end

"""
    _lazy_checkpoint!(pool::AdaptiveArrayPool)

Lightweight checkpoint for lazy mode (`use_typed=false` macro path).

Increments `_current_depth` and pushes bitmask sentinels — but does **not** save
`n_active` for any fixed-slot typed pool. The `_LAZY_MODE_BIT` (bit 15) in
`_touched_type_masks` marks this depth as lazy mode so that
`_record_type_touch!` can trigger lazy first-touch checkpoints.

Existing `others` entries are eagerly checkpointed since there is no per-type
tracking for non-fixed-slot pools; Case B in `_rewind_typed_pool!` handles any
new `others` entries created during the scope (n_active starts at 0 = sentinel).

Performance: ~2ns vs ~540ns for full `checkpoint!`.
"""
@inline function _lazy_checkpoint!(pool::AdaptiveArrayPool)

    pool._current_depth += 1
    # _LAZY_MODE_BIT = lazy mode flag (bits 0–7 are fixed-slot type bits)
    push!(pool._touched_type_masks, _LAZY_MODE_BIT)
    push!(pool._touched_has_others, false)
    depth = pool._current_depth
    # Eagerly checkpoint any pre-existing others entries.
    # New others types created during the scope start at n_active=0 (sentinel covers them).
    for p in values(pool.others)
        _checkpoint_typed_pool!(p, depth)
        @inbounds pool._touched_has_others[depth] = true
    end
    return nothing
end

# ==============================================================================
# State Management - rewind!
# ==============================================================================

"""
    rewind!(pool::AdaptiveArrayPool)

Restore the pool state (n_active counters) from internal stacks.
Uses _checkpoint_depths to accurately determine which entries to pop vs restore.

Only the counters are restored; allocated memory remains for reuse.
Handles touched types by checking _checkpoint_depths for accurate restoration.

**Safety**: If called at global scope (depth=1, no pending checkpoints),
automatically delegates to `reset!` to safely clear all n_active counters.

See also: [`checkpoint!`](@ref), [`reset!`](@ref), [`@with_pool`](@ref)
"""
function rewind!(pool::AdaptiveArrayPool)

    cur_depth = pool._current_depth

    # Safety guard: at global scope (depth=1), no checkpoint to rewind to
    # Delegate to reset! which safely clears all n_active counters
    if cur_depth == 1
        reset!(pool)
        return nothing
    end

    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        _rewind_typed_pool!(tp, cur_depth)
    end

    # Process fallback types
    for tp in values(pool.others)
        _rewind_typed_pool!(tp, cur_depth)
    end

    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1

    return nothing
end

"""
    rewind!(pool::AdaptiveArrayPool, ::Type{T})

Restore state for a specific type only.
Also updates _current_depth and bitmask state.
"""
@inline function rewind!(pool::AdaptiveArrayPool, ::Type{T}) where {T}

    # Safety guard: at global scope (depth=1), delegate to reset!
    if pool._current_depth == 1
        reset!(get_typed_pool!(pool, T))
        return nothing
    end
    _rewind_typed_pool!(get_typed_pool!(pool, T), pool._current_depth)
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

"""
    rewind!(pool::AdaptiveArrayPool, types::Type...)

Restore state for multiple specific types in reverse order.
Decrements _current_depth once after all types are rewound.
"""
@generated function rewind!(pool::AdaptiveArrayPool, types::Type...)

    # Deduplicate types at compile time (e.g., Float64, Float64 → Float64)
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    rewind_exprs = [:(_rewind_typed_pool!(get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in reverse(unique_indices)]
    reset_exprs = [:(reset!(get_typed_pool!(pool, types[$i]))) for i in unique_indices]
    return quote
        # Safety guard: at global scope (depth=1), delegate to reset!
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
# Safety: Structural Invalidation on Rewind (POOL_SAFETY_LV >= 1)
# ==============================================================================
#
# When released, backing vectors are resize!'d to 0 and cached Array/BitArray
# wrappers have their size set to (0,...). This makes stale SubArrays and Arrays
# throw BoundsError on access instead of silently returning corrupted data.
#
# @noinline keeps invalidation code off the inlined hot path of _rewind_typed_pool!.

# No-op fallback for extension types (e.g. CuTypedPool)
_invalidate_released_slots!(::AbstractTypedPool, ::Int) = nothing

@noinline function _invalidate_released_slots!(tp::TypedPool{T}, old_n_active::Int) where {T}
    new_n = tp.n_active
    # Level 2+: poison vectors with NaN/sentinel before structural invalidation
    if POOL_SAFETY_LV[] >= 2
        _poison_released_vectors!(tp, old_n_active)
    end
    # Level 1+: resize backing vectors to length 0 (invalidates SubArrays from acquire!)
    for i in (new_n + 1):old_n_active
        @inbounds resize!(tp.vectors[i], 0)
    end
    # Invalidate N-D Array wrappers from unsafe_acquire! (setfield! size to zeros)
    for N_idx in 1:length(tp.arr_wrappers)
        wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
        wrappers_for_N === nothing && continue
        wrappers = wrappers_for_N::Vector{Any}
        for i in (new_n + 1):min(old_n_active, length(wrappers))
            wrapper = @inbounds wrappers[i]
            wrapper === nothing && continue
            setfield!(wrapper::Array, :size, ntuple(_ -> 0, N_idx))
        end
    end
    return nothing
end

@noinline function _invalidate_released_slots!(tp::BitTypedPool, old_n_active::Int)
    new_n = tp.n_active
    # Level 2+: poison BitVectors (all bits set to true)
    if POOL_SAFETY_LV[] >= 2
        _poison_released_vectors!(tp, old_n_active)
    end
    # Level 1+: resize backing BitVectors to length 0 (invalidates chunks)
    for i in (new_n + 1):old_n_active
        @inbounds resize!(tp.vectors[i], 0)
    end
    # Invalidate N-D BitArray wrappers (setfield! len and dims to zeros)
    for N_idx in 1:length(tp.arr_wrappers)
        wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
        wrappers_for_N === nothing && continue
        wrappers = wrappers_for_N::Vector{Any}
        for i in (new_n + 1):min(old_n_active, length(wrappers))
            wrapper = @inbounds wrappers[i]
            wrapper === nothing && continue
            ba = wrapper::BitArray
            setfield!(ba, :len, 0)
            setfield!(ba, :dims, ntuple(_ -> 0, N_idx))
        end
    end
    return nothing
end

# ==============================================================================
# Internal: Rewind with Orphan Cleanup
# ==============================================================================

# Internal helper for rewind with orphan cleanup (works for any AbstractTypedPool)
# Uses 1-based sentinel pattern: no isempty checks needed (sentinel [0] guarantees non-empty)
@inline function _rewind_typed_pool!(tp::AbstractTypedPool, current_depth::Int)

    # 1. Orphaned Checkpoints Cleanup
    # If there are checkpoints from deeper scopes (depth > current), pop them first.
    # This happens when a nested scope did full checkpoint but typed rewind,
    # leaving orphaned checkpoints that must be cleaned before finding current state.
    while @inbounds tp._checkpoint_depths[end] > current_depth
        pop!(tp._checkpoint_depths)
        pop!(tp._checkpoint_n_active)
    end

    # Capture n_active before restore (for safety invalidation)
    @static if STATIC_POOL_CHECKS
        _old_n_active = tp.n_active
    end

    # 2. Normal Rewind Logic (Sentinel Pattern)
    # Now the stack top is guaranteed to be at depth <= current depth.
    if @inbounds tp._checkpoint_depths[end] == current_depth
        # Checkpointed at current depth: pop and restore
        pop!(tp._checkpoint_depths)
        tp.n_active = pop!(tp._checkpoint_n_active)
    else
        # No checkpoint at current depth (this type was excluded from typed checkpoint)
        # MUST restore n_active from parent checkpoint value!
        # - Untracked acquire may have modified n_active
        # - If sentinel (_checkpoint_n_active=[0]), restores to n_active=0
        tp.n_active = @inbounds tp._checkpoint_n_active[end]
    end

    # 3. Safety: invalidate released slots (Level 1+)
    @static if STATIC_POOL_CHECKS
        if POOL_SAFETY_LV[] >= 1 && _old_n_active > tp.n_active
            _invalidate_released_slots!(tp, _old_n_active)
        end
    end

    return nothing
end

"""
    _lazy_rewind!(pool::AdaptiveArrayPool)

Complete rewind for lazy mode (`use_typed=false` macro path).

Reads the combined mask at the current depth, rewinds only the fixed-slot pools
whose bits are set, handles any `others` entries, then pops the depth metadata.

Called directly from the macro-generated `finally` clause as a single function call
(matching the structure of `_lazy_checkpoint!` for symmetry and performance).
"""
@inline function _lazy_rewind!(pool::AdaptiveArrayPool)

    d = pool._current_depth
    bits = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    _selective_rewind_fixed_slots!(pool, bits)
    if @inbounds(pool._touched_has_others[d])
        for tp in values(pool.others)
            _rewind_typed_pool!(tp, d)
        end
    end
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

"""
    _typed_lazy_checkpoint!(pool::AdaptiveArrayPool, types::Type...)

Typed checkpoint that enables lazy first-touch checkpointing for extra types touched
by helpers (`use_typed=true`, `_can_use_typed_path=false` path).

Calls `checkpoint!(pool, types...)` (checkpoints only the statically-known types),
then sets `_TYPED_LAZY_BIT` (bit 14) in `_touched_type_masks[depth]` to signal typed lazy mode.

`_record_type_touch!` checks `(mask & _MODE_BITS_MASK) != 0` (bit 14 OR bit 15) to trigger a
lazy first-touch checkpoint for each extra type on first acquire, ensuring Case A
(not Case B) applies at rewind and parent `n_active` is preserved correctly.
"""
@inline function _typed_lazy_checkpoint!(pool::AdaptiveArrayPool, types::Type...)
    checkpoint!(pool, types...)
    d = pool._current_depth
    @inbounds pool._touched_type_masks[d] |= _TYPED_LAZY_BIT

    # Eagerly snapshot pre-existing others entries — mirrors _lazy_checkpoint!.
    # _record_type_touch! cannot lazy-checkpoint others types (b==0 branch, no per-type bit).
    # Without this, a helper that re-acquires an already-active others type triggers Case B
    # at rewind and restores the wrong parent n_active value.
    #
    # Also set has_others=true when pool.others is non-empty, so _typed_lazy_rewind!
    # enters the others loop even for tracked non-fixed-slot types (e.g. CPU Float16) that
    # used _acquire_impl! (bypassing _record_type_touch!, leaving has_others=false otherwise).
    # Skip re-snapshot for entries already checkpointed at d by checkpoint!(pool, types...)
    # (e.g. Float16 in types... was just checkpointed above — avoid double-push).
    for p in values(pool.others)
        if @inbounds(p._checkpoint_depths[end]) != d
            _checkpoint_typed_pool!(p, d)
        end
        @inbounds pool._touched_has_others[d] = true
    end
    return nothing
end

"""
    _typed_lazy_rewind!(pool::AdaptiveArrayPool, tracked_mask::UInt16)

Selective rewind for typed mode (`use_typed=true`) fallback path.

Called when `_can_use_typed_path` returns false (helpers touched types beyond the
statically-tracked set). Rewinds only pools whose bits are set in
`tracked_mask | touched_mask`. All touched types have Case A checkpoints,
guaranteed by the `_TYPED_LAZY_BIT` mode set in `_typed_lazy_checkpoint!`.
"""
@inline function _typed_lazy_rewind!(pool::AdaptiveArrayPool, tracked_mask::UInt16)

    d = pool._current_depth
    touched = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    combined = tracked_mask | touched
    _selective_rewind_fixed_slots!(pool, combined)
    if @inbounds(pool._touched_has_others[d])
        for tp in values(pool.others)
            _rewind_typed_pool!(tp, d)
        end
    end
    pop!(pool._touched_type_masks)
    pop!(pool._touched_has_others)
    pool._current_depth -= 1
    return nothing
end

"""
    _selective_rewind_fixed_slots!(pool::AdaptiveArrayPool, mask::UInt16)

Rewind only the fixed-slot typed pools whose bits are set in `mask`.

Each of the 8 fixed-slot pools maps to bits 0–7 (same encoding as `_fixed_slot_bit`).
Bits 8–15 (mode flags) are **not** checked here — callers must strip them
before passing the mask (e.g. `mask & _TYPE_BITS_MASK`).

Unset bits are skipped entirely: for pools that were acquired without a matching
checkpoint, `_rewind_typed_pool!` Case B safely restores from the parent checkpoint.
"""
@inline function _selective_rewind_fixed_slots!(pool::AdaptiveArrayPool, mask::UInt16)

    d = pool._current_depth
    _has_bit(mask, Float64)    && _rewind_typed_pool!(pool.float64, d)
    _has_bit(mask, Float32)    && _rewind_typed_pool!(pool.float32, d)
    _has_bit(mask, Int64)      && _rewind_typed_pool!(pool.int64, d)
    _has_bit(mask, Int32)      && _rewind_typed_pool!(pool.int32, d)
    _has_bit(mask, ComplexF64) && _rewind_typed_pool!(pool.complexf64, d)
    _has_bit(mask, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d)
    _has_bit(mask, Bool)       && _rewind_typed_pool!(pool.bool, d)
    _has_bit(mask, Bit)        && _rewind_typed_pool!(pool.bits, d)
    return nothing
end

# ==============================================================================
# State Management - empty!
# ==============================================================================

"""
    empty!(tp::TypedPool)

Clear all internal storage for TypedPool, releasing all memory.
Restores sentinel values for 1-based sentinel pattern.
"""
function Base.empty!(tp::TypedPool)
    empty!(tp.vectors)
    empty!(tp.arr_wrappers)
    tp.n_active = 0
    # Restore sentinel values (1-based sentinel pattern)
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)   # Sentinel: n_active=0 at depth=0
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)     # Sentinel: depth=0 = no checkpoint
    return tp
end

"""
    empty!(pool::AdaptiveArrayPool)

Completely clear the pool, releasing all stored vectors and resetting all state.

This is useful when you want to free memory or start fresh without creating
a new pool instance.

## Example
```julia
pool = AdaptiveArrayPool()
v = acquire!(pool, Float64, 1000)
# ... use v ...
empty!(pool)  # Release all memory
```

## Warning
Any SubArrays previously acquired from this pool become invalid after `empty!`.
"""
function Base.empty!(pool::AdaptiveArrayPool)
    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        empty!(tp)
    end

    # Others - clear all TypedPools then the IdDict itself
    for tp in values(pool.others)
        empty!(tp)
    end
    empty!(pool.others)

    # Reset type touch tracking state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._touched_type_masks)
    push!(pool._touched_type_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._touched_has_others)
    push!(pool._touched_has_others, false)         # Sentinel: no others

    return pool
end

# ==============================================================================
# State Management - reset!
# ==============================================================================

"""
    reset!(tp::AbstractTypedPool)

Reset state without clearing allocated storage.
Sets `n_active = 0` and restores checkpoint stacks to sentinel state.
"""
function reset!(tp::AbstractTypedPool)
    @static if STATIC_POOL_CHECKS
        _old_n_active = tp.n_active
    end
    tp.n_active = 0
    # Restore sentinel values (1-based sentinel pattern)
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)   # Sentinel: n_active=0 at depth=0
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)     # Sentinel: depth=0 = no checkpoint
    @static if STATIC_POOL_CHECKS
        if POOL_SAFETY_LV[] >= 1 && _old_n_active > 0
            _invalidate_released_slots!(tp, _old_n_active)
        end
    end
    return tp
end

"""
    reset!(pool::AdaptiveArrayPool)

Reset pool state without clearing allocated storage.

This function:
- Resets all `n_active` counters to 0
- Restores all checkpoint stacks to sentinel state
- Resets `_current_depth` and type touch tracking state

Unlike `empty!`, this **preserves** all allocated vectors and N-D wrapper caches
for reuse, avoiding reallocation costs.

## Use Case
When functions that acquire from the pool are called without proper
`checkpoint!/rewind!` management, `n_active` can grow indefinitely.
Use `reset!` to cleanly restore the pool to its initial state while
keeping allocated memory available.

## Example
```julia
pool = AdaptiveArrayPool()

# Some function that acquires without checkpoint management
function compute!(pool)
    v = acquire!(pool, Float64, 100)
    # ... use v ...
    # No rewind! called
end

for _ in 1:1000
    compute!(pool)  # n_active grows each iteration
end

reset!(pool)  # Restore state, keep allocated memory
# Now pool.n_active == 0, but vectors are still available for reuse
```

See also: [`empty!`](@ref), [`rewind!`](@ref)
"""
function reset!(pool::AdaptiveArrayPool)
    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        reset!(tp)
    end

    # Others - reset all TypedPools
    for tp in values(pool.others)
        reset!(tp)
    end

    # Reset type touch tracking state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._touched_type_masks)
    push!(pool._touched_type_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._touched_has_others)
    push!(pool._touched_has_others, false)         # Sentinel: no others

    # Clear borrow registry
    pool._pending_callsite = ""
    pool._borrow_log = nothing

    return pool
end

"""
    reset!(pool::AdaptiveArrayPool, ::Type{T})

Reset state for a specific type only. Clears n_active and checkpoint stacks
to sentinel state while preserving allocated vectors.

See also: [`reset!(::AdaptiveArrayPool)`](@ref), [`rewind!`](@ref)
"""
@inline function reset!(pool::AdaptiveArrayPool, ::Type{T}) where {T}
    reset!(get_typed_pool!(pool, T))
    return pool
end

"""
    reset!(pool::AdaptiveArrayPool, types::Type...)

Reset state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling.

See also: [`reset!(::AdaptiveArrayPool)`](@ref), [`rewind!`](@ref)
"""
@generated function reset!(pool::AdaptiveArrayPool, types::Type...)
    reset_exprs = [:(reset!(get_typed_pool!(pool, types[$i]))) for i in 1:length(types)]
    return quote
        $(reset_exprs...)
        pool
    end
end

# ==============================================================================
# Bitmask Helpers for Typed Path Decisions
# ==============================================================================

"""
    _tracked_mask_for_types(types::Type...) -> UInt16

Compute compile-time bitmask for the types tracked by a typed checkpoint/rewind.
Uses `@generated` for zero-overhead constant folding.

Returns `UInt16(0)` when called with no arguments.
Non-fixed-slot types contribute `UInt16(0)` (their bit is 0).
"""
@generated function _tracked_mask_for_types(types::Type...)
    mask = UInt16(0)
    for i in 1:length(types)
        T = types[i].parameters[1]
        mask |= _fixed_slot_bit(T)
    end
    return :(UInt16($mask))
end

"""
    _can_use_typed_path(pool::AbstractArrayPool, tracked_mask::UInt16) -> Bool

Check if the typed (fast) checkpoint/rewind path is safe to use.

Returns `true` when all touched types at the current depth are a subset
of the tracked types (bitmask subset check) AND no non-fixed-slot types were touched.

The subset check: `(touched_mask & ~tracked_mask) == 0` means every bit set
in `touched_mask` is also set in `tracked_mask`.
"""
@inline function _can_use_typed_path(pool::AbstractArrayPool, tracked_mask::UInt16)
    depth = pool._current_depth
    touched_mask = @inbounds(pool._touched_type_masks[depth]) & _TYPE_BITS_MASK
    has_others = @inbounds pool._touched_has_others[depth]
    return (touched_mask & ~tracked_mask) == UInt16(0) && !has_others
end

# ==============================================================================
# DisabledPool State Management (no-ops)
# ==============================================================================
# DisabledPool doesn't track state, so all operations are no-ops.

checkpoint!(::DisabledPool) = nothing
checkpoint!(::DisabledPool, ::Type) = nothing
checkpoint!(::DisabledPool, types::Type...) = nothing

rewind!(::DisabledPool) = nothing
rewind!(::DisabledPool, ::Type) = nothing
rewind!(::DisabledPool, types::Type...) = nothing

reset!(::DisabledPool) = nothing
reset!(::DisabledPool, ::Type) = nothing
reset!(::DisabledPool, types::Type...) = nothing

Base.empty!(::DisabledPool) = nothing
