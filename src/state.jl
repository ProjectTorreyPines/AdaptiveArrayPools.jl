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
    _runtime_check(pool) && push!(pool._others_ptr_bounds_checkpoints, length(pool._others_ptr_bounds))
    depth = pool._current_depth

    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        _checkpoint_typed_pool!(tp, depth)
    end

    # Others - iterate without allocation via cached values vector
    for p in pool._others_values
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
    _runtime_check(pool) && push!(pool._others_ptr_bounds_checkpoints, length(pool._others_ptr_bounds))
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
        _runtime_check(pool) && push!(pool._others_ptr_bounds_checkpoints, length(pool._others_ptr_bounds))
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
    _runtime_check(pool) && push!(pool._others_ptr_bounds_checkpoints, length(pool._others_ptr_bounds))
    depth = pool._current_depth
    # Eagerly checkpoint any pre-existing others entries.
    # New others types created during the scope start at n_active=0 (sentinel covers them).
    for p in pool._others_values
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
function rewind!(pool::AdaptiveArrayPool{S}) where {S}

    cur_depth = pool._current_depth

    # Safety guard: at global scope (depth=1), no checkpoint to rewind to
    # Delegate to reset! which safely clears all n_active counters
    if cur_depth == 1
        reset!(pool)
        return nothing
    end

    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        _rewind_typed_pool!(tp, cur_depth, S)
    end

    # Process fallback types
    for tp in pool._others_values
        _rewind_typed_pool!(tp, cur_depth, S)
    end

    if S >= 1 && length(pool._others_ptr_bounds_checkpoints) > 1
        resize!(pool._others_ptr_bounds, pop!(pool._others_ptr_bounds_checkpoints))
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
@inline function rewind!(pool::AdaptiveArrayPool{S}, ::Type{T}) where {S, T}

    # Safety guard: at global scope (depth=1), delegate to reset!
    if pool._current_depth == 1
        reset!(get_typed_pool!(pool, T), S)
        return nothing
    end
    _rewind_typed_pool!(get_typed_pool!(pool, T), pool._current_depth, S)
    if S >= 1 && length(pool._others_ptr_bounds_checkpoints) > 1
        resize!(pool._others_ptr_bounds, pop!(pool._others_ptr_bounds_checkpoints))
    end
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
@generated function rewind!(pool::AdaptiveArrayPool{S}, types::Type...) where {S}

    # Deduplicate types at compile time (e.g., Float64, Float64 → Float64)
    seen = Set{Any}()
    unique_indices = Int[]
    for i in eachindex(types)
        if !(types[i] in seen)
            push!(seen, types[i])
            push!(unique_indices, i)
        end
    end
    rewind_exprs = [:(_rewind_typed_pool!(get_typed_pool!(pool, types[$i]), pool._current_depth, $S)) for i in reverse(unique_indices)]
    reset_exprs = [:(reset!(get_typed_pool!(pool, types[$i]), $S)) for i in unique_indices]
    return quote
        # Safety guard: at global scope (depth=1), delegate to reset!
        if pool._current_depth == 1
            $(reset_exprs...)
            return nothing
        end
        $(rewind_exprs...)
        if $S >= 1 && length(pool._others_ptr_bounds_checkpoints) > 1
            resize!(pool._others_ptr_bounds, pop!(pool._others_ptr_bounds_checkpoints))
        end
        pop!(pool._touched_type_masks)
        pop!(pool._touched_has_others)
        pool._current_depth -= 1
        nothing
    end
end

# ==============================================================================
# Safety: Structural Invalidation on Rewind (S >= 1)
# ==============================================================================
#
# When released, backing vectors are resize!'d to 0 and cached Array/BitArray
# wrappers have their size set to (0,...). This makes stale SubArrays and Arrays
# throw BoundsError on access instead of silently returning corrupted data.
#
# @noinline keeps invalidation code off the inlined hot path of _rewind_typed_pool!.

# No-op fallback for extension types (e.g. CuTypedPool)
_invalidate_released_slots!(::AbstractTypedPool, ::Int, ::Int) = nothing
_invalidate_released_slots!(::AbstractTypedPool, ::Int) = nothing  # legacy 2-arg compat

# Zero-dims tuple for wrapper invalidation. Literal tuples for N ≤ 4 avoid
# ntuple(_ -> 0, N) dynamic-dispatch allocation (runtime N → heterogeneous
# return type → boxing). Falls back to ntuple for N > 4 (extremely rare).
@inline function _zero_dims_tuple(N::Int)
    N == 1 && return (0,)
    N == 2 && return (0, 0)
    N == 3 && return (0, 0, 0)
    N == 4 && return (0, 0, 0, 0)
    return ntuple(_ -> 0, N)
end

@noinline function _invalidate_released_slots!(tp::TypedPool{T}, old_n_active::Int, S::Int) where {T}
    new_n = tp.n_active
    # S=1: check for structural mutation before invalidation (wrappers still intact)
    if S >= 1
        _check_wrapper_mutation!(tp, new_n, old_n_active)
    end
    # S=1: poison vectors with NaN/sentinel before structural invalidation
    if S >= 1
        _poison_released_vectors!(tp, old_n_active)
    end
    # S=1: resize backing vectors to length 0 (invalidates SubArrays from acquire!)
    for i in (new_n + 1):old_n_active
        @inbounds resize!(tp.vectors[i], 0)
    end
    # Invalidate N-D Array wrappers (setfield! size to zeros)
    for N_idx in 1:length(tp.arr_wrappers)
        wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
        wrappers_for_N === nothing && continue
        wrappers = wrappers_for_N::Vector{Any}
        for i in (new_n + 1):min(old_n_active, length(wrappers))
            wrapper = @inbounds wrappers[i]
            wrapper === nothing && continue
            setfield!(wrapper::Array, :size, _zero_dims_tuple(N_idx))
        end
    end
    return nothing
end

@noinline function _invalidate_released_slots!(tp::BitTypedPool, old_n_active::Int, S::Int)
    new_n = tp.n_active
    # S=1: check for structural mutation before invalidation (wrappers still intact)
    if S >= 1
        _check_wrapper_mutation!(tp, new_n, old_n_active)
    end
    # S=1: poison BitVectors (all bits set to true)
    if S >= 1
        _poison_released_vectors!(tp, old_n_active)
    end
    # S=1: resize backing BitVectors to length 0 (invalidates chunks)
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
            setfield!(ba, :dims, _zero_dims_tuple(N_idx))
        end
    end
    return nothing
end

# ==============================================================================
# Internal: Rewind with Orphan Cleanup
# ==============================================================================

# Internal helper for rewind with orphan cleanup (works for any AbstractTypedPool)
# Uses 1-based sentinel pattern: no isempty checks needed (sentinel [0] guarantees non-empty)
#
# S parameter: runtime check level (0=off, 1=on). When called from AdaptiveArrayPool{S}
# callers, S is a compile-time constant → `S >= 1` dead-code-eliminates at S=0.
@inline function _rewind_typed_pool!(tp::AbstractTypedPool, current_depth::Int, S::Int)

    # 1. Orphaned Checkpoints Cleanup
    # If there are checkpoints from deeper scopes (depth > current), pop them first.
    # This happens when a nested scope did full checkpoint but typed rewind,
    # leaving orphaned checkpoints that must be cleaned before finding current state.
    while @inbounds tp._checkpoint_depths[end] > current_depth
        pop!(tp._checkpoint_depths)
        pop!(tp._checkpoint_n_active)
    end

    # Capture n_active before restore (compiler eliminates dead variable at S=0)
    _old_n_active = tp.n_active

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
    # At S=0: `0 >= 1` is false → entire branch eliminated (dead code)
    if S >= 1 && _old_n_active > tp.n_active
        _invalidate_released_slots!(tp, _old_n_active, S)
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
@inline function _lazy_rewind!(pool::AdaptiveArrayPool{S}) where {S}

    d = pool._current_depth
    bits = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    _selective_rewind_fixed_slots!(pool, bits)  # S propagated via pool type
    if @inbounds(pool._touched_has_others[d])
        for tp in pool._others_values
            _rewind_typed_pool!(tp, d, S)
        end
    end
    if S >= 1 && length(pool._others_ptr_bounds_checkpoints) > 1
        resize!(pool._others_ptr_bounds, pop!(pool._others_ptr_bounds_checkpoints))
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
    for p in pool._others_values
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
@inline function _typed_lazy_rewind!(pool::AdaptiveArrayPool{S}, tracked_mask::UInt16) where {S}

    d = pool._current_depth
    touched = @inbounds(pool._touched_type_masks[d]) & _TYPE_BITS_MASK
    combined = tracked_mask | touched
    _selective_rewind_fixed_slots!(pool, combined)  # S propagated via pool type
    if @inbounds(pool._touched_has_others[d])
        for tp in pool._others_values
            _rewind_typed_pool!(tp, d, S)
        end
    end
    if S >= 1 && length(pool._others_ptr_bounds_checkpoints) > 1
        resize!(pool._others_ptr_bounds, pop!(pool._others_ptr_bounds_checkpoints))
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
@inline function _selective_rewind_fixed_slots!(pool::AdaptiveArrayPool{S}, mask::UInt16) where {S}

    d = pool._current_depth
    _has_bit(mask, Float64)    && _rewind_typed_pool!(pool.float64, d, S)
    _has_bit(mask, Float32)    && _rewind_typed_pool!(pool.float32, d, S)
    _has_bit(mask, Int64)      && _rewind_typed_pool!(pool.int64, d, S)
    _has_bit(mask, Int32)      && _rewind_typed_pool!(pool.int32, d, S)
    _has_bit(mask, ComplexF64) && _rewind_typed_pool!(pool.complexf64, d, S)
    _has_bit(mask, ComplexF32) && _rewind_typed_pool!(pool.complexf32, d, S)
    _has_bit(mask, Bool)       && _rewind_typed_pool!(pool.bool, d, S)
    _has_bit(mask, Bit)        && _rewind_typed_pool!(pool.bits, d, S)
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
    empty!(tp.slot_extents)
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

    # Others - clear all TypedPools then the IdDict and values cache
    for tp in pool._others_values
        empty!(tp)
    end
    empty!(pool.others)
    empty!(pool._others_values)

    # Reset pre-collected pointer bounds
    empty!(pool._others_ptr_bounds)
    empty!(pool._others_ptr_bounds_checkpoints)
    push!(pool._others_ptr_bounds_checkpoints, 0)  # Sentinel

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
function reset!(tp::AbstractTypedPool, S::Int)
    _old_n_active = tp.n_active
    tp.n_active = 0
    # Restore sentinel values (1-based sentinel pattern)
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)   # Sentinel: n_active=0 at depth=0
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)     # Sentinel: depth=0 = no checkpoint
    if S >= 1 && _old_n_active > 0
        _invalidate_released_slots!(tp, _old_n_active, S)
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
function reset!(pool::AdaptiveArrayPool{S}) where {S}
    # Fixed slots - zero allocation via @generated iteration
    foreach_fixed_slot(pool) do tp
        reset!(tp, S)
    end

    # Others - reset all TypedPools (don't clear _others_values — pools are kept)
    for tp in pool._others_values
        reset!(tp, S)
    end

    # Reset pre-collected pointer bounds
    empty!(pool._others_ptr_bounds)
    empty!(pool._others_ptr_bounds_checkpoints)
    push!(pool._others_ptr_bounds_checkpoints, 0)  # Sentinel

    # Reset type touch tracking state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._touched_type_masks)
    push!(pool._touched_type_masks, UInt16(0))   # Sentinel: no bits set
    empty!(pool._touched_has_others)
    push!(pool._touched_has_others, false)         # Sentinel: no others

    # Clear borrow registry and return-site tracking
    pool._pending_callsite = ""
    pool._pending_return_site = ""
    pool._borrow_log = nothing

    return pool
end

"""
    reset!(pool::AdaptiveArrayPool, ::Type{T})

Reset state for a specific type only. Clears n_active and checkpoint stacks
to sentinel state while preserving allocated vectors.

See also: [`reset!(::AdaptiveArrayPool)`](@ref), [`rewind!`](@ref)
"""
@inline function reset!(pool::AdaptiveArrayPool{S}, ::Type{T}) where {S, T}
    reset!(get_typed_pool!(pool, T), S)
    return pool
end

"""
    reset!(pool::AdaptiveArrayPool, types::Type...)

Reset state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling.

See also: [`reset!(::AdaptiveArrayPool)`](@ref), [`rewind!`](@ref)
"""
@generated function reset!(pool::AdaptiveArrayPool{S}, types::Type...) where {S}
    reset_exprs = [:(reset!(get_typed_pool!(pool, types[$i]), $S)) for i in 1:length(types)]
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
# State Management - trim! (manual inactive-slot trimming)
# ==============================================================================
#
# trim! drops the pool's strong references to INACTIVE retained slots
# (indices n_active+1 : end of each typed pool) while preserving ACTIVE slots
# (1 : n_active). This releases backing buffers (and cached wrappers) so they
# become GC-eligible once no user code still references them. It does NOT promise
# immediate OS/VRAM return.
#
# trim! is a manual operation. It adds no code to the hot acquire path or the
# automatic rewind path.

# Build the public summary NamedTuple from raw `(slots, wrappers, bytes)` counts
# and the gc flag. Single definition point for the `trim!` return shape — every
# overload (CPU/GPU/per-type/zero) funnels through here, so they all return the
# identical concrete `@NamedTuple{...::Int, ..., gc_triggered::Bool}`.
@inline _trim_summary(c::NTuple{3, Int}, gc::Bool) = (;
    slots_released = c[1], wrappers_released = c[2],
    estimated_bytes_released = c[3], gc_triggered = gc,
)

# Zero summary: returned by no-op trims (DisabledPool, nothing to release).
const _ZERO_TRIM_SUMMARY = _trim_summary((0, 0, 0), false)

# Drop cached N-D wrappers for inactive slots. A cached Array/BitArray holds a
# `:ref` into its slot's backing buffer, so leaving inactive wrappers in place
# would pin the very buffers we just detached. Each per-slot wrapper vector
# (indexed by dimensionality N) is truncated to `keep`. Returns the count of
# non-nothing wrapper objects dropped.
function _trim_inactive_wrappers!(tp::AbstractTypedPool, keep::Int)
    released = 0
    for wrappers in tp.arr_wrappers
        wrappers === nothing && continue
        old_len = length(wrappers)
        old_len <= keep && continue
        for i in (keep + 1):old_len
            @inbounds wrappers[i] === nothing || (released += 1)
        end
        resize!(wrappers, keep)
    end
    return released
end

# Estimate the storage held by inactive backing vectors in slots `first:last`.
# Uses `Base.summarysize`, which is capacity-aware: it sizes the backing `Memory`,
# so it still reports the retained capacity for vectors that `rewind!`/`reset!`
# shrank to logical length 0 in runtime-check mode (S >= 1). This is an estimate,
# not a guarantee that RSS dropped. Dispatch point: extensions (e.g. Metal) may
# override for backend-appropriate (GPU) storage sizes.
function _inactive_storage_bytes(tp::AbstractTypedPool, first::Int, last::Int)
    total = 0
    for i in first:last
        @inbounds total += Base.summarysize(tp.vectors[i])
    end
    return total
end

# Truncate `tp.slot_extents` parallel to `tp.vectors` when trimming. `TypedPool` carries
# the per-slot extent record; other pools (e.g. `BitTypedPool`) do not — no-op for them.
_trim_slot_extents!(::AbstractTypedPool, ::Int) = nothing
@inline function _trim_slot_extents!(tp::TypedPool, keep::Int)
    length(tp.slot_extents) > keep && resize!(tp.slot_extents, keep)
    return nothing
end

# Trim one typed pool: drop inactive backing-vector and wrapper references by
# truncating `tp.vectors` and the wrapper caches to the active count. Returns the
# raw `(slots, wrappers, bytes)` counts (bytes measured before truncation). The
# `::NTuple{3, Int}` return type keeps callers concrete even when this is reached
# through dynamic dispatch over the `others` `Vector{Any}`.
function _trim_counts!(tp::AbstractTypedPool)::NTuple{3, Int}
    keep = tp.n_active
    old_len = length(tp.vectors)
    released = old_len - keep
    wrappers_released = _trim_inactive_wrappers!(tp, keep)
    bytes = 0
    if released > 0
        bytes = _inactive_storage_bytes(tp, keep + 1, old_len)
        resize!(tp.vectors, keep)
        _trim_slot_extents!(tp, keep)   # keep the extent record parallel to `vectors`
    end
    return (max(0, released), wrappers_released, bytes)
end

# Sum `_trim_counts!` over every fixed slot, unrolled at compile time. A plain
# `foreach_fixed_slot(pool) do tp ... end` closure would box the accumulators
# (Core.Box) and widen the summary to `Any`; this unrolled fold stays concrete
# (`NTuple{3, Int}`) and allocation-free.
@generated function _trim_fixed_counts!(pool::AdaptiveArrayPool)
    syms = [Symbol(:c, i) for i in 1:length(FIXED_SLOT_FIELDS)]
    assigns = [
        :($(syms[i]) = _trim_counts!(getfield(pool, $(QuoteNode(f)))))
            for (i, f) in enumerate(FIXED_SLOT_FIELDS)
    ]
    sums = [Expr(:call, :+, [:($(s)[$j]) for s in syms]...) for j in 1:3]
    return quote
        Base.@_inline_meta
        $(assigns...)
        ($(sums...),)
    end
end

# Sum `_trim_counts!` over the `others` (non-fixed-slot) typed pools. A plain loop
# (no closure) avoids boxing; the `::NTuple{3, Int}` assertion keeps the running
# total concrete despite dynamic dispatch over the `Any`-typed iterable. Shared by
# the CPU (`_others_values`) and GPU (`values(pool.others)`) backends.
function _trim_others_counts!(others)::NTuple{3, Int}
    slots = 0
    wrappers = 0
    bytes = 0
    for tp in others
        c = _trim_counts!(tp)::NTuple{3, Int}
        slots += c[1]
        wrappers += c[2]
        bytes += c[3]
    end
    return (slots, wrappers, bytes)
end

# ==============================================================================
# State Management - compact! primitive (in-place capacity compaction)
# ==============================================================================
#
# A slot's backing `Vector` length is the *high-water mark* (the largest size ever
# acquired — `_claim_slot!` only grows it). The *current logical size* is recorded
# per-slot in `tp.slot_extents` by `_claim_slot!`. So a slot is "bloated" when its
# backing capacity is far larger than the size actually in use. `compact!` returns
# that excess by swapping the backing's `Memory` for a right-sized one in place
# (keeping the `Vector` object's identity, so views following `parent` stay valid)
# and re-syncing the cached wrappers' `:ref`.

# Allocated backing capacity (Memory length) for a CPU backing vector.
@inline _slot_capacity(v::Vector) = length(getfield(v, :ref).mem)

# Current logical extent of a slot = the size compaction must preserve. This is the
# size of the slot's most recent `_claim_slot!`, recorded in `tp.slot_extents` for
# BOTH the `acquire!` (Array wrapper) and `acquire_view!` (uncached SubArray /
# ReshapedArray) paths — so a live view's extent is never under-counted (the bug a
# wrapper-only scan had: `acquire_view!` caches no wrapper). Returns `::Int` — a
# concrete `Vector{Int}` read, keeping the gate arithmetic in `_maybe_compact_slot!`
# allocation-free (a `length(::Any)` wrapper scan boxed it). 0 if unrecorded.
@inline function _slot_used(tp::TypedPool, slot::Int)::Int
    ext = tp.slot_extents
    return slot <= length(ext) ? (@inbounds ext[slot]) : 0
end

# Shrink one slot's backing to capacity `target` in place: allocate a smaller
# `Memory`, copy the live `used` elements, swap the backing `Vector`'s `:ref`/`:size`
# (identity preserved → views follow), then re-sync the slot's cached wrappers.
function _compact_slot!(tp::TypedPool{T}, slot::Int, target::Int, used::Int) where {T}
    v = @inbounds tp.vectors[slot]
    nv = Vector{T}(undef, target)
    copyto!(nv, 1, v, 1, used)
    setfield!(v, :size, (target,))
    setfield!(v, :ref, getfield(nv, :ref))
    for wrappers in tp.arr_wrappers
        wrappers === nothing && continue
        slot <= length(wrappers) || continue
        w = @inbounds wrappers[slot]
        w === nothing && continue
        setfield!(w, :ref, getfield(v, :ref))   # :size unchanged
    end
    return nothing
end

# Gate + compact one slot. Compacts iff backing capacity is `≥ factor × used` AND
# the reclaim is `≥ min_bytes`. Returns bytes reclaimed (0 if skipped). The generic
# fallback (e.g. `BitTypedPool`, whose backing is not a `Vector`) is a no-op.
_maybe_compact_slot!(::AbstractTypedPool, ::Int, ::Real, ::Real, ::Int) = 0
function _maybe_compact_slot!(tp::TypedPool{T}, slot::Int, factor::Real, shrink_to::Real, min_bytes::Int) where {T}
    used = _slot_used(tp, slot)
    used == 0 && return 0
    cap = _slot_capacity(@inbounds tp.vectors[slot])
    cap >= factor * used || return 0
    target = max(used, ceil(Int, shrink_to * used))
    reclaim = (cap - target) * sizeof(T)
    reclaim >= min_bytes || return 0
    _compact_slot!(tp, slot, target, used)
    return reclaim
end

# Public summary NamedTuple from raw `(slots, bytes)` counts and the gc flag. Single
# definition point for the `compact!` return shape — every overload funnels through
# here, so they all return the identical concrete
# `@NamedTuple{slots_compacted::Int, bytes_reclaimed::Int, gc_triggered::Bool}`.
@inline _compact_summary(c::NTuple{2, Int}, gc::Bool) = (;
    slots_compacted = c[1], bytes_reclaimed = c[2], gc_triggered = gc,
)

# Zero summary: returned by no-op compactions (DisabledPool, nothing bloated).
const _ZERO_COMPACT_SUMMARY = _compact_summary((0, 0), false)

# Compact one typed pool's slots, gating each via `_maybe_compact_slot!`. Returns the
# raw `(slots_compacted, bytes_reclaimed)` counts; the `::NTuple{2, Int}` return keeps
# callers concrete even through the `others` dynamic dispatch.
#
# `active=false` (Tier 1, default): scan only INACTIVE slots (`n_active+1 : end`).
# `active=true`  (Tier 2): scan ALL slots (`1 : end`), so even slots the user is still
# holding are compacted. This is safe because `_compact_slot!` re-syncs every cached
# wrapper's `:ref` (and the backing keeps its `Vector` identity, so views follow), and
# `_maybe_compact_slot!`'s `target ≥ used` guarantee never drops the live elements.
function _compact_counts!(tp::AbstractTypedPool, factor::Real, shrink_to::Real, min_bytes::Int, active::Bool)::NTuple{2, Int}
    slots = 0
    bytes = 0
    start = active ? 1 : (tp.n_active + 1)
    for slot in start:length(tp.vectors)
        r = _maybe_compact_slot!(tp, slot, factor, shrink_to, min_bytes)
        if r > 0
            slots += 1
            bytes += r
        end
    end
    return (slots, bytes)
end

# Sum `_compact_counts!` over every fixed slot, unrolled at compile time — same
# anti-boxing rationale as `_trim_fixed_counts!`: a closure fold would box the
# accumulators (Core.Box) and widen the summary to `Any`; this stays `NTuple{2, Int}`.
@generated function _compact_fixed_counts!(pool::AdaptiveArrayPool, factor::Real, shrink_to::Real, min_bytes::Int, active::Bool)
    syms = [Symbol(:c, i) for i in 1:length(FIXED_SLOT_FIELDS)]
    assigns = [
        :($(syms[i]) = _compact_counts!(getfield(pool, $(QuoteNode(f))), factor, shrink_to, min_bytes, active))
            for (i, f) in enumerate(FIXED_SLOT_FIELDS)
    ]
    sums = [Expr(:call, :+, [:($(s)[$j]) for s in syms]...) for j in 1:2]
    return quote
        Base.@_inline_meta
        $(assigns...)
        ($(sums...),)
    end
end

# Sum `_compact_counts!` over the `others` (non-fixed-slot) typed pools. A plain loop
# (no closure) avoids boxing; the `::NTuple{2, Int}` assertion keeps the running total
# concrete despite dynamic dispatch over the `Any`-typed iterable.
function _compact_others_counts!(others, factor::Real, shrink_to::Real, min_bytes::Int, active::Bool)::NTuple{2, Int}
    slots = 0
    bytes = 0
    for tp in others
        c = _compact_counts!(tp, factor, shrink_to, min_bytes, active)::NTuple{2, Int}
        slots += c[1]
        bytes += c[2]
    end
    return (slots, bytes)
end

# Guard + compact for one element type, returning raw `(slots, bytes)`. Like
# `_trim_one_counts!`, never *creates* a pool: a never-used fallback type is skipped
# so a reclamation call does not surprise-register a fallback pool in `pool.others`.
function _compact_one_counts!(pool::AdaptiveArrayPool, ::Type{T}, factor::Real, shrink_to::Real, min_bytes::Int, active::Bool)::NTuple{2, Int} where {T}
    (!(T <: _FIXED_SLOT_TYPES) && !haskey(pool.others, T)) && return (0, 0)
    return _compact_counts!(get_typed_pool!(pool, T), factor, shrink_to, min_bytes, active)
end

"""
    compact!(pool::AdaptiveArrayPool;
             factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
             active::Bool = false, force_gc::Bool = false)

Shrink the **over-allocated capacity** of backing buffers in place. A slot's backing
length is the high-water mark (the largest size ever acquired); its current logical
size lives in the cached wrapper. A slot is *bloated* when its retained capacity is
`≥ factor ×` that logical size; `compact!` swaps such a backing's `Memory` for one of
size `ceil(shrink_to × used)` (keeping the `Vector`'s identity so views following
`parent` stay valid) and re-syncs the cached wrappers' `:ref`.

Orthogonal to [`trim!`](@ref): `trim!` drops whole inactive *slots*, `compact!` shrinks
the *capacity* of retained ones.

- **`active=false`** (default, Tier 1): touch only **inactive** slots (`n_active+1 : end`),
  i.e. buffers no live array references. Always safe.
- **`active=true`** (Tier 2): also compact **active** slots — buffers the caller is still
  holding. This is `compact!`'s reason for existing over `trim!`: it reclaims peak
  capacity from arrays still in use. The held wrapper and any `view` of it follow the
  in-place swap (wrapper `:ref` re-synced; backing identity preserved), and the
  `target ≥ used` guarantee never drops live elements. Only call it at a synchronous
  point where no other task is mid-access to this pool's arrays.

A slot is compacted only if both gates pass: `capacity ≥ factor × used` **and** the
reclaim `(capacity − target) × sizeof(T) ≥ min_bytes` (default 1 MiB), so tiny buffers
are not churned. Returns `(; slots_compacted, bytes_reclaimed, gc_triggered)`.

`force_gc=true` runs `GC.gc()` after the swaps to make the detached buffers collectable.

!!! note "Julia version"
    Requires Julia 1.12+ (uses `setfield!(:ref/:size)`). On older Julia (the legacy
    pool architecture) `compact!` is a defined no-op that returns a zero summary and
    warns once, so dependent packages compile across the supported Julia range.

See also: [`trim!`](@ref), [`reset!`](@ref), [`empty!`](@ref).
"""
function compact!(
        pool::AdaptiveArrayPool;
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = false, force_gc::Bool = false
    )
    counts = _compact_fixed_counts!(pool, factor, shrink_to, min_bytes, active) .+
        _compact_others_counts!(pool._others_values, factor, shrink_to, min_bytes, active)
    force_gc && GC.gc()
    return _compact_summary(counts, force_gc)
end

"""
    compact!(pool::AdaptiveArrayPool, ::Type{T}; kwargs...)

Compact slots for a single element type `T` only. See [`compact!`](@ref).
"""
function compact!(
        pool::AdaptiveArrayPool, ::Type{T};
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = false, force_gc::Bool = false
    ) where {T}
    counts = _compact_one_counts!(pool, T, factor, shrink_to, min_bytes, active)
    force_gc && GC.gc()
    return _compact_summary(counts, force_gc)
end

"""
    compact!(pool::AdaptiveArrayPool, types::Type...; kwargs...)

Compact slots for several element types at once, running `GC.gc()` (when `force_gc=true`)
a single time after all listed types are compacted. Returns one combined summary. Uses
`@generated` for zero-overhead compile-time unrolling, mirroring [`trim!`](@ref).
See [`compact!`](@ref).
"""
@generated function compact!(
        pool::AdaptiveArrayPool, types::Type...;
        factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
        active::Bool = false, force_gc::Bool = false
    )
    n = length(types)
    syms = [Symbol(:c, i) for i in 1:n]
    assigns = [:($(syms[i]) = _compact_one_counts!(pool, types[$i], factor, shrink_to, min_bytes, active)) for i in 1:n]
    counts_expr = n == 0 ? :((0, 0)) :
        Expr(:tuple, (Expr(:call, :+, (:($(s)[$j]) for s in syms)...) for j in 1:2)...)
    return quote
        $(assigns...)
        counts = $counts_expr
        force_gc && GC.gc()
        return _compact_summary(counts, force_gc)
    end
end

"""
    compact!(; kwargs...)

Compact slots of the current task-local pool (`get_task_local_pool()`).
"""
compact!(;
    factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
    active::Bool = false, force_gc::Bool = false,
) = compact!(
    get_task_local_pool();
    factor = factor, shrink_to = shrink_to, min_bytes = min_bytes, active = active, force_gc = force_gc
)

"""
    trim!(pool::AdaptiveArrayPool; force_gc::Bool = false)

Release the pool's references to **inactive retained slots** (those already
released by `rewind!`/scope exit), keeping all **active** slots intact.

Unlike `reset!` (keeps all buffers) and `empty!` (drops all buffers and resets
state), `trim!` drops only the inactive backing buffers and cached wrappers and
leaves `n_active`, checkpoint stacks, and depth state unchanged.

Returns `(; slots_released, wrappers_released, estimated_bytes_released, gc_triggered)`.

`force_gc=true` runs `GC.gc()` after detaching references, and on backends with a
caching device allocator it also returns the pooled memory to the driver (CUDA:
`CUDA.reclaim()`; Metal/CPU have no further reclaim step). Even so, immediate
OS/VRAM return is not guaranteed — it only asks the runtime to collect now that
the pool no longer holds the buffers.

!!! note "Julia version"
    Actual reclamation requires Julia 1.12+. On older Julia (the legacy pool
    architecture) `trim!` is a defined no-op: it returns a zero summary and warns
    once. It remains callable so dependent packages compile across the full
    supported Julia range.

See also: [`reset!`](@ref), [`empty!`](@ref).
"""
function trim!(pool::AdaptiveArrayPool; force_gc::Bool = false)
    counts = _trim_fixed_counts!(pool) .+ _trim_others_counts!(pool._others_values)
    force_gc && GC.gc()
    return _trim_summary(counts, force_gc)
end

# Guard + trim for one element type, returning raw `(slots, wrappers, bytes)`.
# Shared by the single-type and varargs `trim!` forms. Never *creates* a pool: a
# never-used fallback type is skipped, because `get_typed_pool!` would register a
# new fallback pool in `pool.others` on a miss — a surprising mutation/allocation
# for a reclamation call. Fixed-slot types are always present; fallback types are
# trimmed only if already in the pool. The `::NTuple{3, Int}` return keeps the
# varargs fold type-stable even when the type is not a compile-time constant.
function _trim_one_counts!(pool::AdaptiveArrayPool, ::Type{T})::NTuple{3, Int} where {T}
    (!(T <: _FIXED_SLOT_TYPES) && !haskey(pool.others, T)) && return (0, 0, 0)
    return _trim_counts!(get_typed_pool!(pool, T))
end

"""
    trim!(pool::AdaptiveArrayPool, ::Type{T}; force_gc::Bool = false)

Trim inactive slots for a single element type `T` only. See [`trim!`](@ref).
"""
function trim!(pool::AdaptiveArrayPool, ::Type{T}; force_gc::Bool = false) where {T}
    counts = _trim_one_counts!(pool, T)
    force_gc && GC.gc()
    return _trim_summary(counts, force_gc)
end

"""
    trim!(pool::AdaptiveArrayPool, types::Type...; force_gc::Bool = false)

Trim inactive slots for several element types at once, running `GC.gc()` (when
`force_gc=true`) a single time after all listed types are detached. Returns one
combined summary. Uses `@generated` for zero-overhead compile-time unrolling,
mirroring [`reset!`](@ref). See [`trim!`](@ref).
"""
@generated function trim!(pool::AdaptiveArrayPool, types::Type...; force_gc::Bool = false)
    n = length(types)
    syms = [Symbol(:c, i) for i in 1:n]
    assigns = [:($(syms[i]) = _trim_one_counts!(pool, types[$i])) for i in 1:n]
    counts_expr = n == 0 ? :((0, 0, 0)) :
        Expr(:tuple, (Expr(:call, :+, (:($(s)[$j]) for s in syms)...) for j in 1:3)...)
    return quote
        $(assigns...)
        counts = $counts_expr
        force_gc && GC.gc()
        return _trim_summary(counts, force_gc)
    end
end

"""
    trim!(; force_gc::Bool = false)

Trim inactive slots of the current task-local pool (`get_task_local_pool()`).
"""
trim!(; force_gc::Bool = false) = trim!(get_task_local_pool(); force_gc = force_gc)

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

trim!(::DisabledPool; force_gc::Bool = false) = _ZERO_TRIM_SUMMARY
trim!(::DisabledPool, ::Type{T}; force_gc::Bool = false) where {T} = _ZERO_TRIM_SUMMARY
trim!(::DisabledPool, types::Type...; force_gc::Bool = false) = _ZERO_TRIM_SUMMARY

compact!(
    ::DisabledPool;
    factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
    active::Bool = false, force_gc::Bool = false
) = _ZERO_COMPACT_SUMMARY
compact!(
    ::DisabledPool, ::Type{T};
    factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
    active::Bool = false, force_gc::Bool = false
) where {T} = _ZERO_COMPACT_SUMMARY
compact!(
    ::DisabledPool, types::Type...;
    factor::Real = 10, shrink_to::Real = 1.5, min_bytes::Int = 2^20,
    active::Bool = false, force_gc::Bool = false
) = _ZERO_COMPACT_SUMMARY
