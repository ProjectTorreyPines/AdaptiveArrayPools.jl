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
    # Increment depth and initialize untracked flag
    pool._current_depth += 1
    push!(pool._untracked_flags, false)
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

Also updates _current_depth and _untracked_flags for untracked acquire detection.

~77% faster than full checkpoint! when only one type is used.
"""
@inline function checkpoint!(pool::AdaptiveArrayPool, ::Type{T}) where T
    pool._current_depth += 1
    push!(pool._untracked_flags, false)
    _checkpoint_typed_pool!(get_typed_pool!(pool, T), pool._current_depth)
end

"""
    checkpoint!(pool::AdaptiveArrayPool, types::Type...)

Save state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling. Increments _current_depth once for all types.
"""
@generated function checkpoint!(pool::AdaptiveArrayPool, types::Type...)
    checkpoint_exprs = [:(_checkpoint_typed_pool!(get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in 1:length(types)]
    quote
        pool._current_depth += 1
        push!(pool._untracked_flags, false)
        $(checkpoint_exprs...)
        nothing
    end
end

checkpoint!(::Nothing) = nothing
checkpoint!(::Nothing, ::Type) = nothing
checkpoint!(::Nothing, types::Type...) = nothing

# Internal helper for checkpoint
@inline function _checkpoint_typed_pool!(tp::TypedPool, depth::Int)
    push!(tp._checkpoint_n_active, tp.n_active)
    push!(tp._checkpoint_depths, depth)
    nothing
end

# ==============================================================================
# State Management - rewind!
# ==============================================================================

"""
    rewind!(pool::AdaptiveArrayPool)

Restore the pool state (n_active counters) from internal stacks.
Uses _checkpoint_depths to accurately determine which entries to pop vs restore.

Only the counters are restored; allocated memory remains for reuse.
Handles untracked acquires by checking _checkpoint_depths for accurate restoration.

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

    pop!(pool._untracked_flags)
    pool._current_depth -= 1

    return nothing
end

"""
    rewind!(pool::AdaptiveArrayPool, ::Type{T})

Restore state for a specific type only.
Also updates _current_depth and _untracked_flags.
"""
@inline function rewind!(pool::AdaptiveArrayPool, ::Type{T}) where T
    # Safety guard: at global scope (depth=1), delegate to reset!
    if pool._current_depth == 1
        reset!(get_typed_pool!(pool, T))
        return nothing
    end
    _rewind_typed_pool!(get_typed_pool!(pool, T), pool._current_depth)
    pop!(pool._untracked_flags)
    pool._current_depth -= 1
end

"""
    rewind!(pool::AdaptiveArrayPool, types::Type...)

Restore state for multiple specific types in reverse order.
Decrements _current_depth once after all types are rewound.
"""
@generated function rewind!(pool::AdaptiveArrayPool, types::Type...)
    rewind_exprs = [:(_rewind_typed_pool!(get_typed_pool!(pool, types[$i]), pool._current_depth)) for i in length(types):-1:1]
    reset_exprs = [:(reset!(get_typed_pool!(pool, types[$i]))) for i in 1:length(types)]
    quote
        # Safety guard: at global scope (depth=1), delegate to reset!
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

rewind!(::Nothing) = nothing
rewind!(::Nothing, ::Type) = nothing
rewind!(::Nothing, types::Type...) = nothing

# Internal helper for rewind with orphan cleanup
# Uses 1-based sentinel pattern: no isempty checks needed (sentinel [0] guarantees non-empty)
@inline function _rewind_typed_pool!(tp::TypedPool, current_depth::Int)
    # 1. Orphaned Checkpoints Cleanup
    # If there are checkpoints from deeper scopes (depth > current), pop them first.
    # This happens when a nested scope did full checkpoint but typed rewind,
    # leaving orphaned checkpoints that must be cleaned before finding current state.
    while @inbounds tp._checkpoint_depths[end] > current_depth
        pop!(tp._checkpoint_depths)
        pop!(tp._checkpoint_n_active)
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
    nothing
end

# ==============================================================================
# State Management - empty!
# ==============================================================================

"""
    empty!(tp::TypedPool)

Clear all internal storage of a TypedPool, releasing all memory.
Restores sentinel values for 1-based sentinel pattern.
"""
function Base.empty!(tp::TypedPool)
    empty!(tp.vectors)
    empty!(tp.views)
    empty!(tp.view_lengths)
    # Clear N-D Array cache (N-way)
    empty!(tp.nd_arrays)
    empty!(tp.nd_dims)
    empty!(tp.nd_ptrs)
    empty!(tp.nd_next_way)
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

    # Reset untracked detection state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._untracked_flags)
    push!(pool._untracked_flags, false)       # Sentinel: global scope starts with false

    return pool
end

Base.empty!(::Nothing) = nothing

# ==============================================================================
# State Management - reset!
# ==============================================================================

"""
    reset!(tp::TypedPool)

Reset TypedPool state without clearing allocated storage.

Sets `n_active = 0` and restores checkpoint stacks to sentinel state.
All vectors, views, and N-D arrays are preserved for reuse.

This is useful when you want to "start fresh" without reallocating memory.
"""
function reset!(tp::TypedPool)
    tp.n_active = 0
    # Restore sentinel values (1-based sentinel pattern)
    empty!(tp._checkpoint_n_active)
    push!(tp._checkpoint_n_active, 0)   # Sentinel: n_active=0 at depth=0
    empty!(tp._checkpoint_depths)
    push!(tp._checkpoint_depths, 0)     # Sentinel: depth=0 = no checkpoint
    return tp
end

"""
    reset!(pool::AdaptiveArrayPool)

Reset pool state without clearing allocated storage.

This function:
- Resets all `n_active` counters to 0
- Restores all checkpoint stacks to sentinel state
- Resets `_current_depth` and `_untracked_flags`

Unlike `empty!`, this **preserves** all allocated vectors, views, and N-D arrays
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

    # Reset untracked detection state (1-based sentinel pattern)
    pool._current_depth = 1                   # 1 = global scope (sentinel)
    empty!(pool._untracked_flags)
    push!(pool._untracked_flags, false)       # Sentinel: global scope starts with false

    return pool
end

"""
    reset!(pool::AdaptiveArrayPool, ::Type{T})

Reset state for a specific type only. Clears n_active and checkpoint stacks
to sentinel state while preserving allocated vectors.

See also: [`reset!(::AdaptiveArrayPool)`](@ref), [`rewind!`](@ref)
"""
@inline function reset!(pool::AdaptiveArrayPool, ::Type{T}) where T
    reset!(get_typed_pool!(pool, T))
end

"""
    reset!(pool::AdaptiveArrayPool, types::Type...)

Reset state for multiple specific types. Uses @generated for zero-overhead
compile-time unrolling.

See also: [`reset!(::AdaptiveArrayPool)`](@ref), [`rewind!`](@ref)
"""
@generated function reset!(pool::AdaptiveArrayPool, types::Type...)
    reset_exprs = [:(reset!(get_typed_pool!(pool, types[$i]))) for i in 1:length(types)]
    quote
        $(reset_exprs...)
        nothing
    end
end

reset!(::Nothing) = nothing
reset!(::Nothing, ::Type) = nothing
reset!(::Nothing, types::Type...) = nothing
