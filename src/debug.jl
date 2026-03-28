# ==============================================================================
# Debugging & Safety (Runtime escape detection, RUNTIME_CHECK >= 1)
# ==============================================================================

function _validate_pool_return(val, pool::AdaptiveArrayPool)
    # 0. Check BitArray / BitVector (bit-packed storage)
    if val isa BitArray
        _check_bitchunks_overlap(val, pool)
        return
    end

    # 1. Check SubArray
    if val isa SubArray
        p = parent(val)
        # Use pointer overlap check for ALL Array parents (Vector <: Array)
        # This catches both:
        # - acquire_view!() 1D returns: SubArray backed by pool's internal Vector
        # - view(acquire!()): SubArray backed by acquire!'s Array wrapper
        if p isa Array
            _check_pointer_overlap(p, pool, val)
        elseif p isa BitArray
            _check_bitchunks_overlap(p, pool, val)
        end
        return
    end

    # 2. Check ReshapedArray
    #    - acquire_view!() N-D: ReshapedArray wrapping SubArray of pool Vector
    #    - p isa Array / BitArray branches: defensive (reshape(::Array) returns Array via jl_reshape_array)
    if val isa Base.ReshapedArray
        p = parent(val)
        if p isa Array
            _check_pointer_overlap(p, pool, val)
        elseif p isa SubArray
            pp = parent(p)
            if pp isa Array
                _check_pointer_overlap(pp, pool, val)
            elseif pp isa BitArray
                _check_bitchunks_overlap(pp, pool, val)
            end
        elseif p isa BitArray
            _check_bitchunks_overlap(p, pool, val)
        end
        return
    end

    # 3. Check raw Array (from acquire!) + element recursion
    return if val isa Array
        # Pool vectors always have concrete eltypes — skip overlap check for abstract
        if isconcretetype(eltype(val))
            _check_pointer_overlap(val, pool)
        end
        # Recurse into elements for containers like Vector{SubArray}
        if _eltype_may_contain_arrays(eltype(val))
            for x in val
                _validate_pool_return(x, pool)
            end
        end
    end
end

# Eltype guard: skip element iteration for leaf types (perf optimization in debug mode)
_eltype_may_contain_arrays(::Type{<:Number}) = false
_eltype_may_contain_arrays(::Type{<:AbstractString}) = false
_eltype_may_contain_arrays(::Type{Symbol}) = false
_eltype_may_contain_arrays(::Type{Char}) = false
_eltype_may_contain_arrays(::Type) = true

# Scope-aware boundary: returns the n_active saved at checkpoint for `depth`.
# Vectors with index <= boundary belong to an outer scope and are NOT escapees.
# If this type has no checkpoint at `depth`, it was never touched in this scope → all safe.
@inline function _scope_boundary(tp::AbstractTypedPool, depth::Int)
    @inbounds if tp._checkpoint_depths[end] == depth
        return tp._checkpoint_n_active[end]   # vectors[1:boundary] are from outer scopes
    end
    return tp.n_active   # no checkpoint at this depth → nothing acquired here → all safe
end

# Safe element size: isbitstype → sizeof(T), otherwise → sizeof(Ptr) (conservative bound).
# Avoids sizeof(Array) crash on Julia 1.10 where Array is opaque C-backed without definite size.
@inline _safe_elsize(::Type{T}) where {T} = isbitstype(T) ? sizeof(T) : sizeof(Ptr{Nothing})

# Check if array memory overlaps with any pool vector **acquired in the current scope**.
# `original_val` is the user-visible value (e.g., SubArray) for error reporting;
# `arr` may be its parent Array used for the actual pointer comparison.
function _check_pointer_overlap(arr::Array, pool::AdaptiveArrayPool, original_val = arr)
    arr_ptr = UInt(pointer(arr))
    arr_end = arr_ptr + length(arr) * _safe_elsize(eltype(arr))

    return_site = let rs = pool._pending_return_site
        isempty(rs) ? nothing : rs
    end

    current_depth = pool._current_depth

    # Explicit per-slot calls via @generated — avoids closure allocation (128 bytes)
    _check_all_slots_pointer_overlap(pool, arr_ptr, arr_end, current_depth, return_site, original_val)
    return
end

# Standalone overlap check for a single TypedPool (no closure capture)
@noinline function _check_tp_pointer_overlap(
        tp::AbstractTypedPool, arr_ptr::UInt, arr_end::UInt,
        current_depth::Int, pool::AdaptiveArrayPool, return_site, original_val
    )
    boundary = _scope_boundary(tp, current_depth)
    for i in (boundary + 1):tp.n_active
        v = @inbounds tp.vectors[i]
        v isa Array || continue  # Skip BitVector (checked via _check_bitchunks_overlap)
        v_ptr = UInt(pointer(v))
        v_end = v_ptr + length(v) * _safe_elsize(eltype(v))
        if !(arr_end <= v_ptr || v_end <= arr_ptr)
            callsite = _lookup_borrow_callsite(pool, v)
            _throw_pool_escape_error(original_val, eltype(v), callsite, return_site)
        end
    end
    return
end

# Pre-collected bounds check for others types (zero-alloc when bounds are recorded at S=1).
# Falls back to walking TypedPools directly when bounds are empty (S=0 or direct API calls).
@noinline function _check_others_pointer_overlap(
        pool::AdaptiveArrayPool, arr_ptr::UInt, arr_end::UInt,
        current_depth::Int, return_site, original_val
    )
    bounds = pool._others_ptr_bounds
    if !isempty(bounds)
        # Fast path: pre-collected UInt bounds (zero-alloc)
        @inbounds for i in 1:2:length(bounds)
            v_ptr = bounds[i]
            v_end = bounds[i + 1]
            if !(arr_end <= v_ptr || v_end <= arr_ptr)
                _throw_others_overlap_error(pool, arr_ptr, arr_end, current_depth, return_site, original_val)
                return
            end
        end
    else
        # Fallback for S=0 or direct API calls without macro (bounds not recorded).
        # May allocate — acceptable since this path is rare.
        _psz = UInt(sizeof(Ptr{Nothing}))
        for tp in values(pool.others)
            boundary = _scope_boundary(tp, current_depth)
            for i in (boundary + 1):tp.n_active
                v = @inbounds tp.vectors[i]
                v isa Array || continue
                v_ptr = UInt(pointer(v))
                v_end = v_ptr + UInt(length(v)) * _psz
                if !(arr_end <= v_ptr || v_end <= arr_ptr)
                    callsite = _lookup_borrow_callsite(pool, v)
                    _throw_pool_escape_error(original_val, eltype(v), callsite, return_site)
                end
            end
        end
    end
    return nothing
end

# Error helper for others overlap: walks TypedPools to find actual overlapping vector for message
@noinline function _throw_others_overlap_error(
        pool::AdaptiveArrayPool, arr_ptr::UInt, arr_end::UInt,
        current_depth::Int, return_site, original_val
    )
    _psz = UInt(sizeof(Ptr{Nothing}))
    for tp in values(pool.others)
        boundary = _scope_boundary(tp, current_depth)
        for i in (boundary + 1):tp.n_active
            v = @inbounds tp.vectors[i]
            v isa Array || continue
            v_ptr = UInt(pointer(v))
            v_end = v_ptr + UInt(length(v)) * _psz
            if !(arr_end <= v_ptr || v_end <= arr_ptr)
                callsite = _lookup_borrow_callsite(pool, v)
                _throw_pool_escape_error(original_val, eltype(v), callsite, return_site)
            end
        end
    end
    return
end

# @generated unrolling over FIXED_SLOT_FIELDS — zero-allocation dispatch
@generated function _check_all_slots_pointer_overlap(
        pool::AdaptiveArrayPool, arr_ptr::UInt, arr_end::UInt,
        current_depth::Int, return_site, original_val
    )
    calls = [
        :(
                _check_tp_pointer_overlap(
                    getfield(pool, $(QuoteNode(f))), arr_ptr, arr_end,
                    current_depth, pool, return_site, original_val
                )
            )
            for f in FIXED_SLOT_FIELDS
    ]
    return quote
        Base.@_inline_meta
        $(calls...)
        # Guard: only check others if current scope touched non-fixed-slot types
        # CRITICAL: index is current_depth, NOT current_depth + 1
        _ho_idx = current_depth
        _has_others = if _ho_idx <= length(pool._touched_has_others)
            @inbounds pool._touched_has_others[_ho_idx]
        else
            !isempty(pool.others)  # fallback for direct API calls
        end
        if _has_others
            _check_others_pointer_overlap(pool, arr_ptr, arr_end, current_depth, return_site, original_val)
        end
        nothing
    end
end

"""
    PoolRuntimeEscapeError <: Exception

Thrown at runtime when `_validate_pool_return` detects a pool-backed array
escaping from an `@with_pool` scope (requires `RUNTIME_CHECK >= 1`).

This is the runtime counterpart of [`PoolEscapeError`](@ref) (compile-time).
"""
struct PoolRuntimeEscapeError <: Exception
    val_summary::String
    pool_eltype::String
    callsite::Union{Nothing, String}      # acquire location (S ≥ 1)
    return_site::Union{Nothing, String}   # return location (S ≥ 1)
end

function Base.showerror(io::IO, e::PoolRuntimeEscapeError)
    has_callsite = e.callsite !== nothing

    printstyled(io, "PoolEscapeError"; color = :red, bold = true)
    printstyled(io, " (runtime, RUNTIME_CHECK >= 1)"; color = :light_black)
    println(io)

    println(io)
    printstyled(io, "    "; color = :normal)
    printstyled(io, e.val_summary; color = :red, bold = true)
    println(io)
    printstyled(io, "      ← backed by "; color = :light_black)
    printstyled(io, e.pool_eltype; color = :yellow)
    printstyled(io, " pool memory, will be reclaimed at scope exit\n"; color = :light_black)

    if has_callsite
        # Parse callsite: "file:line" or "file:line\nexpr"
        parts = split(e.callsite, '\n'; limit = 2)
        location = String(parts[1])
        expr_text = length(parts) >= 2 ? String(parts[2]) : nothing

        # Shorten the file path (shorter of relpath vs ~/…-contracted)
        location = _shorten_location(location)

        printstyled(io, "      ← acquired at "; color = :light_black)
        printstyled(io, location; color = :cyan, bold = true)
        println(io)

        if expr_text !== nothing
            printstyled(io, "        "; color = :normal)
            printstyled(io, expr_text; color = :cyan)
            println(io)
        end
    end

    has_return_site = e.return_site !== nothing
    if has_return_site
        parts = split(e.return_site, '\n'; limit = 2)
        location = _shorten_location(String(parts[1]))
        expr_text = length(parts) >= 2 ? String(parts[2]) : nothing

        printstyled(io, "      ← escapes at "; color = :light_black)
        printstyled(io, location; color = :magenta, bold = true)
        println(io)

        if expr_text !== nothing
            printstyled(io, "        "; color = :normal)
            printstyled(io, expr_text; color = :magenta)
            println(io)
        end
    end

    println(io)
    printstyled(io, "  Fix: "; bold = true)
    printstyled(io, "Wrap with "; color = :light_black)
    printstyled(io, "collect()"; bold = true)
    printstyled(io, " to return an owned copy, or compute a scalar result.\n"; color = :light_black)

    return nothing
end

Base.showerror(io::IO, e::PoolRuntimeEscapeError, ::Any; backtrace = true) = showerror(io, e)

@noinline function _throw_pool_escape_error(val, pool_eltype, callsite::Union{Nothing, String} = nothing, return_site::Union{Nothing, String} = nothing)
    throw(PoolRuntimeEscapeError(summary(val), string(pool_eltype), callsite, return_site))
end

# Recursive inspection of container types (Tuple, NamedTuple, Pair, Dict, Set).
# These are common wrapper types in Julia through which pool-backed arrays
# can escape undetected when hidden inside return values.
# Note: Array element recursion is handled in the main function via _eltype_may_contain_arrays.

function _validate_pool_return(val::Tuple, pool::AdaptiveArrayPool)
    for x in val
        _validate_pool_return(x, pool)
    end
    return
end

function _validate_pool_return(val::NamedTuple, pool::AdaptiveArrayPool)
    for x in values(val)
        _validate_pool_return(x, pool)
    end
    return
end

function _validate_pool_return(val::Pair, pool::AdaptiveArrayPool)
    _validate_pool_return(val.first, pool)
    return _validate_pool_return(val.second, pool)
end

function _validate_pool_return(val::AbstractDict, pool::AdaptiveArrayPool)
    for p in val  # each p is a Pair — reuses Pair dispatch
        _validate_pool_return(p, pool)
    end
    return
end

function _validate_pool_return(val::AbstractSet, pool::AdaptiveArrayPool)
    for x in val
        _validate_pool_return(x, pool)
    end
    return
end

_validate_pool_return(val, ::DisabledPool) = nothing
# No-op fallback for pool types without specific validation (overridden by CUDA extension)
_validate_pool_return(val, ::AbstractArrayPool) = nothing

# ==============================================================================
# Leaked Scope Warning (direct-rewind path, RUNTIME_CHECK >= 1)
# ==============================================================================
#
# Detects when entry depth guard fires (inner scope didn't rewind properly).
# @noinline to keep it out of the inlined hot path — only called on error.

@noinline function _warn_leaked_scope(pool::AbstractArrayPool, entry_depth::Int)
    return @error(
        "Leaked @with_pool scope detected! " *
            "Pool depth is $(pool._current_depth), expected $(entry_depth + 1). " *
            "A macro inside @with_pool may have generated an unseen `return`/`break`, " *
            "or an inner scope threw without try-finally protection. " *
            "Consider using @safe_with_pool for exception safety.",
        current_depth = pool._current_depth,
        expected_depth = entry_depth + 1,
    )
end

# ==============================================================================
# Poisoning: Fill released vectors with sentinel values (S >= 1)
# ==============================================================================
#
# Poisons backing vectors with detectable values (NaN, typemax) before
# structural invalidation. This ensures stale references read obviously wrong
# data instead of silently valid old values — especially useful for
# Array wrappers on Julia 1.10 where setfield!(:size) is unavailable
# and structural invalidation can't catch stale access.

_poison_value(::Type{T}) where {T <: AbstractFloat} = T(NaN)
_poison_value(::Type{T}) where {T <: Integer} = typemax(T)
_poison_value(::Type{Complex{T}}) where {T} = Complex{T}(_poison_value(T), _poison_value(T))
_poison_value(::Type{T}) where {T} = zero(T)  # generic fallback (Rational, etc.)

function _poison_fill!(v::Vector{T}) where {T}
    isempty(v) && return nothing
    if !isbitstype(T)
        # non-isbits (reference types): skip poison, resize!(v, 0) handles invalidation
        return nothing
    end
    # isbits: try _poison_value dispatch (NaN, typemax, zero for known types),
    # then duck-type 0 * first(v) for custom structs without zero(T).
    # If neither works, skip poisoning — must not throw during rewind.
    try
        fill!(v, _poison_value(T))
    catch
        try
            fill!(v, 0 * first(v))
        catch
        end
    end
    return nothing
end
_poison_fill!(v::BitVector) = fill!(v, true)

"""
    _poison_released_vectors!(tp::AbstractTypedPool, old_n_active)

Fill released backing vectors (indices `n_active+1:old_n_active`) with sentinel
values. Called from `_invalidate_released_slots!` when `S >= 1`,
before `resize!` zeroes the lengths.
"""
@noinline function _poison_released_vectors!(tp::AbstractTypedPool, old_n_active::Int)
    new_n = tp.n_active
    for i in (new_n + 1):old_n_active
        _poison_fill!(@inbounds tp.vectors[i])
    end
    return nothing
end

# ==============================================================================
# Path Shortening (for readable callsite display)
# ==============================================================================
#
# Picks the shortest human-readable representation of a file path:
# relative to pwd, ~/…-contracted, or the original absolute path.
# Adapted from Infiltrator.jl (src/breakpoints.jl).

function _short_path(f::String)
    contracted = Base.contractuser(f)
    try
        rel = relpath(f)
        return length(rel) < length(contracted) ? rel : contracted
    catch
        return contracted
    end
end

# Shorten "file:line" location string using _short_path
function _shorten_location(location::String)
    colon_idx = findlast(':', location)
    if colon_idx !== nothing
        file = location[1:prevind(location, colon_idx)]
        line_part = location[colon_idx:end]
        return _short_path(file) * line_part
    end
    return location
end

# ==============================================================================
# Borrow Registry: Call-site tracking for acquire! (S >= 1)
# ==============================================================================
#
# Records where each acquire! call originated (file:line) so escape errors
# can point to the exact source location. The macro sets `_pending_callsite`
# before each acquire call; the _*_impl! functions call _record_borrow_from_pending!
# after claiming a slot.

"""
    _record_borrow_from_pending!(pool, tp)

Record the pending callsite for the most recently claimed slot in `tp`.
Called from `_acquire_impl!` / `_acquire_view_impl!` when `S >= 1`.
"""
@noinline function _record_borrow_from_pending!(pool::AdaptiveArrayPool, tp::AbstractTypedPool)
    callsite = pool._pending_callsite
    isempty(callsite) && return nothing
    log = pool._borrow_log
    if log === nothing
        log = IdDict{Any, String}()
        pool._borrow_log = log
    end
    @inbounds log[tp.vectors[tp.n_active]] = callsite
    pool._pending_callsite = ""   # Clear so next _set_pending_callsite! can set a fresh value
    return nothing
end

"""
    _lookup_borrow_callsite(pool, v) -> Union{Nothing, String}

Look up the callsite string for a pool backing vector. Returns `nothing` if
no borrow was recorded (S=0 or non-macro path without callsite info).
"""
@noinline function _lookup_borrow_callsite(pool::AdaptiveArrayPool, v)::Union{Nothing, String}
    log = pool._borrow_log
    log === nothing && return nothing
    return get(log, v, nothing)
end

# ==============================================================================
# Runtime Structural Mutation Detection (S >= 1)
# ==============================================================================
#
# Detects if pool-backed Array wrappers were structurally mutated (resize!, push!, etc.)
# by comparing the wrapper's MemoryRef against the backing vector's MemoryRef at rewind time.
#
# Called from _invalidate_released_slots! BEFORE poison/invalidation zeroes everything.
# Uses @warn (not throw) because throwing during rewind would skip cleanup of other pools.

# No-op fallback for extension types (e.g. CuTypedPool) and legacy (1.10) TypedPool/BitTypedPool
# (legacy structs lack arr_wrappers field — they use N-way nd_arrays cache instead)
_check_wrapper_mutation!(::AbstractTypedPool, ::Int, ::Int) = nothing

# Function barrier: zero-alloc length check for wrappers stored in Vector{Any}.
# length() is an intrinsic that works on ::Any without boxing.
@noinline _wrapper_prod_size(wrapper)::Int = length(wrapper)

# Julia 1.11+: TypedPool uses arr_wrappers (1:1 wrappers) and MemoryRef-based Array internals.
# Must not be defined on 1.10 where TypedPool has no arr_wrappers and Array has no :ref field.
@static if VERSION >= v"1.11-"

    """
        _check_wrapper_mutation!(tp::TypedPool{T}, new_n, old_n)

    Check released slots for structural mutation of cached Array wrappers.
    Compares wrapper's Memory identity and length against the backing vector.

    Called before invalidation (resize! to 0, setfield! size to zeros) while both
    wrapper and backing vector are still intact.
    """
    @noinline function _check_wrapper_mutation!(tp::TypedPool{T}, new_n::Int, old_n::Int) where {T}
        for i in (new_n + 1):old_n
            @inbounds vec = tp.vectors[i]
            # Use ccall for data pointer comparison (avoids boxing MemoryRef{T})
            vec_ptr = ccall(:jl_array_ptr, Ptr{Cvoid}, (Any,), vec)

            for N_idx in 1:length(tp.arr_wrappers)
                wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
                wrappers_for_N === nothing && continue
                wrappers = wrappers_for_N::Vector{Any}
                i > length(wrappers) && continue
                wrapper = @inbounds wrappers[i]
                wrapper === nothing && continue
                wrapper::Array  # safety: ensure wrapper is Array before ccall (TypeError vs segfault)

                # Hot path: pointer comparison via ccall (zero-alloc).
                # Check pointer FIRST — defers _wrapper_prod_size to rare mismatch path
                # to avoid dynamic dispatch boxing on wrapper::Any.
                wrapper_ptr = ccall(:jl_array_ptr, Ptr{Cvoid}, (Any,), wrapper)
                if wrapper_ptr != vec_ptr
                    # Rare path: check if stale (already invalidated by prior rewind)
                    _wrapper_prod_size(wrapper) == 0 && continue
                    dims = getfield(wrapper, :size)
                    @warn "Pool-backed Array{$T,$N_idx} wrapper reallocation detected" *
                        " (slot $i, $(N_idx)D $(dims))." *
                        " resize!/push! changed the wrapper's backing memory." *
                        " Pooling benefits (zero-alloc reuse) may be lost." maxlog = 1
                    return
                end
                # Pointer match → shared Memory → no size check needed
                # (Check 2 removed: when pointers match, wrapper cannot exceed backing capacity)
            end
        end
        return nothing
    end

    """
        _check_wrapper_mutation!(tp::BitTypedPool, new_n, old_n)

    Check released BitArray wrappers for structural mutation.
    BitArrays share their `chunks` Vector{UInt64} with the backing BitVector.
    """
    @noinline function _check_wrapper_mutation!(tp::BitTypedPool, new_n::Int, old_n::Int)
        for i in (new_n + 1):old_n
            @inbounds bv = tp.vectors[i]
            bv_chunks = bv.chunks
            bv_len = length(bv)

            for N_idx in 1:length(tp.arr_wrappers)
                wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
                wrappers_for_N === nothing && continue
                wrappers = wrappers_for_N::Vector{Any}
                i > length(wrappers) && continue
                wrapper = @inbounds wrappers[i]
                wrapper === nothing && continue

                ba = wrapper::BitArray
                # Check 1: chunks identity — detects reallocation
                if ba.chunks !== bv_chunks
                    @warn "Pool-backed BitArray: resize!/push! caused chunks reallocation " *
                        "(slot $i). Pooling benefits (zero-alloc reuse) may be lost; " *
                        "temporary extra memory retention may occur. " *
                        "Consider requesting the exact size via acquire!(pool, Bit, n) if known in advance." maxlog = 1
                    return
                end
                # Check 2: wrapper length exceeds backing
                if ba.len > bv_len
                    @warn "Pool-backed BitArray: wrapper grew beyond backing BitVector " *
                        "(slot $i, wrapper: $(ba.len) bits, backing: $bv_len). " *
                        "Pooling benefits (zero-alloc reuse) may be lost; " *
                        "temporary extra memory retention may occur. " *
                        "Consider requesting the exact size via acquire!(pool, Bit, n) if known in advance." maxlog = 1
                    return
                end
            end
        end
        return nothing
    end

end # @static if VERSION >= v"1.11-"
