# ==============================================================================
# Metal Safety: Poisoning, Escape Detection, Borrow Tracking
# ==============================================================================
# Metal-specific safety implementations for MetalAdaptiveArrayPool{R,S}.
#
# Binary safety system (R=0 off, R=1 all checks):
# - R=0: Zero overhead (all branches dead-code-eliminated)
# - R=1: Poisoning + structural invalidation + escape detection + borrow tracking
#
# Key difference: CPU uses resize!(v, 0) at Level 1 to invalidate stale SubArrays.
# On Metal, resize!(MtlVector, 0) would free GPU memory, so we use
# _resize_to_fit!(vec, 0) instead — sets dims to (0,) while preserving
# the GPU allocation (maxsize). Poisoning fills sentinel data before the shrink.
# arr_wrappers are invalidated by setting wrapper dims to zeros (matches CPU pattern).

using AdaptiveArrayPools: _runtime_check, _validate_pool_return,
    _set_pending_callsite!, _maybe_record_borrow!,
    _invalidate_released_slots!, _zero_dims_tuple,
    _throw_pool_escape_error,
    PoolRuntimeEscapeError

# ==============================================================================
# Poisoning: Fill released MtlArrays with sentinel values (R=1)
# ==============================================================================

_metal_poison_value(::Type{T}) where {T <: AbstractFloat} = T(NaN)
_metal_poison_value(::Type{T}) where {T <: Integer} = typemax(T)
_metal_poison_value(::Type{Complex{T}}) where {T} = Complex{T}(_metal_poison_value(T), _metal_poison_value(T))
_metal_poison_value(::Type{Bool}) = true
_metal_poison_value(::Type{T}) where {T} = zero(T)  # generic fallback

"""
    _metal_poison_fill!(v::MtlArray{T,1})

Fill a MtlArray with a detectable sentinel value (NaN for floats, typemax for ints).
@noinline to avoid inlining GPU kernel launch overhead into hot rewind paths.
"""
@noinline function _metal_poison_fill!(v::MtlArray{T, 1}) where {T}
    length(v) > 0 && Metal.fill!(v, _metal_poison_value(T))
    return nothing
end

# ==============================================================================
# _invalidate_released_slots! for MetalTypedPool (R=1)
# ==============================================================================
#
# Overrides the no-op fallback in base. On Metal:
# - R=0: no-op (base _rewind_typed_pool! gates with S >= 1, so never called)
# - R=1: poison released MtlArrays + invalidate arr_wrappers
# - NO resize!(mtl, 0) — would free GPU memory; use _resize_to_fit! instead

@noinline function AdaptiveArrayPools._invalidate_released_slots!(
        tp::MetalTypedPool{T, S}, old_n_active::Int, safety::Int
    ) where {T, S}
    new_n = tp.n_active
    # Poison released MtlArrays + shrink logical length to 0
    for i in (new_n + 1):old_n_active
        _metal_poison_fill!(@inbounds tp.vectors[i])
        # Shrink logical length to 0 (GPU memory preserved via _resize_to_fit!).
        # Matches CPU behavior where resize!(vec, 0) invalidates SubArray references.
        _resize_to_fit!(@inbounds(tp.vectors[i]), 0)
    end
    # Invalidate arr_wrappers for released slots (matches CPU pattern from src/state.jl)
    for N_idx in 1:length(tp.arr_wrappers)
        wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
        wrappers_for_N === nothing && continue
        wrappers = wrappers_for_N::Vector{Any}
        for i in (new_n + 1):min(old_n_active, length(wrappers))
            wrapper = @inbounds wrappers[i]
            wrapper === nothing && continue
            setfield!(wrapper::MtlArray, :dims, _zero_dims_tuple(N_idx))
        end
    end
    return nothing
end

# ==============================================================================
# Borrow Tracking: Call-site recording (R=1)
# ==============================================================================
#
# Overrides the no-op AbstractArrayPool fallbacks.
# The macro injects pool._pending_callsite = "file:line\nexpr" before acquire calls.
# These functions flush that pending info into the borrow log.

"""Record pending callsite for borrow tracking (compiles to no-op when R=0)."""
@inline function AdaptiveArrayPools._set_pending_callsite!(pool::MetalAdaptiveArrayPool{R, S}, msg::String) where {R, S}
    R >= 1 && isempty(pool._pending_callsite) && (pool._pending_callsite = msg)
    return nothing
end

"""Flush pending callsite into borrow log (compiles to no-op when R=0)."""
@inline function AdaptiveArrayPools._maybe_record_borrow!(pool::MetalAdaptiveArrayPool{R, S}, tp::AbstractTypedPool) where {R, S}
    R >= 1 && _metal_record_borrow_from_pending!(pool, tp)
    return nothing
end

@noinline function _metal_record_borrow_from_pending!(pool::MetalAdaptiveArrayPool, tp::AbstractTypedPool)
    callsite = pool._pending_callsite
    isempty(callsite) && return nothing
    log = pool._borrow_log
    if log === nothing
        log = IdDict{Any, String}()
        pool._borrow_log = log
    end
    @inbounds log[tp.vectors[tp.n_active]] = callsite
    pool._pending_callsite = ""   # Clear for next acquire
    return nothing
end

@noinline function _metal_lookup_borrow_callsite(pool::MetalAdaptiveArrayPool, v)::Union{Nothing, String}
    log = pool._borrow_log
    log === nothing && return nothing
    return get(log, v, nothing)
end

# ==============================================================================
# Escape Detection: _validate_pool_return for MtlArrays (R=1)
# ==============================================================================
#
# MtlArray views share the same device buffer, so device pointer overlap
# detection works correctly. pointer(::MtlArray) returns MtlPointer{T}.

function AdaptiveArrayPools._validate_pool_return(val, pool::MetalAdaptiveArrayPool{R, S}) where {R, S}
    R >= 1 || return nothing
    _validate_metal_return(val, pool)
    return nothing
end

function _validate_metal_return(val, pool::MetalAdaptiveArrayPool)
    # Note: Container recursion (Tuple, NamedTuple, Pair, Dict, Set, AbstractArray)
    # is duplicated from CPU's _validate_pool_return dispatch chain (src/debug.jl).
    # CPU uses multiple dispatch on pool::AdaptiveArrayPool for each container type,
    # which doesn't cover MetalAdaptiveArrayPool. We could add MetalAdaptiveArrayPool methods
    # for each container, but that creates 6+ method definitions vs. this single function.

    # MtlArray (MtlVector, MtlMatrix, etc.)
    if val isa MtlArray
        _check_metal_overlap(val, pool)
        return
    end

    # SubArray / ReshapedArray of MtlArray — defensive code.
    # Current Metal.jl: view(MtlVector, 1:n) returns MtlArray via GPUArrays derive(),
    # NOT SubArray. These branches guard against future Metal.jl behavior changes.
    if val isa SubArray
        p = parent(val)
        if p isa MtlArray
            _check_metal_overlap(p, pool, val)
        end
        return
    end

    if val isa Base.ReshapedArray
        p = parent(val)
        if p isa MtlArray
            _check_metal_overlap(p, pool, val)
        elseif p isa SubArray
            pp = parent(p)
            if pp isa MtlArray
                _check_metal_overlap(pp, pool, val)
            end
        end
        return
    end

    # Tuple
    if val isa Tuple
        for x in val
            _validate_metal_return(x, pool)
        end
        return
    end

    # NamedTuple
    if val isa NamedTuple
        for x in values(val)
            _validate_metal_return(x, pool)
        end
        return
    end

    # Pair
    if val isa Pair
        _validate_metal_return(val.first, pool)
        _validate_metal_return(val.second, pool)
        return
    end

    # AbstractDict
    if val isa AbstractDict
        for p in val
            _validate_metal_return(p, pool)
        end
        return
    end

    # AbstractSet
    if val isa AbstractSet
        for x in val
            _validate_metal_return(x, pool)
        end
        return
    end

    # Array of MtlArrays (element recursion for containers)
    if val isa AbstractArray
        ET = eltype(val)
        if !(ET <: Number) && !(ET <: AbstractString) && ET !== Symbol && ET !== Char
            for x in val
                _validate_metal_return(x, pool)
            end
        end
    end

    return
end

"""
    _check_metal_overlap(arr::MtlArray, pool, original_val=arr)

Check if a MtlArray's device memory overlaps with any pool backing MtlArray.
Throws `PoolRuntimeEscapeError` on overlap.
"""
function _check_metal_overlap(arr::MtlArray, pool::MetalAdaptiveArrayPool, original_val = arr)
    arr_ptr = pointer(arr)
    arr_buf = arr_ptr.buffer
    arr_off = Int(arr_ptr.offset)
    arr_sz = length(arr) * sizeof(eltype(arr))
    arr_end = arr_off + arr_sz

    return_site = let rs = pool._pending_return_site
        isempty(rs) ? nothing : rs
    end

    # Check fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _check_tp_metal_overlap(tp, arr_buf, arr_off, arr_end, pool, return_site, original_val)
    end

    # Check others
    for tp in values(pool.others)
        _check_tp_metal_overlap(tp, arr_buf, arr_off, arr_end, pool, return_site, original_val)
    end
    return
end

@noinline function _check_tp_metal_overlap(
        tp::AbstractTypedPool, abuf, aoff::Int, aend::Int,
        pool::MetalAdaptiveArrayPool, return_site, original_val
    )
    for v in tp.vectors
        vptr = pointer(v)
        vbuf = vptr.buffer
        voff = Int(vptr.offset)
        vsz = length(v) * sizeof(eltype(v))
        vend = voff + vsz
        if abuf === vbuf && !(aend <= voff || vend <= aoff)
            callsite = _metal_lookup_borrow_callsite(pool, v)
            _throw_pool_escape_error(original_val, eltype(v), callsite, return_site)
        end
    end
    return
end
