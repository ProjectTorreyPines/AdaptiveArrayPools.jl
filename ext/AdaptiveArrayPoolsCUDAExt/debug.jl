# ==============================================================================
# CUDA Safety: Poisoning, Escape Detection, Borrow Tracking
# ==============================================================================
# CUDA-specific safety implementations for CuAdaptiveArrayPool{S}.
#
# Safety levels on CUDA differ from CPU:
# - Level 0: Zero overhead (all branches dead-code-eliminated)
# - Level 1: Poisoning (NaN/sentinel fill) + structural invalidation via
#            _resize_to_fit!(vec, 0) + arr_wrappers invalidation (setfield!(:dims, zeros))
# - Level 2: Poisoning + escape detection (_validate_pool_return for CuArrays)
# - Level 3: Full + borrow call-site registry + debug messages
#
# Key difference: CPU uses resize!(v, 0) at Level 1 to invalidate stale SubArrays.
# On CUDA, resize!(CuVector, 0) would free GPU memory, so we use
# _resize_to_fit!(vec, 0) instead — sets dims to (0,) while preserving
# the GPU allocation (maxsize). Poisoning fills sentinel data before the shrink.
# arr_wrappers are invalidated by setting wrapper dims to zeros (matches CPU pattern).

using AdaptiveArrayPools: _safety_level, _validate_pool_return,
    _set_pending_callsite!, _maybe_record_borrow!,
    _invalidate_released_slots!, _zero_dims_tuple,
    _throw_pool_escape_error,
    POOL_DEBUG, POOL_SAFETY_LV,
    PoolRuntimeEscapeError

# ==============================================================================
# Poisoning: Fill released CuVectors with sentinel values (Level 1+)
# ==============================================================================

_cuda_poison_value(::Type{T}) where {T <: AbstractFloat} = T(NaN)
_cuda_poison_value(::Type{T}) where {T <: Integer} = typemax(T)
_cuda_poison_value(::Type{Complex{T}}) where {T} = Complex{T}(_cuda_poison_value(T), _cuda_poison_value(T))
_cuda_poison_value(::Type{Bool}) = true
_cuda_poison_value(::Type{T}) where {T} = zero(T)  # generic fallback

"""
    _cuda_poison_fill!(v::CuVector{T})

Fill a CuVector with a detectable sentinel value (NaN for floats, typemax for ints).
@noinline to avoid inlining GPU kernel launch overhead into hot rewind paths.
"""
@noinline function _cuda_poison_fill!(v::CuVector{T}) where {T}
    length(v) > 0 && CUDA.fill!(v, _cuda_poison_value(T))
    return nothing
end

# ==============================================================================
# _invalidate_released_slots! for CuTypedPool (Level 1+)
# ==============================================================================
#
# Overrides the no-op fallback in base. On CUDA:
# - Level 0: no-op (base _rewind_typed_pool! gates with S >= 1, so never called)
# - Level 1+: poison released CuVectors + invalidate arr_wrappers
# - NO resize!(cuv, 0) — would free GPU memory; use _resize_to_fit! instead

@noinline function AdaptiveArrayPools._invalidate_released_slots!(
        tp::CuTypedPool{T}, old_n_active::Int, S::Int
    ) where {T}
    new_n = tp.n_active
    # Poison released CuVectors + shrink logical length to 0
    for i in (new_n + 1):old_n_active
        _cuda_poison_fill!(@inbounds tp.vectors[i])
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
            setfield!(wrapper::CuArray, :dims, _zero_dims_tuple(N_idx))
        end
    end
    return nothing
end

# ==============================================================================
# Borrow Tracking: Call-site recording (Level 3)
# ==============================================================================
#
# Overrides the no-op AbstractArrayPool fallbacks.
# The macro injects pool._pending_callsite = "file:line\nexpr" before acquire calls.
# These functions flush that pending info into the borrow log.

"""Record pending callsite for borrow tracking (compiles to no-op when S = 0)."""
@inline function AdaptiveArrayPools._set_pending_callsite!(pool::CuAdaptiveArrayPool{S}, msg::String) where {S}
    S >= 1 && isempty(pool._pending_callsite) && (pool._pending_callsite = msg)
    return nothing
end

"""Flush pending callsite into borrow log (compiles to no-op when S = 0)."""
@inline function AdaptiveArrayPools._maybe_record_borrow!(pool::CuAdaptiveArrayPool{S}, tp::AbstractTypedPool) where {S}
    S >= 1 && _cuda_record_borrow_from_pending!(pool, tp)
    return nothing
end

@noinline function _cuda_record_borrow_from_pending!(pool::CuAdaptiveArrayPool, tp::AbstractTypedPool)
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

@noinline function _cuda_lookup_borrow_callsite(pool::CuAdaptiveArrayPool, v)::Union{Nothing, String}
    log = pool._borrow_log
    log === nothing && return nothing
    return get(log, v, nothing)
end

# ==============================================================================
# Escape Detection: _validate_pool_return for CuArrays (Level 2+)
# ==============================================================================
#
# CuArray views share the same device buffer, so device pointer overlap
# detection works correctly. pointer(::CuArray) returns CuPtr{T}.

function AdaptiveArrayPools._validate_pool_return(val, pool::CuAdaptiveArrayPool{S}) where {S}
    (S >= 1 || POOL_DEBUG[]) || return nothing
    _validate_cuda_return(val, pool)
    return nothing
end

function _validate_cuda_return(val, pool::CuAdaptiveArrayPool)
    # Note: Container recursion (Tuple, NamedTuple, Pair, Dict, Set, AbstractArray)
    # is duplicated from CPU's _validate_pool_return dispatch chain (src/debug.jl).
    # CPU uses multiple dispatch on pool::AdaptiveArrayPool for each container type,
    # which doesn't cover CuAdaptiveArrayPool. We could add CuAdaptiveArrayPool methods
    # for each container, but that creates 6+ method definitions vs. this single function.
    # Trade-off: if a new container type is added to the CPU path, it must also be added here.

    # CuArray (CuVector, CuMatrix, etc.)
    if val isa CuArray
        _check_cuda_pointer_overlap(val, pool)
        return
    end

    # SubArray / ReshapedArray of CuArray — defensive code.
    # Current CUDA.jl: view(CuVector, 1:n) returns CuArray via GPUArrays derive(),
    # NOT SubArray. These branches guard against future CUDA.jl behavior changes
    # or user-constructed SubArray{T,N,CuArray} / ReshapedArray wrappers.
    if val isa SubArray
        p = parent(val)
        if p isa CuArray
            _check_cuda_pointer_overlap(p, pool, val)
        end
        return
    end

    if val isa Base.ReshapedArray
        p = parent(val)
        if p isa CuArray
            _check_cuda_pointer_overlap(p, pool, val)
        elseif p isa SubArray
            pp = parent(p)
            if pp isa CuArray
                _check_cuda_pointer_overlap(pp, pool, val)
            end
        end
        return
    end

    # Tuple
    if val isa Tuple
        for x in val
            _validate_cuda_return(x, pool)
        end
        return
    end

    # NamedTuple
    if val isa NamedTuple
        for x in values(val)
            _validate_cuda_return(x, pool)
        end
        return
    end

    # Pair
    if val isa Pair
        _validate_cuda_return(val.first, pool)
        _validate_cuda_return(val.second, pool)
        return
    end

    # AbstractDict
    if val isa AbstractDict
        for p in val
            _validate_cuda_return(p, pool)
        end
        return
    end

    # AbstractSet
    if val isa AbstractSet
        for x in val
            _validate_cuda_return(x, pool)
        end
        return
    end

    # Array of CuArrays (element recursion for containers)
    if val isa AbstractArray
        ET = eltype(val)
        if !(ET <: Number) && !(ET <: AbstractString) && ET !== Symbol && ET !== Char
            for x in val
                _validate_cuda_return(x, pool)
            end
        end
    end

    return
end

"""
    _check_cuda_pointer_overlap(arr::CuArray, pool, original_val=arr)

Check if a CuArray's device memory overlaps with any pool backing CuVector.
Throws `PoolRuntimeEscapeError` on overlap.
"""
function _check_cuda_pointer_overlap(arr::CuArray, pool::CuAdaptiveArrayPool, original_val = arr)
    arr_ptr = UInt(pointer(arr))
    arr_bytes = length(arr) * sizeof(eltype(arr))
    arr_end = arr_ptr + arr_bytes

    return_site = let rs = pool._pending_return_site
        isempty(rs) ? nothing : rs
    end

    # Check fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _check_tp_cuda_overlap(tp, arr_ptr, arr_end, pool, return_site, original_val)
    end

    # Check others
    for tp in values(pool.others)
        _check_tp_cuda_overlap(tp, arr_ptr, arr_end, pool, return_site, original_val)
    end
    return
end

@noinline function _check_tp_cuda_overlap(
        tp::AbstractTypedPool, arr_ptr::UInt, arr_end::UInt,
        pool::CuAdaptiveArrayPool, return_site, original_val
    )
    for v in tp.vectors
        v_ptr = UInt(pointer(v))
        v_bytes = length(v) * sizeof(eltype(v))
        v_end = v_ptr + v_bytes
        if !(arr_end <= v_ptr || v_end <= arr_ptr)
            callsite = _cuda_lookup_borrow_callsite(pool, v)
            _throw_pool_escape_error(original_val, eltype(v), callsite, return_site)
        end
    end
    return
end
