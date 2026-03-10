# ==============================================================================
# CUDA Safety: Poisoning, Escape Detection, Borrow Tracking
# ==============================================================================
# CUDA-specific safety implementations for CuAdaptiveArrayPool{S}.
#
# Safety levels on CUDA differ from CPU:
# - Level 0: Zero overhead (all branches dead-code-eliminated)
# - Level 1: Poisoning (NaN/sentinel fill) + N-way cache invalidation
#            (CUDA equivalent of CPU's resize!/setfield! structural invalidation)
# - Level 2: Poisoning + escape detection (_validate_pool_return for CuArrays)
# - Level 3: Full + borrow call-site registry + debug messages
#
# Key difference: CPU uses resize!(v, 0) at Level 1 to invalidate stale SubArrays.
# On CUDA, resize!(CuVector, 0) frees GPU memory, so we use poisoning instead.

using AdaptiveArrayPools: _safety_level, _validate_pool_return,
    _set_pending_callsite!, _maybe_record_borrow!,
    _invalidate_released_slots!,
    _throw_pool_escape_error, _lookup_borrow_callsite,
    POOL_DEBUG, POOL_SAFETY_LV,
    PoolRuntimeEscapeError, _shorten_location

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
# - Level 1+: poison released CuVectors + invalidate N-way view cache
# - NO resize!(cuv, 0) — would free GPU memory

@noinline function AdaptiveArrayPools._invalidate_released_slots!(
        tp::CuTypedPool{T}, old_n_active::Int, S::Int
    ) where {T}
    new_n = tp.n_active
    # Poison released CuVectors with sentinel values
    for i in (new_n + 1):old_n_active
        _cuda_poison_fill!(@inbounds tp.vectors[i])
    end
    # Invalidate N-way cache entries for released slots.
    # After poisoning, cached views point at poisoned data — clear them so
    # re-acquire creates fresh views instead of returning stale poisoned ones.
    for i in (new_n + 1):old_n_active
        base = (i - 1) * CACHE_WAYS
        for k in 1:CACHE_WAYS
            @inbounds tp.views[base + k] = nothing
            @inbounds tp.view_dims[base + k] = nothing
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

"""Record pending callsite for borrow tracking (compiles to no-op when S < 3)."""
@inline function AdaptiveArrayPools._set_pending_callsite!(pool::CuAdaptiveArrayPool{S}, msg::String) where {S}
    S >= 3 && isempty(pool._pending_callsite) && (pool._pending_callsite = msg)
    return nothing
end

"""Flush pending callsite into borrow log (compiles to no-op when S < 3)."""
@inline function AdaptiveArrayPools._maybe_record_borrow!(pool::CuAdaptiveArrayPool{S}, tp::AbstractTypedPool) where {S}
    S >= 3 && _cuda_record_borrow_from_pending!(pool, tp)
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
    (S >= 2 || POOL_DEBUG[]) || return nothing
    _validate_cuda_return(val, pool)
    return nothing
end

function _validate_cuda_return(val, pool::CuAdaptiveArrayPool)
    # CuArray (CuVector, CuMatrix, etc.)
    if val isa CuArray
        _check_cuda_pointer_overlap(val, pool)
        return
    end

    # SubArray of CuArray (if someone does view(cuarray, ...))
    if val isa SubArray
        p = parent(val)
        if p isa CuArray
            _check_cuda_pointer_overlap(p, pool, val)
        end
        return
    end

    # ReshapedArray (from reshape of CuArray view)
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

    _check = function (tp)
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

    # Check fixed slots
    AdaptiveArrayPools.foreach_fixed_slot(pool) do tp
        _check(tp)
    end

    # Check others
    for tp in values(pool.others)
        _check(tp)
    end
    return
end
