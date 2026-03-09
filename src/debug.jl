# ==============================================================================
# Debugging & Safety (POOL_DEBUG escape detection)
# ==============================================================================

"""
    POOL_DEBUG

Legacy flag for escape detection. Superseded by [`POOL_SAFETY_LV`](@ref).

Setting `POOL_DEBUG[] = true` enables escape detection at `@with_pool` scope exit
(equivalent to `POOL_SAFETY_LV[] >= 2` behavior). Both flags are checked independently.

For new code, prefer `POOL_SAFETY_LV[] = 2`.

Default: `false`
"""
const POOL_DEBUG = Ref(false)

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
        # - acquire!() 1D returns: SubArray backed by pool's internal Vector
        # - view(unsafe_acquire!()): SubArray backed by unsafe_wrap'd Array
        if p isa Array
            _check_pointer_overlap(p, pool, val)
        elseif p isa BitArray
            _check_bitchunks_overlap(p, pool, val)
        end
        return
    end

    # 2. Check ReshapedArray (from acquire! N-D, wraps SubArray of pool Vector)
    if val isa Base.ReshapedArray
        p = parent(val)
        # ReshapedArray wraps SubArray{T,1,Vector{T},...}
        if p isa SubArray
            pp = parent(p)
            if pp isa Array
                _check_pointer_overlap(pp, pool, val)
            elseif pp isa BitArray
                _check_bitchunks_overlap(pp, pool, val)
            end
        end
        return
    end

    # 3. Check raw Array (from unsafe_acquire!) + element recursion
    if val isa Array
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

# Check if array memory overlaps with any pool vector.
# `original_val` is the user-visible value (e.g., SubArray) for error reporting;
# `arr` may be its parent Array used for the actual pointer comparison.
function _check_pointer_overlap(arr::Array, pool::AdaptiveArrayPool, original_val=arr)
    arr_ptr = UInt(pointer(arr))
    arr_len = length(arr) * sizeof(eltype(arr))
    arr_end = arr_ptr + arr_len

    check_overlap = function (tp)
        for v in tp.vectors
            v isa Array || continue  # Skip BitVector (no pointer(); checked via _check_bitchunks_overlap)
            v_ptr = UInt(pointer(v))
            v_len = length(v) * sizeof(eltype(v))
            v_end = v_ptr + v_len
            if !(arr_end <= v_ptr || v_end <= arr_ptr)
                _throw_pool_escape_error(original_val, eltype(v))
            end
        end
        return
    end

    # Check fixed slots
    foreach_fixed_slot(pool) do tp
        check_overlap(tp)
    end

    # Check others
    for tp in values(pool.others)
        check_overlap(tp)
    end
    return
end

@noinline function _throw_pool_escape_error(val, pool_eltype)
    error("Pool escape detected: $(summary(val)) is backed by $(pool_eltype) pool memory. " *
          "Memory will be reclaimed at @with_pool scope exit. " *
          "Return collect(x) or a computed result instead.\n" *
          "Tip: set POOL_SAFETY_LV[] = 3 for acquire!() call-site tracking.")
end

# Recursive inspection of container types (Tuple, NamedTuple, Pair, Dict, Set).
# These are common wrapper types in Julia through which pool-backed arrays
# can escape undetected when hidden inside return values.
# Note: Array element recursion is handled in the main function via _eltype_may_contain_arrays.

function _validate_pool_return(val::Tuple, pool::AdaptiveArrayPool)
    for x in val
        _validate_pool_return(x, pool)
    end
end

function _validate_pool_return(val::NamedTuple, pool::AdaptiveArrayPool)
    for x in values(val)
        _validate_pool_return(x, pool)
    end
end

function _validate_pool_return(val::Pair, pool::AdaptiveArrayPool)
    _validate_pool_return(val.first, pool)
    _validate_pool_return(val.second, pool)
end

function _validate_pool_return(val::AbstractDict, pool::AdaptiveArrayPool)
    for p in val  # each p is a Pair — reuses Pair dispatch
        _validate_pool_return(p, pool)
    end
end

function _validate_pool_return(val::AbstractSet, pool::AdaptiveArrayPool)
    for x in val
        _validate_pool_return(x, pool)
    end
end

_validate_pool_return(val, ::DisabledPool) = nothing

# ==============================================================================
# Poisoning: Fill released vectors with sentinel values (POOL_SAFETY_LV >= 2)
# ==============================================================================
#
# Poisons backing vectors with detectable values (NaN, typemax) before
# structural invalidation. This ensures stale references read obviously wrong
# data instead of silently valid old values — especially useful for
# unsafe_acquire! Array wrappers on Julia 1.10 where setfield!(:size) is
# unavailable and structural invalidation can't catch stale access.

_poison_value(::Type{T}) where {T <: AbstractFloat} = T(NaN)
_poison_value(::Type{T}) where {T <: Integer} = typemax(T)
_poison_value(::Type{Complex{T}}) where {T} = Complex{T}(_poison_value(T), _poison_value(T))
_poison_value(::Type{T}) where {T} = zero(T)  # generic fallback

_poison_fill!(v::Vector{T}) where {T} = fill!(v, _poison_value(T))
_poison_fill!(v::BitVector) = fill!(v, true)

"""
    _poison_released_vectors!(tp::AbstractTypedPool, old_n_active)

Fill released backing vectors (indices `n_active+1:old_n_active`) with sentinel
values. Called from `_invalidate_released_slots!` when `POOL_SAFETY_LV[] >= 2`,
before `resize!` zeroes the lengths.
"""
@noinline function _poison_released_vectors!(tp::AbstractTypedPool, old_n_active::Int)
    new_n = tp.n_active
    for i in (new_n + 1):old_n_active
        _poison_fill!(@inbounds tp.vectors[i])
    end
    return nothing
end
