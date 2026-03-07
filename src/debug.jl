# ==============================================================================
# Debugging & Safety (POOL_DEBUG escape detection)
# ==============================================================================

"""
    POOL_DEBUG

Legacy flag for escape detection. Superseded by [`POOL_SAFETY`](@ref).

Setting `POOL_DEBUG[] = true` enables escape detection at `@with_pool` scope exit
(equivalent to `POOL_SAFETY[] >= 2` behavior). Both flags are checked independently.

For new code, prefer `POOL_SAFETY[] = 2`.

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
            _check_pointer_overlap(p, pool)
        elseif p isa BitArray
            _check_bitchunks_overlap(p, pool)
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
                _check_pointer_overlap(pp, pool)
            elseif pp isa BitArray
                _check_bitchunks_overlap(pp, pool)
            end
        end
        return
    end

    # 3. Check raw Array (from unsafe_acquire!)
    return if val isa Array
        _check_pointer_overlap(val, pool)
    end
end

# Check if array memory overlaps with any pool vector
function _check_pointer_overlap(arr::Array, pool::AdaptiveArrayPool)
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
                error("Safety Violation: The function returned an Array backed by pool memory. This is unsafe as the memory will be reclaimed. Please return a copy (collect) or a scalar.")
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

_validate_pool_return(val, ::DisabledPool) = nothing
