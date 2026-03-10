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

# Check if array memory overlaps with any pool vector.
# `original_val` is the user-visible value (e.g., SubArray) for error reporting;
# `arr` may be its parent Array used for the actual pointer comparison.
function _check_pointer_overlap(arr::Array, pool::AdaptiveArrayPool, original_val = arr)
    arr_ptr = UInt(pointer(arr))
    arr_len = length(arr) * sizeof(eltype(arr))
    arr_end = arr_ptr + arr_len

    return_site = let rs = pool._pending_return_site
        isempty(rs) ? nothing : rs
    end

    check_overlap = function (tp)
        for v in tp.vectors
            v isa Array || continue  # Skip BitVector (no pointer(); checked via _check_bitchunks_overlap)
            v_ptr = UInt(pointer(v))
            v_len = length(v) * sizeof(eltype(v))
            v_end = v_ptr + v_len
            if !(arr_end <= v_ptr || v_end <= arr_ptr)
                callsite = _lookup_borrow_callsite(pool, v)
                _throw_pool_escape_error(original_val, eltype(v), callsite, return_site)
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

"""
    PoolRuntimeEscapeError <: Exception

Thrown at runtime when `_validate_pool_return` detects a pool-backed array
escaping from an `@with_pool` scope (requires `POOL_SAFETY_LV[] >= 2`).

This is the runtime counterpart of [`PoolEscapeError`](@ref) (compile-time).
"""
struct PoolRuntimeEscapeError <: Exception
    val_summary::String
    pool_eltype::String
    callsite::Union{Nothing, String}      # acquire location (LV ≥ 3)
    return_site::Union{Nothing, String}   # return location (LV ≥ 3)
end

function Base.showerror(io::IO, e::PoolRuntimeEscapeError)
    has_callsite = e.callsite !== nothing
    lv_label = has_callsite ? "POOL_SAFETY_LV ≥ 3" : "POOL_SAFETY_LV ≥ 2"

    printstyled(io, "PoolEscapeError"; color = :red, bold = true)
    printstyled(io, " (runtime, ", lv_label, ")"; color = :light_black)
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

    return if !has_callsite
        println(io)
        printstyled(io, "  Tip: "; bold = true)
        printstyled(io, "set "; color = :light_black)
        printstyled(io, "POOL_SAFETY_LV[] = 3"; bold = true)
        printstyled(io, " for acquire!() call-site tracking.\n"; color = :light_black)
    end
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
# No-op fallback for non-CPU pools (e.g. CuAdaptiveArrayPool) that lack borrow tracking fields
_validate_pool_return(val, ::AbstractArrayPool) = nothing

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
# Borrow Registry: Call-site tracking for acquire! (POOL_SAFETY_LV >= 3)
# ==============================================================================
#
# Records where each acquire! call originated (file:line) so escape errors
# can point to the exact source location. The macro sets `_pending_callsite`
# before each acquire call; the _*_impl! functions call _record_borrow_from_pending!
# after claiming a slot.

"""
    _record_borrow_from_pending!(pool, tp)

Record the pending callsite for the most recently claimed slot in `tp`.
Called from `_acquire_impl!` / `_unsafe_acquire_impl!` when `POOL_SAFETY_LV[] >= 3`.
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
no borrow was recorded (LV < 3 or non-macro path without callsite info).
"""
@noinline function _lookup_borrow_callsite(pool::AdaptiveArrayPool, v)::Union{Nothing, String}
    log = pool._borrow_log
    log === nothing && return nothing
    return get(log, v, nothing)
end
