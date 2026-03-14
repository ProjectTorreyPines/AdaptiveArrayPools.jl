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
_poison_value(::Type{T}) where {T} = zero(T)  # generic fallback

_poison_fill!(v::Vector{T}) where {T} = fill!(v, _poison_value(T))
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
            vec_mem = getfield(vec, :ref).mem
            vec_len = length(vec)

            for N_idx in 1:length(tp.arr_wrappers)
                wrappers_for_N = @inbounds tp.arr_wrappers[N_idx]
                wrappers_for_N === nothing && continue
                wrappers = wrappers_for_N::Vector{Any}
                i > length(wrappers) && continue
                wrapper = @inbounds wrappers[i]
                wrapper === nothing && continue

                arr = wrapper::Array
                # Check 1: Memory identity — detects reallocation from resize!/push! beyond capacity
                if getfield(arr, :ref).mem !== vec_mem
                    @warn "Pool-backed Array{$T}: resize!/push! caused memory reallocation " *
                        "(slot $i). Pooling benefits (zero-alloc reuse) may be lost; " *
                        "temporary extra memory retention may occur. " *
                        "Request the exact size via acquire!(pool, T, n)." maxlog = 1
                    return
                end
                # Check 2: wrapper length exceeds backing vector — detects growth beyond backing
                wrapper_len = prod(getfield(arr, :size))
                if wrapper_len > vec_len
                    @warn "Pool-backed Array{$T}: wrapper grew beyond backing vector " *
                        "(slot $i, wrapper: $wrapper_len, backing: $vec_len). " *
                        "Pooling benefits (zero-alloc reuse) may be lost; " *
                        "temporary extra memory retention may occur. " *
                        "Request the exact size via acquire!(pool, T, n)." maxlog = 1
                    return
                end
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
                        "Request the exact size via acquire!(pool, Bit, n)." maxlog = 1
                    return
                end
                # Check 2: wrapper length exceeds backing
                if ba.len > bv_len
                    @warn "Pool-backed BitArray: wrapper grew beyond backing BitVector " *
                        "(slot $i, wrapper: $(ba.len) bits, backing: $bv_len). " *
                        "Pooling benefits (zero-alloc reuse) may be lost; " *
                        "temporary extra memory retention may occur. " *
                        "Request the exact size via acquire!(pool, Bit, n)." maxlog = 1
                    return
                end
            end
        end
        return nothing
    end

end # @static if VERSION >= v"1.11-"
