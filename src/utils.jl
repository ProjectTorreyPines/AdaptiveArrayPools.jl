# ==============================================================================
# Debugging & Safety
# ==============================================================================

"""
    POOL_DEBUG

When `true`, `@with_pool` macros validate that returned values don't
reference pool memory (which would be unsafe).

Default: `false`
"""
const POOL_DEBUG = Ref(false)

function _validate_pool_return(val, pool::AdaptiveArrayPool)
    # 0. Check BitArray / BitVector (bit-packed storage)
    #    Note: _check_bitchunks_overlap is defined in bitarray.jl (included after utils.jl)
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

# ==============================================================================
# Statistics & Pretty Printing
# ==============================================================================

# --- Helper functions for pool_stats (type-specific behavior) ---
_default_type_name(::TypedPool{T}) where {T} = string(T)

_vector_bytes(v::Vector) = Base.summarysize(v)

_count_label(::TypedPool) = "elements"

"""
    pool_stats(tp::AbstractTypedPool; io::IO=stdout, indent::Int=0, name::String="")

Print statistics for a TypedPool or BitTypedPool.
"""
function pool_stats(tp::AbstractTypedPool; io::IO = stdout, indent::Int = 0, name::String = "")
    prefix = " "^indent
    type_name = isempty(name) ? _default_type_name(tp) : name

    n_arrays = length(tp.vectors)
    if n_arrays == 0
        printstyled(io, prefix, type_name, color = :cyan)
        printstyled(io, " (empty)\n", color = :dark_gray)
        return
    end

    total_count = sum(length(v) for v in tp.vectors)
    total_bytes = sum(_vector_bytes(v) for v in tp.vectors)
    bytes_str = Base.format_bytes(total_bytes)

    # Header
    printstyled(io, prefix, type_name, color = :cyan)
    println(io)

    # Stats
    printstyled(io, prefix, "  slots: ", color = :dark_gray)
    printstyled(io, n_arrays, color = :blue)
    printstyled(io, " (active: ", color = :dark_gray)
    printstyled(io, tp.n_active, color = :blue)
    printstyled(io, ")\n", color = :dark_gray)

    printstyled(io, prefix, "  ", _count_label(tp), ": ", color = :dark_gray)
    printstyled(io, total_count, color = :blue)
    printstyled(io, " ($bytes_str)\n", color = :dark_gray)
    return nothing
end

"""
    pool_stats(pool::AdaptiveArrayPool; io::IO=stdout)

Print detailed statistics about pool usage with colored output.

# Example
```julia
pool = AdaptiveArrayPool()
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    pool_stats(pool)
end
```
"""
function pool_stats(pool::AdaptiveArrayPool; io::IO = stdout)
    # Header
    printstyled(io, "AdaptiveArrayPool", bold = true, color = :white)
    println(io)

    has_content = false

    # Fixed slots - use foreach_fixed_slot for consistency
    foreach_fixed_slot(pool) do tp
        if !isempty(tp.vectors)
            has_content = true
            name = _default_type_name(tp) * " (fixed)"
            pool_stats(tp; io, indent = 2, name)
        end
    end

    # Fallback types
    for (T, tp) in pool.others
        has_content = true
        pool_stats(tp; io, indent = 2, name = "$T (fallback)")
    end

    if !has_content
        printstyled(io, "  (empty)\n", color = :dark_gray)
    end
    return nothing
end

"""
    pool_stats(; io::IO=stdout)

Print statistics for all task-local pools (CPU and CUDA if loaded).

# Example
```julia
@with_pool begin
    v = acquire!(pool, Float64, 100)
    pool_stats()  # Shows all pool stats
end
```
"""
function pool_stats(; io::IO = stdout)
    pool_stats(:cpu; io)
    # Show CUDA pools if extension is loaded and pools exist
    try
        pool_stats(Val(:cuda); io)
    catch e
        e isa MethodError || rethrow()
        # CUDA extension not loaded - silently skip
    end
    return nothing
end

"""
    pool_stats(:cpu; io::IO=stdout)

Print statistics for the CPU task-local pool only.
"""
pool_stats(::Val{:cpu}; io::IO = stdout) = pool_stats(get_task_local_pool(); io)
pool_stats(s::Symbol; io::IO = stdout) = pool_stats(Val(s); io)

"""
    pool_stats(:cuda; io::IO=stdout)

Print statistics for CUDA task-local pools.
Requires CUDA.jl to be loaded.
"""
function pool_stats(::Val{:cuda}; io::IO = stdout)
    pools = get_task_local_cuda_pools()  # Throws MethodError if extension not loaded
    for pool in values(pools)
        pool_stats(pool; io)
    end
    return nothing
end

# ==============================================================================
# Base.show (delegates to pool_stats)
# ==============================================================================

# --- Helper for Base.show (full type name for display) ---
_show_type_name(::TypedPool{T}) where {T} = "TypedPool{$T}"

# Compact one-line show for all AbstractTypedPool
function Base.show(io::IO, tp::AbstractTypedPool)
    name = _show_type_name(tp)
    n_vectors = length(tp.vectors)
    return if n_vectors == 0
        print(io, "$name(empty)")
    else
        total = sum(length(v) for v in tp.vectors)
        label = _count_label(tp)
        print(io, "$name(slots=$n_vectors, active=$(tp.n_active), $label=$total)")
    end
end

# Multi-line show for all AbstractTypedPool
function Base.show(io::IO, ::MIME"text/plain", tp::AbstractTypedPool)
    return pool_stats(tp; io, name = _show_type_name(tp))
end

# Compact one-line show for AdaptiveArrayPool
function Base.show(io::IO, pool::AdaptiveArrayPool)
    n_types = Ref(0)
    total_vectors = Ref(0)
    total_active = Ref(0)

    foreach_fixed_slot(pool) do tp
        if !isempty(tp.vectors)
            n_types[] += 1
        end
        total_vectors[] += length(tp.vectors)
        total_active[] += tp.n_active
    end

    n_types[] += length(pool.others)
    for tp in values(pool.others)
        total_vectors[] += length(tp.vectors)
        total_active[] += tp.n_active
    end

    return print(io, "AdaptiveArrayPool(types=$(n_types[]), slots=$(total_vectors[]), active=$(total_active[]))")
end

# Multi-line show for AdaptiveArrayPool
function Base.show(io::IO, ::MIME"text/plain", pool::AdaptiveArrayPool)
    return pool_stats(pool; io)
end
