# ==============================================================================
# CUDA Pool Display & Statistics
# ==============================================================================

using AdaptiveArrayPools: pool_stats, foreach_fixed_slot

# ==============================================================================
# pool_stats for CuTypedPool
# ==============================================================================

"""
    pool_stats(tp::CuTypedPool{T}; io::IO=stdout, indent::Int=0, name::String="")

Print statistics for a CUDA typed pool.
"""
function AdaptiveArrayPools.pool_stats(tp::CuTypedPool{T}; io::IO=stdout, indent::Int=0, name::String="") where {T}
    prefix = " "^indent
    type_name = isempty(name) ? string(T) : name

    n_arrays = length(tp.vectors)
    if n_arrays == 0
        printstyled(io, prefix, type_name, color=:cyan)
        printstyled(io, " (empty)\n", color=:dark_gray)
        return
    end

    # Calculate total elements and memory
    total_elements = sum(length(v) for v in tp.vectors)
    gpu_bytes = sum(sizeof(v) for v in tp.vectors)  # sizeof(CuArray) returns GPU data size
    cpu_bytes = sum(Base.summarysize(v) for v in tp.vectors)
    gpu_str = Base.format_bytes(gpu_bytes)
    cpu_str = Base.format_bytes(cpu_bytes)

    # Header
    printstyled(io, prefix, type_name, color=:cyan)
    printstyled(io, " [GPU]", color=:green)
    println(io)

    # Stats
    printstyled(io, prefix, "  slots: ", color=:dark_gray)
    printstyled(io, n_arrays, color=:blue)
    printstyled(io, " (active: ", color=:dark_gray)
    printstyled(io, tp.n_active, color=:blue)
    printstyled(io, ")\n", color=:dark_gray)

    printstyled(io, prefix, "  elements: ", color=:dark_gray)
    printstyled(io, total_elements, color=:blue)
    printstyled(io, " ($gpu_str GPU + $cpu_str CPU)\n", color=:dark_gray)
end

# ==============================================================================
# pool_stats for CuAdaptiveArrayPool
# ==============================================================================

"""
    pool_stats(pool::CuAdaptiveArrayPool; io::IO=stdout)

Print statistics for a CUDA adaptive array pool.
"""
function AdaptiveArrayPools.pool_stats(pool::CuAdaptiveArrayPool; io::IO=stdout)
    # Header with device info
    printstyled(io, "CuAdaptiveArrayPool", bold=true, color=:green)
    printstyled(io, " (device ", color=:dark_gray)
    printstyled(io, pool.device_id, color=:blue)
    printstyled(io, ")\n", color=:dark_gray)

    has_content = false

    # Fixed slots
    foreach_fixed_slot(pool) do tp
        if !isempty(tp.vectors)
            has_content = true
            T = typeof(tp).parameters[1]
            pool_stats(tp; io, indent=2, name="$T (fixed)")
        end
    end

    # Fallback types
    for (T, tp) in pool.others
        has_content = true
        pool_stats(tp; io, indent=2, name="$T (fallback)")
    end

    if !has_content
        printstyled(io, "  (empty)\n", color=:dark_gray)
    end
    return nothing
end

# ==============================================================================
# Base.show for CuTypedPool
# ==============================================================================

# Compact one-line show
function Base.show(io::IO, tp::CuTypedPool{T}) where {T}
    n_vectors = length(tp.vectors)
    if n_vectors == 0
        print(io, "CuTypedPool{$T}(empty)")
    else
        total = sum(length(v) for v in tp.vectors)
        print(io, "CuTypedPool{$T}(slots=$n_vectors, active=$(tp.n_active), elements=$total)")
    end
end

# Multi-line show
function Base.show(io::IO, ::MIME"text/plain", tp::CuTypedPool{T}) where {T}
    pool_stats(tp; io, name="CuTypedPool{$T}")
end

# ==============================================================================
# Base.show for CuAdaptiveArrayPool
# ==============================================================================

# Compact one-line show
function Base.show(io::IO, pool::CuAdaptiveArrayPool)
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

    print(io, "CuAdaptiveArrayPool(device=$(pool.device_id), types=$(n_types[]), slots=$(total_vectors[]), active=$(total_active[]))")
end

# Multi-line show
function Base.show(io::IO, ::MIME"text/plain", pool::CuAdaptiveArrayPool)
    pool_stats(pool; io)
end
