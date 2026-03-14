# ==============================================================================
# Metal Pool Display & Statistics
# ==============================================================================

using AdaptiveArrayPools: pool_stats, foreach_fixed_slot

# ==============================================================================
# pool_stats for MetalTypedPool
# ==============================================================================

"""
    pool_stats(tp::MetalTypedPool{T,S}; io::IO=stdout, indent::Int=0, name::String="")

Print statistics for a Metal typed pool.
"""
function AdaptiveArrayPools.pool_stats(tp::MetalTypedPool{T, S}; io::IO = stdout, indent::Int = 0, name::String = "") where {T, S}
    prefix = " "^indent
    type_name = isempty(name) ? string(T) : name

    n_arrays = length(tp.vectors)
    if n_arrays == 0
        printstyled(io, prefix, type_name, color = :cyan)
        printstyled(io, " (empty)\n", color = :dark_gray)
        return
    end

    # Calculate total elements and memory
    total_elements = sum(length(v) for v in tp.vectors)
    gpu_bytes = sum(sizeof(v) for v in tp.vectors)  # sizeof(MtlArray) returns GPU data size
    cpu_bytes = sum(Base.summarysize(v) for v in tp.vectors)
    gpu_str = Base.format_bytes(gpu_bytes)
    cpu_str = Base.format_bytes(cpu_bytes)

    # Header
    printstyled(io, prefix, type_name, color = :cyan)
    printstyled(io, " [Metal]", color = :magenta)
    println(io)

    # Stats
    printstyled(io, prefix, "  slots: ", color = :dark_gray)
    printstyled(io, n_arrays, color = :blue)
    printstyled(io, " (active: ", color = :dark_gray)
    printstyled(io, tp.n_active, color = :blue)
    printstyled(io, ")\n", color = :dark_gray)

    printstyled(io, prefix, "  elements: ", color = :dark_gray)
    printstyled(io, total_elements, color = :blue)
    printstyled(io, " ($gpu_str GPU + $cpu_str CPU)\n", color = :dark_gray)
    return nothing
end

# ==============================================================================
# pool_stats for MetalAdaptiveArrayPool
# ==============================================================================

"""
    pool_stats(pool::MetalAdaptiveArrayPool; io::IO=stdout)

Print statistics for a Metal adaptive array pool.
"""
function AdaptiveArrayPools.pool_stats(pool::MetalAdaptiveArrayPool{R, S}; io::IO = stdout) where {R, S}
    # Header with device info and runtime check level
    printstyled(io, "MetalAdaptiveArrayPool", bold = true, color = :magenta)
    printstyled(io, "{$R,$S}", color = :yellow)
    dev_name = try
        string(pool.device_key.name)
    catch
        string(nameof(typeof(pool.device_key)))
    end
    printstyled(io, " (device ", color = :dark_gray)
    printstyled(io, dev_name, color = :blue)
    printstyled(io, ", check=", color = :dark_gray)
    printstyled(io, _metal_check_label(R), color = :yellow)
    printstyled(io, ")\n", color = :dark_gray)

    has_content = false

    # Fixed slots
    foreach_fixed_slot(pool) do tp
        if !isempty(tp.vectors)
            has_content = true
            T = typeof(tp).parameters[1]
            pool_stats(tp; io, indent = 2, name = "$T (fixed)")
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

# Backend dispatch
function AdaptiveArrayPools.pool_stats(::Val{:metal}; io::IO = stdout)
    pools = get_task_local_metal_pools()
    for pool in values(pools)
        pool_stats(pool; io)
    end
    return nothing
end

# ==============================================================================
# Base.show for MetalTypedPool
# ==============================================================================

# Compact one-line show
function Base.show(io::IO, tp::MetalTypedPool{T, S}) where {T, S}
    n_vectors = length(tp.vectors)
    return if n_vectors == 0
        print(io, "MetalTypedPool{$T,$S}(empty)")
    else
        total = sum(length(v) for v in tp.vectors)
        print(io, "MetalTypedPool{$T,$S}(slots=$n_vectors, active=$(tp.n_active), elements=$total)")
    end
end

# Multi-line show
function Base.show(io::IO, ::MIME"text/plain", tp::MetalTypedPool{T, S}) where {T, S}
    return pool_stats(tp; io, name = "MetalTypedPool{$T,$S}")
end

# ==============================================================================
# Base.show for MetalAdaptiveArrayPool
# ==============================================================================

# Compact one-line show
function Base.show(io::IO, pool::MetalAdaptiveArrayPool{R, S}) where {R, S}
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

    return print(io, "MetalAdaptiveArrayPool{$R,$S}(check=$(_metal_check_label(R)), types=$(n_types[]), slots=$(total_vectors[]), active=$(total_active[]))")
end

# Multi-line show
function Base.show(io::IO, ::MIME"text/plain", pool::MetalAdaptiveArrayPool)
    return pool_stats(pool; io)
end
