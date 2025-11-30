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
    if !(val isa SubArray)
        return
    end
    p = parent(val)

    # Check fixed slots
    for tp in (pool.float64, pool.float32, pool.int64, pool.int32, pool.complexf64, pool.bool)
        for v in tp.vectors
            if v === p
                error("Safety Violation: The function returned a SubArray backed by the AdaptiveArrayPool. This is unsafe as the memory will be reclaimed. Please return a copy (collect) or a scalar.")
            end
        end
    end

    # Check others
    for tp in values(pool.others)
        for v in tp.vectors
            if v === p
                error("Safety Violation: The function returned a SubArray backed by the AdaptiveArrayPool. This is unsafe as the memory will be reclaimed. Please return a copy (collect) or a scalar.")
            end
        end
    end
end

_validate_pool_return(val, ::Nothing) = nothing

# ==============================================================================
# Statistics & Pretty Printing
# ==============================================================================

"""
    pool_stats(tp::TypedPool{T}; io::IO=stdout, indent::Int=0, name::String="")

Print statistics for a single TypedPool.
"""
function pool_stats(tp::TypedPool{T}; io::IO=stdout, indent::Int=0, name::String="") where {T}
    prefix = " "^indent
    type_name = isempty(name) ? string(T) : name

    n_arrays = length(tp.vectors)
    if n_arrays == 0
        printstyled(io, prefix, type_name, color=:cyan)
        printstyled(io, " (empty)\n", color=:dark_gray)
        return
    end

    total_elements = sum(length(v) for v in tp.vectors)
    total_bytes = total_elements * sizeof(T)

    # Type name header
    printstyled(io, prefix, type_name, "\n", bold=true, color=:cyan)

    # Details with arrow prefix
    detail_prefix = prefix * "  "

    print(io, detail_prefix, "├─ arrays: ")
    printstyled(io, n_arrays, "\n", color=:yellow)

    print(io, detail_prefix, "├─ active: ")
    active_color = tp.n_active == 0 ? :green : :magenta
    printstyled(io, tp.n_active, "\n", color=active_color)

    print(io, detail_prefix, "├─ elements: ")
    printstyled(io, total_elements, "\n", color=:blue)

    print(io, detail_prefix, "└─ memory: ")
    printstyled(io, _format_bytes(total_bytes), "\n", color=:blue)
end

# Format bytes to human-readable string (matches @time output style)
function _format_bytes(bytes::Integer)
    if bytes < 1024
        return "$(bytes) bytes"
    elseif bytes < 1024^2
        return @sprintf("%.3f KiB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.3f MiB", bytes / 1024^2)
    else
        return @sprintf("%.3f GiB", bytes / 1024^3)
    end
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
function pool_stats(pool::AdaptiveArrayPool; io::IO=stdout)
    # Header
    printstyled(io, "AdaptiveArrayPool", bold=true, color=:white)
    println(io)

    fixed_slots = [
        ("Float64", pool.float64),
        ("Float32", pool.float32),
        ("Int64", pool.int64),
        ("Int32", pool.int32),
        ("ComplexF64", pool.complexf64),
        ("Bool", pool.bool)
    ]

    has_content = false

    # Fixed slots
    for (name, tp) in fixed_slots
        if !isempty(tp.vectors)
            has_content = true
            pool_stats(tp; io, indent=2, name="$name (fixed)")
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
end

"""
    pool_stats(; io::IO=stdout)

Print statistics for the global (Task-local) pool.

# Example
```julia
@with_pool begin
    v = acquire!(pool, Float64, 100)
    pool_stats()  # Shows global pool stats
end
```
"""
pool_stats(; io::IO=stdout) = pool_stats(get_global_pool(); io)

# ==============================================================================
# Base.show (delegates to pool_stats)
# ==============================================================================

# Compact one-line show for TypedPool
function Base.show(io::IO, tp::TypedPool{T}) where {T}
    n_vectors = length(tp.vectors)
    if n_vectors == 0
        print(io, "TypedPool{$T}(empty)")
    else
        total = sum(length(v) for v in tp.vectors)
        print(io, "TypedPool{$T}(vectors=$n_vectors, active=$(tp.n_active), elements=$total)")
    end
end

# Multi-line show for TypedPool
function Base.show(io::IO, ::MIME"text/plain", tp::TypedPool{T}) where {T}
    pool_stats(tp; io, name="TypedPool{$T}")
end

# Compact one-line show for AdaptiveArrayPool
function Base.show(io::IO, pool::AdaptiveArrayPool)
    fixed_slots = (pool.float64, pool.float32, pool.int64, pool.int32, pool.complexf64, pool.bool)
    n_types = count(tp -> !isempty(tp.vectors), fixed_slots) + length(pool.others)
    total_vectors = sum(length(tp.vectors) for tp in fixed_slots; init=0) +
                    sum(length(tp.vectors) for tp in values(pool.others); init=0)
    total_active = sum(tp.n_active for tp in fixed_slots; init=0) +
                   sum(tp.n_active for tp in values(pool.others); init=0)
    print(io, "AdaptiveArrayPool(types=$n_types, vectors=$total_vectors, active=$total_active)")
end

# Multi-line show for AdaptiveArrayPool
function Base.show(io::IO, ::MIME"text/plain", pool::AdaptiveArrayPool)
    pool_stats(pool; io)
end
