# ==============================================================================
# Debugging & Safety
# ==============================================================================

"""
    POOL_DEBUG

When `true`, `@use_pool` macros validate that returned values don't
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

"""
    pool_stats(pool::AdaptiveArrayPool)

Print statistics about pool usage (for debugging/profiling).
"""
function pool_stats(pool::AdaptiveArrayPool)
    println("AdaptiveArrayPool Statistics:")

    # Fixed slots
    for (name, tp) in [
        ("Float64", pool.float64),
        ("Float32", pool.float32),
        ("Int64", pool.int64),
        ("Int32", pool.int32),
        ("ComplexF64", pool.complexf64),
        ("Bool", pool.bool)
    ]
        if !isempty(tp.vectors)
            total_elements = sum(length(v) for v in tp.vectors; init=0)
            println("  $name (fixed slot):")
            println("    Vectors: $(length(tp.vectors))")
            println("    In Use:  $(tp.in_use)")
            println("    Total elements: $total_elements")
        end
    end

    # Others
    for (T, tp) in pool.others
        total_elements = sum(length(v) for v in tp.vectors; init=0)
        println("  $T (fallback):")
        println("    Vectors: $(length(tp.vectors))")
        println("    In Use:  $(tp.in_use)")
        println("    Total elements: $total_elements")
    end
end
