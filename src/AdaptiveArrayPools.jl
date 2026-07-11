module AdaptiveArrayPools

using Printf
using Preferences: @load_preference
import Random
# Extend (and re-export) Random's rand!/randn! with pool-constructor methods.
# Re-exporting the SAME binding means no conflict warning when a user also
# does `using Random` (see `rand!`/`randn!` in convenience.jl).
import Random: rand!, randn!

# Public API
export AdaptiveArrayPool, acquire!, pool_stats, get_task_local_pool
export acquire_view!, acquire_array!  # Explicit naming variants
export zeros!, ones!, trues!, falses!, similar!, reshape!, default_eltype  # Convenience functions
export rand!, randn!  # Random-array convenience constructors (re-exported from Random)
export Bit  # Sentinel type for BitArray (use with acquire!, trues!, falses!)
export @with_pool, @maybe_with_pool, @safe_with_pool, @safe_maybe_with_pool
export STATIC_POOLING, MAYBE_POOLING, RUNTIME_CHECK, ESCAPE_LINT
export PoolEscapeError, EscapePoint, PoolMutationError, MutationPoint
export checkpoint!, rewind!, reset!, trim!, compact!
export enable_auto_manage!, disable_auto_manage!, auto_manage_enabled, AUTO_MANAGE
export get_task_local_cuda_pool, get_task_local_cuda_pools  # CUDA (stubs, overridden by extension)
export get_task_local_metal_pool, get_task_local_metal_pools  # Metal (stubs, overridden by extension)

# Extension API (for GPU backends)
export AbstractTypedPool, AbstractArrayPool  # For subtyping
export DisabledPool, DISABLED_CPU, pooling_enabled  # Disabled pool support
# Note: Extensions add methods to _get_pool_for_backend(::Val{:backend}) directly

# Expansion-time incidental-tail escape severity: "error" (default) | "warn" | "off".
# Read once at package load; a compile-time constant like RUNTIME_CHECK.
const ESCAPE_LINT = let v = @load_preference("escape_lint", "error")
    v in ("error", "warn", "off") ||
        error("escape_lint preference must be \"error\", \"warn\", or \"off\" (got \"$v\")")
    v
end

# All includes grouped under a single version branch
@static if VERSION >= v"1.12-"
    include("types.jl")
    include("utils.jl")
    include("acquire.jl")
    include("bitarray.jl")
    include("convenience.jl")
    include("state.jl")
    include("auto_manage.jl")
    include("task_local_pool.jl")
    include("debug.jl")
    include("macros.jl")
else
    export CACHE_WAYS, set_cache_ways!  # N-way cache configuration (legacy only)
    include("legacy/types.jl")
    include("utils.jl")
    include("legacy/acquire.jl")
    include("legacy/bitarray.jl")
    include("convenience.jl")
    include("legacy/state.jl")
    include("legacy/auto_manage.jl")
    include("task_local_pool.jl")
    include("debug.jl")
    include("macros.jl")
end

end # module
