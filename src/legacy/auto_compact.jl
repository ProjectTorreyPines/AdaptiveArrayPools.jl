# ==============================================================================
# Auto-Compact — legacy (Julia < 1.12) no-op shims.
#
# The modern src/auto_compact.jl is NOT included on the legacy path, but the SHARED
# macros.jl and task_local_pool.jl reference AUTO_COMPACT / _maybe_auto_compact! /
# register_auto_compact!. Define them here as no-ops so the shared code compiles and the
# generated @with_pool scope-exit hook DCEs away (AUTO_COMPACT === false). The public
# enable/disable/enabled API stays callable across the whole supported Julia range.
# Upgrade to Julia 1.12+ for actual background auto-compaction.
# ==============================================================================

# Compile-time OFF: constant-folds the shared @with_pool scope-exit hook to nothing.
const AUTO_COMPACT = false

# Referenced (and DCE'd) by get_task_local_pool's `AUTO_COMPACT && register_auto_compact!`.
register_auto_compact!(pool) = nothing

# Referenced (and DCE'd) by the @with_pool scope-exit hook.
@inline _maybe_auto_compact!(::Any) = nothing

"""
    enable_auto_compact!(; interval = 30.0, factor = 10, shrink_to = 1.5,
                           min_bytes = 2^20, active = true)

No-op on Julia < 1.12 (the legacy pool architecture has no capacity compaction). Warns
once. Upgrade to Julia 1.12+ for background auto-compaction.
"""
function enable_auto_compact!(;
        interval::Real = 30.0, factor::Real = 10, shrink_to::Real = 1.5,
        min_bytes::Int = 2^20, active::Bool = true,
    )
    @warn "enable_auto_compact! is a no-op on Julia < 1.12 (legacy pool architecture). " *
        "Upgrade to Julia 1.12+ for background auto-compaction." maxlog = 1
    return nothing
end

"""
    disable_auto_compact!()

No-op on Julia < 1.12. See [`enable_auto_compact!`](@ref).
"""
disable_auto_compact!() = nothing

"""
    auto_compact_enabled() -> Bool

Always `false` on Julia < 1.12 (no auto-compaction). See [`enable_auto_compact!`](@ref).
"""
auto_compact_enabled() = false
