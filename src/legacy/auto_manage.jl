# ==============================================================================
# Auto-Manage — legacy (Julia < 1.12) no-op shims.
#
# The modern src/auto_manage.jl is NOT included on the legacy path, but the SHARED
# macros.jl and task_local_pool.jl reference AUTO_MANAGE / _maybe_auto_manage! /
# register_auto_manage!. Define them here as no-ops so the shared code compiles and the
# generated @with_pool scope-entry hook DCEs away (AUTO_MANAGE === false). The public
# enable/disable/enabled API stays callable across the whole supported Julia range.
# Upgrade to Julia 1.12+ for actual background auto-management.
# ==============================================================================

# Compile-time OFF: constant-folds the shared @with_pool scope-entry hook to nothing.
const AUTO_MANAGE = false

# Referenced (and DCE'd) by get_task_local_pool's `AUTO_MANAGE && register_auto_manage!`.
register_auto_manage!(pool) = nothing

# Referenced (and DCE'd) by the @with_pool scope-entry hook.
@inline _maybe_auto_manage!(::Any) = nothing

"""
    enable_auto_manage!(; interval = 30.0, trim_interval = 120.0, factor = 10,
                           shrink_to = 1.5, min_bytes = 2^20, active = true)

No-op on Julia < 1.12 (the legacy pool architecture has no capacity management). Warns
once. Upgrade to Julia 1.12+ for background auto-management. Accepts (and ignores) the same
keywords as the 1.12+ method, so the public API is call-compatible across all supported Julia.
"""
function enable_auto_manage!(;
        interval::Real = 30.0, trim_interval::Real = 120.0, factor::Real = 10,
        shrink_to::Real = 1.5, min_bytes::Int = 2^20, active::Bool = true,
    )
    @warn "enable_auto_manage! is a no-op on Julia < 1.12 (legacy pool architecture). " *
        "Upgrade to Julia 1.12+ for background auto-management." maxlog = 1
    return nothing
end

"""
    disable_auto_manage!()

No-op on Julia < 1.12. See [`enable_auto_manage!`](@ref).
"""
disable_auto_manage!() = nothing

"""
    auto_manage_enabled() -> Bool

Always `false` on Julia < 1.12 (no auto-management). See [`enable_auto_manage!`](@ref).
"""
auto_manage_enabled() = false
