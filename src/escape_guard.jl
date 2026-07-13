# ==============================================================================
# EscapedPoolArray — inert guard replacing a pool-backed value that escapes a
# block-form `@with_pool` scope as its implicit tail value.
#
# The macro cannot see its own call site, so whether a block's value is used
# (`out = @with_pool ...`) or discarded (loop body, bare statement) is
# undecidable at expansion time. Instead of erroring on safe-but-flagged code
# or warning on buggy code, the expansion replaces the escaping value with this
# guard: discarded → completely inert; used → throws `EscapedPoolUseError`
# with full provenance at the first trapped operation.
#
# Deliberately metadata-only — the guard does NOT hold the array: it must not
# extend the rewound buffer's GC lifetime, and "unwrap and use anyway" must be
# impossible. `typeof`/`size` are recorded at the tail-capture point (before
# rewind), where the array is still valid.
#
# Version-independent (shared by the modern and legacy trees, like macros.jl).
# ==============================================================================

struct EscapedPoolArray
    var::Symbol                    # source variable name, or :expression for anonymous tails
    arraytype::Type                # typeof(x) at escape, e.g. Matrix{Float64}
    dims::Tuple{Vararg{Int}}       # size(x) at escape; () when not an AbstractArray
    file::Union{String, Nothing}   # @with_pool scope location
    line::Union{Int, Nothing}
end

# Expansion-emitted constructor: records provenance, discards the array.
function EscapedPoolArray(
        x, var::Symbol, file::Union{String, Nothing}, line::Union{Int, Nothing},
    )
    return EscapedPoolArray(var, typeof(x), x isa AbstractArray ? size(x) : (), file, line)
end

"""
    EscapedPoolUseError <: Exception

Thrown on the first use of an [`EscapedPoolArray`](@ref) guard — a pool-backed
array that escaped its `@with_pool` scope as the block's implicit tail value.
The array was recycled by the scope's rewind, so any use of the escaped value
is invalid; the guard defers the error to the moment of actual use (a merely
discarded escape stays silent).
"""
struct EscapedPoolUseError <: Exception
    guard::EscapedPoolArray
    op::Symbol                     # the trapped operation that was attempted
end

# "`x` (3×2 Matrix{Float64})" / "an anonymous expression (Vector{Float64})"
function _show_guard_identity(io::IO, g::EscapedPoolArray)
    if g.var === :expression
        print(io, "an anonymous expression")
    else
        print(io, "`", g.var, "`")
    end
    if isempty(g.dims)
        print(io, " (", g.arraytype, ")")
    else
        print(io, " (", join(g.dims, "×"), " ", g.arraytype, ")")
    end
    return nothing
end

_guard_location(g::EscapedPoolArray) =
    g.file === nothing ? nothing : (g.line === nothing ? g.file : "$(g.file):$(g.line)")

# Non-throwing, informative display: the REPL showing a discarded-but-displayed
# guard must explain itself, not crash.
function Base.show(io::IO, g::EscapedPoolArray)
    print(io, "EscapedPoolArray: pool-backed array ")
    _show_guard_identity(io, g)
    print(io, " escaped its `@with_pool` scope")
    loc = _guard_location(g)
    loc !== nothing && print(io, " at ", loc)
    print(io, " — invalid after the scope rewinds; any use throws EscapedPoolUseError")
    return nothing
end

function Base.showerror(io::IO, e::EscapedPoolUseError)
    g = e.guard
    printstyled(io, "EscapedPoolUseError"; color = :red, bold = true)
    print(io, ": `", e.op, "` on a pool-backed array that escaped its `@with_pool` scope\n\n")
    printstyled(io, "  Escaped value: "; color = :light_black)
    _show_guard_identity(io, g)
    print(io, "\n")
    loc = _guard_location(g)
    if loc !== nothing
        printstyled(io, "  Scope location: "; color = :light_black)
        print(io, loc, "\n")
    end
    printstyled(
        io,
        "\n  A pool-backed array is only valid inside its `@with_pool` scope — the\n" *
            "  scope's rewind recycles it, so the escaping value was replaced by this\n" *
            "  inert guard at scope exit.\n";
        color = :light_black,
    )
    print(io, "\n  Fix options:\n")
    print(io, "    • materialize an owned copy inside the scope: `collect(x)` / `copy(x)`\n")
    print(io, "    • return a scalar or other non-pool value instead\n")
    print(io, "    • end the block with `nothing` if the value is not meant to be used\n")
    printstyled(
        io,
        "\n    If this escape looks like a false positive, please file an issue\n" *
            "    with a minimal reproducer so we can improve the escape detector.\n";
        color = :light_black,
    )
    return nothing
end

# Trap surface: the common array entry points all throw with provenance. Any
# operation not listed here falls through to a MethodError that names the
# guard type — self-documenting. Identity/no-dispatch operations (`===`,
# `isnothing`, `typeof`) intentionally keep working so harmless plumbing
# around a discarded value stays silent.
for f in (
        :getindex, :setindex!, :size, :length, :axes, :iterate,
        :firstindex, :lastindex, :similar, :copy, :view, :vec, :keys,
    )
    @eval Base.$f(g::EscapedPoolArray, args...) = throw(EscapedPoolUseError(g, $(QuoteNode(f))))
end
Base.broadcastable(g::EscapedPoolArray) = throw(EscapedPoolUseError(g, :broadcastable))
