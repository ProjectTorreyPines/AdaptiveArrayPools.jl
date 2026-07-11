# ==============================================================================
# Macros for AdaptiveArrayPools
# ==============================================================================

# The typed-path macro upgrades (parametric static types, per-scope tp hoisting,
# nested-macrocall transform guard) target the modern (Julia >= 1.12) tree only.
# The legacy tree keeps its pre-existing expansion byte-for-byte; the escape
# lint below is version-independent and stays active everywhere.
const _MACRO_TYPED_UPGRADES = VERSION >= v"1.12-"

# ==============================================================================
# PoolEscapeError — Compile-time escape detection error
# ==============================================================================

"""Per-return-point escape detail: which expression, at which line, leaks which vars.

`incidental` carries the `(kind, detail)` pair for a Stage-2 incidental-tail escape
(a direct acquire call / broadcast-assign / assignment tail — see `_incidental_exposure`)
and is `nothing` for a Stage-1 intentional-return escape. Storing it at throw time
lets `showerror` render the message without re-classifying the expression, and is
what distinguishes the two error layouts (no `vars`-emptiness sentinel)."""
struct EscapePoint
    expr::Any
    line::Union{Int, Nothing}
    vars::Vector{Symbol}
    incidental::Union{Nothing, Tuple{Symbol, Any}}
end

# Stage-1 points carry no incidental classification.
EscapePoint(expr, line, vars) = EscapePoint(expr, line, vars, nothing)

"""Per-variable declaration site: where an escaping variable was assigned."""
struct DeclarationSite
    var::Symbol
    expr::Any
    line::Union{Int, Nothing}
    file::Union{Symbol, Nothing}
end

"""
    PoolEscapeError <: Exception

Thrown at macro expansion time when pool-backed variables are detected in
return position of `@with_pool` / `@maybe_with_pool` blocks.

This is a compile-time check with zero runtime cost.
"""
struct PoolEscapeError <: Exception
    vars::Vector{Symbol}
    file::Union{String, Nothing}
    line::Union{Int, Nothing}
    points::Vector{EscapePoint}
    var_info::Dict{Symbol, Tuple{Symbol, Vector{Symbol}}}  # var => (kind, source_vars)
    declarations::Vector{DeclarationSite}
end

PoolEscapeError(vars, file, line, points) =
    PoolEscapeError(vars, file, line, points, Dict{Symbol, Tuple{Symbol, Vector{Symbol}}}(), DeclarationSite[])

PoolEscapeError(vars, file, line, points, var_info) =
    PoolEscapeError(vars, file, line, points, var_info, DeclarationSite[])

"""Render an expression with escaped variable names highlighted in red.
Handles return, tuple, NamedTuple, array literal; falls back to print for others."""
function _render_return_expr(io::IO, expr, escaped::Set{Symbol})
    return if expr isa Symbol
        if expr in escaped
            printstyled(io, string(expr); color = :red, bold = true)
        else
            print(io, expr)
        end
    elseif expr isa Expr
        if expr.head == :return && !isempty(expr.args)
            printstyled(io, "return "; color = :light_black)
            _render_return_expr(io, expr.args[1], escaped)
        elseif expr.head == :tuple
            print(io, "(")
            for (i, arg) in enumerate(expr.args)
                i > 1 && print(io, ", ")
                _render_return_expr(io, arg, escaped)
            end
            print(io, ")")
        elseif expr.head == :(=) && length(expr.args) >= 2
            # NamedTuple key = value — only highlight value
            print(io, expr.args[1], " = ")
            _render_return_expr(io, expr.args[2], escaped)
        elseif expr.head == :vect
            print(io, "[")
            for (i, arg) in enumerate(expr.args)
                i > 1 && print(io, ", ")
                _render_return_expr(io, arg, escaped)
            end
            print(io, "]")
        else
            print(io, expr)
        end
    else
        print(io, expr)
    end
end

"""Render a `PoolEscapeError` that came from an incidental-tail pattern: the escaping
thing is a tail *expression* (an acquire call, a broadcast-assign, or an assignment),
not a directly-returned named variable — see `_incidental_exposure`. Each point's
`incidental` field carries the `(kind, detail)` classified at throw time; reuses
`_lint_message` so the wording matches the `escape_lint = \"warn\"` path exactly."""
function _showerror_incidental_tail(io::IO, e::PoolEscapeError)
    printstyled(io, "PoolEscapeError"; color = :red, bold = true)
    printstyled(io, " (compile-time)"; color = :light_black)
    println(io)
    println(io)
    for pt in e.points
        kind, detail = pt.incidental
        printstyled(io, "  "; color = :normal)
        print(io, _lint_message(kind, detail, pt.expr))
        println(io)
        loc = _format_point_location(e.file, pt.line)
        if loc !== nothing
            printstyled(io, "  ["; color = :magenta, bold = true)
            printstyled(io, loc; color = :magenta, bold = true)
            printstyled(io, "] "; color = :magenta, bold = true)
            println(io)
        end
    end
    println(io)
    printstyled(io, "  False positive?\n"; bold = true)
    printstyled(io, "    Please file an issue at "; color = :light_black)
    printstyled(io, "https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/issues"; bold = true)
    return printstyled(io, "\n    with a minimal reproducer so we can improve the escape detector.\n"; color = :light_black)
end

function Base.showerror(io::IO, e::PoolEscapeError)
    # Incidental-tail escapes carry a classified `incidental` on their points and no
    # named variable — render separately. Stage-1 and Stage-2 points never mix in a
    # single error (Stage 1 throws before Stage 2 runs), so the first point decides.
    if !isempty(e.points) && e.points[1].incidental !== nothing
        return _showerror_incidental_tail(io, e)
    end

    # Header
    printstyled(io, "PoolEscapeError"; color = :red, bold = true)
    printstyled(io, " (compile-time)"; color = :light_black)
    println(io)

    # Descriptive message
    println(io)
    n = length(e.vars)
    if n == 1
        printstyled(io, "  The following variable escapes the @with_pool scope:\n"; color = :light_black)
    else
        printstyled(io, "  The following ", n, " variables escape the @with_pool scope:\n"; color = :light_black)
    end

    # Escaped variables — one per line with classification
    println(io)
    for v in e.vars
        printstyled(io, "    "; color = :normal)
        printstyled(io, string(v); color = :red, bold = true)
        kind, sources = get(e.var_info, v, (:pool_buffer, Symbol[]))
        if kind === :container
            src_str = join(string.(sources), ", ")
            printstyled(io, "  ← wraps pool variable"; color = :light_black)
            length(sources) > 1 && printstyled(io, "s"; color = :light_black)
            printstyled(io, " (", src_str, ")\n"; color = :light_black)
        elseif kind === :alias
            printstyled(io, "  ← alias of pool variable (", string(sources[1]), ")\n"; color = :light_black)
        elseif kind === :pool_array
            printstyled(io, "  ← pool-acquired array\n"; color = :light_black)
        elseif kind === :pool_bitarray
            printstyled(io, "  ← pool-acquired BitArray\n"; color = :light_black)
        elseif kind === :pool_view
            printstyled(io, "  ← pool-acquired view\n"; color = :light_black)
        else
            printstyled(io, "  ← pool-backed temporary\n"; color = :light_black)
        end
    end

    # Declaration sites — where each escaping variable was assigned
    if !isempty(e.declarations)
        println(io)
        printstyled(io, "  Declarations:\n"; bold = true)
        for (idx, decl) in enumerate(e.declarations)
            printstyled(io, "    [", idx, "]  "; color = :light_black)
            printstyled(io, string(decl.expr); color = :cyan)
            # Fall back to macro source file when body LineNumberNode has :none (REPL/eval)
            decl_file = (decl.file !== nothing && decl.file !== :none) ? decl.file : e.file
            loc = _format_location_str(decl_file, decl.line)
            if loc !== nothing
                printstyled(io, "  ["; color = :cyan, bold = true)
                printstyled(io, loc; color = :cyan, bold = true)
                printstyled(io, "] "; color = :cyan, bold = true)
            end
            println(io)
        end
    end

    # Escaping return points with highlighted expressions
    if !isempty(e.points)
        println(io)
        label = length(e.points) == 1 ? "  Escaping return:" : "  Escaping returns:"
        printstyled(io, label, "\n"; bold = true)
        escaped_set = Set{Symbol}(e.vars)
        for (idx, pt) in enumerate(e.points)
            printstyled(io, "    [", idx, "]  "; color = :light_black)
            _render_return_expr(io, pt.expr, escaped_set)
            loc = _format_point_location(e.file, pt.line)
            if loc !== nothing
                printstyled(io, "  ["; color = :magenta, bold = true)
                printstyled(io, loc; color = :magenta, bold = true)
                printstyled(io, "] "; color = :magenta, bold = true)
            end
            println(io)
        end
    end

    # Suggestion 1: fix — collect targets are direct pool vars + container sources
    println(io)
    collect_targets = Symbol[]
    has_containers = false
    for v in e.vars
        vkind, vsources = get(e.var_info, v, (:pool_buffer, Symbol[]))
        if vkind === :container
            append!(collect_targets, vsources)
            has_containers = true
        else
            push!(collect_targets, v)
        end
    end
    unique!(collect_targets)
    sort!(collect_targets)
    collects_str = join(["collect($v)" for v in collect_targets], ", ")
    printstyled(io, "  Fix: "; bold = true)
    printstyled(io, "Use "; color = :light_black)
    printstyled(io, collects_str; bold = true)
    printstyled(io, " to return owned copies.\n"; color = :light_black)
    if has_containers
        printstyled(io, "       Copy pool variables before wrapping in containers.\n"; color = :light_black)
    end
    printstyled(io, "       Or use a regular Julia array (zeros()/Array{T}()) if it must outlive the pool scope.\n"; color = :light_black)

    # Suggestion 2: false positive → file issue
    println(io)
    printstyled(io, "  False positive?\n"; bold = true)
    printstyled(io, "    Please file an issue at "; color = :light_black)
    printstyled(io, "https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/issues"; bold = true)
    return printstyled(io, "\n    with a minimal reproducer so we can improve the escape detector.\n"; color = :light_black)
end

# Location formatting helpers (uses _short_path from debug.jl)
function _format_location_str(file, line)
    file_str = file !== nothing ? string(file) : nothing
    # Skip "none" — Julia's placeholder for REPL/eval contexts
    if file_str !== nothing && file_str != "none"
        short = _short_path(file_str)
        return line !== nothing ? short * ":" * string(line) : short
    elseif line !== nothing
        return "line " * string(line)
    end
    return nothing
end

function _format_point_location(file::Union{String, Nothing}, line::Union{Int, Nothing})
    # Skip "none" — Julia's placeholder for REPL/eval contexts
    if file !== nothing && file != "none"
        short = _short_path(file)
        return line !== nothing ? short * ":" * string(line) : short
    elseif line !== nothing
        return "line " * string(line)
    end
    return nothing
end

# Suppress stacktrace — LoadError delegates to this via showerror(io, ex.error, bt)
Base.showerror(io::IO, e::PoolEscapeError, ::Any; backtrace = true) = showerror(io, e)


# ==============================================================================
# Backend Dispatch (for extensibility)
# ==============================================================================

"""
    _get_pool_for_backend(::Val{:cpu}) -> AdaptiveArrayPool

Get task-local pool for the specified backend.

Extensions add methods for their backends (e.g., `Val{:cuda}`).
Using `Val{Symbol}` enables compile-time dispatch and full inlining,
achieving zero overhead compared to Dict-based registry.

## Example (in CUDA extension)
```julia
@inline AdaptiveArrayPools._get_pool_for_backend(::Val{:cuda}) = get_task_local_cuda_pool()
```
"""
@inline _get_pool_for_backend(::Val{:cpu}) = get_task_local_pool()

# Fallback with helpful error message (marked @noinline to keep hot path fast)
@noinline function _get_pool_for_backend(::Val{B}) where {B}
    error("Pool backend :$B is not available. Load the extension first (e.g., `using CUDA` for :cuda).")
end

# ==============================================================================
# @with_pool Macro
# ==============================================================================

"""
    @with_pool pool_name expr
    @with_pool expr
    @with_pool :backend pool_name expr
    @with_pool :backend expr

Executes code within a pooling scope with automatic lifecycle management.
Calls `checkpoint!` on entry and inserts `rewind!` at every exit point
(implicit return, explicit `return`, `break`, `continue`).

If `pool_name` is omitted, a hidden variable is used (useful when you don't
need to reference the pool directly).

## Backend Selection
Use a symbol to specify the pool backend:
- `:cpu` - CPU pools (default)
- `:cuda` - GPU pools (requires `using CUDA`)

```julia
# CPU (default)
@with_pool pool begin ... end

# GPU via CUDA
@with_pool :cuda pool begin ... end
```

## Function Definition
Wrap function definitions to inject pool lifecycle into the body:

```julia
# Long form function
@with_pool pool function compute_stats(data)
    tmp = acquire!(pool, Float64, length(data))
    tmp .= data
    mean(tmp), std(tmp)
end

# Short form function
@with_pool pool fast_sum(data) = begin
    tmp = acquire!(pool, eltype(data), length(data))
    tmp .= data
    sum(tmp)
end
```

## Block Usage
```julia
# With explicit pool name
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 1.0
    sum(v)
end

# Without pool name (for simple blocks)
@with_pool begin
    inner_function()  # inner function can use get_task_local_pool()
end
```

## Nesting
Nested `@with_pool` blocks work correctly - each maintains its own checkpoint.

```julia
@with_pool p1 begin
    v1 = acquire!(p1, Float64, 10)
    inner = @with_pool p2 begin
        v2 = acquire!(p2, Float64, 5)
        sum(v2)
    end
    # v1 is still valid here
    sum(v1) + inner
end
```

## Escape Detection

`@with_pool` statically analyzes the scope body at macro-expansion time and
rejects code whose return value would be a pool-backed array — a
[`PoolEscapeError`](@ref), thrown at expansion time (zero runtime cost). This
always covers the direct-return patterns (a bare variable, `return v`,
container literals like `(v, w)` / `[v]`), and three additional "incidental"
tail patterns that don't *look* like a `return` but still expose a pool-backed
array as the scope's value:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    acquire!(pool, Float64, 100)   # ← direct acquire-call tail
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 0.0                        # ← broadcast-assign tail (evaluates to `v`, the LHS)
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    y = v                           # ← assignment tail (evaluates to its RHS)
end
```

If the scope's last expression is meant to be discarded (a "run it for its
side effects" scope), end the block with `nothing`:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 0.0
    nothing   # ← fixed: block no longer returns a pool-backed value
end
```

Severity for these three incidental-tail patterns is controlled by the
`escape_lint` preference (via `Preferences.jl`, read once at package load as
the `ESCAPE_LINT` compile-time constant):
- `"error"` (default) — throws `PoolEscapeError`, same as the direct-return patterns.
- `"warn"` — prints the same diagnostic via `@warn` and continues (migration escape hatch).
- `"off"` — disables Stage-2 (incidental-tail) checking; direct-return patterns still always error.

```julia
using Preferences
Preferences.set_preferences!("AdaptiveArrayPools", "escape_lint" => "warn")
```

See also the "Compile-Time Detection" page in the manual for full error-message
examples and known static-analysis limitations (opaque function calls, `let`
blocks).

## Exception Behavior

`@with_pool` does **not** use `try-finally` (for inlining performance). Implications:

1. **Uncaught exceptions**: If an exception propagates out of all `@with_pool` scopes,
   pool state is invalid. Call `reset!(pool)` or use a fresh pool.
2. **Caught exceptions (nested)**: If an inner `@with_pool` throws and an outer scope
   catches, the outer scope's exit will clean up leaked inner scopes automatically
   (deferred recovery). Do not use pool operations inside the catch block.
3. **`PoolRuntimeEscapeError`**: After this error fires, the pool is poisoned.
   Fix the bug in your code and restart.
4. For full exception safety (`try-finally` guarantee), use [`@safe_with_pool`](@ref).
"""
macro with_pool(pool_name, expr)
    return _generate_pool_code(pool_name, expr, true; source = __source__)
end

macro with_pool(expr)
    pool_name = gensym(:pool)
    return _generate_pool_code(pool_name, expr, true; source = __source__)
end

# Backend-specific variants: @with_pool :cuda pool begin ... end
macro with_pool(backend::QuoteNode, pool_name, expr)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, true; source = __source__)
end

macro with_pool(backend::QuoteNode, expr)
    pool_name = gensym(:pool)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, true; source = __source__)
end

"""
    @maybe_with_pool pool_name expr
    @maybe_with_pool expr

Conditionally enables pooling based on `MAYBE_POOLING[]`.
If disabled, `pool_name` is bound to a `DisabledPool` sentinel (e.g. `DISABLED_CPU` on CPU),
and `acquire!` falls back to standard allocation.

Useful for libraries that want to let users control pooling behavior at runtime.

Like `@with_pool`, does **not** use `try-finally` — see `@with_pool` for exception
behavior details. For exception safety, use [`@safe_maybe_with_pool`](@ref).

## Function Definition
Like `@with_pool`, wrap function definitions:

```julia
@maybe_with_pool pool function process_data(data)
    tmp = acquire!(pool, Float64, length(data))  # Conditionally pooled
    tmp .= data
    sum(tmp)
end
```

## Block Usage
```julia
MAYBE_POOLING[] = false
@maybe_with_pool pool begin
    v = acquire!(pool, Float64, 100)  # Falls back to Vector{Float64}(undef, 100)
end
```
"""
macro maybe_with_pool(pool_name, expr)
    return _generate_pool_code(pool_name, expr, false; source = __source__)
end

macro maybe_with_pool(expr)
    pool_name = gensym(:pool)
    return _generate_pool_code(pool_name, expr, false; source = __source__)
end

# Backend-specific variants: @maybe_with_pool :cuda pool begin ... end
macro maybe_with_pool(backend::QuoteNode, pool_name, expr)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, false; source = __source__)
end

macro maybe_with_pool(backend::QuoteNode, expr)
    pool_name = gensym(:pool)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, false; source = __source__)
end

# ==============================================================================
# @safe_with_pool / @safe_maybe_with_pool — Exception-Safe Variants
# ==============================================================================

"""
    @safe_with_pool pool_name expr
    @safe_with_pool expr
    @safe_with_pool :backend pool_name expr
    @safe_with_pool :backend expr

Like [`@with_pool`](@ref) but uses `try-finally` to guarantee pool cleanup even
when exceptions are thrown. Use this when code inside the pool scope may throw
and you need the pool to remain in a valid state afterward.

Performance note: `try-finally` prevents Julia's compiler from inlining the pool
scope, resulting in ~35-73% overhead compared to `@with_pool`. Prefer `@with_pool`
for hot paths and use `@safe_with_pool` only when exception safety is required.

See also: [`@with_pool`](@ref), [`@safe_maybe_with_pool`](@ref)
"""
macro safe_with_pool(pool_name, expr)
    return _generate_pool_code(pool_name, expr, true; safe = true, source = __source__)
end

macro safe_with_pool(expr)
    pool_name = gensym(:pool)
    return _generate_pool_code(pool_name, expr, true; safe = true, source = __source__)
end

macro safe_with_pool(backend::QuoteNode, pool_name, expr)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, true; safe = true, source = __source__)
end

macro safe_with_pool(backend::QuoteNode, expr)
    pool_name = gensym(:pool)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, true; safe = true, source = __source__)
end

"""
    @safe_maybe_with_pool pool_name expr
    @safe_maybe_with_pool expr
    @safe_maybe_with_pool :backend pool_name expr
    @safe_maybe_with_pool :backend expr

Like [`@maybe_with_pool`](@ref) but uses `try-finally` for exception safety.
Combines the runtime pooling toggle of `@maybe_with_pool` with the exception
guarantees of `@safe_with_pool`.

See also: [`@maybe_with_pool`](@ref), [`@safe_with_pool`](@ref)
"""
macro safe_maybe_with_pool(pool_name, expr)
    return _generate_pool_code(pool_name, expr, false; safe = true, source = __source__)
end

macro safe_maybe_with_pool(expr)
    pool_name = gensym(:pool)
    return _generate_pool_code(pool_name, expr, false; safe = true, source = __source__)
end

macro safe_maybe_with_pool(backend::QuoteNode, pool_name, expr)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, false; safe = true, source = __source__)
end

macro safe_maybe_with_pool(backend::QuoteNode, expr)
    pool_name = gensym(:pool)
    return _generate_pool_code_with_backend(backend.value, pool_name, expr, false; safe = true, source = __source__)
end

# ==============================================================================
# Internal: DisabledPool Expression Generator
# ==============================================================================

"""
    _disabled_pool_expr(backend::Symbol) -> Expr

Generate expression for DisabledPool singleton based on backend.
Used when pooling is disabled to preserve backend context.
"""
function _disabled_pool_expr(backend::Symbol)
    return if backend == :cpu
        :($DISABLED_CPU)
    else
        :($(DisabledPool{backend}()))
    end
end

# ==============================================================================
# Internal: Source Location Helpers
# ==============================================================================

"""
    _find_first_lnn_index(args) -> Union{Int, Nothing}

Find the index of the first LineNumberNode in the leading prefix of `args`.

Scans sequentially, skipping `Expr(:meta, ...)` nodes (inserted by `@inline`,
`@inbounds`, etc.). Returns `nothing` as soon as a non-meta, non-LNN expression
is encountered—this prevents matching LNNs deeper in the AST.

# Example AST prefix patterns
- `[Expr(:meta,:inline), LNN, ...]` → returns 2
- `[LNN, ...]` → returns 1
- `[Expr(:meta,:inline), Expr(:call,...), LNN, ...]` → returns nothing (stopped at call)
"""
function _find_first_lnn_index(args)
    for (i, arg) in enumerate(args)
        if arg isa LineNumberNode
            return i
        elseif arg isa Expr && arg.head === :meta
            continue
        else
            return nothing
        end
    end
    return nothing
end

"""
    _ensure_body_has_toplevel_lnn(body, source)

Ensure body has a LineNumberNode pointing to user source at the top level.
- Scans first few args to handle Expr(:meta, ...) from @inline etc.
- If first LNN points to user file (same as source.file), preserve it
- If first LNN points elsewhere (e.g., macros.jl), replace with source LNN
- If no LNN exists, prepend source LNN
- If source.file === :none (REPL/eval), don't clobber valid file LNNs

Returns a new Expr to avoid mutating the original AST.
"""
function _ensure_body_has_toplevel_lnn(body, source::Union{LineNumberNode, Nothing})
    source === nothing && return body
    # Don't clobber valid file info with :none from REPL/eval
    source.file === :none && return body
    source_lnn = LineNumberNode(source.line, source.file)

    if body isa Expr && body.head === :block && !isempty(body.args)
        lnn_idx = _find_first_lnn_index(body.args)
        if lnn_idx !== nothing
            existing_lnn = body.args[lnn_idx]
            # Check if LNN already points to user file
            if existing_lnn.file == source.file
                return body  # User file LNN already present
            else
                # Replace macros.jl LNN with source LNN
                new_args = copy(body.args)
                new_args[lnn_idx] = source_lnn
                return Expr(:block, new_args...)
            end
        end
        # No LNN found, prepend source LNN
        return Expr(:block, source_lnn, body.args...)
    elseif body isa Expr && body.head === :block
        # Empty block
        return Expr(:block, source_lnn)
    else
        # Non-block body
        return Expr(:block, source_lnn, body)
    end
end

"""
    _fix_generated_lnn!(expr, source)

Replace all macro-generated LineNumberNodes with user source location.

When `quote...end` blocks in macros.jl generate code, every line gets a LNN
pointing to macros.jl. This causes Julia's stack traces to show macros.jl
instead of the user's call site. This function walks the entire AST and
replaces every LNN whose file differs from `source.file` with the source LNN.

User code (inserted via `esc()`) retains its own LNNs since those already
point to the user's file and won't be replaced.

If source.file === :none (REPL/eval), don't clobber valid file LNNs.
Modifies expr in-place and returns it.
"""
function _fix_generated_lnn!(expr, source::Union{LineNumberNode, Nothing})
    source === nothing && return expr
    # Don't clobber valid file info with :none from REPL/eval
    source.file === :none && return expr
    source_lnn = LineNumberNode(source.line, source.file)

    if expr isa Expr
        for (i, arg) in enumerate(expr.args)
            if arg isa LineNumberNode && arg.file != source.file
                expr.args[i] = source_lnn
            elseif arg isa Expr
                _fix_generated_lnn!(arg, source)
            end
        end
    end
    return expr
end

# ==============================================================================
# Internal: Compile-Time Type Assertion Helper
# ==============================================================================

"""
    _pool_type_for_backend(::Val{B}) -> Type

Returns the concrete pool type for a given backend, used at macro expansion time
to generate direct type assertions. Extensions override this for their backends.

CPU returns `AdaptiveArrayPool`, CUDA extension returns `CuAdaptiveArrayPool`.
"""
_pool_type_for_backend(::Val{:cpu}) = AdaptiveArrayPool
_pool_type_for_backend(::Val{B}) where {B} = nothing  # unregistered backend — runtime fallback

"""
    _wrap_with_dispatch(pool_name_esc, pool_getter, inner_body; backend=:cpu)

Direct type assertion: generates `let pool = getter::PoolType{RUNTIME_CHECK}`.

Since `RUNTIME_CHECK` is a compile-time `const Int`, the pool type parameter S
is resolved at compile time. `_runtime_check(pool)` returns a compile-time Bool,
enabling dead-code elimination of all safety branches when `RUNTIME_CHECK = 0`.

The pool type is resolved at macro expansion time via `_pool_type_for_backend`,
which extensions override (e.g., CUDA adds `CuAdaptiveArrayPool`).
"""
function _wrap_with_dispatch(pool_name_esc, pool_getter, inner_body; backend::Symbol = :cpu)
    PoolType = _pool_type_for_backend(Val{backend}())
    if PoolType === nothing
        # Unregistered backend: no type assertion, runtime will error in pool getter.
        return Expr(:let, Expr(:(=), pool_name_esc, pool_getter), inner_body)
    end
    _PT = GlobalRef(parentmodule(PoolType), nameof(PoolType))
    _RC = GlobalRef(@__MODULE__, :RUNTIME_CHECK)
    # RUNTIME_CHECK is const Int → compiler resolves to literal S, zero branching.
    concrete_t = :($_PT{$_RC})
    return Expr(:let, Expr(:(=), pool_name_esc, :($pool_getter::$concrete_t)), inner_body)
end

# ==============================================================================
# Internal: Code Generation
# ==============================================================================

function _generate_pool_code(pool_name, expr, force_enable; safe::Bool = false, source::Union{LineNumberNode, Nothing} = nothing)
    # Compile-time check: if pooling disabled, use DisabledPool to preserve backend context
    if !STATIC_POOLING
        disabled_pool = _disabled_pool_expr(:cpu)
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            # Function definition: inject local pool = DisabledPool at start of body
            return _generate_function_pool_code(pool_name, expr, force_enable, true, :cpu; safe, source)
        else
            return quote
                local $(esc(pool_name)) = $disabled_pool
                $(esc(expr))
            end
        end
    end

    # Check if function definition
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        return _generate_function_pool_code(pool_name, expr, force_enable, false; safe, source)
    end

    # Compile-time escape detection (zero runtime cost)
    _esc = _check_compile_time_escape(expr, pool_name, source)
    _esc !== nothing && return :(throw($_esc))

    # Compile-time container-escape warning (conservative, may have false positives)
    _warn_compile_time_container_escape(expr, pool_name, source)

    # Compile-time reassignment-escape warning (v = f(v) ambiguity)
    _warn_compile_time_reassign_escape(expr, pool_name, source)

    # Compile-time structural mutation detection (zero runtime cost)
    _check_structural_mutation(expr, pool_name, source)

    # Block logic — shared with backend-specific code generation.
    # NOTE: `@with_pool` (force_enable) and `@maybe_with_pool` (!force_enable) reuse
    # this single `inner`; the only difference below is the runtime MAYBE_POOLING[]
    # gate. So they never need manual syncing — any change to _generate_block_inner
    # (or the escape/mutation checks above, which run before this branch) applies to
    # both. The axis that DOES diverge is safe ↔ non-safe (see _generate_block_inner).
    inner = _generate_block_inner(pool_name, expr, safe, source)

    if force_enable
        return _wrap_with_dispatch(esc(pool_name), :(get_task_local_pool()), inner)
    else
        # Split branches completely to avoid Union boxing
        enabled_branch = _wrap_with_dispatch(esc(pool_name), :(get_task_local_pool()), inner)
        return quote
            if $MAYBE_POOLING[]
                $enabled_branch
            else
                # let block isolates scope — prevents user variables from being
                # captured by the dispatch closure in the if-branch (Core.Box)
                let $(esc(pool_name)) = $DISABLED_CPU
                    $(esc(expr))
                end
            end
        end
    end
end

# ==============================================================================
# Internal: Shared Block-Form Inner Body Generator
# ==============================================================================
#
# Shared between _generate_pool_code (CPU) and _generate_pool_code_with_backend.
# Produces the `inner` quote block containing checkpoint → body → validate → rewind.

"""
    _generate_block_inner(pool_name, expr, safe, source) -> Expr

Generate the inner body for block-form `@with_pool`. Handles both safe (try-finally)
and direct-rewind paths. Used by both CPU and backend-specific code generators.

Does NOT handle the outer dispatch wrapper or MAYBE_POOLING branching — callers
handle those after receiving the inner body.
"""
function _generate_block_inner(pool_name, expr, safe::Bool, source)
    # @goto safety check (direct-rewind path only)
    if !safe
        _check_unsafe_goto(expr)
    end

    all_types = _extract_acquire_types(expr, pool_name)
    local_vars = _extract_local_assignments(expr)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)
    use_typed = !has_dynamic && !isempty(static_types)

    # Generate checkpoint/rewind calls (esc'd, for inner body template)
    if use_typed
        checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
        rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
    else
        checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
        rewind_call = _generate_lazy_rewind_call(esc(pool_name))
    end

    transformed_expr = use_typed ? _transform_acquire_calls(expr, pool_name) : expr
    tp_bindings = Expr[]
    if use_typed && _MACRO_TYPED_UPGRADES
        tp_vars, transformed_expr = _hoist_typed_pools(transformed_expr, static_types)
        for (t, v) in tp_vars
            push!(tp_bindings, :(local $(esc(v)) = $get_typed_pool!($(esc(pool_name)), $(esc(t)))))
        end
    end
    transformed_expr = _inject_pending_callsite(transformed_expr, pool_name, expr)

    # ── safe ↔ non-safe divergence (the axis that MUST be kept in sync) ──────────
    # These two branches are fundamentally different control flow, NOT a shared body
    # with a flag: `safe` wraps the work in try/finally (rewind guaranteed even when
    # an exception escapes the outermost scope); non-safe uses a direct rewind +
    # entry-depth guard + leaked-scope cleanup + break/continue injection.
    #
    # The split is deliberate and measurement-justified: try/finally costs a fixed
    # ~4–11 ns/scope on Julia 1.12+ (≈20% on a hot typed scope, persists with real
    # work) — unacceptable for the zero-overhead default, so `safe` stays opt-in.
    #
    # ⚠️ Any change to the checkpoint/rewind/tp-hoisting contract above MUST be
    # verified on BOTH branches — the parametrized divergence matrix in
    # test/test_macros.jl ("divergence matrix") is the mechanical guard. In
    # particular, tp bindings must stay inside the try (see below): emitting them
    # before it would leak the checkpoint if get_typed_pool! throws.
    if safe
        transformed_expr = _transform_return_stmts(transformed_expr, pool_name)
        return quote
            $(_auto_manage_hook(pool_name))
            $checkpoint_call
            # Hoisted tp bindings live INSIDE the try: get_typed_pool! can allocate
            # (fallback slow path) and thus throw, and the whole point of the safe
            # form is that the finally rewind runs after the checkpoint no matter
            # what. Emitting them before the try would leak the checkpoint on throw.
            try
                $(tp_bindings...)
                local _result = $(esc(transformed_expr))
                if $_RUNTIME_CHECK_REF($(esc(pool_name)))
                    $_validate_pool_return(_result, $(esc(pool_name)))
                end
                _result
            finally
                $rewind_call
            end
        end
    else
        entry_depth_var = gensym(:_entry_depth)
        raw_rewind = _generate_raw_rewind_call(pool_name, use_typed, static_types)
        raw_guard = _generate_raw_entry_depth_guard(pool_name, entry_depth_var)

        transformed_expr = _transform_return_stmts(
            transformed_expr, pool_name;
            rewind_call = raw_rewind,
            entry_depth_guard = raw_guard
        )
        transformed_expr = _transform_break_continue(transformed_expr, raw_rewind, raw_guard)

        return quote
            local $(esc(entry_depth_var)) = $(esc(pool_name))._current_depth
            $(_auto_manage_hook(pool_name))
            $checkpoint_call
            $(tp_bindings...)
            local _result = $(esc(transformed_expr))
            # Leaked scope cleanup BEFORE validation: if an inner @with_pool threw
            # without rewind, _current_depth is still the inner depth. Validation
            # uses _current_depth via _scope_boundary, so we must normalize first.
            if $_RUNTIME_CHECK_REF($(esc(pool_name))) && $(esc(pool_name))._current_depth > $(esc(entry_depth_var)) + 1
                $_WARN_LEAKED_SCOPE_REF($(esc(pool_name)), $(esc(entry_depth_var)))
            end
            while $(esc(pool_name))._current_depth > $(esc(entry_depth_var)) + 1
                $_REWIND_REF($(esc(pool_name)))
            end
            if $_RUNTIME_CHECK_REF($(esc(pool_name)))
                $_validate_pool_return(_result, $(esc(pool_name)))
            end
            $rewind_call
            _result
        end
    end
end

"""
    _generate_function_inner(pool_name, expr, safe, source)

Shared helper for function-form code generation (both CPU and backend variants).
Like `_generate_block_inner` but does NOT apply `_transform_break_continue` —
`break`/`continue` cannot exit a function scope.
"""
function _generate_function_inner(pool_name, expr, safe::Bool, source)
    # @goto safety check (direct-rewind path only)
    if !safe
        _check_unsafe_goto(expr)
    end

    all_types = _extract_acquire_types(expr, pool_name)
    local_vars = _extract_local_assignments(expr)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)
    use_typed = !has_dynamic && !isempty(static_types)

    # Generate checkpoint/rewind calls (esc'd, for inner body template)
    if use_typed
        checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
        rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
    else
        checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
        rewind_call = _generate_lazy_rewind_call(esc(pool_name))
    end

    transformed_expr = use_typed ? _transform_acquire_calls(expr, pool_name) : expr
    tp_bindings = Expr[]
    if use_typed && _MACRO_TYPED_UPGRADES
        tp_vars, transformed_expr = _hoist_typed_pools(transformed_expr, static_types)
        for (t, v) in tp_vars
            push!(tp_bindings, :(local $(esc(v)) = $get_typed_pool!($(esc(pool_name)), $(esc(t)))))
        end
    end
    transformed_expr = _inject_pending_callsite(transformed_expr, pool_name, expr)

    if safe
        transformed_expr = _transform_return_stmts(transformed_expr, pool_name)
        return quote
            $(_auto_manage_hook(pool_name))
            $checkpoint_call
            # Hoisted tp bindings live INSIDE the try (see _generate_block_inner):
            # get_typed_pool! can throw on the fallback slow path, and the finally
            # rewind must run after the checkpoint regardless.
            try
                $(tp_bindings...)
                local _result = $(esc(transformed_expr))
                if $_RUNTIME_CHECK_REF($(esc(pool_name)))
                    $_validate_pool_return(_result, $(esc(pool_name)))
                end
                _result
            finally
                $rewind_call
            end
        end
    else
        entry_depth_var = gensym(:_entry_depth)
        raw_rewind = _generate_raw_rewind_call(pool_name, use_typed, static_types)
        raw_guard = _generate_raw_entry_depth_guard(pool_name, entry_depth_var)

        # Function form: transform returns with rewind, but NO break/continue transform
        transformed_expr = _transform_return_stmts(
            transformed_expr, pool_name;
            rewind_call = raw_rewind,
            entry_depth_guard = raw_guard
        )

        return quote
            local $(esc(entry_depth_var)) = $(esc(pool_name))._current_depth
            $(_auto_manage_hook(pool_name))
            $checkpoint_call
            $(tp_bindings...)
            local _result = $(esc(transformed_expr))
            # Leaked scope cleanup BEFORE validation: if an inner @with_pool threw
            # without rewind, _current_depth is still the inner depth. Validation
            # uses _current_depth via _scope_boundary, so we must normalize first.
            if $_RUNTIME_CHECK_REF($(esc(pool_name))) && $(esc(pool_name))._current_depth > $(esc(entry_depth_var)) + 1
                $_WARN_LEAKED_SCOPE_REF($(esc(pool_name)), $(esc(entry_depth_var)))
            end
            while $(esc(pool_name))._current_depth > $(esc(entry_depth_var)) + 1
                $_REWIND_REF($(esc(pool_name)))
            end
            if $_RUNTIME_CHECK_REF($(esc(pool_name)))
                $_validate_pool_return(_result, $(esc(pool_name)))
            end
            $rewind_call
            _result
        end
    end
end

# ==============================================================================
# Internal: Backend-Specific Code Generation
# ==============================================================================

"""
    _generate_pool_code_with_backend(backend, pool_name, expr, force_enable)

Generate pool code for a specific backend (e.g., :cuda, :cpu).
Uses `_get_pool_for_backend(Val{backend}())` for zero-overhead dispatch.

Includes type-specific checkpoint/rewind optimization (same as regular @with_pool).
"""
function _generate_pool_code_with_backend(backend::Symbol, pool_name, expr, force_enable::Bool; safe::Bool = false, source::Union{LineNumberNode, Nothing} = nothing)
    # Compile-time check: if pooling disabled, use DisabledPool to preserve backend context
    if !STATIC_POOLING
        disabled_pool = _disabled_pool_expr(backend)
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            return _generate_function_pool_code_with_backend(backend, pool_name, expr, force_enable, true; safe, source)
        else
            return quote
                local $(esc(pool_name)) = $disabled_pool
                $(esc(expr))
            end
        end
    end

    # Runtime check for @maybe_with_pool :backend (force_enable=false)
    if !force_enable
        disabled_pool = _disabled_pool_expr(backend)
        # Check if function definition
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            return _generate_function_pool_code_with_backend(backend, pool_name, expr, false, false; safe, source)
        end

        # Compile-time escape detection (zero runtime cost)
        _esc = _check_compile_time_escape(expr, pool_name, source)
        _esc !== nothing && return :(throw($_esc))

        # Compile-time container-escape warning (conservative, may have false positives)
        _warn_compile_time_container_escape(expr, pool_name, source)

        # Compile-time reassignment-escape warning (v = f(v) ambiguity)
        _warn_compile_time_reassign_escape(expr, pool_name, source)

        # Compile-time structural mutation detection (zero runtime cost)
        _check_structural_mutation(expr, pool_name, source)

        # Block logic with runtime check
        inner = _generate_block_inner(pool_name, expr, safe, source)
        pool_getter = :($_get_pool_for_backend($(Val{backend}())))
        enabled_branch = _wrap_with_dispatch(esc(pool_name), pool_getter, inner; backend)
        return quote
            if $MAYBE_POOLING[]
                $enabled_branch
            else
                let $(esc(pool_name)) = $disabled_pool
                    $(esc(expr))
                end
            end
        end
    end

    # Check if function definition
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        return _generate_function_pool_code_with_backend(backend, pool_name, expr, true, false; safe, source)
    end

    # Compile-time escape detection (zero runtime cost)
    _esc = _check_compile_time_escape(expr, pool_name, source)
    _esc !== nothing && return :(throw($_esc))

    # Compile-time container-escape warning (conservative, may have false positives)
    _warn_compile_time_container_escape(expr, pool_name, source)

    # Compile-time reassignment-escape warning (v = f(v) ambiguity)
    _warn_compile_time_reassign_escape(expr, pool_name, source)

    # Compile-time structural mutation detection (zero runtime cost)
    _check_structural_mutation(expr, pool_name, source)

    # Block logic (force_enable=true path)
    inner = _generate_block_inner(pool_name, expr, safe, source)
    pool_getter = :($_get_pool_for_backend($(Val{backend}())))
    return _wrap_with_dispatch(esc(pool_name), pool_getter, inner; backend)
end

"""
    _generate_function_pool_code_with_backend(backend, pool_name, func_def, force_enable, disable_pooling)

Generate function code for a specific backend (e.g., :cuda).
Wraps the function body with pool getter, checkpoint, and rewind.

When `disable_pooling=true` (STATIC_POOLING=false), generates DisabledPool binding.
When `force_enable=true` (@with_pool), always uses the real pool.
When `force_enable=false` (@maybe_with_pool), generates MAYBE_POOLING[] runtime check.
"""
function _generate_function_pool_code_with_backend(backend::Symbol, pool_name, func_def, force_enable::Bool, disable_pooling::Bool; safe::Bool = false, source::Union{LineNumberNode, Nothing} = nothing)
    def_head = func_def.head
    call_expr = func_def.args[1]
    body = func_def.args[2]

    if disable_pooling
        disabled_pool = _disabled_pool_expr(backend)
        new_body = quote
            local $(esc(pool_name)) = $disabled_pool
            $(esc(body))
        end
        # Ensure new_body has source location for proper stack traces
        new_body = _ensure_body_has_toplevel_lnn(new_body, source)
        return Expr(def_head, esc(call_expr), new_body)
    end

    # Compile-time escape detection (zero runtime cost)
    _esc = _check_compile_time_escape(body, pool_name, source)
    _esc !== nothing && return :(throw($_esc))

    # Compile-time container-escape warning (conservative, may have false positives)
    _warn_compile_time_container_escape(body, pool_name, source)

    # Compile-time reassignment-escape warning (v = f(v) ambiguity)
    _warn_compile_time_reassign_escape(body, pool_name, source)

    # Compile-time structural mutation detection (zero runtime cost)
    _check_structural_mutation(body, pool_name, source)

    # Function body inner — no break/continue transform (can't break out of a function)
    inner = _generate_function_inner(pool_name, body, safe, source)

    # Use Val{backend}() for compile-time dispatch
    pool_getter = :($_get_pool_for_backend($(Val{backend}())))

    if force_enable
        new_body = quote
            $(_wrap_with_dispatch(esc(pool_name), pool_getter, inner; backend))
        end
    else
        disabled_pool = _disabled_pool_expr(backend)
        enabled_branch = _wrap_with_dispatch(esc(pool_name), pool_getter, inner; backend)
        new_body = quote
            if $MAYBE_POOLING[]
                $enabled_branch
            else
                let $(esc(pool_name)) = $disabled_pool
                    $(esc(body))
                end
            end
        end
    end

    # Ensure new_body has source location for proper stack traces
    new_body = _ensure_body_has_toplevel_lnn(new_body, source)
    _fix_generated_lnn!(new_body, source)  # Fix generated LNNs for accurate stack traces
    return Expr(def_head, esc(call_expr), new_body)
end

function _generate_function_pool_code(pool_name, func_def, force_enable, disable_pooling, backend::Symbol = :cpu; safe::Bool = false, source::Union{LineNumberNode, Nothing} = nothing)
    def_head = func_def.head
    call_expr = func_def.args[1]
    body = func_def.args[2]

    if disable_pooling
        disabled_pool = _disabled_pool_expr(backend)
        new_body = quote
            local $(esc(pool_name)) = $disabled_pool
            $(esc(body))
        end
        # Ensure new_body has source location for proper stack traces
        new_body = _ensure_body_has_toplevel_lnn(new_body, source)
        return Expr(def_head, esc(call_expr), new_body)
    end

    # Compile-time escape detection (zero runtime cost)
    _esc = _check_compile_time_escape(body, pool_name, source)
    _esc !== nothing && return :(throw($_esc))

    # Compile-time container-escape warning (conservative, may have false positives)
    _warn_compile_time_container_escape(body, pool_name, source)

    # Compile-time reassignment-escape warning (v = f(v) ambiguity)
    _warn_compile_time_reassign_escape(body, pool_name, source)

    # Compile-time structural mutation detection (zero runtime cost)
    _check_structural_mutation(body, pool_name, source)

    # Function body inner — no break/continue transform (can't break out of a function)
    inner = _generate_function_inner(pool_name, body, safe, source)

    if force_enable
        new_body = _wrap_with_dispatch(esc(pool_name), :(get_task_local_pool()), inner)
    else
        disabled_pool = _disabled_pool_expr(backend)
        enabled_branch = _wrap_with_dispatch(esc(pool_name), :(get_task_local_pool()), inner)
        new_body = quote
            if $MAYBE_POOLING[]
                $enabled_branch
            else
                let $(esc(pool_name)) = $disabled_pool
                    $(esc(body))
                end
            end
        end
    end

    # Ensure new_body has source location for proper stack traces
    new_body = _ensure_body_has_toplevel_lnn(new_body, source)
    _fix_generated_lnn!(new_body, source)  # Fix generated LNNs for accurate stack traces
    return Expr(def_head, esc(call_expr), new_body)
end

function _is_function_def(expr)
    if expr.head == :function
        return true
    end
    if expr.head == :(=) && length(expr.args) >= 1
        lhs = expr.args[1]
        while Meta.isexpr(lhs, [:where, :(::)])
            lhs = lhs.args[1]
        end
        return Meta.isexpr(lhs, :call)
    end
    return false
end

# ==============================================================================
# Internal: Type Extraction for Optimized checkpoint!/rewind!
# ==============================================================================

"""
    _extract_local_assignments(expr, locals=Set{Symbol}()) -> Set{Symbol}

Find all symbols that are assigned locally in the expression body.
These cannot be used for typed checkpoint since they're defined after checkpoint!.

Detects patterns like: `T = eltype(x)`, `local T = ...`, etc.
"""
function _extract_local_assignments(expr, locals = Set{Symbol}())
    if expr isa Expr
        if expr.head == :(=) && length(expr.args) >= 1
            lhs = expr.args[1]
            # Simple assignment: T = ...
            if lhs isa Symbol
                push!(locals, lhs)
                # Typed assignment: T::Type = ...
            elseif Meta.isexpr(lhs, :(::)) && length(lhs.args) >= 1 && lhs.args[1] isa Symbol
                push!(locals, lhs.args[1])
            end
        elseif expr.head == :local
            # local T or local T = ...
            for arg in expr.args
                if arg isa Symbol
                    push!(locals, arg)
                elseif Meta.isexpr(arg, :(=)) && arg.args[1] isa Symbol
                    push!(locals, arg.args[1])
                end
            end
        end
        # Recurse
        for arg in expr.args
            _extract_local_assignments(arg, locals)
        end
    end
    return locals
end

"""
    _extract_acquire_types(expr, target_pool) -> Set{Any}

Extract type arguments from acquire/convenience function calls in an expression.
Only extracts types from calls where the first argument matches `target_pool`.
This prevents AST pollution when multiple pools are used in the same block.

Supported functions:
- `acquire!` and its alias `acquire_array!`
- `acquire_view!`
- `zeros!`, `ones!`, `similar!`

Handles various forms:
- `acquire!(pool, Type, dims...)`: extracts Type directly
- `acquire!(pool, x)`: generates `eltype(x)` expression
- `zeros!(pool, dims...)` / `ones!(pool, dims...)`: Float64 (default)
- `zeros!(pool, Type, dims...)` / `ones!(pool, Type, dims...)`: extracts Type
- `similar!(pool, x)`: generates `eltype(x)` expression
- `similar!(pool, x, Type, ...)`: extracts Type
"""
function _extract_acquire_types(expr, target_pool, types = Set{Any}())
    if expr isa Expr
        # Match: function calls with pool argument
        if expr.head == :call && length(expr.args) >= 3
            fn = expr.args[1]
            pool_arg = expr.args[2]

            # Only process if pool argument matches our target pool
            if pool_arg == target_pool
                # All acquire function names (including aliases)
                acquire_names = (:acquire!, :acquire_view!, :acquire_array!)

                # Get function name (handle qualified names)
                fn_name = fn
                if fn isa Expr && fn.head == :. && length(fn.args) >= 2
                    qn = fn.args[end]
                    if qn isa QuoteNode
                        fn_name = qn.value
                    end
                end

                nargs = length(expr.args)

                # acquire!/acquire_view!/acquire_array!
                if fn in acquire_names || fn_name in acquire_names
                    if nargs >= 4
                        # acquire!(pool, Type, dims...) - traditional form
                        push!(types, expr.args[3])
                    elseif nargs == 3
                        # acquire!(pool, x) - similar-style form
                        push!(types, Expr(:call, :eltype, expr.args[3]))
                    end
                    # trues!/falses! (always uses Bit type)
                elseif fn in (:trues!, :falses!) || fn_name in (:trues!, :falses!)
                    push!(types, :Bit)
                    # zeros!/ones!/rand!/randn!
                    # NOTE: for rand!(pool, S, dims) (collection/range form), the
                    # third arg S is not a syntactic type, so the default_eltype
                    # branch is taken here (a harmless over-registration). The real
                    # eltype(S) is recorded at runtime inside `_rand_impl!`.
                elseif fn in (:zeros!, :ones!, :rand!, :randn!) || fn_name in (:zeros!, :ones!, :rand!, :randn!)
                    if nargs >= 3
                        third_arg = expr.args[3]
                        # Check if third arg looks like a type (Symbol starting with uppercase or curly)
                        if _looks_like_type(third_arg)
                            push!(types, third_arg)
                        else
                            # No type specified, use default_eltype(pool) - resolved at compile time
                            # CPU: Float64, CUDA: Float32 (via default_eltype dispatch)
                            push!(types, Expr(:call, :default_eltype, target_pool))
                        end
                    end
                    # similar!
                elseif fn in (:similar!,) || fn_name in (:similar!,)
                    if nargs == 3
                        # similar!(pool, x) - same type as x
                        push!(types, Expr(:call, :eltype, expr.args[3]))
                    elseif nargs >= 4
                        fourth_arg = expr.args[4]
                        if _looks_like_type(fourth_arg)
                            # similar!(pool, x, Type, ...) - explicit type
                            push!(types, fourth_arg)
                        else
                            # similar!(pool, x, dims...) - same type as x
                            push!(types, Expr(:call, :eltype, expr.args[3]))
                        end
                    end
                    # reshape!
                elseif fn in (:reshape!,) || fn_name in (:reshape!,)
                    # reshape!(pool, A, dims...) — extract eltype(A) from second arg
                    if nargs >= 3
                        push!(types, Expr(:call, :eltype, expr.args[3]))
                    end
                end
            end
        end
        # Recurse into sub-expressions
        for arg in expr.args
            _extract_acquire_types(arg, target_pool, types)
        end
    end
    return types
end

"""
    _looks_like_type(expr) -> Bool

Heuristic to check if an expression looks like a type.
Returns true for: uppercase Symbols (Float64, Int), curly expressions (Vector{T}), GlobalRef to types.
"""
function _looks_like_type(expr)
    if expr isa Symbol
        s = string(expr)
        return !isempty(s) && isuppercase(first(s))
    elseif expr isa Expr && expr.head == :curly
        return true
    elseif expr isa GlobalRef
        return true
    end
    return false
end

"""
    _uses_local_var(expr, local_vars) -> Bool

Check if an expression uses any local variable (recursively).
Handles field access (x.y.z) and indexing (x[i]) by checking the base variable.

This is used to detect cases like `acquire!(pool, cp1d.t_i_average)` where
`cp1d` is defined locally - the eltype expression can't be evaluated at
checkpoint time since cp1d doesn't exist yet.
"""
function _uses_local_var(expr, local_vars)
    if expr isa Symbol
        return expr in local_vars
    elseif expr isa Expr
        if expr.head == :. && !isempty(expr.args)
            # Field access: cp1d.t_i_average → check if cp1d is local
            return _uses_local_var(expr.args[1], local_vars)
        elseif expr.head == :ref && !isempty(expr.args)
            # Indexing: arr[i] → check if arr is local
            return _uses_local_var(expr.args[1], local_vars)
        else
            # Other expressions - check all args recursively
            return any(_uses_local_var(arg, local_vars) for arg in expr.args)
        end
    end
    return false
end

"""
    _filter_static_types(types, local_vars=Set{Symbol}()) -> (static_types, has_dynamic)

Filter types for typed checkpoint/rewind generation.

- Symbols NOT in local_vars are passed through (type parameters, global types)
- Symbols IN local_vars trigger fallback (defined after checkpoint!)
- Parametric types like `Vector{Float64}` are static iff every free name inside
  the curly resolves outside the block (global type or `where` param); a curly
  over a local name (e.g. `Vector{T}` with `T` assigned in-block) triggers fallback
- `eltype(x)` expressions: usable if `x` does NOT reference a local variable

Type parameters (T, S from `where` clause) resolve to concrete types at runtime.
Local variables (T = eltype(x)) are defined after checkpoint! and cannot be used.
"""
function _filter_static_types(types, local_vars = Set{Symbol}())
    static_types = Any[]
    has_dynamic = false

    for t in types
        if t isa Symbol
            if t in local_vars
                # Local variable like T = eltype(x) - defined after checkpoint!
                # Must fall back to full checkpoint
                has_dynamic = true
            else
                # Type parameter or global type - safe to use
                push!(static_types, t)
            end
        elseif t isa Expr
            if t.head == :curly
                # Parametric type literal like Vector{Float64} / Foo{T}.
                # Static iff every free name resolves outside the block
                # (global type or `where` param) — same rule as eltype(x) below.
                # Legacy tree (< 1.12): keep the pre-existing conservative behavior
                # of always falling back to dynamic for curly type literals.
                if !_MACRO_TYPED_UPGRADES
                    has_dynamic = true
                elseif _uses_local_var(t, local_vars)
                    has_dynamic = true
                else
                    push!(static_types, t)
                end
            elseif t.head == :call && length(t.args) >= 2 && t.args[1] == :eltype
                # eltype(x) expression from acquire!(pool, x) form
                inner_arg = t.args[2]
                if _uses_local_var(inner_arg, local_vars)
                    # x (or its base in x.field) is defined locally
                    # Can't use at checkpoint time (checkpoint runs before definition)
                    has_dynamic = true
                else
                    # x is external (function param, global, etc.) - safe to use
                    push!(static_types, t)
                end
            elseif t.head == :call && length(t.args) >= 2 && t.args[1] == :default_eltype
                # default_eltype(pool) expression from zeros!(pool, 10) etc.
                # This is a compile-time constant (Float64 for CPU, Float32 for CUDA)
                # Safe to use - pool type is known at compile time
                inner_arg = t.args[2]
                if _uses_local_var(inner_arg, local_vars)
                    has_dynamic = true
                else
                    push!(static_types, t)
                end
            else
                # Other expressions - treat as dynamic
                has_dynamic = true
            end
        else
            # GlobalRef or other concrete type reference
            push!(static_types, t)
        end
    end

    return static_types, has_dynamic
end

"""
    _generate_typed_checkpoint_call(pool_expr, types)

Generate bitmask-aware checkpoint call. When types are known at compile time,
emits a conditional:
- if touched types ⊆ tracked types → typed checkpoint (fast path)
- otherwise → `_typed_lazy_checkpoint!` (typed checkpoint + set bit 14 for
  lazy first-touch checkpointing of extra types touched by helpers)
"""
function _generate_typed_checkpoint_call(pool_expr, types)
    if isempty(types)
        return :($checkpoint!($pool_expr))   # fallback for direct external calls (unreachable via macro)
    else
        escaped_types = [esc(t) for t in types]
        typed_call = :($checkpoint!($pool_expr, $(escaped_types...)))
        lazy_call = :($_typed_lazy_checkpoint!($pool_expr, $(escaped_types...)))
        return quote
            if $_can_use_typed_path($pool_expr, $_tracked_mask_for_types($(escaped_types...)))
                $typed_call
            else
                $lazy_call
            end
        end
    end
end

"""
    _generate_typed_rewind_call(pool_expr, types)

Generate bitmask-aware rewind call. When types are known at compile time,
emits a conditional:
- if touched types ⊆ tracked types → typed rewind (fast path)
- otherwise → `_typed_lazy_rewind!` (rewinds tracked | touched mask;
  all touched types have Case A checkpoints via bit 14 lazy mode)
"""
function _generate_typed_rewind_call(pool_expr, types)
    if isempty(types)
        return :($rewind!($pool_expr))       # fallback for direct external calls (unreachable via macro)
    else
        escaped_types = [esc(t) for t in types]
        typed_call = :($rewind!($pool_expr, $(escaped_types...)))
        selective_call = :(
            $_typed_lazy_rewind!(
                $pool_expr,
                $_tracked_mask_for_types($(escaped_types...))
            )
        )
        return quote
            if $_can_use_typed_path($pool_expr, $_tracked_mask_for_types($(escaped_types...)))
                $typed_call
            else
                $selective_call
            end
        end
    end
end

"""
    _generate_lazy_checkpoint_call(pool_expr)

Generate a depth-only checkpoint call for dynamic-selective mode (`use_typed=false`).
Much lighter than full `checkpoint!`: only increments depth and pushes bitmask sentinels.
"""
function _generate_lazy_checkpoint_call(pool_expr)
    return :($_lazy_checkpoint!($pool_expr))
end

"""
    _generate_lazy_rewind_call(pool_expr)

Generate selective rewind code for dynamic-selective mode (`use_typed=false`).
Delegates to `_lazy_rewind!` — a single function call, symmetric
with `_lazy_checkpoint!` for checkpoint. This avoids `let`-block overhead
in `finally` clauses (which can impair Julia's type inference and cause boxing).
"""
function _generate_lazy_rewind_call(pool_expr)
    return :($_lazy_rewind!($pool_expr))
end

# ==============================================================================
# Internal: Raw (Un-Escaped) Rewind/Guard Generators for Direct-Rewind Path
# ==============================================================================
#
# These generate Expr nodes using raw pool_name symbols (NOT esc'd) and GlobalRef
# function references. They are embedded inside the un-escaped AST processed by
# _transform_return_stmts and _transform_break_continue. The outer esc() applied
# to the full transformed_expr handles escaping for all embedded nodes at once.

"""
    _generate_raw_rewind_call(pool_name, use_typed, static_types) -> Expr

Generate un-escaped rewind call for embedding in AST transforms.
Uses GlobalRef function references and raw pool_name symbol.
"""
function _generate_raw_rewind_call(pool_name, use_typed::Bool, static_types)
    if !use_typed || isempty(static_types)
        return Expr(:call, _LAZY_REWIND_REF, pool_name)
    else
        typed_call = Expr(:call, _REWIND_REF, pool_name, static_types...)
        mask_call = Expr(:call, _TRACKED_MASK_REF, static_types...)
        selective_call = Expr(:call, _TYPED_LAZY_REWIND_REF, pool_name, mask_call)
        condition = Expr(:call, _CAN_USE_TYPED_PATH_REF, pool_name, mask_call)
        return Expr(:if, condition, typed_call, selective_call)
    end
end

"""
    _generate_raw_entry_depth_guard(pool_name, entry_depth_var) -> Expr

Generate un-escaped entry depth guard for cleaning up leaked inner scopes.

Produces: `while pool._current_depth > _entry_depth + 1; rewind!(pool); end`
Uses full `rewind!(pool)` (not typed/lazy) because leaked inner scope may have
touched types outside this scope's static type set.
"""
function _generate_raw_entry_depth_guard(pool_name, entry_depth_var)
    depth_access = Expr(:., pool_name, QuoteNode(:_current_depth))
    condition = Expr(:call, :>, depth_access, Expr(:call, :+, entry_depth_var, 1))
    body = Expr(:call, _REWIND_REF, pool_name)
    return Expr(:while, condition, body)
end

# ==============================================================================
# Internal: Acquire Call Transformation
# ==============================================================================

"""
    _transform_acquire_calls(expr, pool_name) -> Expr

Transform acquire!/acquire_view!/convenience function calls to their _impl! counterparts.
Only transforms calls where the first argument matches `pool_name`.

This allows macro-transformed code to bypass the type touch recording overhead,
since the macro already knows about these calls at compile time.

Transformation rules:
- `acquire!(pool, ...)` → `_acquire_impl!(pool, ...)`
- `acquire_array!(pool, ...)` → `_acquire_impl!(pool, ...)`
- `acquire_view!(pool, ...)` → `_acquire_view_impl!(pool, ...)`
- `zeros!(pool, ...)` → `_zeros_impl!(pool, ...)`
- `ones!(pool, ...)` → `_ones_impl!(pool, ...)`
- `similar!(pool, ...)` → `_similar_impl!(pool, ...)`
"""
# Module-qualified references for transformed acquire calls
# Using GlobalRef ensures the function is looked up in AdaptiveArrayPools, not the caller's module
const _ACQUIRE_IMPL_REF = GlobalRef(@__MODULE__, :_acquire_impl!)
const _ACQUIRE_VIEW_IMPL_REF = GlobalRef(@__MODULE__, :_acquire_view_impl!)
const _ZEROS_IMPL_REF = GlobalRef(@__MODULE__, :_zeros_impl!)
const _ONES_IMPL_REF = GlobalRef(@__MODULE__, :_ones_impl!)
const _TRUES_IMPL_REF = GlobalRef(@__MODULE__, :_trues_impl!)
const _FALSES_IMPL_REF = GlobalRef(@__MODULE__, :_falses_impl!)
const _SIMILAR_IMPL_REF = GlobalRef(@__MODULE__, :_similar_impl!)
const _RESHAPE_IMPL_REF = GlobalRef(@__MODULE__, :_reshape_impl!)
const _RAND_IMPL_REF = GlobalRef(@__MODULE__, :_rand_impl!)
const _RANDN_IMPL_REF = GlobalRef(@__MODULE__, :_randn_impl!)

# `@with_pool`/`@maybe_with_pool`/`@safe_with_pool`/`@safe_maybe_with_pool` each
# establish their own independent checkpoint/rewind scope and will run this same
# transform on their own body once Julia expands them. `_transform_acquire_calls`
# must not recurse into a nested macrocall for one of these: doing so renames the
# inner scope's `acquire!` to `_acquire_impl!` before the inner macro ever sees it,
# so the inner macro's own (unrelated) type-extraction pass finds no `acquire!`
# calls and silently demotes to the dynamic/lazy path (bypassing its own typed
# checkpoint/rewind for the type it actually uses).
const _WITH_POOL_FAMILY_MACROS = (
    Symbol("@with_pool"), Symbol("@maybe_with_pool"),
    Symbol("@safe_with_pool"), Symbol("@safe_maybe_with_pool"),
)

function _is_nested_with_pool_macrocall(expr)
    # Legacy tree (< 1.12): keep the old recurse-into-nested-macrocall expansion,
    # byte-identical to what shipped before the typed-path macro upgrades.
    _MACRO_TYPED_UPGRADES || return false
    expr isa Expr && expr.head === :macrocall && !isempty(expr.args) || return false
    fn = expr.args[1]
    fn isa Symbol && return fn in _WITH_POOL_FAMILY_MACROS
    # Module-qualified form: `AdaptiveArrayPools.@with_pool ...` etc.
    # Match on the FINAL name regardless of the module path prefix — same policy
    # as the qualified-name handling for acquire!/zeros!/etc. below.
    fn isa Expr && fn.head === :. && length(fn.args) >= 2 || return false
    qn = fn.args[end]
    return qn isa QuoteNode && qn.value in _WITH_POOL_FAMILY_MACROS
end

function _transform_acquire_calls(expr, pool_name)
    if expr isa Expr
        # Independent nested scope — leave untouched, it transforms its own body.
        _is_nested_with_pool_macrocall(expr) && return expr

        # Handle call expressions
        if expr.head == :call && length(expr.args) >= 2
            fn = expr.args[1]
            pool_arg = expr.args[2]

            # Only transform if pool argument matches
            if pool_arg == pool_name
                # Check for acquire functions (including qualified names)
                if fn == :acquire! || fn == :acquire_array!
                    expr = Expr(:call, _ACQUIRE_IMPL_REF, expr.args[2:end]...)
                elseif fn == :acquire_view!
                    expr = Expr(:call, _ACQUIRE_VIEW_IMPL_REF, expr.args[2:end]...)
                elseif fn == :zeros!
                    expr = Expr(:call, _ZEROS_IMPL_REF, expr.args[2:end]...)
                elseif fn == :ones!
                    expr = Expr(:call, _ONES_IMPL_REF, expr.args[2:end]...)
                elseif fn == :trues!
                    expr = Expr(:call, _TRUES_IMPL_REF, expr.args[2:end]...)
                elseif fn == :falses!
                    expr = Expr(:call, _FALSES_IMPL_REF, expr.args[2:end]...)
                elseif fn == :similar!
                    expr = Expr(:call, _SIMILAR_IMPL_REF, expr.args[2:end]...)
                elseif fn == :reshape!
                    expr = Expr(:call, _RESHAPE_IMPL_REF, expr.args[2:end]...)
                elseif fn == :rand!
                    expr = Expr(:call, _RAND_IMPL_REF, expr.args[2:end]...)
                elseif fn == :randn!
                    expr = Expr(:call, _RANDN_IMPL_REF, expr.args[2:end]...)
                elseif fn isa Expr && fn.head == :. && length(fn.args) >= 2
                    # Qualified name: AdaptiveArrayPools.acquire! etc.
                    qn = fn.args[end]
                    if qn == QuoteNode(:acquire!) || qn == QuoteNode(:acquire_array!)
                        expr = Expr(:call, _ACQUIRE_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:acquire_view!)
                        expr = Expr(:call, _ACQUIRE_VIEW_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:zeros!)
                        expr = Expr(:call, _ZEROS_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:ones!)
                        expr = Expr(:call, _ONES_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:trues!)
                        expr = Expr(:call, _TRUES_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:falses!)
                        expr = Expr(:call, _FALSES_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:similar!)
                        expr = Expr(:call, _SIMILAR_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:reshape!)
                        expr = Expr(:call, _RESHAPE_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:rand!)
                        expr = Expr(:call, _RAND_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:randn!)
                        expr = Expr(:call, _RANDN_IMPL_REF, expr.args[2:end]...)
                    end
                end
            end
        end

        # Recursively transform sub-expressions
        # Create new args array to avoid mutating original
        new_args = Any[_transform_acquire_calls(arg, pool_name) for arg in expr.args]
        return Expr(expr.head, new_args...)
    end
    return expr
end

# ==============================================================================
# Internal: Per-Scope Typed-Pool Hoisting
# ==============================================================================
#
# After `_transform_acquire_calls` rewrites `acquire!`/`zeros!`/etc. to their
# `_*_impl!` GlobalRef forms, this pass replaces the literal type argument of
# each such call with a per-scope local variable bound to the looked-up
# `AbstractTypedPool`, so `get_typed_pool!` runs once per static type per scope
# instead of once per acquire call.

# Impl refs whose 2nd argument is a type literal replaceable by a hoisted tp.
const _HOISTABLE_IMPL_REFS = (
    _ACQUIRE_IMPL_REF, _ACQUIRE_VIEW_IMPL_REF,
    _ZEROS_IMPL_REF, _ONES_IMPL_REF, _RAND_IMPL_REF, _RANDN_IMPL_REF,
)

"""
    _hoist_typed_pools(expr, static_types) -> (tp_vars, rewritten_expr)

For each static type expression, allocate a gensym and replace the type argument
of transformed `_*_impl!` calls (structural `==` match) with that variable.
Returns the type→gensym map (ordered as `static_types`) and the rewritten body.
The caller emits `local var = get_typed_pool!(pool, T)` bindings after checkpoint.

Only types actually substituted into a hoistable call get a binding: a static type
reached solely through a non-hoistable wrapper (`similar!`/`reshape!`/`trues!`/
`falses!`, absent from `_HOISTABLE_IMPL_REFS`) would otherwise produce a dead
`local tp = get_typed_pool!(...)` binding.
"""
function _hoist_typed_pools(expr, static_types)
    lookup = Dict{Any, Symbol}()
    for t in static_types
        lookup[t] = gensym(:_aap_tp)
    end
    used = Set{Any}()
    rewritten = _rewrite_hoisted_calls(expr, lookup, used)
    tp_vars = Pair{Any, Symbol}[t => lookup[t] for t in static_types if t in used]
    return tp_vars, rewritten
end

function _rewrite_hoisted_calls(expr, lookup, used)
    expr isa Expr || return expr
    if expr.head == :call && length(expr.args) >= 3 &&
            (expr.args[1] in _HOISTABLE_IMPL_REFS) && haskey(lookup, expr.args[3])
        t = expr.args[3]
        push!(used, t)
        rest = Any[_rewrite_hoisted_calls(a, lookup, used) for a in expr.args[4:end]]
        return Expr(:call, expr.args[1], expr.args[2], lookup[t], rest...)
    end
    return Expr(expr.head, Any[_rewrite_hoisted_calls(a, lookup, used) for a in expr.args]...)
end

# ==============================================================================
# Internal: Borrow Callsite Injection (S = 1)
# ==============================================================================
#
# Second-pass AST transformation that inserts `pool._pending_callsite = "file:line"`
# before each statement containing an acquire call. This enables borrow registry
# error messages to show WHERE the problematic acquire originated.
#
# Works with both typed path (_*_impl! GlobalRefs) and dynamic path (original
# acquire!/zeros!/etc. calls). Always injected — gated at runtime by
# `_runtime_check(pool)` (dead-code-eliminated when S=0).

const _RUNTIME_CHECK_REF = GlobalRef(@__MODULE__, :_runtime_check)

# GlobalRefs for direct-rewind path (no try-finally):
# Used by _transform_return_stmts and _transform_break_continue to inject
# rewind calls into the un-escaped AST (outer esc() handles escaping).
const _WARN_LEAKED_SCOPE_REF = GlobalRef(@__MODULE__, :_warn_leaked_scope)
const _REWIND_REF = GlobalRef(@__MODULE__, :rewind!)
const _LAZY_REWIND_REF = GlobalRef(@__MODULE__, :_lazy_rewind!)

# Auto-manage scope-ENTRY hook, generated before the checkpoint in every `@with_pool`.
# Gated by the `AUTO_MANAGE` compile-time const (constant-folded → DCE'd to nothing when
# off → zero cost). Dispatches through `_maybe_auto_manage!` so non-CPU (GPU) pools safely
# no-op. Placed BEFORE the checkpoint, so `_current_depth == 1` ⇒ the OUTERMOST scope is
# being entered from global (nothing borrowed → safe). Servicing the flag at entry handles
# every exit type of the previous scope (return/break/throw), with no compaction in `finally`.
const _AUTO_MANAGE_REF = GlobalRef(@__MODULE__, :AUTO_MANAGE)
const _MAYBE_AUTO_MANAGE_REF = GlobalRef(@__MODULE__, :_maybe_auto_manage!)
_auto_manage_hook(pool_name) = :($_AUTO_MANAGE_REF && $_MAYBE_AUTO_MANAGE_REF($(esc(pool_name))))
const _TYPED_LAZY_REWIND_REF = GlobalRef(@__MODULE__, :_typed_lazy_rewind!)
const _CAN_USE_TYPED_PATH_REF = GlobalRef(@__MODULE__, :_can_use_typed_path)
const _TRACKED_MASK_REF = GlobalRef(@__MODULE__, :_tracked_mask_for_types)

"""Set of all transformed `_*_impl!` function names (GlobalRef targets)."""
const _IMPL_FUNC_NAMES = Set{Symbol}(
    [
        :_acquire_impl!, :_acquire_view_impl!,
        :_zeros_impl!, :_ones_impl!, :_trues_impl!, :_falses_impl!,
        :_similar_impl!, :_reshape_impl!,
        :_rand_impl!, :_randn_impl!,
    ]
)

"""
    _contains_acquire_call(expr, pool_name) -> Bool

Detect if `expr` (or any sub-expression) contains a pool acquire call.
Matches both transformed (`GlobalRef`-based `_*_impl!`) and original
(`acquire!`, `zeros!`, etc.) call forms.
"""
function _contains_acquire_call(expr, pool_name)
    expr isa Expr || return false
    if expr.head == :call && length(expr.args) >= 2
        fn = expr.args[1]
        # Transformed _*_impl! calls (GlobalRef from typed path)
        if fn isa GlobalRef && fn.name in _IMPL_FUNC_NAMES
            return true
        end
        # Original acquire calls (dynamic path, or pre-transform)
        if _is_acquire_call(expr, pool_name)
            return true
        end
    end
    return any(arg -> _contains_acquire_call(arg, pool_name), expr.args)
end

"""
    _find_acquire_call_expr(expr, pool_name) -> Union{Expr, Nothing}

Find the first acquire call expression in `expr` targeting `pool_name`.
Returns the original call Expr (e.g., `:(zeros!(pool, Float64, 10))`) or `nothing`.
Used to capture the user's source expression for debug display.
"""
function _find_acquire_call_expr(expr, pool_name)
    expr isa Expr || return nothing
    if _is_acquire_call(expr, pool_name)
        return expr
    end
    for arg in expr.args
        result = _find_acquire_call_expr(arg, pool_name)
        result !== nothing && return result
    end
    return nothing
end

"""
    _inject_pending_callsite(expr, pool_name, original_expr=expr) -> Expr

Walk block-level statements, track `LineNumberNode`s, and insert
`_runtime_check(pool) && (pool._pending_callsite = "file:line\\nexpr")`
before each statement containing a pool acquire call.

When `original_expr` differs from `expr` (i.e., after `_transform_acquire_calls`),
the original untransformed AST is used to extract the user's source expression
(e.g., `zeros!(pool, Float64, 10)` instead of `_zeros_impl!(pool, Float64, 10)`).

Only processes `:block` expressions. Non-block expressions are recursed
into to find nested blocks.
"""
function _inject_pending_callsite(expr, pool_name, original_expr = expr)
    expr isa Expr || return expr
    if expr.head == :block
        new_args = Any[]
        current_lnn = nothing
        orig_args = (original_expr isa Expr && original_expr.head == :block) ? original_expr.args : nothing
        for (i, arg) in enumerate(expr.args)
            if arg isa LineNumberNode
                current_lnn = arg
                push!(new_args, arg)
            else
                orig_arg = (orig_args !== nothing && i <= length(orig_args)) ? orig_args[i] : arg
                processed = _inject_pending_callsite(arg, pool_name, orig_arg)
                if current_lnn !== nothing && _contains_acquire_call(processed, pool_name)
                    # Use the full original statement for debug display
                    expr_text = string(orig_arg)
                    callsite_str = isempty(expr_text) ?
                        "$(current_lnn.file):$(current_lnn.line)" :
                        "$(current_lnn.file):$(current_lnn.line)\n$(expr_text)"
                    inject = Expr(
                        :&&,
                        Expr(:call, _RUNTIME_CHECK_REF, pool_name),
                        Expr(
                            :(=),
                            Expr(:., pool_name, QuoteNode(:_pending_callsite)),
                            callsite_str
                        )
                    )
                    push!(new_args, inject)
                end
                push!(new_args, processed)
            end
        end
        return Expr(:block, new_args...)
    else
        orig_expr_args = (original_expr isa Expr) ? original_expr.args : nothing
        new_args = Any[]
        for (i, arg) in enumerate(expr.args)
            orig_arg = (orig_expr_args !== nothing && i <= length(orig_expr_args)) ? orig_expr_args[i] : arg
            push!(new_args, _inject_pending_callsite(arg, pool_name, orig_arg))
        end
        return Expr(expr.head, new_args...)
    end
end

# ==============================================================================
# Internal: Return Statement Validation (S = 1)
# ==============================================================================
#
# Transforms `return expr` → `begin local _ret = expr; validate(_ret); return _ret end`
# so that explicit `return` statements in @with_pool function bodies are validated
# before exiting. Without this, `return` bypasses the post-body _validate_pool_return
# check because it exits the function before that line is reached.
#
# Stops recursion at :function and :-> boundaries (nested function return statements
# belong to the inner function, not the @with_pool scope).

const _VALIDATE_POOL_RETURN_REF = GlobalRef(@__MODULE__, :_validate_pool_return)

"""
    _transform_return_stmts(expr, pool_name; rewind_call=nothing, entry_depth_guard=nothing) -> Expr

Walk AST and wrap explicit `return value` statements with escape validation.
Generates: `local _ret = value; if _runtime_check(pool) validate(_ret, pool); end; return _ret`

When `rewind_call` and `entry_depth_guard` are provided (direct-rewind path,
`safe=false`), they are inserted after validation but before `return`:
  `local _ret = value; validate; entry_depth_guard; rewind_call; return _ret`

When `nothing` (safe path / try-finally), behavior is unchanged — rewind
happens in the `finally` clause instead.

Does NOT recurse into nested `:function` or `:->` expressions (inner functions
have their own `return` semantics).
"""
function _transform_return_stmts(
        expr, pool_name, current_lnn = nothing;
        rewind_call = nothing,
        entry_depth_guard = nothing
    )
    expr isa Expr || return expr

    # Don't recurse into nested function definitions or quoted AST
    if expr.head in (:function, :->, :quote)
        return expr
    end

    if expr.head == :return && length(expr.args) >= 1
        value_expr = expr.args[1]
        # Bare return (return nothing) — skip validation but still need rewind
        if value_expr === nothing
            if rewind_call !== nothing
                return Expr(:block, entry_depth_guard, rewind_call, expr)
            end
            return expr
        end
        # Recurse into the value expression first (may contain nested returns in ternary etc.)
        value_expr = _transform_return_stmts(
            value_expr, pool_name, current_lnn;
            rewind_call, entry_depth_guard
        )
        retvar = gensym(:_pool_ret)

        # Build return-site string for S=1 display (e.g. "file:line\nreturn v")
        return_site_str = if current_lnn !== nothing
            "$(current_lnn.file):$(current_lnn.line)\n$(string(expr))"
        else
            ""
        end

        # Conditionally set _pending_return_site before validation
        validate_expr = if !isempty(return_site_str)
            Expr(
                :block,
                Expr(
                    :&&,
                    Expr(:call, _RUNTIME_CHECK_REF, pool_name),
                    Expr(
                        :(=),
                        Expr(:., pool_name, QuoteNode(:_pending_return_site)),
                        return_site_str
                    )
                ),
                Expr(:call, _VALIDATE_POOL_RETURN_REF, retvar, pool_name)
            )
        else
            Expr(:call, _VALIDATE_POOL_RETURN_REF, retvar, pool_name)
        end

        # Build statement list: validate → [guard → rewind] → return
        stmts = Any[
            Expr(:local, Expr(:(=), retvar, value_expr)),
            Expr(:if, Expr(:call, _RUNTIME_CHECK_REF, pool_name), validate_expr),
        ]
        if rewind_call !== nothing
            push!(stmts, entry_depth_guard)
            push!(stmts, rewind_call)
        end
        push!(stmts, Expr(:return, retvar))
        return Expr(:block, stmts...)
    end

    # For blocks, track LineNumberNodes
    if expr.head == :block
        new_args = Any[]
        lnn = current_lnn
        for arg in expr.args
            if arg isa LineNumberNode
                lnn = arg
                push!(new_args, arg)
            else
                push!(
                    new_args, _transform_return_stmts(
                        arg, pool_name, lnn;
                        rewind_call, entry_depth_guard
                    )
                )
            end
        end
        return Expr(:block, new_args...)
    end

    # Other expressions: recurse with current_lnn
    new_args = Any[
        _transform_return_stmts(
                arg, pool_name, current_lnn;
                rewind_call, entry_depth_guard
            ) for arg in expr.args
    ]
    return Expr(expr.head, new_args...)
end

# ==============================================================================
# Internal: Break/Continue Transformation (Direct-Rewind Path)
# ==============================================================================
#
# For block-form @with_pool (NOT function form), `break` and `continue` at the
# pool scope level exit the pool scope (the block is inside a loop). Without
# try-finally, we must insert rewind before these statements.
#
# The walker SKIPS :for/:while bodies — break/continue inside nested loops
# belong to those loops, not the pool scope. Also skips :function/:-> bodies.

"""
    _transform_break_continue(expr, rewind_call, entry_depth_guard) -> Expr

Walk AST and insert entry depth guard + rewind before `break`/`continue` statements
that would exit the pool scope. Only used for block-form `@with_pool` (not function form).

Skips `:for`, `:while` bodies (break/continue there are for those loops).
Skips `:function`, `:->` bodies (inner function scope boundary).
"""
function _transform_break_continue(expr, rewind_call, entry_depth_guard)
    expr isa Expr || return expr

    # Don't recurse into nested functions or quoted AST
    expr.head in (:function, :->, :quote) && return expr

    # Don't recurse into loop bodies — break/continue there are for those loops
    expr.head in (:for, :while) && return expr

    # Transform bare break/continue at pool-block level
    if expr.head in (:break, :continue)
        return Expr(:block, entry_depth_guard, rewind_call, expr)
    end

    # Recurse into other expressions (if, try, let, block, etc.)
    new_args = Any[
        _transform_break_continue(arg, rewind_call, entry_depth_guard)
            for arg in expr.args
    ]
    return Expr(expr.head, new_args...)
end

# ==============================================================================
# Internal: @goto Safety Check (Direct-Rewind Path)
# ==============================================================================

"""
    _collect_local_gotos_and_labels(expr) -> (gotos::Set{Symbol}, labels::Set{Symbol})

Walk the body AST and collect all `@goto` target symbols and `@label` names.
Skips `:function`/`:->` bodies (inner functions have their own scope).

At macro expansion time, `@goto`/`@label` are `:macrocall` nodes, not `:symbolicgoto`.
"""
function _collect_local_gotos_and_labels(expr)
    gotos = Set{Symbol}()
    labels = Set{Symbol}()

    function walk(node)
        node isa Expr || return

        if node.head === :macrocall && length(node.args) >= 3
            name = node.args[1]
            target = node.args[3]
            if name === Symbol("@goto") && target isa Symbol
                push!(gotos, target)
            elseif name === Symbol("@label") && target isa Symbol
                push!(labels, target)
            end
        end

        # Skip nested function bodies (separate scope) and quoted AST (not executable here)
        node.head in (:function, :->, :quote) && return

        for arg in node.args
            walk(arg)
        end
        return
    end

    walk(expr)
    return gotos, labels
end

"""
    _check_unsafe_goto(expr)

Hard error if the body contains any `@goto` that targets a label NOT defined
within the same body. Such jumps would bypass `rewind!` insertion.

Internal jumps (`@goto label` where `@label label` exists in the body) are safe
and allowed — they don't exit the pool scope.
"""
function _check_unsafe_goto(expr)
    gotos, labels = _collect_local_gotos_and_labels(expr)
    unsafe = setdiff(gotos, labels)
    return if !isempty(unsafe)
        targets = join(unsafe, ", ")
        error(
            "Pool scope: @goto to external label(s) ($targets) detected. " *
                "This would bypass rewind! and corrupt pool state. " *
                "Use the @safe_* variant (e.g., @safe_with_pool) for @goto across pool boundaries."
        )
    end
end

# ==============================================================================
# Internal: Compile-Time Escape Detection
# ==============================================================================
#
# Detects common pool escape patterns at macro expansion time (zero runtime cost).
# - Error: bare acquired variable as last expression (100% escape)
# - Warning: acquired variable inside tuple/array literal (likely escape)
#
# This catches the most common beginner mistake — returning a pool-backed array
# from @with_pool — before the code even runs.

"""
    _ALL_ACQUIRE_NAMES

Set of all function names that return pool-backed arrays.
Used by `_extract_acquired_vars` to identify assignments like `v = acquire!(pool, ...)`.
"""
const _ALL_ACQUIRE_NAMES = Set{Symbol}(
    [
        :acquire!, :acquire_view!, :acquire_array!,
        :zeros!, :ones!, :similar!, :reshape!,
        :trues!, :falses!,
        :rand!, :randn!,
    ]
)

"""Function names that return views (SubArray) from pool memory."""
const _VIEW_ACQUIRE_NAMES = Set{Symbol}(
    [
        :acquire_view!,
    ]
)

"""Function names that return raw Arrays backed by pool memory."""
const _ARRAY_ACQUIRE_NAMES = Set{Symbol}(
    [
        :acquire!, :acquire_array!,
        :zeros!, :ones!, :similar!, :reshape!,
        :rand!, :randn!,
    ]
)

"""Function names that return BitArrays from pool memory."""
const _BITARRAY_ACQUIRE_NAMES = Set{Symbol}(
    [
        :trues!, :falses!,
    ]
)

"""
    _is_acquire_call(expr, target_pool) -> Bool

Check if an expression is a call to any pool acquire/convenience function
targeting `target_pool`.
"""
function _is_acquire_call(expr, target_pool)
    if !(expr isa Expr && expr.head == :call && length(expr.args) >= 2)
        return false
    end
    fn = expr.args[1]
    pool_arg = expr.args[2]
    pool_arg == target_pool || return false

    # Direct name
    if fn isa Symbol
        return fn in _ALL_ACQUIRE_NAMES
    end
    # Qualified name: Module.acquire!
    if fn isa Expr && fn.head == :. && length(fn.args) >= 2
        qn = fn.args[end]
        if qn isa QuoteNode && qn.value isa Symbol
            return qn.value in _ALL_ACQUIRE_NAMES
        end
    end
    return false
end

"""
    _acquire_call_kind(expr, target_pool) -> Union{Symbol, Nothing}

Return the classification of an acquire call: `:pool_view`, `:pool_array`, `:pool_bitarray`,
or `nothing` if not an acquire call.
"""
function _acquire_call_kind(expr, target_pool)
    if !(expr isa Expr && expr.head == :call && length(expr.args) >= 2)
        return nothing
    end
    fn = expr.args[1]
    pool_arg = expr.args[2]
    pool_arg == target_pool || return nothing

    fname = nothing
    if fn isa Symbol
        fname = fn
    elseif fn isa Expr && fn.head == :. && length(fn.args) >= 2
        qn = fn.args[end]
        if qn isa QuoteNode && qn.value isa Symbol
            fname = qn.value
        end
    end
    fname === nothing && return nothing

    fname in _VIEW_ACQUIRE_NAMES && return :pool_view
    fname in _ARRAY_ACQUIRE_NAMES && return :pool_array
    fname in _BITARRAY_ACQUIRE_NAMES && return :pool_bitarray
    return nothing
end

"""
    _extract_acquired_vars(expr, target_pool) -> Set{Symbol}

Walk AST to find variable names assigned from acquire/convenience calls.
Returns the set of symbols that hold pool-backed arrays.

Only top-level assignments in a block are tracked (not inside branches).
Handles both simple assignment (`v = acquire!(...)`) and tuple destructuring
(`(v, w) = (acquire!(...), expr)`).
"""
function _extract_acquired_vars(expr, target_pool, vars = Set{Symbol}())
    if expr isa Expr
        if expr.head == :block
            # Walk top-level statements only (for flat reassignment tracking)
            for arg in expr.args
                _extract_acquired_vars(arg, target_pool, vars)
            end
        elseif expr.head == :(=) && length(expr.args) >= 2
            lhs = expr.args[1]
            rhs = expr.args[2]
            if lhs isa Symbol && _is_acquire_call(rhs, target_pool)
                push!(vars, lhs)
            elseif lhs isa Symbol && rhs isa Symbol && rhs in vars
                # Simple alias: d = z where z is acquired
                push!(vars, lhs)
            elseif lhs isa Symbol && _literal_contains_acquired(rhs, vars)
                # Container wrapping: d = (z,), d = [z, w], etc.
                push!(vars, lhs)
            elseif Meta.isexpr(lhs, :tuple) && Meta.isexpr(rhs, :tuple)
                # Destructuring with tuple literal RHS: (v, w) = (acquire!(...), expr)
                for (l, r) in zip(lhs.args, rhs.args)
                    if l isa Symbol && _is_acquire_call(r, target_pool)
                        push!(vars, l)
                    elseif l isa Symbol && r isa Symbol && r in vars
                        # Destructuring alias: (a, d) = (..., z)
                        push!(vars, l)
                    end
                end
            end
            # Recurse into RHS (for nested blocks with acquire calls)
            _extract_acquired_vars(rhs, target_pool, vars)
        else
            # Skip scope-introducing expressions whose `return` does NOT
            # return from the enclosing function (function, ->, macro).
            # NOTE: `let` is NOT skipped — `return` inside `let` exits the
            # enclosing function, so pool variables acquired there can escape.
            if expr.head in (:function, :(->), :macro)
                return vars
            end
            for arg in expr.args
                _extract_acquired_vars(arg, target_pool, vars)
            end
        end
    end
    return vars
end

"""
    _get_last_expression(expr) -> Any

Return the last non-LineNumberNode expression from a block.
For non-block expressions, returns the expression itself.
"""
function _get_last_expression(expr)
    if expr isa Expr && expr.head == :block
        for i in length(expr.args):-1:1
            arg = expr.args[i]
            arg isa LineNumberNode && continue
            return _get_last_expression(arg)
        end
        return nothing
    end
    return expr
end

"""
    _collect_all_return_values(expr) -> Vector{Tuple{Any, Union{Int,Nothing}}}

Collect all (expression, line) pairs that could be returned from a block/function body:
- Explicit `return expr` statements anywhere in the body (recursive, skips nested functions)
- Implicit returns: the last expression, recursing into if/else/elseif branches
"""
function _collect_all_return_values(expr)
    values = Tuple{Any, Union{Int, Nothing}}[]
    _collect_explicit_returns!(values, expr, nothing)
    last_expr, last_line = _get_last_expression_with_line(expr)
    if last_expr !== nothing
        _collect_implicit_return_values!(values, last_expr, last_line)
    end
    return values
end

"""Walk AST to find all explicit `return expr` statements with line numbers.
Tracks LineNumberNodes through blocks. Skips nested function definitions."""
function _collect_explicit_returns!(values, expr, current_line::Union{Int, Nothing})
    expr isa Expr || return
    expr.head in (:function, :(->)) && return
    if expr.head == :return
        push!(values, (expr, current_line))
        return
    end
    return if expr.head == :block
        line = current_line
        for arg in expr.args
            if arg isa LineNumberNode
                line = arg.line
            else
                _collect_explicit_returns!(values, arg, line)
            end
        end
    else
        for arg in expr.args
            _collect_explicit_returns!(values, arg, current_line)
        end
    end
end

"""Return the last non-LineNumberNode expression from a block, together with its line.
Recurses into nested blocks."""
function _get_last_expression_with_line(expr, default_line::Union{Int, Nothing} = nothing)
    if !(expr isa Expr && expr.head == :block)
        return (expr, default_line)
    end
    for i in length(expr.args):-1:1
        arg = expr.args[i]
        arg isa LineNumberNode && continue
        # Find the LineNumberNode preceding this expression
        line = default_line
        for j in (i - 1):-1:1
            if expr.args[j] isa LineNumberNode
                line = expr.args[j].line
                break
            end
        end
        return _get_last_expression_with_line(arg, line)
    end
    return (nothing, default_line)
end

"""Expand implicit return values by recursing into if/elseif/else branches.
Non-branch expressions are collected as (expr, line) pairs."""
function _collect_implicit_return_values!(values, expr, current_line::Union{Int, Nothing})
    return if expr isa Expr && expr.head in (:if, :elseif)
        for i in 2:length(expr.args)
            branch = expr.args[i]
            if branch isa Expr && branch.head in (:if, :elseif)
                _collect_implicit_return_values!(values, branch, current_line)
            else
                last_expr, last_line = _get_last_expression_with_line(branch, current_line)
                if last_expr !== nothing
                    _collect_implicit_return_values!(values, last_expr, last_line)
                end
            end
        end
    else
        push!(values, (expr, current_line))
    end
end


"""
    _extract_ordered_acquired(expr, target_pool) -> Set{Symbol}

Order-aware extraction of pool-tainted variables. Combines the logic of
`_extract_acquired_vars` and `_remove_flat_reassigned!` into a single forward
pass over top-level block statements, correctly handling statement ordering.

For top-level assignments, taint is added/removed as statements are scanned
in source order. For nested blocks (if/for/while), recursion adds taint
conservatively (never removes inside branches — can't resolve control flow).

Tuple destructuring is handled atomically: all RHS values are evaluated against
the current taint set before any LHS updates are applied.
"""
function _extract_ordered_acquired(expr, target_pool)
    if !(expr isa Expr && expr.head == :block)
        # Not a block — fall back to recursive extraction
        return _extract_acquired_vars(expr, target_pool)
    end
    tainted = Set{Symbol}()
    for arg in expr.args
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :(=) && length(arg.args) >= 2
            lhs = arg.args[1]
            rhs = arg.args[2]
            if lhs isa Symbol
                if _is_acquire_call(rhs, target_pool)
                    push!(tainted, lhs)
                elseif rhs isa Symbol && rhs in tainted
                    # Alias: d = v where v is tainted
                    push!(tainted, lhs)
                elseif _literal_contains_acquired(rhs, tainted)
                    # Container wrapping: d = (v,), d = [v, w]
                    push!(tainted, lhs)
                else
                    # Non-pool reassignment — remove taint
                    delete!(tainted, lhs)
                end
            elseif Meta.isexpr(lhs, :tuple)
                if Meta.isexpr(rhs, :tuple) && length(rhs.args) == length(lhs.args)
                    # Atomic destructuring: evaluate all RHS against current taint,
                    # then apply all LHS changes
                    rhs_tainted = [
                        _is_acquire_call(r, target_pool) ||
                            (r isa Symbol && r in tainted) for r in rhs.args
                    ]
                    for (i, l) in enumerate(lhs.args)
                        l isa Symbol || continue
                        if rhs_tainted[i]
                            push!(tainted, l)
                        else
                            delete!(tainted, l)
                        end
                    end
                else
                    # Opaque RHS (function call) — all destructured vars become untainted
                    for l in lhs.args
                        l isa Symbol && delete!(tainted, l)
                    end
                end
            end
            # Recurse into RHS for nested blocks containing acquire calls
            _extract_acquired_vars(rhs, target_pool, tainted)
        else
            # Recurse into non-assignment expressions (if/for/while/etc.)
            _extract_acquired_vars(arg, target_pool, tainted)
        end
    end
    return tainted
end

"""
    _find_direct_exposure(expr, acquired) -> Set{Symbol}

Check if the expression directly exposes any acquired variable.
Only catches high-confidence patterns:
- Bare Symbol: `v`
- Explicit return: `return v`
- Tuple/array literal containing a var: `(v, w)`, `[v, w]`
- NamedTuple-style kw: `(a=v,)`

Does NOT recurse into function calls (can't know what `f(v)` returns).
"""

"""Check if a literal expression (symbol, identity call, tuple/vect) transitively contains any acquired var."""
function _literal_contains_acquired(expr, acquired)
    expr isa Symbol && return expr in acquired
    if expr isa Expr
        # identity(x) — transparent, look through
        if expr.head == :call && length(expr.args) >= 2 && expr.args[1] === :identity
            return _literal_contains_acquired(expr.args[2], acquired)
        end
        if expr.head in (:tuple, :vect)
            for arg in expr.args
                if Meta.isexpr(arg, :(=)) && length(arg.args) >= 2
                    _literal_contains_acquired(arg.args[2], acquired) && return true
                elseif Meta.isexpr(arg, :kw) && length(arg.args) >= 2
                    _literal_contains_acquired(arg.args[2], acquired) && return true
                else
                    _literal_contains_acquired(arg, acquired) && return true
                end
            end
        end
    end
    return false
end

"""Check if a call target is `identity` or `Base.identity`."""
_is_identity_call(x) = x === :identity ||
    (x isa Expr && x.head == :. && x.args == [:Base, QuoteNode(:identity)])

function _find_direct_exposure(expr, acquired)
    found = Set{Symbol}()
    if expr isa Symbol
        # Bare variable: v
        if expr in acquired
            push!(found, expr)
        end
    elseif expr isa Expr
        if expr.head == :return
            # return v, return (v, w), etc.
            for arg in expr.args
                union!(found, _find_direct_exposure(arg, acquired))
            end
        elseif expr.head in (:tuple, :vect)
            # (v, w), [v, w], nested (a, (b, v)), (key=(a, v),)
            for arg in expr.args
                if Meta.isexpr(arg, :(=)) && length(arg.args) >= 2
                    # NamedTuple: (a=v,) or (a=(b,v),)
                    union!(found, _find_direct_exposure(arg.args[2], acquired))
                else
                    union!(found, _find_direct_exposure(arg, acquired))
                end
            end
        elseif expr.head == :parameters
            # (; a=v) style named tuple parameters
            for arg in expr.args
                if Meta.isexpr(arg, :kw) && length(arg.args) >= 2
                    union!(found, _find_direct_exposure(arg.args[2], acquired))
                end
            end
        elseif expr.head == :call && length(expr.args) >= 2 && _is_identity_call(expr.args[1])
            # identity(x) / Base.identity(x) — transparent, look through
            union!(found, _find_direct_exposure(expr.args[2], acquired))
        end
    end
    return found
end

"""Dotted (broadcast) assignment head: :.=, :.+=, :.*=, …"""
function _is_dotted_assign_head(h)
    h isa Symbol || return false
    s = String(h)
    return length(s) >= 2 && startswith(s, ".") && endswith(s, "=")
end

"""
    _incidental_exposure(expr, tainted, pool_name) -> Union{Nothing, Tuple{Symbol, Any}}

Detect the three incidental pool-backed tail patterns the direct-exposure ERROR
does not cover (their value escapes as the block's return value, but the syntax
does not *look* like a return):

- `(:acquire_call, expr)`     — tail is a direct acquire-family call
- `(:broadcast_assign, var)`  — tail is `x .= v` / `x .op= v` with `x` acquired
                                 (also `x[...] .= v` on an acquired base)
- `(:assign, var)`            — tail is `x = <acquire call>` or `x = <acquired var>`
                                 (assignment evaluates to its RHS value)

`ret_expr` values arrive here either as the bare tail expression (implicit
return) or as the whole `Expr(:return, ...)` node (explicit `return`, per
`_collect_all_return_values`). An explicit `return` is unwrapped first — one
recursion into `expr.args[1]` — mirroring `_find_direct_exposure`'s handling
of `:return`, so `return acquire!(...)`, `return (v .= 0.0)`, etc. are caught
exactly like their implicit-tail equivalents. A bare `return` (no value) is
safe and falls through untouched.
"""
function _incidental_exposure(expr, tainted, pool_name)
    expr isa Expr || return nothing
    if expr.head == :return && !isempty(expr.args)
        return _incidental_exposure(expr.args[1], tainted, pool_name)
    end
    if _is_acquire_call(expr, pool_name)
        return (:acquire_call, expr)
    elseif _is_dotted_assign_head(expr.head) && length(expr.args) >= 1
        lhs = expr.args[1]
        base = Meta.isexpr(lhs, :ref) ? lhs.args[1] : lhs
        if base isa Symbol && base in tainted
            return (:broadcast_assign, base)
        end
    elseif expr.head == :(=) && length(expr.args) >= 2
        rhs = expr.args[2]
        if _is_acquire_call(rhs, pool_name)
            return (:assign, expr.args[1])
        elseif rhs isa Symbol && rhs in tainted
            return (:assign, rhs)
        end
    end
    return nothing
end

"""Report an incidental-tail escape at `severity`: `"error"` throws a `PoolEscapeError`
(storing the classified `(kind, detail)` on the point so `showerror` renders directly),
`"warn"` emits an expansion-time warning. Separated from the const-gated call site so
both severities are unit-testable — `ESCAPE_LINT` is a load-time constant, so the call
site only ever exercises one branch per session."""
function _report_incidental_escape(severity, kind, detail, ret_expr, ret_line, file, line)
    if severity == "error"
        point = EscapePoint(ret_expr, ret_line, Symbol[], (kind, detail))
        throw(PoolEscapeError(Symbol[], file, line, [point]))
    else # "warn"
        @warn _lint_message(kind, detail, ret_expr) _file = file _line = something(ret_line, line, 0)
    end
    return
end

"""Build the human-readable expansion-time lint message for an incidental-tail escape.
Used both for the `escape_lint = "warn"` path and for rendering `PoolEscapeError`
(via `showerror`) when the error originates from an incidental tail rather than an
intentional-return pattern."""
function _lint_message(kind, detail, ret_expr)
    tail = sprint(Base.show_unquoted, ret_expr)
    what = kind === :acquire_call ?
        "is a direct acquire call — its pool-backed array" :
        kind === :broadcast_assign ?
        "evaluates to the pool-backed array `$(detail)`" :
        "assigns a pool-backed array, and the assignment's value"
    return string(
        "the scope's last expression `", tail, "` ", what,
        " becomes the scope's return value and escapes",
        " (a pool array is invalid after the scope rewinds).",
        " If the value is meant to be discarded, end the block with `nothing`.",
        " [escape_lint preference: \"error\" (default) | \"warn\" | \"off\"]"
    )
end


"""Collect acquired variable names contained in a literal expression (symbol, tuple, vect)."""
function _collect_acquired_in_literal(expr, acquired_keys::Set{Symbol})
    found = Symbol[]
    _collect_acquired_in_literal!(found, expr, acquired_keys)
    return found
end

function _collect_acquired_in_literal!(found, expr, acquired_keys)
    return if expr isa Symbol
        expr in acquired_keys && push!(found, expr)
    elseif expr isa Expr
        if expr.head == :call && length(expr.args) >= 2 && expr.args[1] === :identity
            _collect_acquired_in_literal!(found, expr.args[2], acquired_keys)
        elseif expr.head in (:tuple, :vect)
            for arg in expr.args
                if Meta.isexpr(arg, :(=)) && length(arg.args) >= 2
                    _collect_acquired_in_literal!(found, arg.args[2], acquired_keys)
                elseif Meta.isexpr(arg, :kw) && length(arg.args) >= 2
                    _collect_acquired_in_literal!(found, arg.args[2], acquired_keys)
                else
                    _collect_acquired_in_literal!(found, arg, acquired_keys)
                end
            end
        end
    end
end

"""
    _classify_escaped_vars(expr, target_pool, escaped, acquired)

Classify each escaped variable by its origin for better error messages:
- `:pool_view` — from acquire_view! (returns SubArray)
- `:pool_array` — from acquire!, zeros!, etc. (returns Array)
- `:pool_bitarray` — from trues!, falses! (returns BitArray)
- `:alias` — alias of another acquired variable (e.g., `d = v`)
- `:container` — wraps acquired variables in a literal (e.g., `d = [v, 1]`)

Returns `Dict{Symbol, Tuple{Symbol, Vector{Symbol}}}` mapping var → (kind, source_vars).
"""
function _classify_escaped_vars(expr, target_pool, escaped::Vector{Symbol}, acquired::Set{Symbol})
    info = Dict{Symbol, Tuple{Symbol, Vector{Symbol}}}()
    escaped_set = Set(escaped)
    _classify_walk!(info, expr, target_pool, escaped_set, acquired)
    return info
end

function _classify_walk!(info, expr, target_pool, escaped_set, acquired)
    expr isa Expr || return
    return if expr.head == :block
        for arg in expr.args
            _classify_walk!(info, arg, target_pool, escaped_set, acquired)
        end
    elseif expr.head == :(=) && length(expr.args) >= 2
        lhs = expr.args[1]
        rhs = expr.args[2]
        if lhs isa Symbol && lhs in escaped_set
            kind = _acquire_call_kind(rhs, target_pool)
            if kind !== nothing
                info[lhs] = (kind, Symbol[])
            elseif rhs isa Symbol && rhs in acquired
                info[lhs] = (:alias, [rhs])
            else
                sources = _collect_acquired_in_literal(rhs, acquired)
                if !isempty(sources)
                    info[lhs] = (:container, sort!(sources))
                end
            end
        end
        # Recurse into RHS for nested blocks
        _classify_walk!(info, rhs, target_pool, escaped_set, acquired)
    else
        for arg in expr.args
            _classify_walk!(info, arg, target_pool, escaped_set, acquired)
        end
    end
end

"""
    _extract_declaration_sites(expr, escaped)

Walk the AST to find assignment sites for escaped variables.
Returns a `Vector{DeclarationSite}` sorted by line number.
"""
function _extract_declaration_sites(expr, escaped::Set{Symbol})
    sites = DeclarationSite[]
    seen = Set{Symbol}()
    _collect_declaration_sites!(sites, seen, expr, escaped, nothing, nothing)
    sort!(sites; by = s -> something(s.line, typemax(Int)))
    return sites
end

function _collect_declaration_sites!(sites, seen, expr, escaped, current_line, current_file)
    expr isa Expr || return
    return if expr.head == :block
        line = current_line
        file = current_file
        for arg in expr.args
            if arg isa LineNumberNode
                line = arg.line
                file = arg.file
            else
                _collect_declaration_sites!(sites, seen, arg, escaped, line, file)
            end
        end
    elseif expr.head == :(=) && length(expr.args) >= 2
        lhs = expr.args[1]
        if lhs isa Symbol && lhs in escaped && lhs ∉ seen
            push!(sites, DeclarationSite(lhs, expr, current_line, current_file))
            push!(seen, lhs)
        elseif Meta.isexpr(lhs, :tuple)
            for l in lhs.args
                if l isa Symbol && l in escaped && l ∉ seen
                    push!(sites, DeclarationSite(l, expr, current_line, current_file))
                    push!(seen, l)
                end
            end
        end
        _collect_declaration_sites!(sites, seen, expr.args[2], escaped, current_line, current_file)
    else
        for arg in expr.args
            _collect_declaration_sites!(sites, seen, arg, escaped, current_line, current_file)
        end
    end
end

"""
    _check_compile_time_escape(expr, pool_name, source)

Compile-time (macro expansion time) escape detection.

Checks if the block/function body's return expression directly contains
a pool-backed variable. This catches the most common beginner mistake
at zero runtime cost.

All detected escapes are errors — bare symbol (`v`), `return v`, and
container patterns (`(v, w)`, `[v]`, `(key=v,)`) (Stage 1), plus the three
incidental-tail patterns — direct acquire-call tail, broadcast-assign tail,
assignment tail — gated by the `ESCAPE_LINT` preference (Stage 2, default "error").

Skipped when `STATIC_POOLING = false` (pooling disabled, acquire returns normal arrays).
"""
function _check_compile_time_escape(expr, pool_name, source::Union{LineNumberNode, Nothing})
    # Order-aware extraction: single forward pass that tracks taint per statement order
    acquired = _extract_ordered_acquired(expr, pool_name)

    # Collect ALL return points: explicit returns + implicit (last expr / if-else branches)
    return_values = _collect_all_return_values(expr)
    isempty(return_values) && return

    # ---- Stage 1: intentional-return patterns → ERROR (existing behavior) ----
    if !isempty(acquired)
        # Check each return point for direct exposure of acquired vars
        all_escaped = Set{Symbol}()
        points = EscapePoint[]
        seen_lines = Set{Int}()
        for (ret_expr, ret_line) in return_values
            # Deduplicate: explicit + implicit scanners can find the same return
            if ret_line !== nothing && ret_line in seen_lines
                continue
            end
            point_escaped = _find_direct_exposure(ret_expr, acquired)
            if !isempty(point_escaped)
                push!(points, EscapePoint(ret_expr, ret_line, sort!(collect(point_escaped))))
                union!(all_escaped, point_escaped)
                ret_line !== nothing && push!(seen_lines, ret_line)
            end
        end
        if !isempty(all_escaped)
            sorted = sort!(collect(all_escaped))
            var_info = _classify_escaped_vars(expr, pool_name, sorted, acquired)
            declarations = _extract_declaration_sites(expr, all_escaped)
            file = source !== nothing ? string(source.file) : nothing
            line = source !== nothing ? source.line : nothing
            throw(PoolEscapeError(sorted, file, line, points, var_info, declarations))
        end
    end

    # ---- Stage 2: incidental-tail patterns (error by default) ----
    # Reached whenever Stage 1 found nothing to throw on — including when
    # `acquired` is empty (e.g. a bare `acquire!(pool, ...)` tail with no
    # assigned variable at all: Stage 1 has nothing to track, but the call's
    # result still escapes as the scope's return value).
    ESCAPE_LINT == "off" && return
    file = source !== nothing ? string(source.file) : nothing
    line = source !== nothing ? source.line : nothing
    for (ret_expr, ret_line) in return_values
        hit = _incidental_exposure(ret_expr, acquired, pool_name)
        hit === nothing && continue
        kind, detail = hit
        _report_incidental_escape(ESCAPE_LINT, kind, detail, ret_expr, ret_line, file, line)
    end
    return
end

# ==============================================================================
# Compile-time container-escape WARNING (conservative, may have false positives)
# ==============================================================================
#
# Detects patterns like:
#   vac = (wv=zeros!(pool, ...), grri=acquire!(pool, ...))
#   return vac.wv   # ← pool-backed array escaping via dot access
#
# This is a WARNING (not error) because:
# - `return vac.name` where `name="hello"` is safe (false positive)
# - Macro can't distinguish pool-backed vs non-pool fields in a container
# - Runtime check (RUNTIME_CHECK=1) catches true positives precisely

"""
    _tuple_contains_acquire_call(expr, target_pool) -> Bool

Check if a tuple/vect literal contains any acquire calls as values.
Unlike `_literal_contains_acquired`, this checks the RHS calls directly,
so it works even when `vars` is empty (solves the chicken-and-egg problem).
"""
function _tuple_contains_acquire_call(expr, target_pool)
    expr isa Expr || return false
    if expr.head in (:tuple, :vect)
        for arg in expr.args
            if Meta.isexpr(arg, :(=)) && length(arg.args) >= 2
                # NamedTuple value: (wv=zeros!(pool, ...), ...)
                _is_acquire_call(arg.args[2], target_pool) && return true
                _tuple_contains_acquire_call(arg.args[2], target_pool) && return true
            elseif Meta.isexpr(arg, :kw) && length(arg.args) >= 2
                _is_acquire_call(arg.args[2], target_pool) && return true
                _tuple_contains_acquire_call(arg.args[2], target_pool) && return true
            else
                _is_acquire_call(arg, target_pool) && return true
                _tuple_contains_acquire_call(arg, target_pool) && return true
            end
        end
    end
    return false
end

"""
    _extract_container_vars(expr, target_pool) -> Set{Symbol}

Order-aware extraction of container variables (assigned from tuple/vect literals
containing acquire calls). Single forward pass for top-level block statements;
removes container taint when a variable is reassigned to a non-acquire container.
E.g., `vac = (wv=zeros!(pool, ...), ...)` → returns `{:vac}`.
"""
function _extract_container_vars(expr, target_pool)
    if !(expr isa Expr && expr.head == :block)
        containers = Set{Symbol}()
        _extract_container_vars!(containers, expr, target_pool)
        return containers
    end
    containers = Set{Symbol}()
    for arg in expr.args
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :(=) && length(arg.args) >= 2
            lhs = arg.args[1]
            rhs = arg.args[2]
            if lhs isa Symbol
                if _tuple_contains_acquire_call(rhs, target_pool)
                    push!(containers, lhs)
                else
                    # Reassigned to non-acquire value — no longer a container
                    delete!(containers, lhs)
                end
            end
            # Recurse into RHS for nested blocks
            _extract_container_vars!(containers, rhs, target_pool)
        else
            # Recurse into non-assignment expressions (if/for/while/etc.)
            _extract_container_vars!(containers, arg, target_pool)
        end
    end
    return containers
end

function _extract_container_vars!(containers, expr, target_pool)
    expr isa Expr || return
    # Skip scope-introducing expressions whose `return` does NOT
    # return from the enclosing function (same as _extract_acquired_vars)
    if expr.head in (:function, :(->), :macro)
        return
    end
    if expr.head == :block
        for arg in expr.args
            _extract_container_vars!(containers, arg, target_pool)
        end
    elseif expr.head == :(=) && length(expr.args) >= 2
        lhs = expr.args[1]
        rhs = expr.args[2]
        if lhs isa Symbol && _tuple_contains_acquire_call(rhs, target_pool)
            push!(containers, lhs)
        end
        _extract_container_vars!(containers, rhs, target_pool)
    else
        for arg in expr.args
            _extract_container_vars!(containers, arg, target_pool)
        end
    end
    return
end

"""
    _find_dot_access_exposure(expr, containers) -> Vector{Tuple{Any, Union{Int,Nothing}, Symbol}}

Find container symbols exposed via dot access (e.g., `container.field`)
in return expressions. Returns `(ret_expr, line, container_sym)` tuples.
"""
function _find_dot_access_exposure(return_values, containers)
    results = Tuple{Any, Union{Int, Nothing}, Symbol}[]
    for (ret_expr, ret_line) in return_values
        syms = _collect_dot_access_syms(ret_expr, containers)
        for sym in syms
            push!(results, (ret_expr, ret_line, sym))
        end
    end
    return results
end

function _collect_dot_access_syms(expr, containers)
    found = Symbol[]
    if expr isa Expr
        if expr.head == :return
            for arg in expr.args
                append!(found, _collect_dot_access_syms(arg, containers))
            end
        elseif expr.head in (:tuple, :vect)
            for arg in expr.args
                if Meta.isexpr(arg, :(=)) && length(arg.args) >= 2
                    append!(found, _collect_dot_access_syms(arg.args[2], containers))
                else
                    append!(found, _collect_dot_access_syms(arg, containers))
                end
            end
        elseif expr.head == :. && length(expr.args) >= 1
            base = expr.args[1]
            if base isa Symbol && base in containers
                push!(found, base)
            end
        end
    end
    return found
end

"""Extract the assignment expression for a container variable from the AST."""
function _find_container_declaration(expr, var::Symbol)
    if expr isa Expr
        if expr.head == :(=) && length(expr.args) >= 2 && expr.args[1] === var
            return expr
        end
        for arg in expr.args
            result = _find_container_declaration(arg, var)
            result !== nothing && return result
        end
    end
    return nothing
end

"""Find the LineNumberNode preceding a given expression in a block."""
function _find_line_for_expr(block, target_expr)
    if block isa Expr && block.head == :block
        current_line = nothing
        for arg in block.args
            if arg isa LineNumberNode
                current_line = arg.line
            elseif arg === target_expr || (arg isa Expr && _contains_expr(arg, target_expr))
                return current_line
            end
        end
    end
    return nothing
end

function _contains_expr(expr, target)
    expr === target && return true
    if expr isa Expr
        for arg in expr.args
            _contains_expr(arg, target) && return true
        end
    end
    return false
end

"""
    _warn_compile_time_container_escape(expr, pool_name, source)

Emit a compile-time WARNING (not error) when pool-backed arrays may escape
via container dot access (e.g., `vac = (wv=zeros!(pool,...)); return vac.wv`).

This is conservative — may have false positives when the accessed field is not
pool-backed (e.g., `return vac.name` where `name="hello"`).
"""
function _warn_compile_time_container_escape(expr, pool_name, source::Union{LineNumberNode, Nothing})
    containers = _extract_container_vars(expr, pool_name)
    isempty(containers) && return

    return_values = _collect_all_return_values(expr)
    isempty(return_values) && return

    exposures = _find_dot_access_exposure(return_values, containers)
    isempty(exposures) && return

    exposed_vars = sort!(unique([sym for (_, _, sym) in exposures]))
    file = source !== nothing ? string(source.file) : nothing
    io = stderr

    # Header (matches PoolEscapeError style)
    printstyled(io, "PoolContainerEscapeWarning"; color = :yellow, bold = true)
    printstyled(io, " (compile-time, conservative)"; color = :light_black)
    println(io)

    # Descriptive message
    println(io)
    n = length(exposed_vars)
    if n == 1
        printstyled(io, "  The following container may expose pool memory from the @with_pool scope:\n"; color = :light_black)
    else
        printstyled(io, "  The following ", n, " containers may expose pool memory from the @with_pool scope:\n"; color = :light_black)
    end

    # Variables — one per line (matches PoolEscapeError)
    println(io)
    for var in exposed_vars
        printstyled(io, "    "; color = :normal)
        printstyled(io, string(var); color = :yellow, bold = true)
        printstyled(io, "  ← container holds pool-backed arrays (zeros!, acquire!, etc.)\n"; color = :light_black)
    end

    # Declarations — numbered, with [location] (matches PoolEscapeError)
    decl_idx = 0
    for var in exposed_vars
        decl_expr = _find_container_declaration(expr, var)
        if decl_expr !== nothing
            if decl_idx == 0
                println(io)
                printstyled(io, "  Declarations:\n"; bold = true)
            end
            decl_idx += 1
            decl_line = _find_line_for_expr(expr, decl_expr)
            printstyled(io, "    [", decl_idx, "]  "; color = :light_black)
            printstyled(io, string(decl_expr); color = :cyan)
            loc = _format_location_str(file, decl_line)
            if loc !== nothing
                printstyled(io, "  ["; color = :cyan, bold = true)
                printstyled(io, loc; color = :cyan, bold = true)
                printstyled(io, "] "; color = :cyan, bold = true)
            end
            println(io)
        end
    end

    # Escaping returns — numbered, with [location] (matches PoolEscapeError)
    println(io)
    n_returns = length(unique(string(r) for (r, _, _) in exposures))
    label = n_returns == 1 ? "  Escaping return:" : "  Escaping returns:"
    printstyled(io, label, "\n"; bold = true)
    seen = Set{String}()
    ret_idx = 0
    for (ret_expr, ret_line, _) in exposures
        s = string(ret_expr)
        s in seen && continue
        push!(seen, s)
        ret_idx += 1
        printstyled(io, "    [", ret_idx, "]  "; color = :light_black)
        printstyled(io, s; color = :magenta)
        loc = _format_point_location(file, ret_line)
        if loc !== nothing
            printstyled(io, "  ["; color = :magenta, bold = true)
            printstyled(io, loc; color = :magenta, bold = true)
            printstyled(io, "] "; color = :magenta, bold = true)
        end
        println(io)
    end

    # Note: this is a conservative check — may be a false positive
    println(io)
    printstyled(io, "  Note: "; color = :light_black)
    printstyled(io, "This is a conservative warning — it may be a false positive if\n"; color = :light_black)
    printstyled(io, "        the accessed field is not pool-backed. For precise runtime detection:\n"; color = :light_black)
    printstyled(io, "        julia> "; color = :light_black)
    printstyled(io, "using Preferences; set_preferences!(\"AdaptiveArrayPools\", \"runtime_check\" => true)"; color = :light_black, bold = true)
    println(io)
    printstyled(io, "        Then restart Julia (compile-time constant, requires fresh session).\n"; color = :light_black)
    println(io)

    return
end

# ==============================================================================
# Compile-time reassignment-escape WARNING
# ==============================================================================
#
# Detects patterns like:
#   v = acquire!(pool, Float64, 10)
#   v = f(v)            # ← ambiguous: f may return the same pool-backed array
#   return v            # ← pool memory may escape
#
# This is a WARNING (not error) because:
# - f(v) might return a new array (e.g., custom transform) — safe
# - f(v) might return the same array (e.g., identity, reshape) — unsafe
# - Macro can't resolve f at compile time
# - Known-safe functions (collect, copy, deepcopy) are excluded

"""
Functions known to return a new independent array (not a view or alias
of the input). Reassignment through these is always safe.
"""
const _SAFE_COPY_FUNCTIONS = Set{Symbol}([:collect, :copy, :deepcopy, :similar])

"""
    _rhs_call_contains_sym(rhs, sym) -> Bool

Check if a `:call` expression transitively contains `sym` as an argument.
Returns `false` for non-call expressions.
"""
function _rhs_call_contains_sym(rhs, sym::Symbol)
    rhs isa Expr || return false
    if rhs.head == :call
        # Check arguments (skip function name at position 1)
        for i in 2:length(rhs.args)
            arg = rhs.args[i]
            arg === sym && return true
            _rhs_call_contains_sym(arg, sym) && return true
        end
    elseif rhs.head in (:ref, :., :comprehension, :generator)
        for arg in rhs.args
            arg === sym && return true
            _rhs_call_contains_sym(arg, sym) && return true
        end
    end
    return false
end

"""
    _is_safe_copy_call(rhs) -> Bool

Check if `rhs` is a call to a known-safe function that always returns a new
independent array: `collect`, `copy`, `deepcopy`, broadcast calls (`.+`, `f.(v)`).
"""
function _is_safe_copy_call(rhs)
    rhs isa Expr || return false
    # Broadcast function call: f.(v) → Expr(:., :f, Expr(:tuple, :v))
    # (Distinguished from dot-access a.field by tuple second arg)
    if rhs.head == :. && length(rhs.args) >= 2 && Meta.isexpr(rhs.args[2], :tuple)
        return true
    end
    rhs.head == :call || return false
    fname = rhs.args[1]
    # Dotted operator: .+, .*, .-, etc. — broadcast, always allocates new array
    if fname isa Symbol && startswith(string(fname), ".")
        return true
    end
    # Handle Module.func form: e.g., Base.collect(v)
    if Meta.isexpr(fname, :.)
        fname = fname.args[end]
        fname isa QuoteNode && (fname = fname.value)
    end
    return fname isa Symbol && fname in _SAFE_COPY_FUNCTIONS
end

"""
    _find_reassign_maybe_tainted(expr, target_pool) -> Set{Symbol}

Forward pass that identifies variables which were pool-tainted and then
reassigned to a function call containing themselves as an argument.

These variables MAY still reference pool memory (e.g., `v = identity(v)`)
or MAY be safe (e.g., `v = custom_transform(v)` returning a new array).
"""
function _find_reassign_maybe_tainted(expr, target_pool)
    maybe_tainted = Set{Symbol}()
    if !(expr isa Expr && expr.head == :block)
        return maybe_tainted
    end
    tainted = Set{Symbol}()
    for arg in expr.args
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :(=) && length(arg.args) >= 2
            lhs = arg.args[1]
            rhs = arg.args[2]
            if lhs isa Symbol
                if _is_acquire_call(rhs, target_pool)
                    push!(tainted, lhs)
                    delete!(maybe_tainted, lhs)
                elseif rhs isa Symbol && rhs in tainted
                    push!(tainted, lhs)
                elseif rhs isa Symbol && rhs in maybe_tainted
                    push!(maybe_tainted, lhs)
                elseif _literal_contains_acquired(rhs, tainted)
                    push!(tainted, lhs)
                elseif _literal_contains_acquired(rhs, maybe_tainted)
                    push!(maybe_tainted, lhs)
                else
                    # Non-pool reassignment — check if it's an ambiguous call
                    if lhs in tainted && !_is_safe_copy_call(rhs) &&
                            _rhs_call_contains_sym(rhs, lhs)
                        push!(maybe_tainted, lhs)
                    else
                        delete!(maybe_tainted, lhs)
                    end
                    delete!(tainted, lhs)
                end
            elseif Meta.isexpr(lhs, :tuple)
                if Meta.isexpr(rhs, :tuple) && length(rhs.args) == length(lhs.args)
                    # Atomic destructuring: evaluate all RHS against current taint
                    for (l, r) in zip(lhs.args, rhs.args)
                        l isa Symbol || continue
                        if _is_acquire_call(r, target_pool)
                            push!(tainted, l)
                            delete!(maybe_tainted, l)
                        elseif r isa Symbol && r in tainted
                            push!(tainted, l)
                        elseif r isa Symbol && r in maybe_tainted
                            push!(maybe_tainted, l)
                        else
                            delete!(tainted, l)
                            delete!(maybe_tainted, l)
                        end
                    end
                else
                    # Opaque RHS (function call returning tuple) — all become untainted.
                    # Unlike simple `v = f(v)`, destructuring implies a transform,
                    # so we don't mark as maybe-tainted.
                    for l in lhs.args
                        l isa Symbol || continue
                        delete!(tainted, l)
                        delete!(maybe_tainted, l)
                    end
                end
            end
        else
            # Recurse into nested blocks for acquire calls (conservative: add only)
            _extract_acquired_vars(arg, target_pool, tainted)
        end
    end
    return maybe_tainted
end

"""
    _warn_compile_time_reassign_escape(expr, pool_name, source)

Emit a compile-time WARNING when a pool-backed variable is reassigned to a
function call containing itself (e.g., `v = f(v)`) and then escapes the scope.

The function `f` might return the same pool-backed array (identity, reshape)
or a new independent array. Since the macro can't resolve this, it warns
rather than errors.
"""
function _warn_compile_time_reassign_escape(expr, pool_name, source::Union{LineNumberNode, Nothing})
    maybe_tainted = _find_reassign_maybe_tainted(expr, pool_name)
    isempty(maybe_tainted) && return

    return_values = _collect_all_return_values(expr)
    isempty(return_values) && return

    # Check each return point for exposure of maybe-tainted vars
    escaped = Set{Symbol}()
    for (ret_expr, _) in return_values
        union!(escaped, _find_direct_exposure(ret_expr, maybe_tainted))
    end
    isempty(escaped) && return

    escaped_sorted = sort!(collect(escaped))
    file = source !== nothing ? string(source.file) : nothing
    io = stderr

    # Header
    printstyled(io, "PoolReassignEscapeWarning"; color = :yellow, bold = true)
    printstyled(io, " (compile-time, conservative)"; color = :light_black)
    println(io)

    # Description
    println(io)
    n = length(escaped_sorted)
    if n == 1
        printstyled(io, "  A pool-acquired variable was reassigned and may still reference pool memory:\n"; color = :light_black)
    else
        printstyled(io, "  ", n, " pool-acquired variables were reassigned and may still reference pool memory:\n"; color = :light_black)
    end

    # Variables
    println(io)
    for var in escaped_sorted
        printstyled(io, "    "; color = :normal)
        printstyled(io, string(var); color = :yellow, bold = true)
        printstyled(io, "  ← reassigned from call containing itself (may be same array)\n"; color = :light_black)
    end

    # Declarations
    declarations = _extract_declaration_sites(expr, escaped)
    if !isempty(declarations)
        println(io)
        printstyled(io, "  Declarations:\n"; bold = true)
        for (i, decl) in enumerate(declarations)
            printstyled(io, "    [", i, "]  "; color = :light_black)
            printstyled(io, string(decl.expr); color = :cyan)
            loc = _format_location_str(file, decl.line)
            if loc !== nothing
                printstyled(io, "  ["; color = :cyan, bold = true)
                printstyled(io, loc; color = :cyan, bold = true)
                printstyled(io, "] "; color = :cyan, bold = true)
            end
            println(io)
        end
    end

    # Escaping returns — collect first to determine singular/plural
    esc_returns = Tuple{Any, Union{Int, Nothing}}[]
    seen_strs = Set{String}()
    for (ret_expr, ret_line) in return_values
        point_escaped = _find_direct_exposure(ret_expr, maybe_tainted)
        isempty(point_escaped) && continue
        s = string(ret_expr)
        s in seen_strs && continue
        push!(seen_strs, s)
        push!(esc_returns, (ret_expr, ret_line))
    end
    println(io)
    label = length(esc_returns) == 1 ? "  Escaping return:" : "  Escaping returns:"
    printstyled(io, label, "\n"; bold = true)
    for (ret_idx, (ret_expr, ret_line)) in enumerate(esc_returns)
        s = string(ret_expr)
        printstyled(io, "    [", ret_idx, "]  "; color = :light_black)
        printstyled(io, s; color = :magenta)
        loc = _format_point_location(file, ret_line)
        if loc !== nothing
            printstyled(io, "  ["; color = :magenta, bold = true)
            printstyled(io, loc; color = :magenta, bold = true)
            printstyled(io, "] "; color = :magenta, bold = true)
        end
        println(io)
    end

    # Fix suggestion
    println(io)
    printstyled(io, "  Fix: "; bold = true)
    printstyled(io, "If "; color = :light_black)
    printstyled(io, "f(v)"; color = :cyan)
    printstyled(io, " returns the same array, use "; color = :light_black)
    printstyled(io, "collect(v)"; color = :green, bold = true)
    printstyled(io, " before returning.\n"; color = :light_black)
    printstyled(io, "       If it returns a new independent array, this warning is a false positive.\n"; color = :light_black)
    println(io)
    printstyled(io, "  Note: "; color = :light_black)
    printstyled(io, "For precise runtime detection:\n"; color = :light_black)
    printstyled(io, "        julia> "; color = :light_black)
    printstyled(io, "using Preferences; set_preferences!(\"AdaptiveArrayPools\", \"runtime_check\" => true)"; color = :light_black, bold = true)
    println(io)
    printstyled(io, "        Then restart Julia (compile-time constant, requires fresh session).\n"; color = :light_black)
    println(io)

    return
end

# ==============================================================================
# PoolMutationError — Compile-time structural mutation detection
# ==============================================================================

"""Per-mutation-site detail: which expression, at which line, mutates which var."""
struct MutationPoint
    expr::Any
    line::Union{Int, Nothing}
    var::Symbol
    func::Symbol
end

"""
    PoolMutationError <: Exception

Thrown at macro expansion time when structural mutation functions (`resize!`,
`push!`, `pop!`, etc.) are called on pool-backed variables within `@with_pool` /
`@maybe_with_pool` blocks.

Pool-backed arrays share memory with the pool's backing storage. Structural
mutations may cause pooling benefits (zero-alloc reuse) to be lost and
temporary extra memory retention until the next `acquire!` at the same slot.

This is a compile-time check with zero runtime cost.
"""
struct PoolMutationError <: Exception
    file::Union{String, Nothing}
    line::Union{Int, Nothing}
    points::Vector{MutationPoint}
    declarations::Vector{DeclarationSite}
end

function Base.showerror(io::IO, e::PoolMutationError)
    # Header
    printstyled(io, "PoolMutationError"; color = :red, bold = true)
    printstyled(io, " (compile-time)"; color = :light_black)
    println(io)

    # Descriptive message
    println(io)
    n = length(e.points)
    if n == 1
        printstyled(io, "  Structural mutation of pool-backed array detected:\n"; color = :light_black)
    else
        printstyled(io, "  ", n, " structural mutations of pool-backed arrays detected:\n"; color = :light_black)
    end

    # Declaration sites
    if !isempty(e.declarations)
        println(io)
        printstyled(io, "  Declarations:\n"; bold = true)
        for (idx, decl) in enumerate(e.declarations)
            printstyled(io, "    [", idx, "]  "; color = :light_black)
            printstyled(io, string(decl.expr); color = :cyan)
            decl_file = (decl.file !== nothing && decl.file !== :none) ? decl.file : e.file
            loc = _format_location_str(decl_file, decl.line)
            if loc !== nothing
                printstyled(io, "  ["; color = :cyan, bold = true)
                printstyled(io, loc; color = :cyan, bold = true)
                printstyled(io, "] "; color = :cyan, bold = true)
            end
            println(io)
        end
    end

    # Mutation points
    println(io)
    label = n == 1 ? "  Dangerous call:" : "  Dangerous calls:"
    printstyled(io, label, "\n"; bold = true)
    for (idx, pt) in enumerate(e.points)
        printstyled(io, "    [", idx, "]  "; color = :light_black)
        printstyled(io, string(pt.func); color = :red, bold = true)
        printstyled(io, "("; color = :normal)
        printstyled(io, string(pt.var); color = :red, bold = true)
        printstyled(io, ", ...)"; color = :normal)
        loc = _format_point_location(e.file, pt.line)
        if loc !== nothing
            printstyled(io, "  ["; color = :magenta, bold = true)
            printstyled(io, loc; color = :magenta, bold = true)
            printstyled(io, "] "; color = :magenta, bold = true)
        end
        println(io)
    end

    # Suggestion
    println(io)
    printstyled(io, "  Tip: "; bold = true)
    printstyled(io, "Consider requesting the exact size via "; color = :light_black)
    printstyled(io, "acquire!(pool, T, n)"; bold = true)
    printstyled(io, " if known in advance.\n"; color = :light_black)
    printstyled(io, "       resize!/push!/pop! may trigger memory reallocation — pooling benefits\n"; color = :light_black)
    printstyled(io, "       (zero-alloc reuse) may be lost; temporary extra memory retention may occur.\n"; color = :light_black)

    # False positive
    println(io)
    printstyled(io, "  False positive?\n"; bold = true)
    printstyled(io, "    Please file an issue at "; color = :light_black)
    printstyled(io, "https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/issues"; bold = true)
    return printstyled(io, "\n    with a minimal reproducer so we can improve the mutation detector.\n"; color = :light_black)
end

# Suppress stacktrace
Base.showerror(io::IO, e::PoolMutationError, ::Any; backtrace = true) = showerror(io, e)

# ==============================================================================
# Internal: Structural Mutation Detection
# ==============================================================================

"""Set of function names that structurally mutate arrays (resize, grow, shrink)."""
const _STRUCTURAL_MUTATION_NAMES = Set{Symbol}(
    [
        :resize!, :push!, :pop!, :pushfirst!, :popfirst!,
        :append!, :prepend!, :deleteat!, :insert!, :splice!,
        :empty!, :sizehint!,
    ]
)

"""
    _is_mutation_call(expr, acquired) -> Union{Tuple{Symbol, Symbol}, Nothing}

Check if `expr` is a call to a structural mutation function with a pool-backed
variable as the first argument. Returns `(func_name, var_name)` or `nothing`.
"""
function _is_mutation_call(expr, acquired::Set{Symbol})
    expr isa Expr && expr.head == :call && length(expr.args) >= 2 || return nothing

    fn = expr.args[1]
    first_arg = expr.args[2]
    first_arg isa Symbol && first_arg in acquired || return nothing

    # Direct name: resize!(v, ...)
    if fn isa Symbol && fn in _STRUCTURAL_MUTATION_NAMES
        return (fn, first_arg)
    end
    # Qualified name: Base.resize!(v, ...)
    if fn isa Expr && fn.head == :. && length(fn.args) >= 2
        qn = fn.args[end]
        if qn isa QuoteNode && qn.value isa Symbol && qn.value in _STRUCTURAL_MUTATION_NAMES
            return (qn.value, first_arg)
        end
    end
    return nothing
end

"""
    _find_mutation_calls(expr, acquired) -> Vector{MutationPoint}

Walk AST to find all structural mutation calls on pool-backed variables.
Tracks LineNumberNodes for precise error locations. Skips nested function
definitions (lambdas within @with_pool are separate scopes).
"""
function _find_mutation_calls(expr, acquired::Set{Symbol})
    points = MutationPoint[]
    _find_mutation_calls!(points, expr, acquired, nothing)
    return points
end

function _find_mutation_calls!(points, expr, acquired, current_line::Union{Int, Nothing})
    expr isa Expr || return

    # Skip nested function definitions (they have their own scope)
    expr.head in (:function, :(->)) && return

    # Track line numbers
    if expr.head == :block
        line = current_line
        for arg in expr.args
            if arg isa LineNumberNode
                line = arg.line
            else
                _find_mutation_calls!(points, arg, acquired, line)
            end
        end
        return
    end

    # Check if this is a mutation call
    result = _is_mutation_call(expr, acquired)
    if result !== nothing
        func, var = result
        push!(points, MutationPoint(expr, current_line, var, func))
    end

    # Recurse into sub-expressions
    for arg in expr.args
        if arg isa LineNumberNode
            current_line = arg.line
        else
            _find_mutation_calls!(points, arg, acquired, current_line)
        end
    end
    return
end

"""
    _check_structural_mutation(expr, pool_name, source)

Compile-time (macro expansion time) structural mutation detection.

Checks if any pool-backed variable is passed to `resize!`, `push!`, `pop!`,
or other functions that change array structure. These operations may cause
pooling benefits (zero-alloc reuse) to be lost and temporary extra memory
retention, because pool-backed arrays share memory with backing storage.

Skipped when `STATIC_POOLING = false` (pooling disabled, acquire returns normal arrays).
"""
function _check_structural_mutation(expr, pool_name, source::Union{LineNumberNode, Nothing})
    acquired = _extract_ordered_acquired(expr, pool_name)
    isempty(acquired) && return nothing

    points = _find_mutation_calls(expr, acquired)
    isempty(points) && return nothing

    # Collect declaration sites for the mutated variables
    mutated_vars = Set{Symbol}(pt.var for pt in points)
    declarations = _extract_declaration_sites(expr, mutated_vars)

    file = source !== nothing ? string(source.file) : nothing
    line = source !== nothing ? source.line : nothing
    throw(PoolMutationError(file, line, points, declarations))
end
