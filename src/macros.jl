# ==============================================================================
# Macros for AdaptiveArrayPools
# ==============================================================================

# ==============================================================================
# PoolEscapeError — Compile-time escape detection error
# ==============================================================================

"""Per-return-point escape detail: which expression, at which line, leaks which vars."""
struct EscapePoint
    expr::Any
    line::Union{Int, Nothing}
    vars::Vector{Symbol}
end

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

function Base.showerror(io::IO, e::PoolEscapeError)
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

    # Block logic — shared with backend-specific code generation
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
    transformed_expr = _inject_pending_callsite(transformed_expr, pool_name, expr)

    if safe
        transformed_expr = _transform_return_stmts(transformed_expr, pool_name)
        return quote
            $checkpoint_call
            try
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

        transformed_expr = _transform_return_stmts(transformed_expr, pool_name;
                                                    rewind_call = raw_rewind,
                                                    entry_depth_guard = raw_guard)
        transformed_expr = _transform_break_continue(transformed_expr, raw_rewind, raw_guard)

        return quote
            local $(esc(entry_depth_var)) = $(esc(pool_name))._current_depth
            $checkpoint_call
            local _result = $(esc(transformed_expr))
            if $_RUNTIME_CHECK_REF($(esc(pool_name)))
                $_validate_pool_return(_result, $(esc(pool_name)))
            end
            while $(esc(pool_name))._current_depth > $(esc(entry_depth_var)) + 1
                $rewind!($(esc(pool_name)))
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
    transformed_expr = _inject_pending_callsite(transformed_expr, pool_name, expr)

    if safe
        transformed_expr = _transform_return_stmts(transformed_expr, pool_name)
        return quote
            $checkpoint_call
            try
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
        transformed_expr = _transform_return_stmts(transformed_expr, pool_name;
                                                    rewind_call = raw_rewind,
                                                    entry_depth_guard = raw_guard)

        return quote
            local $(esc(entry_depth_var)) = $(esc(pool_name))._current_depth
            $checkpoint_call
            local _result = $(esc(transformed_expr))
            if $_RUNTIME_CHECK_REF($(esc(pool_name)))
                $_validate_pool_return(_result, $(esc(pool_name)))
            end
            while $(esc(pool_name))._current_depth > $(esc(entry_depth_var)) + 1
                $rewind!($(esc(pool_name)))
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

    # Block logic (force_enable=true path)
    inner = _generate_block_inner(pool_name, expr, safe, source)
    pool_getter = :($_get_pool_for_backend($(Val{backend}())))
    return _wrap_with_dispatch(esc(pool_name), pool_getter, inner; backend)
end

"""
    _generate_function_pool_code_with_backend(backend, pool_name, func_def, force_enable, disable_pooling)

Generate function code for a specific backend (e.g., :cuda).
Wraps the function body with pool getter, checkpoint, try-finally, rewind.

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
- `acquire!` and its alias `acquire_view!`
- `unsafe_acquire!` and its alias `acquire_array!`
- `zeros!`, `ones!`, `similar!`
- `unsafe_zeros!`, `unsafe_ones!`, `unsafe_similar!`

Handles various forms:
- `[unsafe_]acquire!(pool, Type, dims...)`: extracts Type directly
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
                acquire_names = (:acquire!, :unsafe_acquire!, :acquire_view!, :acquire_array!)

                # Get function name (handle qualified names)
                fn_name = fn
                if fn isa Expr && fn.head == :. && length(fn.args) >= 2
                    qn = fn.args[end]
                    if qn isa QuoteNode
                        fn_name = qn.value
                    end
                end

                nargs = length(expr.args)

                # acquire!/unsafe_acquire!/acquire_view!/acquire_array!
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
                    # zeros!/ones!/unsafe_zeros!/unsafe_ones!
                elseif fn in (:zeros!, :ones!, :unsafe_zeros!, :unsafe_ones!) || fn_name in (:zeros!, :ones!, :unsafe_zeros!, :unsafe_ones!)
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
                    # similar!/unsafe_similar!
                elseif fn in (:similar!, :unsafe_similar!) || fn_name in (:similar!, :unsafe_similar!)
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
- Parametric types like Vector{T} trigger fallback
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
                # Parametric type like Vector{Float64} - can't use as Type argument
                has_dynamic = true
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

Transform acquire!/unsafe_acquire!/convenience function calls to their _impl! counterparts.
Only transforms calls where the first argument matches `pool_name`.

This allows macro-transformed code to bypass the type touch recording overhead,
since the macro already knows about these calls at compile time.

Transformation rules:
- `acquire!(pool, ...)` → `_acquire_impl!(pool, ...)`
- `acquire_view!(pool, ...)` → `_acquire_impl!(pool, ...)`
- `unsafe_acquire!(pool, ...)` → `_unsafe_acquire_impl!(pool, ...)`
- `acquire_array!(pool, ...)` → `_unsafe_acquire_impl!(pool, ...)`
- `zeros!(pool, ...)` → `_zeros_impl!(pool, ...)`
- `ones!(pool, ...)` → `_ones_impl!(pool, ...)`
- `similar!(pool, ...)` → `_similar_impl!(pool, ...)`
"""
# Module-qualified references for transformed acquire calls
# Using GlobalRef ensures the function is looked up in AdaptiveArrayPools, not the caller's module
const _ACQUIRE_IMPL_REF = GlobalRef(@__MODULE__, :_acquire_impl!)
const _UNSAFE_ACQUIRE_IMPL_REF = GlobalRef(@__MODULE__, :_unsafe_acquire_impl!)
const _ZEROS_IMPL_REF = GlobalRef(@__MODULE__, :_zeros_impl!)
const _ONES_IMPL_REF = GlobalRef(@__MODULE__, :_ones_impl!)
const _TRUES_IMPL_REF = GlobalRef(@__MODULE__, :_trues_impl!)
const _FALSES_IMPL_REF = GlobalRef(@__MODULE__, :_falses_impl!)
const _SIMILAR_IMPL_REF = GlobalRef(@__MODULE__, :_similar_impl!)
const _UNSAFE_ZEROS_IMPL_REF = GlobalRef(@__MODULE__, :_unsafe_zeros_impl!)
const _UNSAFE_ONES_IMPL_REF = GlobalRef(@__MODULE__, :_unsafe_ones_impl!)
const _UNSAFE_SIMILAR_IMPL_REF = GlobalRef(@__MODULE__, :_unsafe_similar_impl!)
const _RESHAPE_IMPL_REF = GlobalRef(@__MODULE__, :_reshape_impl!)

function _transform_acquire_calls(expr, pool_name)
    if expr isa Expr
        # Handle call expressions
        if expr.head == :call && length(expr.args) >= 2
            fn = expr.args[1]
            pool_arg = expr.args[2]

            # Only transform if pool argument matches
            if pool_arg == pool_name
                # Check for acquire functions (including qualified names)
                if fn == :acquire! || fn == :acquire_view!
                    expr = Expr(:call, _ACQUIRE_IMPL_REF, expr.args[2:end]...)
                elseif fn == :unsafe_acquire! || fn == :acquire_array!
                    expr = Expr(:call, _UNSAFE_ACQUIRE_IMPL_REF, expr.args[2:end]...)
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
                elseif fn == :unsafe_zeros!
                    expr = Expr(:call, _UNSAFE_ZEROS_IMPL_REF, expr.args[2:end]...)
                elseif fn == :unsafe_ones!
                    expr = Expr(:call, _UNSAFE_ONES_IMPL_REF, expr.args[2:end]...)
                elseif fn == :unsafe_similar!
                    expr = Expr(:call, _UNSAFE_SIMILAR_IMPL_REF, expr.args[2:end]...)
                elseif fn isa Expr && fn.head == :. && length(fn.args) >= 2
                    # Qualified name: AdaptiveArrayPools.acquire! etc.
                    qn = fn.args[end]
                    if qn == QuoteNode(:acquire!) || qn == QuoteNode(:acquire_view!)
                        expr = Expr(:call, _ACQUIRE_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:unsafe_acquire!) || qn == QuoteNode(:acquire_array!)
                        expr = Expr(:call, _UNSAFE_ACQUIRE_IMPL_REF, expr.args[2:end]...)
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
                    elseif qn == QuoteNode(:unsafe_zeros!)
                        expr = Expr(:call, _UNSAFE_ZEROS_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:unsafe_ones!)
                        expr = Expr(:call, _UNSAFE_ONES_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:unsafe_similar!)
                        expr = Expr(:call, _UNSAFE_SIMILAR_IMPL_REF, expr.args[2:end]...)
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
const _REWIND_REF = GlobalRef(@__MODULE__, :rewind!)
const _LAZY_REWIND_REF = GlobalRef(@__MODULE__, :_lazy_rewind!)
const _TYPED_LAZY_REWIND_REF = GlobalRef(@__MODULE__, :_typed_lazy_rewind!)
const _CAN_USE_TYPED_PATH_REF = GlobalRef(@__MODULE__, :_can_use_typed_path)
const _TRACKED_MASK_REF = GlobalRef(@__MODULE__, :_tracked_mask_for_types)

"""Set of all transformed `_*_impl!` function names (GlobalRef targets)."""
const _IMPL_FUNC_NAMES = Set{Symbol}(
    [
        :_acquire_impl!, :_unsafe_acquire_impl!,
        :_zeros_impl!, :_ones_impl!, :_trues_impl!, :_falses_impl!,
        :_similar_impl!, :_reshape_impl!,
        :_unsafe_zeros_impl!, :_unsafe_ones_impl!, :_unsafe_similar_impl!,
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
function _transform_return_stmts(expr, pool_name, current_lnn = nothing;
                                  rewind_call = nothing,
                                  entry_depth_guard = nothing)
    expr isa Expr || return expr

    # Don't recurse into nested function definitions (return belongs to inner function)
    if expr.head in (:function, :->)
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
        value_expr = _transform_return_stmts(value_expr, pool_name, current_lnn;
                                              rewind_call, entry_depth_guard)
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
                push!(new_args, _transform_return_stmts(arg, pool_name, lnn;
                                                         rewind_call, entry_depth_guard))
            end
        end
        return Expr(:block, new_args...)
    end

    # Other expressions: recurse with current_lnn
    new_args = Any[_transform_return_stmts(arg, pool_name, current_lnn;
                                            rewind_call, entry_depth_guard) for arg in expr.args]
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

    # Don't recurse into nested functions
    expr.head in (:function, :->) && return expr

    # Don't recurse into loop bodies — break/continue there are for those loops
    expr.head in (:for, :while) && return expr

    # Transform bare break/continue at pool-block level
    if expr.head in (:break, :continue)
        return Expr(:block, entry_depth_guard, rewind_call, expr)
    end

    # Recurse into other expressions (if, try, let, block, etc.)
    new_args = Any[_transform_break_continue(arg, rewind_call, entry_depth_guard)
                   for arg in expr.args]
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

        # Skip nested function bodies (separate scope)
        node.head in (:function, :->) && return

        for arg in node.args
            walk(arg)
        end
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
    if !isempty(unsafe)
        targets = join(unsafe, ", ")
        error("@with_pool: @goto to external label(s) ($targets) detected. " *
              "This would bypass rewind! and corrupt pool state. " *
              "Use @safe_with_pool for exception-safe behavior with @goto.")
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
        :acquire!, :unsafe_acquire!, :acquire_view!, :acquire_array!,
        :zeros!, :ones!, :similar!, :reshape!,
        :unsafe_zeros!, :unsafe_ones!, :unsafe_similar!,
        :trues!, :falses!,
    ]
)

"""Function names that return views (SubArray) from pool memory."""
const _VIEW_ACQUIRE_NAMES = Set{Symbol}(
    [
        :acquire!, :acquire_view!,
        :zeros!, :ones!, :similar!,
        :unsafe_zeros!, :unsafe_ones!, :unsafe_similar!,
        :reshape!,
    ]
)

"""Function names that return raw Arrays backed by pool memory (unsafe_wrap)."""
const _ARRAY_ACQUIRE_NAMES = Set{Symbol}(
    [
        :unsafe_acquire!, :acquire_array!,
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
    _remove_flat_reassigned!(expr, acquired, target_pool)

Walk top-level statements in order and remove variables from `acquired`
if they are reassigned to a non-acquire call. Handles both simple assignment
(`v = expr`) and tuple destructuring (`(a, v) = expr`).
Only handles flat (non-branching) reassignment — conditional is conservatively kept.
"""
function _remove_flat_reassigned!(expr, acquired, target_pool)
    if !(expr isa Expr && expr.head == :block)
        return
    end
    for arg in expr.args
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :(=) && length(arg.args) >= 2
            lhs = arg.args[1]
            rhs = arg.args[2]
            if lhs isa Symbol && lhs in acquired && !_is_acquire_call(rhs, target_pool) &&
                    !(rhs isa Symbol && rhs in acquired) &&  # keep aliases
                    !_literal_contains_acquired(rhs, acquired)  # keep container wrapping
                delete!(acquired, lhs)
            elseif Meta.isexpr(lhs, :tuple)
                # Destructuring: (a, v, b) = expr
                if Meta.isexpr(rhs, :tuple) && length(rhs.args) == length(lhs.args)
                    # RHS is tuple literal — check each element pair
                    for (l, r) in zip(lhs.args, rhs.args)
                        if l isa Symbol && l in acquired && !_is_acquire_call(r, target_pool) &&
                                !(r isa Symbol && r in acquired)  # keep aliases
                            delete!(acquired, l)
                        end
                    end
                else
                    # RHS is function call or opaque expression —
                    # acquired var is now reassigned to unknown value, remove it
                    for l in lhs.args
                        if l isa Symbol && l in acquired
                            delete!(acquired, l)
                        end
                    end
                end
            end
        end
    end
    return
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
- `:pool_view` — from acquire!, zeros!, etc. (returns SubArray)
- `:pool_array` — from unsafe_acquire! (returns Array via unsafe_wrap)
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
container patterns (`(v, w)`, `[v]`, `(key=v,)`).

Skipped when `STATIC_POOLING = false` (pooling disabled, acquire returns normal arrays).
"""
function _check_compile_time_escape(expr, pool_name, source::Union{LineNumberNode, Nothing})
    # Extract variables assigned from acquire calls
    acquired = _extract_acquired_vars(expr, pool_name)
    isempty(acquired) && return

    # Remove vars that were unconditionally reassigned to non-acquire values
    _remove_flat_reassigned!(expr, acquired, pool_name)
    isempty(acquired) && return

    # Collect ALL return points: explicit returns + implicit (last expr / if-else branches)
    return_values = _collect_all_return_values(expr)
    isempty(return_values) && return

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
    isempty(all_escaped) && return

    sorted = sort!(collect(all_escaped))
    var_info = _classify_escaped_vars(expr, pool_name, sorted, acquired)
    declarations = _extract_declaration_sites(expr, all_escaped)
    file = source !== nothing ? string(source.file) : nothing
    line = source !== nothing ? source.line : nothing
    throw(PoolEscapeError(sorted, file, line, points, var_info, declarations))
end
