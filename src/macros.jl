# ==============================================================================
# Macros for AdaptiveArrayPools
# ==============================================================================

# ==============================================================================
# PoolEscapeError — Compile-time escape detection error
# ==============================================================================

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
end

function Base.showerror(io::IO, e::PoolEscapeError)
    # Header
    printstyled(io, "PoolEscapeError"; bold=true)
    printstyled(io, " (compile-time)"; color=:light_black)
    if e.file !== nothing
        printstyled(io, " at "; color=:light_black)
        printstyled(io, string(e.file, ":", e.line); color=:light_black)
    end
    println(io)

    # Escaped variables — one per line
    println(io)
    for v in e.vars
        printstyled(io, "    "; color=:normal)
        printstyled(io, string(v); color=:red, bold=true)
        printstyled(io, "  ← temporary array, must not escape @with_pool scope\n"; color=:light_black)
    end

    # Suggestion 1: trace the definition
    println(io)
    vars_str = join([string(v) for v in e.vars], ", ")
    printstyled(io, "  Fix: "; bold=true)
    println(io, "Trace where ", vars_str, " are assigned.")
    collects_str = join(["collect($v)" for v in e.vars], ", ")
    println(io, "       If from acquire!() / zeros!() / similar!(), use ", collects_str)
    println(io, "       to return owned copies.")

    # Suggestion 2: false positive escape hatch
    println(io)
    printstyled(io, "  False positive?\n"; bold=true)
    printstyled(io, "    Wrap with identity() to suppress this check.\n"; color=:light_black)
end

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
Calls `checkpoint!` on entry and `rewind!` on exit (even if errors occur).

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
    _fix_try_body_lnn!(expr, source)

Fix LineNumberNodes inside try blocks to point to user source.
Julia's stack trace uses the LAST LNN before error location for line numbers.
By replacing the first LNN in try body with source LNN, we ensure correct
line numbers in stack traces.

Scans first few args to handle Expr(:meta, ...) from @inline etc.
If source.file === :none (REPL/eval), don't clobber valid file LNNs.
Modifies expr in-place and returns it.
"""
function _fix_try_body_lnn!(expr, source::Union{LineNumberNode, Nothing})
    source === nothing && return expr
    # Don't clobber valid file info with :none from REPL/eval
    source.file === :none && return expr
    source_lnn = LineNumberNode(source.line, source.file)

    if expr isa Expr
        if expr.head === :try && length(expr.args) >= 1
            try_body = expr.args[1]
            if try_body isa Expr && try_body.head === :block && !isempty(try_body.args)
                lnn_idx = _find_first_lnn_index(try_body.args)
                if lnn_idx !== nothing
                    existing_lnn = try_body.args[lnn_idx]
                    if existing_lnn.file != source.file
                        # Replace macros.jl LNN with source LNN
                        try_body.args[lnn_idx] = source_lnn
                    end
                end
            end
        end
        # Recurse into all args
        for arg in expr.args
            _fix_try_body_lnn!(arg, source)
        end
    end
    return expr
end

# ==============================================================================
# Internal: Code Generation
# ==============================================================================

function _generate_pool_code(pool_name, expr, force_enable; source::Union{LineNumberNode, Nothing} = nothing)
    # Compile-time check: if pooling disabled, use DisabledPool to preserve backend context
    if !STATIC_POOLING
        disabled_pool = _disabled_pool_expr(:cpu)
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            # Function definition: inject local pool = DisabledPool at start of body
            return _generate_function_pool_code(pool_name, expr, force_enable, true, :cpu; source)
        else
            return quote
                local $(esc(pool_name)) = $disabled_pool
                $(esc(expr))
            end
        end
    end

    # Check if function definition
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        return _generate_function_pool_code(pool_name, expr, force_enable, false; source)
    end

    # Compile-time escape detection (zero runtime cost)
    _check_compile_time_escape(expr, pool_name, source)

    # Block logic
    # Extract types from acquire! calls for optimized checkpoint/rewind
    # Only extract types for calls to the target pool (pool_name)
    all_types = _extract_acquire_types(expr, pool_name)
    local_vars = _extract_local_assignments(expr)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)

    # Use typed checkpoint/rewind if all types are static, otherwise fallback to full
    use_typed = !has_dynamic && !isempty(static_types)

    # For typed path: transform acquire! → _acquire_impl! (bypasses type touch recording)
    # For dynamic path: keep acquire! untransformed so _record_type_touch! is called
    transformed_expr = use_typed ? _transform_acquire_calls(expr, pool_name) : expr

    if use_typed
        checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
    else
        checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
    end

    if use_typed
        rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
    else
        rewind_call = _generate_lazy_rewind_call(esc(pool_name))
    end

    if force_enable
        return quote
            local $(esc(pool_name)) = get_task_local_pool()
            $checkpoint_call
            try
                local _result = $(esc(transformed_expr))
                if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                    $_validate_pool_return(_result, $(esc(pool_name)))
                end
                _result
            finally
                $rewind_call
            end
        end
    else
        # Split branches completely to avoid Union boxing
        return quote
            if $MAYBE_POOLING[]
                local $(esc(pool_name)) = get_task_local_pool()
                $checkpoint_call
                try
                    local _result = $(esc(transformed_expr))
                    if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                        $_validate_pool_return(_result, $(esc(pool_name)))
                    end
                    _result
                finally
                    $rewind_call
                end
            else
                local $(esc(pool_name)) = $DISABLED_CPU
                $(esc(expr))
            end
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
function _generate_pool_code_with_backend(backend::Symbol, pool_name, expr, force_enable::Bool; source::Union{LineNumberNode, Nothing} = nothing)
    # Compile-time check: if pooling disabled, use DisabledPool to preserve backend context
    if !STATIC_POOLING
        disabled_pool = _disabled_pool_expr(backend)
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            return _generate_function_pool_code_with_backend(backend, pool_name, expr, force_enable, true; source)
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
            return _generate_function_pool_code_with_backend(backend, pool_name, expr, false, false; source)
        end

        # Compile-time escape detection (zero runtime cost)
        _check_compile_time_escape(expr, pool_name, source)

        # Block logic with runtime check
        all_types = _extract_acquire_types(expr, pool_name)
        local_vars = _extract_local_assignments(expr)
        static_types, has_dynamic = _filter_static_types(all_types, local_vars)
        use_typed = !has_dynamic && !isempty(static_types)
        # For typed path: transform acquire! → _acquire_impl! (bypasses type touch recording)
        # For dynamic path: keep acquire! untransformed so _record_type_touch! is called
        transformed_expr = use_typed ? _transform_acquire_calls(expr, pool_name) : expr
        pool_getter = :($_get_pool_for_backend($(Val{backend}())))

        if use_typed
            checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
            rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
        else
            checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
            rewind_call = _generate_lazy_rewind_call(esc(pool_name))
        end

        return quote
            if $MAYBE_POOLING[]
                local $(esc(pool_name)) = $pool_getter
                $checkpoint_call
                try
                    local _result = $(esc(transformed_expr))
                    if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                        $_validate_pool_return(_result, $(esc(pool_name)))
                    end
                    _result
                finally
                    $rewind_call
                end
            else
                local $(esc(pool_name)) = $disabled_pool
                $(esc(expr))
            end
        end
    end

    # Check if function definition
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        return _generate_function_pool_code_with_backend(backend, pool_name, expr, true, false; source)
    end

    # Compile-time escape detection (zero runtime cost)
    _check_compile_time_escape(expr, pool_name, source)

    # Block logic: Extract types from acquire! calls for optimized checkpoint/rewind
    all_types = _extract_acquire_types(expr, pool_name)
    local_vars = _extract_local_assignments(expr)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)

    # Use typed checkpoint/rewind if all types are static, otherwise fallback to full
    use_typed = !has_dynamic && !isempty(static_types)

    # For typed path: transform acquire! → _acquire_impl! (bypasses type touch recording)
    # For dynamic path: keep acquire! untransformed so _record_type_touch! is called
    transformed_expr = use_typed ? _transform_acquire_calls(expr, pool_name) : expr

    # Use Val{backend}() for compile-time dispatch - fully inlinable
    pool_getter = :($_get_pool_for_backend($(Val{backend}())))

    if use_typed
        checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
    else
        checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
    end

    if use_typed
        rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
    else
        rewind_call = _generate_lazy_rewind_call(esc(pool_name))
    end

    return quote
        local $(esc(pool_name)) = $pool_getter
        $checkpoint_call
        try
            local _result = $(esc(transformed_expr))
            if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                $_validate_pool_return(_result, $(esc(pool_name)))
            end
            _result
        finally
            $rewind_call
        end
    end
end

"""
    _generate_function_pool_code_with_backend(backend, pool_name, func_def, force_enable, disable_pooling)

Generate function code for a specific backend (e.g., :cuda).
Wraps the function body with pool getter, checkpoint, try-finally, rewind.

When `disable_pooling=true` (STATIC_POOLING=false), generates DisabledPool binding.
When `force_enable=true` (@with_pool), always uses the real pool.
When `force_enable=false` (@maybe_with_pool), generates MAYBE_POOLING[] runtime check.
"""
function _generate_function_pool_code_with_backend(backend::Symbol, pool_name, func_def, force_enable::Bool, disable_pooling::Bool; source::Union{LineNumberNode, Nothing} = nothing)
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
    _check_compile_time_escape(body, pool_name, source)

    # Analyze body for types
    all_types = _extract_acquire_types(body, pool_name)
    local_vars = _extract_local_assignments(body)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)
    use_typed = !has_dynamic && !isempty(static_types)

    # For typed path: transform acquire! → _acquire_impl! (bypasses type touch recording)
    # For dynamic path: keep acquire! untransformed so _record_type_touch! is called
    transformed_body = use_typed ? _transform_acquire_calls(body, pool_name) : body

    # Use Val{backend}() for compile-time dispatch
    pool_getter = :($_get_pool_for_backend($(Val{backend}())))

    if use_typed
        checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
        rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
    else
        checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
        rewind_call = _generate_lazy_rewind_call(esc(pool_name))
    end

    if force_enable
        new_body = quote
            local $(esc(pool_name)) = $pool_getter
            $checkpoint_call
            try
                local _result = begin
                    $(esc(transformed_body))
                end
                if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                    $_validate_pool_return(_result, $(esc(pool_name)))
                end
                _result
            finally
                $rewind_call
            end
        end
    else
        disabled_pool = _disabled_pool_expr(backend)
        new_body = quote
            if $MAYBE_POOLING[]
                local $(esc(pool_name)) = $pool_getter
                $checkpoint_call
                try
                    local _result = begin
                        $(esc(transformed_body))
                    end
                    if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                        $_validate_pool_return(_result, $(esc(pool_name)))
                    end
                    _result
                finally
                    $rewind_call
                end
            else
                local $(esc(pool_name)) = $disabled_pool
                $(esc(body))
            end
        end
    end

    # Ensure new_body has source location for proper stack traces
    new_body = _ensure_body_has_toplevel_lnn(new_body, source)
    _fix_try_body_lnn!(new_body, source)  # Fix try block LNNs for accurate stack traces
    return Expr(def_head, esc(call_expr), new_body)
end

function _generate_function_pool_code(pool_name, func_def, force_enable, disable_pooling, backend::Symbol = :cpu; source::Union{LineNumberNode, Nothing} = nothing)
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
    _check_compile_time_escape(body, pool_name, source)

    # Analyze body for types
    all_types = _extract_acquire_types(body, pool_name)
    local_vars = _extract_local_assignments(body)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)
    use_typed = !has_dynamic && !isempty(static_types)

    # For typed path: transform acquire! → _acquire_impl! (bypasses type touch recording)
    # For dynamic path: keep acquire! untransformed so _record_type_touch! is called
    transformed_body = use_typed ? _transform_acquire_calls(body, pool_name) : body

    if use_typed
        checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
    else
        checkpoint_call = _generate_lazy_checkpoint_call(esc(pool_name))
    end

    if use_typed
        rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
    else
        rewind_call = _generate_lazy_rewind_call(esc(pool_name))
    end

    if force_enable
        new_body = quote
            local $(esc(pool_name)) = get_task_local_pool()
            $checkpoint_call
            try
                local _result = begin
                    $(esc(transformed_body))
                end
                if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                    $_validate_pool_return(_result, $(esc(pool_name)))
                end
                _result
            finally
                $rewind_call
            end
        end
    else
        disabled_pool = _disabled_pool_expr(backend)
        new_body = quote
            if $MAYBE_POOLING[]
                local $(esc(pool_name)) = get_task_local_pool()
                $checkpoint_call
                try
                    local _result = begin
                        $(esc(transformed_body))
                    end
                    if ($POOL_SAFETY_LV[] >= 2 || $POOL_DEBUG[])
                        $_validate_pool_return(_result, $(esc(pool_name)))
                    end
                    _result
                finally
                    $rewind_call
                end
            else
                local $(esc(pool_name)) = $disabled_pool
                $(esc(body))
            end
        end
    end

    # Ensure new_body has source location for proper stack traces
    new_body = _ensure_body_has_toplevel_lnn(new_body, source)
    _fix_try_body_lnn!(new_body, source)  # Fix try block LNNs for accurate stack traces
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
const _ALL_ACQUIRE_NAMES = Set{Symbol}([
    :acquire!, :unsafe_acquire!, :acquire_view!, :acquire_array!,
    :zeros!, :ones!, :similar!, :reshape!,
    :unsafe_zeros!, :unsafe_ones!, :unsafe_similar!,
    :trues!, :falses!,
])

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
            elseif Meta.isexpr(lhs, :tuple) && Meta.isexpr(rhs, :tuple)
                # Destructuring with tuple literal RHS: (v, w) = (acquire!(...), expr)
                for (l, r) in zip(lhs.args, rhs.args)
                    if l isa Symbol && _is_acquire_call(r, target_pool)
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
    _collect_all_return_values(expr) -> Vector

Collect all expressions that could be returned from a block/function body:
- Explicit `return expr` statements anywhere in the body (recursive, skips nested functions)
- Implicit returns: the last expression, recursing into if/else/elseif branches
"""
function _collect_all_return_values(expr)
    values = Any[]
    _collect_explicit_returns!(values, expr)
    last = _get_last_expression(expr)
    if last !== nothing
        _collect_implicit_return_values!(values, last)
    end
    return values
end

"""Walk AST to find all explicit `return expr` statements.
Skips nested function definitions (their returns belong to a different scope)."""
function _collect_explicit_returns!(values, expr)
    expr isa Expr || return
    # Nested functions have their own scope — don't recurse
    expr.head in (:function, :(->)) && return
    if expr.head == :return
        push!(values, expr)
        return
    end
    for arg in expr.args
        _collect_explicit_returns!(values, arg)
    end
end

"""Expand implicit return values by recursing into if/elseif/else branches.
Non-branch expressions are collected as-is."""
function _collect_implicit_return_values!(values, expr)
    if expr isa Expr && expr.head in (:if, :elseif)
        # args[1] = condition, args[2] = then-branch, args[3] = else/elseif (optional)
        for i in 2:length(expr.args)
            branch = expr.args[i]
            if branch isa Expr && branch.head in (:if, :elseif)
                _collect_implicit_return_values!(values, branch)
            else
                last = _get_last_expression(branch)
                if last !== nothing
                    _collect_implicit_return_values!(values, last)
                end
            end
        end
    else
        push!(values, expr)
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
            if lhs isa Symbol && lhs in acquired && !_is_acquire_call(rhs, target_pool)
                delete!(acquired, lhs)
            elseif Meta.isexpr(lhs, :tuple)
                # Destructuring: (a, v, b) = expr
                if Meta.isexpr(rhs, :tuple) && length(rhs.args) == length(lhs.args)
                    # RHS is tuple literal — check each element pair
                    for (l, r) in zip(lhs.args, rhs.args)
                        if l isa Symbol && l in acquired && !_is_acquire_call(r, target_pool)
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
            # (v, w) or [v, w]
            for arg in expr.args
                if arg isa Symbol && arg in acquired
                    push!(found, arg)
                elseif Meta.isexpr(arg, :(=)) && length(arg.args) >= 2
                    # NamedTuple: (a=v,)
                    val = arg.args[2]
                    if val isa Symbol && val in acquired
                        push!(found, val)
                    end
                end
            end
        elseif expr.head == :parameters
            # (a=v,) style named tuple parameters
            for arg in expr.args
                if Meta.isexpr(arg, :kw) && length(arg.args) >= 2
                    val = arg.args[2]
                    if val isa Symbol && val in acquired
                        push!(found, val)
                    end
                end
            end
        end
    end
    return found
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
    escaped = Set{Symbol}()
    for ret_expr in return_values
        union!(escaped, _find_direct_exposure(ret_expr, acquired))
    end
    isempty(escaped) && return

    sorted = sort!(collect(escaped))
    file = source !== nothing ? string(source.file) : nothing
    line = source !== nothing ? source.line : nothing
    throw(PoolEscapeError(sorted, file, line))
end
