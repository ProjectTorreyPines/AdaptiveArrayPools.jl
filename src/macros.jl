# ==============================================================================
# Macros for AdaptiveArrayPools
# ==============================================================================

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
@noinline function _get_pool_for_backend(::Val{B}) where B
    error("Pool backend :$B not available. Did you forget to load the extension (e.g., `using CUDA`)?")
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
    _generate_pool_code(pool_name, expr, true)
end

macro with_pool(expr)
    pool_name = gensym(:pool)
    _generate_pool_code(pool_name, expr, true)
end

# Backend-specific variants: @with_pool :cuda pool begin ... end
macro with_pool(backend::QuoteNode, pool_name, expr)
    _generate_pool_code_with_backend(backend.value, pool_name, expr, true)
end

macro with_pool(backend::QuoteNode, expr)
    pool_name = gensym(:pool)
    _generate_pool_code_with_backend(backend.value, pool_name, expr, true)
end

"""
    @maybe_with_pool pool_name expr
    @maybe_with_pool expr

Conditionally enables pooling based on `MAYBE_POOLING_ENABLED[]`.
If disabled, `pool_name` becomes `nothing`, and `acquire!` falls back to standard allocation.

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
MAYBE_POOLING_ENABLED[] = false
@maybe_with_pool pool begin
    v = acquire!(pool, Float64, 100)  # Falls back to Vector{Float64}(undef, 100)
end
```
"""
macro maybe_with_pool(pool_name, expr)
    _generate_pool_code(pool_name, expr, false)
end

macro maybe_with_pool(expr)
    pool_name = gensym(:pool)
    _generate_pool_code(pool_name, expr, false)
end

# ==============================================================================
# Internal: Code Generation
# ==============================================================================

function _generate_pool_code(pool_name, expr, force_enable)
    # Compile-time check: if pooling disabled, just run expr with pool=nothing
    if !USE_POOLING
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            # Function definition: inject local pool = nothing at start of body
            return _generate_function_pool_code(pool_name, expr, force_enable, true)
        else
            return quote
                local $(esc(pool_name)) = $(nothing)
                $(esc(expr))
            end
        end
    end

    # Check if function definition
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        return _generate_function_pool_code(pool_name, expr, force_enable, false)
    end

    # Block logic
    # Extract types from acquire! calls for optimized checkpoint/rewind
    # Only extract types for calls to the target pool (pool_name)
    all_types = _extract_acquire_types(expr, pool_name)
    local_vars = _extract_local_assignments(expr)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)

    # Use typed checkpoint/rewind if all types are static, otherwise fallback to full
    use_typed = !has_dynamic && !isempty(static_types)

    # Transform acquire! calls to _acquire_impl! (bypasses untracked marking)
    transformed_expr = _transform_acquire_calls(expr, pool_name)

    # For typed checkpoint, add _untracked_flags check for fallback to full checkpoint
    # This protects parent scope arrays when entering nested @with_pool
    if use_typed
        typed_checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
        checkpoint_call = quote
            if @inbounds $(esc(pool_name))._untracked_flags[$(esc(pool_name))._current_depth]
                $checkpoint!($(esc(pool_name)))  # Full checkpoint (parent had untracked)
            else
                $typed_checkpoint_call  # Fast typed checkpoint
            end
        end
    else
        checkpoint_call = :($checkpoint!($(esc(pool_name))))
    end

    # For typed checkpoint, add _untracked_flags check for fallback to full rewind
    if use_typed
        typed_rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
        rewind_call = quote
            if @inbounds $(esc(pool_name))._untracked_flags[$(esc(pool_name))._current_depth]
                $rewind!($(esc(pool_name)))  # Full rewind (untracked detected)
            else
                $typed_rewind_call  # Fast typed rewind
            end
        end
    else
        rewind_call = :($rewind!($(esc(pool_name))))
    end

    if force_enable
        return quote
            local $(esc(pool_name)) = get_task_local_pool()
            $checkpoint_call
            try
                local _result = $(esc(transformed_expr))
                if $POOL_DEBUG[]
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
            if $MAYBE_POOLING_ENABLED[]
                local $(esc(pool_name)) = get_task_local_pool()
                $checkpoint_call
                try
                    local _result = $(esc(transformed_expr))
                    if $POOL_DEBUG[]
                        $_validate_pool_return(_result, $(esc(pool_name)))
                    end
                    _result
                finally
                    $rewind_call
                end
            else
                local $(esc(pool_name)) = $(nothing)
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
function _generate_pool_code_with_backend(backend::Symbol, pool_name, expr, ::Bool)
    # Compile-time check: if pooling disabled, just run expr with pool=nothing
    if !USE_POOLING
        return quote
            local $(esc(pool_name)) = $(nothing)
            $(esc(expr))
        end
    end

    # Extract types from acquire! calls for optimized checkpoint/rewind
    all_types = _extract_acquire_types(expr, pool_name)
    local_vars = _extract_local_assignments(expr)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)

    # Use typed checkpoint/rewind if all types are static, otherwise fallback to full
    use_typed = !has_dynamic && !isempty(static_types)

    # Transform acquire! calls to _acquire_impl! (bypasses untracked marking)
    transformed_expr = _transform_acquire_calls(expr, pool_name)

    # Use Val{backend}() for compile-time dispatch - fully inlinable
    pool_getter = :($_get_pool_for_backend($(Val{backend}())))

    # Generate checkpoint call (typed or full)
    if use_typed
        typed_checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
        checkpoint_call = quote
            if @inbounds $(esc(pool_name))._untracked_flags[$(esc(pool_name))._current_depth]
                $checkpoint!($(esc(pool_name)))  # Full checkpoint (parent had untracked)
            else
                $typed_checkpoint_call  # Fast typed checkpoint
            end
        end
    else
        checkpoint_call = :($checkpoint!($(esc(pool_name))))
    end

    # Generate rewind call (typed or full)
    if use_typed
        typed_rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
        rewind_call = quote
            if @inbounds $(esc(pool_name))._untracked_flags[$(esc(pool_name))._current_depth]
                $rewind!($(esc(pool_name)))  # Full rewind (untracked detected)
            else
                $typed_rewind_call  # Fast typed rewind
            end
        end
    else
        rewind_call = :($rewind!($(esc(pool_name))))
    end

    return quote
        local $(esc(pool_name)) = $pool_getter
        $checkpoint_call
        try
            local _result = $(esc(transformed_expr))
            if $POOL_DEBUG[]
                $_validate_pool_return(_result, $(esc(pool_name)))
            end
            _result
        finally
            $rewind_call
        end
    end
end

function _generate_function_pool_code(pool_name, func_def, force_enable, disable_pooling)
    def_head = func_def.head
    call_expr = func_def.args[1]
    body = func_def.args[2]

    if disable_pooling
        new_body = quote
            local $(esc(pool_name)) = $(nothing)
            $(esc(body))
        end
        return Expr(def_head, esc(call_expr), new_body)
    end

    # Analyze body for types
    all_types = _extract_acquire_types(body, pool_name)
    local_vars = _extract_local_assignments(body)
    static_types, has_dynamic = _filter_static_types(all_types, local_vars)
    use_typed = !has_dynamic && !isempty(static_types)

    # Transform acquire! calls to _acquire_impl! (bypasses untracked marking)
    transformed_body = _transform_acquire_calls(body, pool_name)

    # For typed checkpoint, add _untracked_flags check for fallback to full checkpoint
    # This protects parent scope arrays when entering nested @with_pool
    if use_typed
        typed_checkpoint_call = _generate_typed_checkpoint_call(esc(pool_name), static_types)
        checkpoint_call = quote
            if @inbounds $(esc(pool_name))._untracked_flags[$(esc(pool_name))._current_depth]
                $checkpoint!($(esc(pool_name)))  # Full checkpoint (parent had untracked)
            else
                $typed_checkpoint_call  # Fast typed checkpoint
            end
        end
    else
        checkpoint_call = :($checkpoint!($(esc(pool_name))))
    end

    # For typed checkpoint, add _untracked_flags check for fallback to full rewind
    if use_typed
        typed_rewind_call = _generate_typed_rewind_call(esc(pool_name), static_types)
        rewind_call = quote
            if @inbounds $(esc(pool_name))._untracked_flags[$(esc(pool_name))._current_depth]
                $rewind!($(esc(pool_name)))  # Full rewind (untracked detected)
            else
                $typed_rewind_call  # Fast typed rewind
            end
        end
    else
        rewind_call = :($rewind!($(esc(pool_name))))
    end

    if force_enable
        new_body = quote
            local $(esc(pool_name)) = get_task_local_pool()
            $checkpoint_call
            try
                $(esc(transformed_body))
            finally
                $rewind_call
            end
        end
    else
        new_body = quote
            if $MAYBE_POOLING_ENABLED[]
                local $(esc(pool_name)) = get_task_local_pool()
                $checkpoint_call
                try
                    $(esc(transformed_body))
                finally
                    $rewind_call
                end
            else
                local $(esc(pool_name)) = $(nothing)
                $(esc(body))
            end
        end
    end

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
function _extract_local_assignments(expr, locals=Set{Symbol}())
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

Extract type arguments from acquire function calls in an expression.
Only extracts types from calls where the first argument matches `target_pool`.
This prevents AST pollution when multiple pools are used in the same block.

Supported functions:
- `acquire!` and its alias `acquire_view!`
- `unsafe_acquire!` and its alias `acquire_array!`

Handles two forms:
- `[unsafe_]acquire!(pool, Type, dims...)` (3+ func args): extracts Type (2nd arg) directly
- `acquire!(pool, x)` (2 func args): generates `eltype(x)` expression for the array
  (Note: `unsafe_acquire!` / `acquire_array!` does not have the 2-arg form)
"""
function _extract_acquire_types(expr, target_pool, types=Set{Any}())
    if expr isa Expr
        # Match: acquire!/acquire_view!/unsafe_acquire!/acquire_array!(pool, ...)
        if expr.head == :call && length(expr.args) >= 3
            fn = expr.args[1]
            # All acquire function names (including aliases)
            acquire_names = (:acquire!, :unsafe_acquire!, :acquire_view!, :acquire_array!)
            acquire_quotenodes = (QuoteNode(:acquire!), QuoteNode(:unsafe_acquire!),
                                  QuoteNode(:acquire_view!), QuoteNode(:acquire_array!))
            is_acquire = fn in acquire_names ||
                         (fn isa Expr && fn.head == :. && length(fn.args) >= 2 &&
                          fn.args[end] in acquire_quotenodes)
            if is_acquire
                # Check if the pool argument matches our target pool
                pool_arg = expr.args[2]
                if pool_arg == target_pool
                    nargs = length(expr.args)
                    if nargs >= 4
                        # acquire!(pool, Type, dims...) - traditional form
                        type_arg = expr.args[3]
                        push!(types, type_arg)
                    elseif nargs == 3
                        # acquire!(pool, x) - similar-style form
                        # Type is eltype of the array argument
                        array_arg = expr.args[3]
                        type_expr = Expr(:call, :eltype, array_arg)
                        push!(types, type_expr)
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
function _filter_static_types(types, local_vars=Set{Symbol}())
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

Generate checkpoint!(pool, T1, T2, ...) call expression.
"""
function _generate_typed_checkpoint_call(pool_expr, types)
    if isempty(types)
        return :($checkpoint!($pool_expr))
    else
        # esc types so they resolve in caller's namespace (Float64, not AdaptiveArrayPools.Float64)
        escaped_types = [esc(t) for t in types]
        return :($checkpoint!($pool_expr, $(escaped_types...)))
    end
end

"""
    _generate_typed_rewind_call(pool_expr, types)

Generate rewind!(pool, T1, T2, ...) call expression.
Types are passed in original order; rewind! handles reversal internally.
"""
function _generate_typed_rewind_call(pool_expr, types)
    if isempty(types)
        return :($rewind!($pool_expr))
    else
        escaped_types = [esc(t) for t in types]
        return :($rewind!($pool_expr, $(escaped_types...)))
    end
end

# ==============================================================================
# Internal: Acquire Call Transformation
# ==============================================================================

"""
    _transform_acquire_calls(expr, pool_name) -> Expr

Transform acquire!/unsafe_acquire! calls to their _impl! counterparts.
Only transforms calls where the first argument matches `pool_name`.

This allows macro-transformed code to bypass the untracked marking overhead,
since the macro already knows about these calls at compile time.

Transformation rules:
- `acquire!(pool, ...)` → `_acquire_impl!(pool, ...)`
- `acquire_view!(pool, ...)` → `_acquire_impl!(pool, ...)`
- `unsafe_acquire!(pool, ...)` → `_unsafe_acquire_impl!(pool, ...)`
- `acquire_array!(pool, ...)` → `_unsafe_acquire_impl!(pool, ...)`
"""
# Module-qualified references for transformed acquire calls
# Using GlobalRef ensures the function is looked up in AdaptiveArrayPools, not the caller's module
const _ACQUIRE_IMPL_REF = GlobalRef(@__MODULE__, :_acquire_impl!)
const _UNSAFE_ACQUIRE_IMPL_REF = GlobalRef(@__MODULE__, :_unsafe_acquire_impl!)

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
                elseif fn isa Expr && fn.head == :. && length(fn.args) >= 2
                    # Qualified name: AdaptiveArrayPools.acquire! etc.
                    qn = fn.args[end]
                    if qn == QuoteNode(:acquire!) || qn == QuoteNode(:acquire_view!)
                        expr = Expr(:call, _ACQUIRE_IMPL_REF, expr.args[2:end]...)
                    elseif qn == QuoteNode(:unsafe_acquire!) || qn == QuoteNode(:acquire_array!)
                        expr = Expr(:call, _UNSAFE_ACQUIRE_IMPL_REF, expr.args[2:end]...)
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
