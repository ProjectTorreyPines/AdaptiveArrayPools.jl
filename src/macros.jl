# ==============================================================================
# Macros for AdaptiveArrayPools (v3: Simplified Lifecycle Management)
# ==============================================================================

"""
    @with_pool pool_name expr
    @with_pool expr

Executes code within a pooling scope with automatic lifecycle management.
Calls `checkpoint!` on entry and `rewind!` on exit (even if errors occur).

If `pool_name` is omitted, a hidden variable is used (useful when you don't
need to reference the pool directly).

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
    inner_function()  # inner function can use get_global_pool()
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

    checkpoint_call = use_typed ? _generate_typed_checkpoint_call(esc(pool_name), static_types) : :($checkpoint!($(esc(pool_name))))
    rewind_call = use_typed ? _generate_typed_rewind_call(esc(pool_name), static_types) : :($rewind!($(esc(pool_name))))

    if force_enable
        return quote
            local $(esc(pool_name)) = get_global_pool()
            $checkpoint_call
            try
                local _result = $(esc(expr))
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
                local $(esc(pool_name)) = get_global_pool()
                $checkpoint_call
                try
                    local _result = $(esc(expr))
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

    checkpoint_call = use_typed ? _generate_typed_checkpoint_call(esc(pool_name), static_types) : :($checkpoint!($(esc(pool_name))))
    rewind_call = use_typed ? _generate_typed_rewind_call(esc(pool_name), static_types) : :($rewind!($(esc(pool_name))))

    if force_enable
        new_body = quote
            local $(esc(pool_name)) = get_global_pool()
            $checkpoint_call
            try
                $(esc(body))
            finally
                $rewind_call
            end
        end
    else
        new_body = quote
            if $MAYBE_POOLING_ENABLED[]
                local $(esc(pool_name)) = get_global_pool()
                $checkpoint_call
                try
                    $(esc(body))
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

Extract type arguments from acquire!(target_pool, Type, ...) calls in an expression.
Only extracts types from calls where the first argument matches `target_pool`.
This prevents AST pollution when multiple pools are used in the same block.
"""
function _extract_acquire_types(expr, target_pool, types=Set{Any}())
    if expr isa Expr
        # Match: acquire!(pool, Type, ...)
        if expr.head == :call && length(expr.args) >= 3
            fn = expr.args[1]
            if fn == :acquire! || (fn isa Expr && fn.head == :. &&
                                   length(fn.args) >= 2 && fn.args[end] == QuoteNode(:acquire!))
                # Check if the pool argument matches our target pool
                pool_arg = expr.args[2]
                if pool_arg == target_pool
                    type_arg = expr.args[3]
                    push!(types, type_arg)
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
    _filter_static_types(types, local_vars=Set{Symbol}()) -> (static_types, has_dynamic)

Filter types for typed checkpoint/rewind generation.

- Symbols NOT in local_vars are passed through (type parameters, global types)
- Symbols IN local_vars trigger fallback (defined after checkpoint!)
- Parametric types like Vector{T} trigger fallback

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
        elseif t isa Expr && t.head == :curly
            # Parametric type like Vector{Float64} - can't use as Type argument
            has_dynamic = true
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
