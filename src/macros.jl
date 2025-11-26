# ==============================================================================
# Macros (v2: using checkpoint!/rewind! without state object)
# ==============================================================================

"""
    @use_pool pool expr
    @use_pool pool function_definition

Executes code with automatic pool state management.

## Block Mode
```julia
pool = AdaptiveArrayPool()
result = @use_pool pool begin
    v = acquire!(pool, Float64, 100)
    sum(v)
end
# Pool state restored here
```

## Function Definition Mode
```julia
@use_pool pool function compute(x)
    temp = acquire!(pool, Float64, length(x))
    temp .= x .* 2
    sum(temp)
end
# `pool` keyword argument is auto-injected
compute([1.0, 2.0])              # pool=nothing (allocates)
compute([1.0, 2.0]; pool=mypool) # uses pool
```
"""
macro use_pool(pool, expr)
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        # Function definition mode: inject pool arg and wrap body
        expr = _inject_pool_arg(pool, expr)
        return _wrap_function_body_with_pool(pool, expr)
    end

    # Block mode: wrap in checkpoint!/rewind!
    quote
        local _pool = $(esc(pool))
        $checkpoint!(_pool)
        try
            local _result = $(esc(expr))
            if $POOL_DEBUG[] && _pool !== nothing
                $_validate_pool_return(_result, _pool)
            end
            _result
        finally
            $rewind!(_pool)
        end
    end
end

"""
    @use_global_pool pool_name expr
    @use_global_pool pool_name function_definition

Binds `pool_name` to the global (task-local) pool.
Always uses the pool regardless of `ENABLE_POOLING`.

## Example
```julia
@use_global_pool pool function fast_compute(n)
    v = acquire!(pool, Float64, n)
    v .= 1.0
    sum(v)
end
```
"""
macro use_global_pool(pool_name, expr)
    _generate_global_pool_code(pool_name, expr, true)
end

"""
    @maybe_use_global_pool pool_name expr
    @maybe_use_global_pool pool_name function_definition

Conditionally binds `pool_name` to the global pool based on `ENABLE_POOLING[]`.
If disabled, `pool_name` becomes `nothing`, and `acquire!` falls back to standard allocation.

Useful for libraries that want to let users control pooling behavior.

## Example
```julia
@maybe_use_global_pool pool function compute(n)
    v = acquire!(pool, Float64, n)  # Allocates if ENABLE_POOLING[] == false
    sum(v)
end

ENABLE_POOLING[] = false
compute(100)  # Normal allocation

ENABLE_POOLING[] = true
compute(100)  # Uses pool
```
"""
macro maybe_use_global_pool(pool_name, expr)
    _generate_global_pool_code(pool_name, expr, false)
end

function _generate_global_pool_code(pool_name, expr, force_enable)
    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        def_head = expr.head
        call_expr = expr.args[1]
        body = expr.args[2]

        if force_enable
            # Always use pool - no Union type
            new_body = quote
                local $(esc(pool_name)) = get_global_pool()
                $checkpoint!($(esc(pool_name)))
                try
                    $(esc(body))
                finally
                    $rewind!($(esc(pool_name)))
                end
            end
        else
            # Split branches completely to avoid Union{Nothing, AdaptiveArrayPool} boxing
            new_body = quote
                if $ENABLE_POOLING[]
                    local $(esc(pool_name)) = get_global_pool()
                    $checkpoint!($(esc(pool_name)))
                    try
                        $(esc(body))
                    finally
                        $rewind!($(esc(pool_name)))
                    end
                else
                    local $(esc(pool_name)) = nothing
                    $(esc(body))
                end
            end
        end

        return Expr(def_head, esc(call_expr), new_body)
    else
        # Block mode
        if force_enable
            return quote
                local $(esc(pool_name)) = get_global_pool()
                $checkpoint!($(esc(pool_name)))
                try
                    $(esc(expr))
                finally
                    $rewind!($(esc(pool_name)))
                end
            end
        else
            # Split branches completely to avoid Union boxing
            return quote
                if $ENABLE_POOLING[]
                    local $(esc(pool_name)) = get_global_pool()
                    $checkpoint!($(esc(pool_name)))
                    try
                        $(esc(expr))
                    finally
                        $rewind!($(esc(pool_name)))
                    end
                else
                    local $(esc(pool_name)) = nothing
                    $(esc(expr))
                end
            end
        end
    end
end

# --- Helper Functions for Macros ---

function _wrap_function_body_with_pool(pool_sym, func_def)
    def_head = func_def.head
    call_expr = func_def.args[1]
    body = func_def.args[2]

    new_body = quote
        $checkpoint!($(esc(pool_sym)))
        try
            local _result = $(esc(body))
            if $POOL_DEBUG[] && $(esc(pool_sym)) !== nothing
                $_validate_pool_return(_result, $(esc(pool_sym)))
            end
            _result
        finally
            $rewind!($(esc(pool_sym)))
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

function _inject_pool_arg(pool_sym, func_def)
    if !isa(pool_sym, Symbol)
        return func_def
    end

    call_expr = func_def.args[1]
    target_expr = call_expr
    while Meta.isexpr(target_expr, [:where, :(::)])
        target_expr = target_expr.args[1]
    end

    if !Meta.isexpr(target_expr, :call)
        return func_def
    end

    args = target_expr.args[2:end]
    arg_names = Symbol[]
    for arg in args
        if Meta.isexpr(arg, :parameters)
            for kw in arg.args
                push!(arg_names, _get_arg_name(kw))
            end
        else
            push!(arg_names, _get_arg_name(arg))
        end
    end

    if pool_sym in arg_names
        return func_def
    end

    # Inject keyword argument with Union type for flexibility
    new_kwarg = Expr(:kw, Expr(:(::), pool_sym, :(Union{Nothing, AdaptiveArrayPool})), nothing)

    if length(target_expr.args) >= 2 && Meta.isexpr(target_expr.args[2], :parameters)
        push!(target_expr.args[2].args, new_kwarg)
    else
        params = Expr(:parameters, new_kwarg)
        insert!(target_expr.args, 2, params)
    end

    return func_def
end

function _get_arg_name(arg)
    if Meta.isexpr(arg, :kw)
        arg = arg.args[1]
    end
    if Meta.isexpr(arg, :(::))
        return length(arg.args) == 2 ? arg.args[1] : :_
    end
    return arg
end
