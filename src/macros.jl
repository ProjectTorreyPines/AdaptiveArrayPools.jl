# ==============================================================================
# Macros (v2: using checkpoint!/rewind! without state object)
# ==============================================================================

"""
    @with_pool pool expr
    @with_pool pool function_definition

Executes code with an explicit pool object and automatic state management.
This is for advanced use cases where you manage your own pool instance.

For most users, prefer `@use_pool` which uses the global (task-local) pool.

## Block Mode
```julia
pool = AdaptiveArrayPool()
result = @with_pool pool begin
    v = acquire!(pool, Float64, 100)
    sum(v)
end
# Pool state restored here
```

## Function Definition Mode
```julia
@with_pool pool function compute(x)
    temp = acquire!(pool, Float64, length(x))
    temp .= x .* 2
    sum(temp)
end
# `pool` keyword argument is auto-injected
compute([1.0, 2.0])              # pool=nothing (allocates)
compute([1.0, 2.0]; pool=mypool) # uses pool
```
"""
macro with_pool(pool, expr)
    # Compile-time check: if pooling disabled, just run expr with pool=nothing
    if !USE_POOLING
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            # Don't inject pool arg - just define function with pool=nothing inside
            def_head = expr.head
            call_expr = expr.args[1]
            body = expr.args[2]
            new_body = quote
                local $(esc(pool)) = nothing
                $(esc(body))
            end
            return Expr(def_head, esc(call_expr), new_body)
        end
        return quote
            local $(esc(pool)) = nothing
            $(esc(expr))
        end
    end

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
    @use_pool pool_name expr
    @use_pool pool_name function_definition

Binds `pool_name` to the global (task-local) pool and executes code with
automatic state management. This is the recommended macro for most use cases.

## Block Mode
```julia
result = @use_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 1.0
    sum(v)
end
```

## Function Definition Mode
```julia
@use_pool pool function fast_compute(n)
    v = acquire!(pool, Float64, n)
    v .= 1.0
    sum(v)
end
```
"""
macro use_pool(pool_name, expr)
    _generate_pool_code(pool_name, expr, true)
end

"""
    @maybe_use_pool pool_name expr
    @maybe_use_pool pool_name function_definition

Conditionally binds `pool_name` to the global pool based on `MAYBE_POOLING_ENABLED[]`.
If disabled, `pool_name` becomes `nothing`, and `acquire!` falls back to standard allocation.

Useful for libraries that want to let users control pooling behavior at runtime.

## Example
```julia
@maybe_use_pool pool function compute(n)
    v = acquire!(pool, Float64, n)  # Allocates if MAYBE_POOLING_ENABLED[] == false
    sum(v)
end

MAYBE_POOLING_ENABLED[] = false
compute(100)  # Normal allocation

MAYBE_POOLING_ENABLED[] = true
compute(100)  # Uses pool
```
"""
macro maybe_use_pool(pool_name, expr)
    _generate_pool_code(pool_name, expr, false)
end

function _generate_pool_code(pool_name, expr, force_enable)
    # Compile-time check: if pooling disabled, just run expr with pool=nothing
    if !USE_POOLING
        if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
            def_head = expr.head
            call_expr = expr.args[1]
            body = expr.args[2]
            new_body = quote
                local $(esc(pool_name)) = nothing
                $(esc(body))
            end
            return Expr(def_head, esc(call_expr), new_body)
        else
            return quote
                local $(esc(pool_name)) = nothing
                $(esc(expr))
            end
        end
    end

    # Extract types from acquire! calls for optimized checkpoint/rewind
    target_expr = Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr) ? expr.args[2] : expr
    all_types = _extract_acquire_types(target_expr)
    static_types, has_dynamic = _filter_static_types(all_types)

    # Use typed checkpoint/rewind if all types are static, otherwise fallback to full
    use_typed = !has_dynamic && !isempty(static_types)

    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        def_head = expr.head
        call_expr = expr.args[1]
        body = expr.args[2]

        if force_enable
            # Always use pool - no Union type
            checkpoint_call = use_typed ? _generate_typed_checkpoint_call(esc(pool_name), static_types) : :($checkpoint!($(esc(pool_name))))
            rewind_call = use_typed ? _generate_typed_rewind_call(esc(pool_name), static_types) : :($rewind!($(esc(pool_name))))

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
            # Split branches completely to avoid Union{Nothing, AdaptiveArrayPool} boxing
            checkpoint_call = use_typed ? _generate_typed_checkpoint_call(esc(pool_name), static_types) : :($checkpoint!($(esc(pool_name))))
            rewind_call = use_typed ? _generate_typed_rewind_call(esc(pool_name), static_types) : :($rewind!($(esc(pool_name))))

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
                    local $(esc(pool_name)) = nothing
                    $(esc(body))
                end
            end
        end

        return Expr(def_head, esc(call_expr), new_body)
    else
        # Block mode
        checkpoint_call = use_typed ? _generate_typed_checkpoint_call(esc(pool_name), static_types) : :($checkpoint!($(esc(pool_name))))
        rewind_call = use_typed ? _generate_typed_rewind_call(esc(pool_name), static_types) : :($rewind!($(esc(pool_name))))

        if force_enable
            return quote
                local $(esc(pool_name)) = get_global_pool()
                $checkpoint_call
                try
                    $(esc(expr))
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
                        $(esc(expr))
                    finally
                        $rewind_call
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

# --- Type Extraction for Optimized checkpoint!/rewind! ---

"""
    _extract_acquire_types(expr) -> (static_types, has_dynamic)

Extract type arguments from acquire!(pool, Type, ...) calls in an expression.
Returns a tuple of (static_types::Vector, has_dynamic::Bool).

Static types (like Float64, Int64) can be used for typed checkpoint.
Dynamic types (like T where T is a variable) require fallback to full checkpoint.
"""
function _extract_acquire_types(expr, types=Set{Any}())
    if expr isa Expr
        # Match: acquire!(pool, Type, ...)
        if expr.head == :call && length(expr.args) >= 3
            fn = expr.args[1]
            if fn == :acquire! || (fn isa Expr && fn.head == :. &&
                                   length(fn.args) >= 2 && fn.args[end] == QuoteNode(:acquire!))
                type_arg = expr.args[3]
                push!(types, type_arg)
            end
        end
        # Recurse into sub-expressions
        for arg in expr.args
            _extract_acquire_types(arg, types)
        end
    end
    return types
end

"""
    _filter_static_types(types) -> (static_types, has_dynamic)

Separate static types (known concrete types) from dynamic types (type parameters).
"""
function _filter_static_types(types)
    # Known concrete types that are safe to use for typed checkpoint
    # Includes aliases (Int = Int64, UInt = UInt64, etc.)
    known_types = Set{Symbol}([
        :Float64, :Float32, :Float16,
        :Int64, :Int32, :Int16, :Int8, :Int,      # Int is alias for Int64
        :UInt64, :UInt32, :UInt16, :UInt8, :UInt, # UInt is alias for UInt64
        :ComplexF64, :ComplexF32,
        :Bool, :Char, :String
    ])

    static_types = Any[]
    has_dynamic = false
    for t in types
        if t isa Symbol
            if t in known_types
                # Known concrete type like Float64 - use it
                push!(static_types, t)
            elseif length(string(t)) == 1
                # Single-letter symbol like T, S, N - likely type parameter
                has_dynamic = true
            else
                # Multi-letter symbol like MyOwnType - assume concrete type
                push!(static_types, t)
            end
        elseif t isa Expr && t.head == :curly
            # Parametric type like Vector{Float64} - treat as dynamic for safety
            has_dynamic = true
        else
            # GlobalRef or other concrete type
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
