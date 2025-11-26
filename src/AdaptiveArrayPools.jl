module AdaptiveArrayPools

export AdaptiveArrayPool, acquire!, pool_stats
export @use_pool, @use_global_pool, @maybe_use_global_pool
export ENABLE_POOLING, POOL_DEBUG

# Note: mark!/reset! are NOT exported to avoid conflict with Base
# Users should use: import AdaptiveArrayPools: mark!, reset!

# ==============================================================================
# Core Data Structures (v2: with saved_stack for zero-allocation mark/reset)
# ==============================================================================

"""
    TypedPool{T}

Internal structure managing a list of vectors for a specific type `T`.
Includes `saved_stack` for nested mark/reset support with zero allocation.
"""
mutable struct TypedPool{T}
    vectors::Vector{Vector{T}}   # Actual memory storage
    in_use::Int                  # Number of currently checked-out vectors
    saved_stack::Vector{Int}     # Stack for nested mark/reset (zero alloc after warmup)
end

TypedPool{T}() where {T} = TypedPool{T}(Vector{T}[], 0, Int[])

"""
    checkout!(tp::TypedPool{T}, n::Int) -> SubArray

Internal function to get a vector view of size `n` from the typed pool.
"""
function checkout!(tp::TypedPool{T}, n::Int) where {T}
    tp.in_use += 1
    if tp.in_use > length(tp.vectors)
        push!(tp.vectors, Vector{T}(undef, n))
        return view(tp.vectors[end], 1:n)
    end
    v = tp.vectors[tp.in_use]
    if length(v) < n
        resize!(v, n)
    end
    return view(v, 1:n)
end

# ==============================================================================
# AdaptiveArrayPool (v2: Fixed Slots + Fallback)
# ==============================================================================

"""
    AdaptiveArrayPool

A high-performance memory pool supporting multiple data types.

## v2 Features
- **Fixed Slots**: Float64, Float32, Int64, Int32, ComplexF64, Bool have dedicated fields (zero Dict lookup)
- **Fallback**: Other types use IdDict (still fast, but with lookup overhead)
- **Zero Allocation**: mark!/reset! use internal stacks, no allocation after warmup

## Thread Safety
This pool is **NOT thread-safe**. Use one pool per Task via `get_global_pool()`.
"""
struct AdaptiveArrayPool
    # Fixed Slots: common types with zero lookup overhead
    float64::TypedPool{Float64}
    float32::TypedPool{Float32}
    int64::TypedPool{Int64}
    int32::TypedPool{Int32}
    complexf64::TypedPool{ComplexF64}
    bool::TypedPool{Bool}

    # Fallback: rare types
    others::IdDict{DataType, Any}
end

function AdaptiveArrayPool()
    AdaptiveArrayPool(
        TypedPool{Float64}(),
        TypedPool{Float32}(),
        TypedPool{Int64}(),
        TypedPool{Int32}(),
        TypedPool{ComplexF64}(),
        TypedPool{Bool}(),
        IdDict{DataType, Any}()
    )
end

# ==============================================================================
# Type Dispatch (Zero-cost for Fixed Slots)
# ==============================================================================

# Fast Path: compile-time dispatch, fully inlined
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Float64}) = p.float64
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Float32}) = p.float32
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Int64}) = p.int64
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Int32}) = p.int32
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{ComplexF64}) = p.complexf64
@inline get_typed_pool!(p::AdaptiveArrayPool, ::Type{Bool}) = p.bool

# Slow Path: rare types via IdDict
@inline function get_typed_pool!(p::AdaptiveArrayPool, ::Type{T}) where {T}
    get!(p.others, T) do
        TypedPool{T}()
    end::TypedPool{T}
end

# ==============================================================================
# Acquisition API
# ==============================================================================

"""
    acquire!(pool, Type{T}, n) -> SubArray
    acquire!(pool, Type{T}, dims...) -> ReshapedArray

Acquire a view of an array of type `T` with size `n` or dimensions `dims`.

Returns a `SubArray` (1D) or `ReshapedArray` (multi-dimensional) backed by the pool.
After the enclosing `@use_pool` block ends, the memory is reclaimed for reuse.

## Example
```julia
@use_global_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 1.0
    sum(v)
end
```
"""
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, n::Int) where {T}
    tp = get_typed_pool!(pool, T)
    return checkout!(tp, n)
end

# Multi-dimensional support (Flat Buffer + Reshape)
@inline function acquire!(pool::AdaptiveArrayPool, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    total_len = prod(dims)
    flat_view = acquire!(pool, T, total_len)
    return reshape(flat_view, dims)
end

# Fallback: When pool is `nothing` (e.g. pooling disabled), allocate normally
@inline function acquire!(::Nothing, ::Type{T}, n::Int) where {T}
    Vector{T}(undef, n)
end

@inline function acquire!(::Nothing, ::Type{T}, dims::Vararg{Int, N}) where {T, N}
    Array{T, N}(undef, dims)
end

# ==============================================================================
# State Management (v2: Zero-Allocation mark!/reset!)
# ==============================================================================

"""
    mark!(pool::AdaptiveArrayPool)

Save the current pool state to internal stacks. No return value.

This is called automatically by `@use_pool` and related macros.
After warmup, this function has **zero allocation**.

See also: [`reset!`](@ref), [`@use_pool`](@ref)
"""
function mark!(pool::AdaptiveArrayPool)
    # Fixed slots - direct field access, no Dict lookup
    push!(pool.float64.saved_stack, pool.float64.in_use)
    push!(pool.float32.saved_stack, pool.float32.in_use)
    push!(pool.int64.saved_stack, pool.int64.in_use)
    push!(pool.int32.saved_stack, pool.int32.in_use)
    push!(pool.complexf64.saved_stack, pool.complexf64.in_use)
    push!(pool.bool.saved_stack, pool.bool.in_use)

    # Others - iterate without allocation (values() returns iterator)
    for p in values(pool.others)
        push!(p.saved_stack, p.in_use)
    end

    return nothing
end

mark!(::Nothing) = nothing

"""
    reset!(pool::AdaptiveArrayPool)

Restore the pool state from internal stacks. No arguments needed.

Handles edge case: types added after mark! get their in_use set to 0.

See also: [`mark!`](@ref), [`@use_pool`](@ref)
"""
function reset!(pool::AdaptiveArrayPool)
    # Fixed slots
    pool.float64.in_use = pop!(pool.float64.saved_stack)
    pool.float32.in_use = pop!(pool.float32.saved_stack)
    pool.int64.in_use = pop!(pool.int64.saved_stack)
    pool.int32.in_use = pop!(pool.int32.saved_stack)
    pool.complexf64.in_use = pop!(pool.complexf64.saved_stack)
    pool.bool.in_use = pop!(pool.bool.saved_stack)

    # Others - handle edge case: new types added after mark!
    for p in values(pool.others)
        if isempty(p.saved_stack)
            # Type was added after mark! - reset to 0
            p.in_use = 0
        else
            p.in_use = pop!(p.saved_stack)
        end
    end

    return nothing
end

reset!(::Nothing) = nothing

# ==============================================================================
# Global Pool (Task Local Storage) & Configuration
# ==============================================================================

"""
    ENABLE_POOLING

Global flag to enable/disable pooling. When `false`, `@maybe_use_global_pool`
will use `nothing` as the pool, causing `acquire!` to allocate normally.

Default: `true`
"""
const ENABLE_POOLING = Ref(true)

"""
    get_global_pool() -> AdaptiveArrayPool

Retrieves (or creates) the `AdaptiveArrayPool` for the current Task.

Each Task gets its own pool instance via `task_local_storage()`,
ensuring thread safety without locks.
"""
function get_global_pool()
    get!(task_local_storage(), :ADAPTIVE_ARRAY_POOL) do
        AdaptiveArrayPool()
    end::AdaptiveArrayPool
end

# ==============================================================================
# Debugging & Safety
# ==============================================================================

"""
    POOL_DEBUG

When `true`, `@use_pool` macros validate that returned values don't
reference pool memory (which would be unsafe).

Default: `false`
"""
const POOL_DEBUG = Ref(false)

function _validate_pool_return(val, pool::AdaptiveArrayPool)
    if !(val isa SubArray)
        return
    end
    p = parent(val)

    # Check fixed slots
    for tp in (pool.float64, pool.float32, pool.int64, pool.int32, pool.complexf64, pool.bool)
        for v in tp.vectors
            if v === p
                error("Safety Violation: The function returned a SubArray backed by the AdaptiveArrayPool. This is unsafe as the memory will be reclaimed. Please return a copy (collect) or a scalar.")
            end
        end
    end

    # Check others
    for tp in values(pool.others)
        for v in tp.vectors
            if v === p
                error("Safety Violation: The function returned a SubArray backed by the AdaptiveArrayPool. This is unsafe as the memory will be reclaimed. Please return a copy (collect) or a scalar.")
            end
        end
    end
end

_validate_pool_return(val, ::Nothing) = nothing

"""
    pool_stats(pool::AdaptiveArrayPool)

Print statistics about pool usage (for debugging/profiling).
"""
function pool_stats(pool::AdaptiveArrayPool)
    println("AdaptiveArrayPool Statistics:")

    # Fixed slots
    for (name, tp) in [
        ("Float64", pool.float64),
        ("Float32", pool.float32),
        ("Int64", pool.int64),
        ("Int32", pool.int32),
        ("ComplexF64", pool.complexf64),
        ("Bool", pool.bool)
    ]
        if !isempty(tp.vectors)
            total_elements = sum(length(v) for v in tp.vectors; init=0)
            println("  $name (fixed slot):")
            println("    Vectors: $(length(tp.vectors))")
            println("    In Use:  $(tp.in_use)")
            println("    Total elements: $total_elements")
        end
    end

    # Others
    for (T, tp) in pool.others
        total_elements = sum(length(v) for v in tp.vectors; init=0)
        println("  $T (fallback):")
        println("    Vectors: $(length(tp.vectors))")
        println("    In Use:  $(tp.in_use)")
        println("    Total elements: $total_elements")
    end
end

# ==============================================================================
# Macros (v2: using mark!/reset! without state object)
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

    # Block mode: wrap in mark!/reset!
    quote
        local _pool = $(esc(pool))
        $mark!(_pool)
        try
            local _result = $(esc(expr))
            if $POOL_DEBUG[] && _pool !== nothing
                $_validate_pool_return(_result, _pool)
            end
            _result
        finally
            $reset!(_pool)
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
    pool_instance = if force_enable
        :(get_global_pool())
    else
        :(ENABLE_POOLING[] ? get_global_pool() : nothing)
    end

    if Meta.isexpr(expr, [:function, :(=)]) && _is_function_def(expr)
        def_head = expr.head
        call_expr = expr.args[1]
        body = expr.args[2]

        new_body = quote
            local $(esc(pool_name)) = $pool_instance
            $mark!($(esc(pool_name)))
            try
                $(esc(body))
            finally
                $reset!($(esc(pool_name)))
            end
        end

        return Expr(def_head, esc(call_expr), new_body)
    else
        return quote
            local $(esc(pool_name)) = $pool_instance
            $mark!($(esc(pool_name)))
            try
                $(esc(expr))
            finally
                $reset!($(esc(pool_name)))
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
        $mark!($(esc(pool_sym)))
        try
            local _result = $(esc(body))
            if $POOL_DEBUG[] && $(esc(pool_sym)) !== nothing
                $_validate_pool_return(_result, $(esc(pool_sym)))
            end
            _result
        finally
            $reset!($(esc(pool_sym)))
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

end # module
