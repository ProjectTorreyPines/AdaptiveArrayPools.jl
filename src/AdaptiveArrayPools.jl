module AdaptiveArrayPools

export AdaptiveArrayPool, acquire!, mark, reset, pool_stats
export @use_pool, @use_global_pool, @maybe_use_global_pool
export ENABLE_POOLING, POOL_DEBUG

# ==============================================================================
# Core Data Structures
# ==============================================================================

"""
    TypedPool{T}

Internal structure managing a list of vectors for a specific type `T`.
"""
mutable struct TypedPool{T}
    vectors::Vector{Vector{T}}
    in_use::Int
end

TypedPool{T}() where {T} = TypedPool{T}(Vector{T}[], 0)

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

"""
    AdaptiveArrayPool

A container that manages memory pools for multiple data types.
It uses an `IdDict` to map `DataType` to specific `TypedPool{T}` instances.
"""
struct AdaptiveArrayPool
    pools::IdDict{DataType, Any} 
end

AdaptiveArrayPool() = AdaptiveArrayPool(IdDict{DataType, Any}())

# Helper to get or create a typed pool
@inline function get_typed_pool!(pool::AdaptiveArrayPool, ::Type{T}) where {T}
    if !haskey(pool.pools, T)
        pool.pools[T] = TypedPool{T}()
    end
    # Type assertion for type stability in downstream code
    return pool.pools[T]::TypedPool{T}
end

# ==============================================================================
# Acquisition API
# ==============================================================================

"""
    acquire!(pool, Type{T}, n)
    acquire!(pool, Type{T}, dims...)

Acquire a view of an array of type `T` with size `n` or dimensions `dims`.
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
# State Management (Mark & Reset)
# ==============================================================================

"""
    mark(pool)

Snapshots the current state of the pool (usage counts).
"""
function mark(pool::AdaptiveArrayPool)
    # Snapshot current usage counts for all active types
    # We use IdDict to minimize overhead, though a custom struct or Tuple might be faster in extreme cases.
    return IdDict{DataType, Int}(T => p.in_use for (T, p) in pool.pools)
end

mark(::Nothing) = nothing

"""
    reset(pool, state)

Restores the pool to a previously marked state.
"""
function reset(pool::AdaptiveArrayPool, state::IdDict{DataType, Int})
    # 1. Restore state for pools that existed at mark time
    for (T, count) in state
        if haskey(pool.pools, T)
            pool.pools[T].in_use = count
        end
    end
    
    # 2. Reset any pools created AFTER mark (they shouldn't exist in the caller's scope)
    for (T, p) in pool.pools
        if !haskey(state, T)
            p.in_use = 0
        end
    end
end

reset(::Nothing, ::Nothing) = nothing

# ==============================================================================
# Global Pool (Task Local Storage) & Configuration
# ==============================================================================

const ENABLE_POOLING = Ref(true)

"""
    get_global_pool()

Retrieves (or creates) the `AdaptiveArrayPool` for the current Task.
"""
function get_global_pool()
    get!(task_local_storage(), :ADAPTIVE_ARRAY_POOL) do
        AdaptiveArrayPool()
    end
end

# ==============================================================================
# Debugging & Safety
# ==============================================================================

const POOL_DEBUG = Ref(false)

function _validate_pool_return(val, pool::AdaptiveArrayPool)
    if !(val isa SubArray)
        return
    end
    p = parent(val)
    
    for subpool in values(pool.pools)
        for v in subpool.vectors
            if v === p
                error("Safety Violation: The function returned a SubArray backed by the AdaptiveArrayPool. This is unsafe as the memory will be reclaimed. Please return a copy (collect) or a scalar.")
            end
        end
    end
end

function pool_stats(pool::AdaptiveArrayPool)
    println("AdaptiveArrayPool Statistics:")
    for (T, tp) in pool.pools
        total_elements = sum(length(v) for v in tp.vectors; init=0)
        total_bytes = total_elements * sizeof(T)
        println("  Type $T:")
        println("    Vectors: $(length(tp.vectors))")
        println("    In Use:  $(tp.in_use)")
        println("    Memory:  $(total_bytes) bytes")
    end
end

# ==============================================================================
# Macros
# ==============================================================================

"""
    @use_pool pool_name function_definition
    @use_pool pool_name expression

Executes code ensuring that the `pool` state is restored to its value at the beginning.
If `pool_name` is not in the arguments of the function definition, it injects it as a keyword argument.
"""
macro use_pool(pool, expr)
    if Meta.isexpr(expr, [:function, :(=)])
        expr = _inject_pool_arg(pool, expr)
    end

    quote
        local _p = $(esc(pool))
        local _state = mark(_p)
        try
            local _result = $(esc(expr))
            
            if POOL_DEBUG[] && _p !== nothing
                _validate_pool_return(_result, _p)
            end
            
            _result
        finally
            reset(_p, _state)
        end
    end
end

"""
    @use_global_pool pool_name function_definition

Binds `pool_name` to the global (task-local) pool.
Always uses the pool regardless of `ENABLE_POOLING`.
"""
macro use_global_pool(pool_name, expr)
    _generate_global_pool_code(pool_name, expr, true)
end

"""
    @maybe_use_global_pool pool_name function_definition

Conditionally binds `pool_name` to the global pool based on `ENABLE_POOLING[]`.
If disabled, `pool_name` becomes `nothing`, and `acquire!` falls back to standard allocation.
"""
macro maybe_use_global_pool(pool_name, expr)
    _generate_global_pool_code(pool_name, expr, false)
end

function _generate_global_pool_code(pool_name, expr, force_enable)
    # Logic to determine pool instance
    pool_instance = if force_enable
        :(get_global_pool())
    else
        :(ENABLE_POOLING[] ? get_global_pool() : nothing)
    end

    if Meta.isexpr(expr, [:function, :(=)])
        def_head = expr.head
        call_expr = expr.args[1]
        body = expr.args[2]
        
        new_body = quote
            local $(esc(pool_name)) = $pool_instance
            local _state = mark($(esc(pool_name)))
            try
                $(esc(body))
            finally
                reset($(esc(pool_name)), _state)
            end
        end
        
        return Expr(def_head, esc(call_expr), new_body)
    else
        return quote
            local $(esc(pool_name)) = $pool_instance
            local _state = mark($(esc(pool_name)))
            try
                $(esc(expr))
            finally
                reset($(esc(pool_name)), _state)
            end
        end
    end
end

# --- Helper Functions for Macros ---

function _inject_pool_arg(pool_sym, func_def)
    if !isa(pool_sym, Symbol) return func_def end
    call_expr = func_def.args[1]
    target_expr = call_expr
    while Meta.isexpr(target_expr, [:where, :(::)])
        target_expr = target_expr.args[1]
    end
    if !Meta.isexpr(target_expr, :call) return func_def end

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

    if pool_sym in arg_names return func_def end

    # Inject keyword argument
    # We use Union{Nothing, AdaptiveArrayPool} to be safe
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
    if Meta.isexpr(arg, :kw) arg = arg.args[1] end
    if Meta.isexpr(arg, :(::))
        return length(arg.args) == 2 ? arg.args[1] : :_
    end
    return arg
end

end # module
