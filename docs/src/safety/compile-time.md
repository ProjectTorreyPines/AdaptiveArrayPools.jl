# Compile-Time Detection

The `@with_pool` macro statically analyzes your code at macro expansion time — zero runtime cost, always active.

## Escape Analysis (`PoolEscapeError`)

Rejects any expression that would return a pool-backed array. Tracks aliases, containers, and convenience wrappers:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    w = v            # alias tracked
    w                # ← PoolEscapeError: w escapes
end

@with_pool pool function bad()
    A = acquire!(pool, Float64, 3, 3)
    return A         # ← PoolEscapeError: explicit return
end
```

Error messages include the variable name, declaration site, and escaping return expression:

```
ERROR: LoadError: PoolEscapeError (compile-time)

  The following variable escapes the @with_pool scope:

    v  ← pool-acquired array

  Declarations:
    [1]  v = acquire!(pool, Float64, 100)  [myfile.jl:2]

  Escaping return:
    [1]  v  [myfile.jl:3]

  Fix: Use collect(v) to return owned copies.
       Or use a regular Julia array (zeros()/Array{T}()) if it must outlive the pool scope.

in expression starting at myfile.jl:1
```

## Mutation Analysis (`PoolMutationError`)

Structural mutations (`resize!`, `push!`, `append!`, etc.) on pool-backed arrays are rejected:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    push!(v, 1.0)   # ← PoolMutationError
end
```

Detected functions: `resize!`, `push!`, `pop!`, `pushfirst!`, `popfirst!`, `append!`, `prepend!`, `insert!`, `deleteat!`, `splice!`, `sizehint!`, `empty!`.

## Limitations

Static analysis can't see through opaque function calls:

```julia
function sneaky(v)
    return v  # opaque — macro can't see inside
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    sneaky(v)  # escapes through opaque call — not caught at compile time
end
```

That's what the [runtime layer](runtime.md) is for.
