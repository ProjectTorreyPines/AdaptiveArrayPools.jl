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

### Incidental Tail Patterns

Beyond the direct-return patterns above, three additional tail shapes are
checked — they don't *look* like a `return`, but the scope's last expression
still evaluates to a pool-backed array, which then escapes as the scope's
value:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    acquire!(pool, Float64, 100)   # ← direct acquire-call tail
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    x .= v                          # ← broadcast-assign tail
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    y = v                           # ← assignment tail
end
```

Fix: end the block with `nothing` if the value is meant to be discarded:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    x .= v
    nothing   # ← no longer escapes
end
```

These three patterns are gated by the `escape_lint` preference (default
`"error"`, matching the direct-return patterns above). Set `"warn"` to
downgrade to a `@warn` diagnostic (migration escape hatch), or `"off"` to
disable this stage of checking entirely:

```julia
using Preferences
Preferences.set_preferences!("AdaptiveArrayPools", "escape_lint" => "warn")
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
