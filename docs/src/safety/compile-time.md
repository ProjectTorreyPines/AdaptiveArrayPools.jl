# Compile-Time Detection

The `@with_pool` macro statically analyzes your code at macro expansion time — zero runtime cost, always active.

## Escape Analysis (`PoolEscapeError`)

Detects any tail expression that would deliver a pool-backed array out of the
scope. Tracks aliases, containers, and convenience wrappers. **Severity is
form-based**, because the two forms differ in what the macro can know:

- **Function form** and **explicit `return`** (either form): the value
  definitively reaches the enclosing function's caller — a `begin` block is
  not a function boundary, so `return` inside a block form returns from the
  *enclosing function*. These always throw `PoolEscapeError` at expansion.
- **Block form, implicit last expression**: the macro cannot see its own call
  site, so whether the block's value is used (`out = @with_pool ...`) or
  discarded (a loop body, a bare statement) is undecidable. These emit a
  `@warn` and the expansion **replaces the escaping value with an inert
  `EscapedPoolArray` guard**: discarded → completely silent; used → the first
  operation throws `EscapedPoolUseError` with the variable name, shape, and
  scope location.

```julia
@with_pool pool function bad()
    A = acquire!(pool, Float64, 3, 3)
    return A         # ← PoolEscapeError: function return, always an error
end

out = @with_pool pool begin
    v = acquire!(pool, Float64, 100)
    w = v            # alias tracked
    w                # ← @warn + guard: `out` is an EscapedPoolArray
end
out[1]               # ← EscapedPoolUseError: names `w`, its shape, and the scope

# The guard is inert when the value is never used — this common
# copy-out-and-discard shape runs silently and correctly:
Threads.@threads for k in 1:nmodels
    @with_pool pool begin
        y = acquire!(pool, Float64, m, n)
        compute!(y, k)
        results[:, :, k] = y   # copy runs; the yielded `y` is guarded; @threads discards it
    end
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
    v .= 0.0                        # ← broadcast-assign tail (evaluates to `v`)
end

@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    y = v                           # ← assignment tail
end
```

Fix: end the block with `nothing` if the value is meant to be discarded —
this also silences the warning:

```julia
@with_pool pool begin
    v = acquire!(pool, Float64, 100)
    v .= 0.0
    nothing   # ← no longer escapes
end
```

In block form these follow the same warn + `EscapedPoolArray`-guard treatment
as all implicit tails; in function form or under an explicit `return`, they
throw. The `escape_lint` preference tunes the **block-form** severity:

- `"warn"` (default) — `@warn` diagnostic + the guard rewrite described above.
- `"error"` — block-form implicit tails also throw `PoolEscapeError`
  (the strict pre-guard behavior; no guard rewrite needed).
- `"off"` — no warning and no guard rewrite (the block tail escapes the raw
  pool array, as before this feature). Function-form and explicit-`return`
  direct escapes still always error.

```julia
using Preferences
Preferences.set_preferences!("AdaptiveArrayPools", "escape_lint" => "error")
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
