# @with_pool Macro LineNumberNode Improvement Plan

## Goal
Utilize `__source__` and `LineNumberNode` to improve coverage, stack trace, and debugging

## Target File
- `/Users/yoo/.julia/dev/AdaptiveArrayPools/src/macros.jl`

---

## Phase 1: Add Helper Functions

### 1.1 LineNumberNode Insertion Helper (New)

**Location**: Add before `_generate_pool_code` function

```julia
"""
    _maybe_add_source_location!(expr, source)

Insert source location LineNumberNode at the beginning of an Expr block.
No-op if source is nothing or expr is not an Expr(:block, ...).
"""
function _maybe_add_source_location!(expr::Expr, source::Union{LineNumberNode,Nothing})
    if source !== nothing && expr.head === :block
        pushfirst!(expr.args, LineNumberNode(source.line, source.file))
    end
    return expr
end
_maybe_add_source_location!(expr, ::Nothing) = expr
```

**Benefits**: Common application across all return paths, reduces risk of omission/drift

### 1.2 Function Body LineNumberNode Correction Helper (New)

**Goal**: Correct with `__source__` **only when no LNN exists at the body top level**

> **Background**: `body` is the **user code AST** obtained from `func_def.args[2]`.
> Existing LNNs point to **user file lines** and must be preserved.
> The problem is **short function forms** like `f(x) = ...` that have no LNN.

```julia
"""
    _has_toplevel_lnn(body) -> Bool

Check if body has a LineNumberNode at the top level (within first few args).
More robust than checking only args[1], handles Expr(:meta) etc.
"""
function _has_toplevel_lnn(body)
    body isa Expr && body.head === :block || return false
    # Check first 3 args for LNN (handles :meta, :line annotations, etc.)
    for i in 1:min(3, length(body.args))
        body.args[i] isa LineNumberNode && return true
    end
    return false
end

"""
    _ensure_body_has_toplevel_lnn(body, source)

Ensure body has a LineNumberNode at the top level.
- If body already has a top-level LNN, preserve it (user file line info)
- If not, prepend source LNN (macro call location as fallback)

Returns a new Expr to avoid mutating the original AST.
"""
function _ensure_body_has_toplevel_lnn(body, source::Union{LineNumberNode,Nothing})
    source === nothing && return body

    # Check if top-level LNN already exists (robust check)
    if _has_toplevel_lnn(body)
        return body  # Preserve existing user file LNN
    end

    # No top-level LNN → add source as fallback (no mutation)
    lnn = LineNumberNode(source.line, source.file)
    if body isa Expr && body.head === :block
        return Expr(:block, lnn, body.args...)
    else
        return Expr(:block, lnn, body)
    end
end
```

**Benefits**:
- **User body LNN preserved**: If existing top-level LNN exists, keep it (accurate body line)
- **Short function form handling**: If no LNN, correct with `__source__`
- **Mutation prevention**: Returns new Expr to protect original AST

---

## Phase 2: Modify Helper Function Signatures

### 2.1 Keyword Argument Approach (Recommended)

To avoid the risk of fixed `:cpu` default, add `source` as keyword argument:

| Function | Search Pattern | Change |
|----------|----------------|--------|
| `_generate_pool_code` | `function _generate_pool_code(pool_name, expr, force_enable)` | `(...; source::Union{LineNumberNode,Nothing}=nothing)` |
| `_generate_pool_code_with_backend` | `function _generate_pool_code_with_backend(backend::Symbol, pool_name, expr, force_enable::Bool)` | `(...; source::Union{LineNumberNode,Nothing}=nothing)` |
| `_generate_function_pool_code` | `function _generate_function_pool_code(pool_name, func_def, force_enable, disable_pooling, backend::Symbol=:cpu)` | `(...; source::Union{LineNumberNode,Nothing}=nothing)` |
| `_generate_function_pool_code_with_backend` | `function _generate_function_pool_code_with_backend(backend::Symbol, pool_name, func_def, disable_pooling::Bool)` | `(...; source::Union{LineNumberNode,Nothing}=nothing)` |

**Benefits**: Minimal changes to existing call sites, solves `backend` default value issue

---

## Phase 3: Pass source to Internal Function Calls

### Inside `_generate_pool_code` (Search: `_generate_function_pool_code(pool_name`)

```julia
# Before
return _generate_function_pool_code(pool_name, expr, force_enable, true, :cpu)
return _generate_function_pool_code(pool_name, expr, force_enable, false)

# After (pass as keyword argument)
return _generate_function_pool_code(pool_name, expr, force_enable, true, :cpu; source)
return _generate_function_pool_code(pool_name, expr, force_enable, false; source)
```

### Inside `_generate_pool_code_with_backend` (Search: `_generate_function_pool_code_with_backend(backend`)

```julia
# After
_generate_function_pool_code_with_backend(backend, pool_name, expr, ...; source)
```

---

## Phase 4: LineNumberNode Insertion

Call helper before returning each `quote ... end` block:

```julia
result = quote
    # ... generated code ...
end
_maybe_add_source_location!(result, source)
return result
```

### Insertion Locations (Based on Search Patterns)

**`_generate_pool_code`** (Search: `function _generate_pool_code`):
- `return quote ... end` in `!USE_POOLING` branch
- `return quote ... end` in `force_enable` branch
- `return quote ... end` in `else` branch

**`_generate_pool_code_with_backend`** (Search: `function _generate_pool_code_with_backend`):
- All `return quote ... end` with same pattern

**`_generate_function_pool_code`** (Search: `function _generate_function_pool_code`):
- After `transformed_body` creation: `transformed_body = _ensure_body_has_toplevel_lnn(transformed_body, source)`
- Also correct `body` in `disable_pooling` path: `body = _ensure_body_has_toplevel_lnn(body, source)`
- Then `new_body = quote ... end`
- **(Optional) To make wrapper appear as call-site**: Add `_maybe_add_source_location!(new_body, source)`

**`_generate_function_pool_code_with_backend`** (Search: `function _generate_function_pool_code_with_backend`):
- Apply `_ensure_body_has_toplevel_lnn(..., source)` to `transformed_body` and `body` before constructing `new_body`
- **(Optional)** Can also apply `_maybe_add_source_location!` to `new_body`

> **Core Principle**: Preserve if top-level LNN exists (user line), correct with `__source__` if not (short function form)
> **Note**: Wrapper code (checkpoint/try/finally) lines may still point to macros.jl. Inserting LNN to `new_body` improves this but is not required.

---

## Phase 5: Macro Definition Modifications

**@with_pool** (Search: `macro with_pool`):
```julia
macro with_pool(pool_name, expr)
    _generate_pool_code(pool_name, expr, true; source=__source__)
end
# Same pattern for remaining 3
```

**@maybe_with_pool** (Search: `macro maybe_with_pool`):
```julia
macro maybe_with_pool(pool_name, expr)
    _generate_pool_code(pool_name, expr, false; source=__source__)
end
# Same pattern for remaining 3
```

---

## Phase 6: Testing

### 6.1 Robust Search Helpers (test/test_macro_expansion.jl)

```julia
"""
    find_linenumbernode_with_line(expr, target_line) -> Union{LineNumberNode, Nothing}

Recursively search for a LineNumberNode matching target_line.
More robust than checking only the first LNN (handles block forms where
_maybe_add_source_location! may insert LNN before user code LNN).
"""
function find_linenumbernode_with_line(expr, target_line::Int)
    if expr isa LineNumberNode && expr.line == target_line
        return expr
    elseif expr isa Expr
        for arg in expr.args
            result = find_linenumbernode_with_line(arg, target_line)
            result !== nothing && return result
        end
    end
    return nothing
end

"""
    has_valid_linenumbernode(expr) -> Bool

Check if expr contains any LineNumberNode with valid line info.
"""
function has_valid_linenumbernode(expr)
    if expr isa LineNumberNode
        return expr.line > 0 && expr.file !== :none
    elseif expr isa Expr
        for arg in expr.args
            has_valid_linenumbernode(arg) && return true
        end
    end
    return false
end

"""
    get_function_body(expr) -> Union{Expr, Nothing}

Extract function body from a function definition expression.
Handles both `function f() ... end` and `f() = ...` forms.
"""
function get_function_body(expr)
    if expr isa Expr
        if expr.head === :function && length(expr.args) >= 2
            return expr.args[2]
        elseif expr.head === :(=) && expr.args[1] isa Expr && expr.args[1].head === :call
            return expr.args[2]
        end
        # Recurse for wrapped expressions
        for arg in expr.args
            result = get_function_body(arg)
            result !== nothing && return result
        end
    end
    return nothing
end
```

### 6.2 Test Cases (Full Coverage)

> **Test Strategy**: Verify "existence of LNN matching expected line" rather than "first LNN".
> In block forms, `_maybe_add_source_location!` may insert additional LNNs,
> so checking for existence of LNN with specific line is more robust.

```julia
@testset "Source location preservation" begin
    # Test 1: @with_pool block form
    @testset "@with_pool block" begin
        expected_line = @__LINE__ + 2
        expr = @macroexpand @with_pool pool begin
            v = acquire!(pool, Float64, 10)
        end
        # Check if LNN matching expected line exists
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
        @test lnn.file !== :none
        # At minimum, valid LNN must exist
        @test has_valid_linenumbernode(expr)
    end

    # Test 2: @with_pool function form
    @testset "@with_pool function" begin
        expected_line = @__LINE__ + 2
        func_expr = @macroexpand @with_pool pool function test_func(n)
            acquire!(pool, Float64, n)
        end
        body = get_function_body(func_expr)
        @test body !== nothing
        lnn = find_linenumbernode_with_line(body, expected_line)
        @test lnn !== nothing
    end

    # Test 3: @maybe_with_pool
    @testset "@maybe_with_pool" begin
        expected_line = @__LINE__ + 2
        expr = @macroexpand @maybe_with_pool pool begin
            v = acquire!(pool, Float64, 10)
        end
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
    end

    # Test 4: Backend variant (@with_pool :cpu)
    @testset "@with_pool :cpu backend" begin
        expected_line = @__LINE__ + 2
        expr = @macroexpand @with_pool :cpu pool begin
            v = acquire!(pool, Float64, 10)
        end
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
    end

    # Test 5: Without pool name (implicit gensym)
    @testset "@with_pool without pool name" begin
        expected_line = @__LINE__ + 2
        expr = @macroexpand @with_pool begin
            inner_function()
        end
        lnn = find_linenumbernode_with_line(expr, expected_line)
        @test lnn !== nothing
    end

    # Test 6: Short-form function (f(x) = ...) - Case without LNN, corrected with __source__
    @testset "@with_pool short function" begin
        expected_line = @__LINE__ + 1
        func_expr = @macroexpand @with_pool pool test_func(x) = acquire!(pool, Float64, x)
        body = get_function_body(func_expr)
        @test body !== nothing
        # Short function originally has no LNN, so corrected with __source__
        lnn = find_linenumbernode_with_line(body, expected_line)
        @test lnn !== nothing
    end
end
```

### 6.3 Verification Command
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Expected Results

| Item | Before Improvement | After Improvement |
|------|-------------------|-------------------|
| Coverage | signature uncovered | Properly mapped |
| Stack trace | macros.jl:XXX | Original source:line |
| Breakpoint | Inside macros.jl | Improved to inside body |

---

## Considerations

1. **Use Keyword Arguments**: Add `source` as keyword arg to minimize impact on existing call sites
2. **Use Helper Functions**: Use `_maybe_add_source_location!` for consistent insertion across all paths
3. **Body Line Correction**: Use `_ensure_body_has_toplevel_lnn` to preserve top-level LNN, correct with `__source__` if not present
4. **Robust Tests**: Search-based verification resistant to AST structure changes + line number accuracy verification
5. **esc() Interaction**: `LineNumberNode` is unrelated to hygiene → insert at quote block top
6. **try-finally**: Lines inside wrapper still point to macros.jl (acceptable)
7. **CUDA Extension**: Only registers backend dispatch, no macro definitions → no changes needed

---

## Change Summary

| Phase | Work | Estimated Change |
|-------|------|-----------------|
| 1 | Add helper functions (3: `_maybe_add_source_location!`, `_has_toplevel_lnn`, `_ensure_body_has_toplevel_lnn`) | +35 lines |
| 2 | Modify signatures (4 functions) | 4 lines modified |
| 3 | Modify internal calls | ~5 lines modified |
| 4 | LineNumberNode insertion | ~10 lines added |
| 5 | Modify macro definitions (8) | 8 lines modified |
| 6 | Add tests (3 helpers + 6 tests) | +80 lines |
| **Total** | | ~140 lines |
