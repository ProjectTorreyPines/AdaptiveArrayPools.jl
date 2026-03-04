using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: get_typed_pool!, _lazy_checkpoint!, _lazy_rewind!,
    _typed_lazy_checkpoint!, _typed_lazy_rewind!, _tracked_mask_for_types,
    _record_type_touch!, _checkpoint_typed_pool!, _rewind_typed_pool!,
    _LAZY_MODE_BIT, _TYPED_LAZY_BIT, _MODE_BITS_MASK, _TYPE_BITS_MASK,
    _fixed_slot_bit, _can_use_typed_path, checkpoint!, rewind!

# ==============================================================================
# Helper: inspect pool.others state
# ==============================================================================

"""Get n_active for a fallback type (0 if type not in pool.others)."""
function others_n_active(pool, ::Type{T}) where {T}
    haskey(pool.others, T) ? pool.others[T].n_active : 0
end

"""Get checkpoint stack length for a fallback type."""
function others_stack_len(pool, ::Type{T}) where {T}
    haskey(pool.others, T) ? length(pool.others[T]._checkpoint_depths) : 0
end

"""Get all n_active values for pool.others entries."""
function all_others_n_active(pool)
    Dict(T => tp.n_active for (T, tp) in pool.others)
end

# ==============================================================================
# Type definitions (must be at module scope, not inside @testset)
# ==============================================================================

struct MyTestElement
    x::Float64
    y::Float64
end

# --- ForwardDiff.Dual-like parametric type ---
# Simulates: Dual{Tag{typeof(f)}, V, N}
# Key properties:
#   - Different Tag/N params = different concrete types = separate pool entries
#   - Each unique Dual variant occupies its own IdDict slot in pool.others
struct FakeTag{F} end
struct FakeDual{Tag, V<:Real, N}
    value::V
    partials::NTuple{N, V}
end
FakeDual{Tag, V, N}(v::V) where {Tag, V, N} = FakeDual{Tag, V, N}(v, ntuple(_ -> zero(V), Val(N)))

# Type aliases for readability
const Dual_f1_11 = FakeDual{FakeTag{:f1}, Float64, 11}
const Dual_f1_4  = FakeDual{FakeTag{:f1}, Float64, 4}
const Dual_f2_11 = FakeDual{FakeTag{:f2}, Float64, 11}

# ==============================================================================
# 1. Multiple Distinct Fallback Types in Single Scope
# ==============================================================================

@testset "Fallback Reclamation" begin

@testset "1. Multiple distinct fallback types in single scope" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    v1 = acquire!(pool, UInt8, 10)
    v2 = acquire!(pool, Float16, 20)
    v3 = acquire!(pool, Int16, 30)

    @test others_n_active(pool, UInt8) == 1
    @test others_n_active(pool, Float16) == 1
    @test others_n_active(pool, Int16) == 1

    rewind!(pool)

    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
    @test others_n_active(pool, Int16) == 0
end

@testset "1b. Multiple arrays per fallback type" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    for _ in 1:5
        acquire!(pool, UInt8, 10)
        acquire!(pool, Float16, 20)
    end
    @test others_n_active(pool, UInt8) == 5
    @test others_n_active(pool, Float16) == 5

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

# ==============================================================================
# 2. Deeply Nested Scopes (3+ levels) with Fallback Types
# ==============================================================================

@testset "2. Deep nesting (5 levels) with fallback types" begin
    pool = AdaptiveArrayPool()
    fallback_types = [UInt8, Float16, Int16, UInt16, Int8]

    # Acquire one per type at each depth level
    for depth_level in 1:5
        checkpoint!(pool)
        v = acquire!(pool, fallback_types[depth_level], 10 * depth_level)
        @test others_n_active(pool, fallback_types[depth_level]) == 1
    end

    # All 5 types active
    for (i, T) in enumerate(fallback_types)
        @test others_n_active(pool, T) == 1
    end

    # Unwind — each type should revert as we go
    for depth_level in 5:-1:1
        rewind!(pool)
        @test others_n_active(pool, fallback_types[depth_level]) == 0
    end
end

@testset "2b. Same fallback type across nested depths" begin
    pool = AdaptiveArrayPool()

    # Level 1: acquire 1 UInt8
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    @test others_n_active(pool, UInt8) == 1

    # Level 2: acquire 2 more
    checkpoint!(pool)
    acquire!(pool, UInt8, 20)
    acquire!(pool, UInt8, 30)
    @test others_n_active(pool, UInt8) == 3

    # Level 3: acquire 1 more
    checkpoint!(pool)
    acquire!(pool, UInt8, 40)
    @test others_n_active(pool, UInt8) == 4

    # Unwind level 3
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 3

    # Unwind level 2
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 1

    # Unwind level 1
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 3. @with_pool Macro with Fallback Types
# ==============================================================================

@testset "3. @with_pool macro with fallback types" begin
    # The macro uses lazy checkpoint/rewind (use_typed=false path) when
    # types aren't statically extractable
    function helper_fallback!(pool)
        acquire!(pool, UInt8, 50)
        acquire!(pool, Float16, 50)
    end

    result = @with_pool pool begin
        helper_fallback!(pool)
        others_n_active(pool, UInt8)
    end
    @test result == 1

    # After scope: get a fresh pool and verify it's clean
    fresh_pool = AdaptiveArrayPool()
    @test isempty(fresh_pool.others)
end

@testset "3b. @with_pool with static fallback type" begin
    # acquire!(pool, UInt8, ...) — UInt8 is not a fixed slot, so macro
    # goes through lazy path (since _fixed_slot_bit(UInt8) == 0)
    result = @with_pool pool begin
        v = acquire!(pool, UInt8, 100)
        length(v)
    end
    @test result == 100
end

@testset "3c. Nested @with_pool with fallback types" begin
    result = @with_pool p1 begin
        a = acquire!(p1, UInt8, 10)
        inner = @with_pool p2 begin
            b = acquire!(p2, UInt8, 20)
            others_n_active(p2, UInt8)
        end
        # After inner scope rewinds, the task-local pool's UInt8 n_active
        # should be back to 1 (only 'a' from outer scope)
        (inner, others_n_active(p1, UInt8))
    end
    @test result[1] == 2   # inner scope had 2 (1 from outer + 1 from inner)
    @test result[2] == 1   # after inner rewind, back to 1
end

# ==============================================================================
# 4. Lazy Mode with Fallback Types
# ==============================================================================

@testset "4. Lazy checkpoint/rewind with fallback types" begin
    pool = AdaptiveArrayPool()

    _lazy_checkpoint!(pool)
    v1 = acquire!(pool, UInt8, 10)
    v2 = acquire!(pool, Float16, 20)

    @test others_n_active(pool, UInt8) == 1
    @test others_n_active(pool, Float16) == 1
    @test pool._touched_has_others[pool._current_depth] == true

    _lazy_rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

@testset "4b. Lazy mode: pre-existing others get eagerly checkpointed" begin
    pool = AdaptiveArrayPool()

    # Pre-populate at global scope
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    @test others_n_active(pool, UInt8) == 1
    rewind!(pool)

    # UInt8 pool exists now with n_active=0
    @test haskey(pool.others, UInt8)
    @test others_n_active(pool, UInt8) == 0

    # Now use lazy mode — pre-existing UInt8 should be eagerly checkpointed
    _lazy_checkpoint!(pool)
    acquire!(pool, UInt8, 20)
    @test others_n_active(pool, UInt8) == 1

    _lazy_rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

@testset "4c. Lazy mode: new fallback type created during scope" begin
    pool = AdaptiveArrayPool()

    _lazy_checkpoint!(pool)
    # Int16 doesn't exist yet — created inside lazy scope
    v = acquire!(pool, Int16, 10)
    @test others_n_active(pool, Int16) == 1

    _lazy_rewind!(pool)
    # Should revert to 0 (auto-checkpoint sentinel covers new types)
    @test others_n_active(pool, Int16) == 0
end

# ==============================================================================
# 5. Typed-Lazy Mode with Fallback Types
# ==============================================================================

@testset "5. Typed-lazy checkpoint/rewind with fallback types" begin
    pool = AdaptiveArrayPool()

    # Pre-populate a fallback type
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0

    # Parent acquires UInt8
    checkpoint!(pool)
    acquire!(pool, UInt8, 5)
    @test others_n_active(pool, UInt8) == 1

    # Child typed-lazy scope tracking Float64
    _typed_lazy_checkpoint!(pool, Float64)
    acquire!(pool, UInt8, 15)  # Helper touches fallback type
    @test others_n_active(pool, UInt8) == 2

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    # Must restore parent's n_active=1 (NOT 0)
    @test others_n_active(pool, UInt8) == 1

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

@testset "5b. Typed-lazy with new fallback type in child scope" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    # Parent doesn't use any fallback types

    _typed_lazy_checkpoint!(pool, Float64)
    # Child creates new fallback type
    acquire!(pool, Int16, 10)
    @test others_n_active(pool, Int16) == 1

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test others_n_active(pool, Int16) == 0

    rewind!(pool)
end

# ==============================================================================
# 6. Mixed Fixed + Fallback Types
# ==============================================================================

@testset "6. Mixed fixed and fallback types rewind correctly" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    f64_v = acquire!(pool, Float64, 10)
    u8_v = acquire!(pool, UInt8, 20)
    i32_v = acquire!(pool, Int32, 30)
    f16_v = acquire!(pool, Float16, 40)

    @test pool.float64.n_active == 1
    @test pool.int32.n_active == 1
    @test others_n_active(pool, UInt8) == 1
    @test others_n_active(pool, Float16) == 1

    rewind!(pool)

    @test pool.float64.n_active == 0
    @test pool.int32.n_active == 0
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

@testset "6b. Lazy mode: mixed fixed + fallback" begin
    pool = AdaptiveArrayPool()

    _lazy_checkpoint!(pool)
    acquire!(pool, Float64, 10)
    acquire!(pool, UInt8, 20)
    acquire!(pool, Int32, 30)
    acquire!(pool, Float16, 40)

    @test pool.float64.n_active == 1
    @test pool.int32.n_active == 1
    @test others_n_active(pool, UInt8) == 1
    @test others_n_active(pool, Float16) == 1

    _lazy_rewind!(pool)

    @test pool.float64.n_active == 0
    @test pool.int32.n_active == 0
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

# ==============================================================================
# 7. Fallback Type Rewind → Re-acquire Cycle
# ==============================================================================

@testset "7. Rewind then re-acquire fallback type reuses memory" begin
    pool = AdaptiveArrayPool()

    # First cycle
    checkpoint!(pool)
    v1 = acquire!(pool, UInt8, 100)
    v1 .= 0x42
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0

    # Second cycle — should reuse existing backing vector
    checkpoint!(pool)
    v2 = acquire!(pool, UInt8, 100)
    @test others_n_active(pool, UInt8) == 1
    # The backing vector should be reused (same object)
    @test parent(v1) === parent(v2)
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 8. Checkpoint Stack Invariants (No Stack Leak)
# ==============================================================================

@testset "8. Full checkpoint: no stack leak over many iterations" begin
    pool = AdaptiveArrayPool()

    # Pre-populate
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    rewind!(pool)

    uint8_pool = pool.others[UInt8]
    initial_stack_len = length(uint8_pool._checkpoint_depths)

    for _ in 1:100
        checkpoint!(pool)
        acquire!(pool, UInt8, 10)
        rewind!(pool)
    end

    @test length(uint8_pool._checkpoint_depths) == initial_stack_len
    @test others_n_active(pool, UInt8) == 0
end

@testset "8b. Lazy checkpoint: no stack leak over many iterations" begin
    pool = AdaptiveArrayPool()

    # Pre-populate
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    rewind!(pool)

    uint8_pool = pool.others[UInt8]
    initial_stack_len = length(uint8_pool._checkpoint_depths)

    for _ in 1:100
        _lazy_checkpoint!(pool)
        acquire!(pool, UInt8, 10)
        _lazy_rewind!(pool)
    end

    @test length(uint8_pool._checkpoint_depths) == initial_stack_len
    @test others_n_active(pool, UInt8) == 0
end

@testset "8c. Typed-lazy checkpoint: no stack leak over many iterations" begin
    pool = AdaptiveArrayPool()

    # Pre-populate
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    rewind!(pool)

    uint8_pool = pool.others[UInt8]
    initial_stack_len = length(uint8_pool._checkpoint_depths)

    for _ in 1:100
        _typed_lazy_checkpoint!(pool, Float64)
        acquire!(pool, UInt8, 10)
        _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    end

    @test length(uint8_pool._checkpoint_depths) == initial_stack_len
    @test others_n_active(pool, UInt8) == 0
end

@testset "8d. @with_pool macro: no stack leak over many iterations" begin
    pool_ref = Ref{AdaptiveArrayPool}()

    # Pre-populate
    @with_pool pool begin
        acquire!(pool, UInt8, 10)
        pool_ref[] = pool
    end

    uint8_pool = pool_ref[].others[UInt8]
    initial_stack_len = length(uint8_pool._checkpoint_depths)

    for _ in 1:100
        @with_pool pool begin
            acquire!(pool, UInt8, 10)
        end
    end

    @test length(uint8_pool._checkpoint_depths) == initial_stack_len
    @test uint8_pool.n_active == 0
end

# ==============================================================================
# 9. n_active Monotonicity (Memory Leak Detection)
# ==============================================================================

@testset "9. n_active doesn't grow over repeated checkpoint/rewind cycles" begin
    pool = AdaptiveArrayPool()
    fallback_types = [UInt8, Float16, Int16]

    # Pre-populate all types
    checkpoint!(pool)
    for T in fallback_types
        acquire!(pool, T, 10)
    end
    rewind!(pool)

    # Run 200 iterations — n_active should always return to 0
    for iter in 1:200
        checkpoint!(pool)
        for T in fallback_types
            acquire!(pool, T, 10)
        end
        rewind!(pool)

        for T in fallback_types
            n = others_n_active(pool, T)
            if n != 0
                @test n == 0  # Will show which type leaked
                @info "LEAK DETECTED" iteration=iter type=T n_active=n
                break
            end
        end
    end

    # Final check
    for T in fallback_types
        @test others_n_active(pool, T) == 0
    end
end

@testset "9b. Lazy mode: n_active doesn't grow over iterations" begin
    pool = AdaptiveArrayPool()

    for iter in 1:200
        _lazy_checkpoint!(pool)
        acquire!(pool, UInt8, 10)
        acquire!(pool, Float16, 20)
        _lazy_rewind!(pool)
    end

    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

@testset "9c. @with_pool: n_active doesn't grow over iterations" begin
    for iter in 1:200
        @with_pool pool begin
            acquire!(pool, UInt8, 10)
            acquire!(pool, Float16, 20)
        end
    end

    # Verify task-local pool is clean
    pool = AdaptiveArrayPools.get_task_local_pool()
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

# ==============================================================================
# 10. Backing Vector Count Stability (Pool Growth Detection)
# ==============================================================================

@testset "10. Pool vectors don't grow over checkpoint/rewind cycles" begin
    pool = AdaptiveArrayPool()

    # Warmup: first cycle creates backing vectors
    checkpoint!(pool)
    acquire!(pool, UInt8, 100)
    acquire!(pool, UInt8, 200)
    acquire!(pool, Float16, 50)
    rewind!(pool)

    # Record vector counts after warmup
    u8_vec_count = length(pool.others[UInt8].vectors)
    f16_vec_count = length(pool.others[Float16].vectors)

    # Run many iterations — vector count should stay stable
    for _ in 1:100
        checkpoint!(pool)
        acquire!(pool, UInt8, 100)
        acquire!(pool, UInt8, 200)
        acquire!(pool, Float16, 50)
        rewind!(pool)
    end

    @test length(pool.others[UInt8].vectors) == u8_vec_count
    @test length(pool.others[Float16].vectors) == f16_vec_count
end

@testset "10b. Lazy mode: pool vectors don't grow" begin
    pool = AdaptiveArrayPool()

    # Warmup
    _lazy_checkpoint!(pool)
    acquire!(pool, UInt8, 100)
    acquire!(pool, Float16, 50)
    _lazy_rewind!(pool)

    u8_vec_count = length(pool.others[UInt8].vectors)
    f16_vec_count = length(pool.others[Float16].vectors)

    for _ in 1:100
        _lazy_checkpoint!(pool)
        acquire!(pool, UInt8, 100)
        acquire!(pool, Float16, 50)
        _lazy_rewind!(pool)
    end

    @test length(pool.others[UInt8].vectors) == u8_vec_count
    @test length(pool.others[Float16].vectors) == f16_vec_count
end

# ==============================================================================
# 11. unsafe_acquire! with Fallback Types
# ==============================================================================

@testset "11. unsafe_acquire! with fallback types" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    v = unsafe_acquire!(pool, UInt8, 10)
    @test v isa Array{UInt8, 1}
    @test length(v) == 10
    @test others_n_active(pool, UInt8) == 1

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

@testset "11b. unsafe_acquire! N-D with fallback types" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    m = unsafe_acquire!(pool, UInt8, 3, 4)
    @test m isa Array{UInt8, 2}
    @test size(m) == (3, 4)
    @test others_n_active(pool, UInt8) == 1

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 12. Convenience Functions with Fallback Types
# ==============================================================================

@testset "12. zeros!/ones!/similar! with fallback types" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    z = zeros!(pool, UInt8, 10)
    @test all(z .== 0)
    @test others_n_active(pool, UInt8) == 1

    o = ones!(pool, UInt8, 10)
    @test all(o .== 1)
    @test others_n_active(pool, UInt8) == 2

    src = UInt8[1, 2, 3]
    s = similar!(pool, src)
    @test length(s) == 3
    @test others_n_active(pool, UInt8) == 3

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 13. Exception Safety
# ==============================================================================

@testset "13. Exception during fallback acquire doesn't leak" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    @test others_n_active(pool, UInt8) == 1

    try
        checkpoint!(pool)
        acquire!(pool, UInt8, 20)
        @test others_n_active(pool, UInt8) == 2
        error("simulated failure")
    catch
        rewind!(pool)
    end

    @test others_n_active(pool, UInt8) == 1
    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

@testset "13b. @with_pool exception safety with fallback types" begin
    try
        @with_pool pool begin
            acquire!(pool, UInt8, 10)
            acquire!(pool, Float16, 20)
            error("simulated failure")
        end
    catch
    end

    # After exception + rewind via finally, pool should be clean
    pool = AdaptiveArrayPools.get_task_local_pool()
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
end

# ==============================================================================
# 14. Depth Tracking Consistency
# ==============================================================================

@testset "14. _current_depth returns to 1 after cleanup" begin
    pool = AdaptiveArrayPool()
    @test pool._current_depth == 1

    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    @test pool._current_depth == 2

    checkpoint!(pool)
    acquire!(pool, Float16, 20)
    @test pool._current_depth == 3

    rewind!(pool)
    @test pool._current_depth == 2

    rewind!(pool)
    @test pool._current_depth == 1
end

@testset "14b. _touched_has_others stack cleaned properly" begin
    pool = AdaptiveArrayPool()
    @test length(pool._touched_has_others) == 1  # sentinel

    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    @test length(pool._touched_has_others) == 2
    @test pool._touched_has_others[2] == true

    rewind!(pool)
    @test length(pool._touched_has_others) == 1  # back to sentinel
end

# ==============================================================================
# 15. Custom Struct Types as Fallback
# ==============================================================================

@testset "15. Custom struct type as fallback" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    v = acquire!(pool, MyTestElement, 5)
    @test v isa SubArray
    @test length(v) == 5
    @test eltype(v) == MyTestElement
    @test others_n_active(pool, MyTestElement) == 1

    rewind!(pool)
    @test others_n_active(pool, MyTestElement) == 0
end

@testset "15b. Custom struct: repeated cycles don't leak" begin
    pool = AdaptiveArrayPool()

    for _ in 1:50
        checkpoint!(pool)
        acquire!(pool, MyTestElement, 10)
        rewind!(pool)
    end

    @test others_n_active(pool, MyTestElement) == 0
    @test length(pool.others[MyTestElement].vectors) == 1  # reuses single backing
end

# ==============================================================================
# 16. Full Mode ↔ Lazy Mode Transitions with Fallback
# ==============================================================================

@testset "16. Parent full checkpoint, child lazy, fallback touched" begin
    pool = AdaptiveArrayPool()

    # Pre-populate UInt8
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    rewind!(pool)

    # Parent: full checkpoint, acquires UInt8
    checkpoint!(pool)
    acquire!(pool, UInt8, 5)
    @test others_n_active(pool, UInt8) == 1

    # Child: lazy checkpoint, touches same fallback type
    _lazy_checkpoint!(pool)
    acquire!(pool, UInt8, 15)
    @test others_n_active(pool, UInt8) == 2

    _lazy_rewind!(pool)
    @test others_n_active(pool, UInt8) == 1  # parent's UInt8 preserved

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

@testset "16b. Parent lazy, child full checkpoint, fallback touched" begin
    pool = AdaptiveArrayPool()

    _lazy_checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    @test others_n_active(pool, UInt8) == 1

    # Child: full checkpoint
    checkpoint!(pool)
    acquire!(pool, UInt8, 20)
    @test others_n_active(pool, UInt8) == 2

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 1

    _lazy_rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

@testset "16c. Parent full, child typed-lazy (Float64), helper touches fallback" begin
    pool = AdaptiveArrayPool()

    # Pre-populate
    checkpoint!(pool)
    acquire!(pool, UInt8, 10)
    rewind!(pool)

    # Parent full checkpoint
    checkpoint!(pool)
    acquire!(pool, UInt8, 5)
    @test others_n_active(pool, UInt8) == 1

    # Child typed-lazy tracking Float64, helper acquires UInt8
    _typed_lazy_checkpoint!(pool, Float64)
    acquire!(pool, Float64, 10)  # tracked type
    acquire!(pool, UInt8, 15)    # untracked fallback
    @test others_n_active(pool, UInt8) == 2

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test others_n_active(pool, UInt8) == 1  # parent's UInt8 preserved
    @test pool.float64.n_active == 0  # tracked type also cleaned

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 17. Stress Test: Simulated Realistic Workload
# ==============================================================================

@testset "17. Realistic workload: nested function calls with fallback types" begin
    pool = AdaptiveArrayPool()

    function inner_compute!(pool)
        a = acquire!(pool, UInt8, 100)
        b = acquire!(pool, Float16, 50)
        a .= 0x01
        b .= Float16(2.0)
        sum(a) + sum(b)
    end

    function middle_compute!(pool)
        checkpoint!(pool)
        try
            x = acquire!(pool, Float64, 10)
            x .= 1.0
            result = inner_compute!(pool)
            return sum(x) + result
        finally
            rewind!(pool)
        end
    end

    # Outer scope
    for _ in 1:100
        checkpoint!(pool)
        try
            r = middle_compute!(pool)
            @test r ≈ 10.0 + 100.0 + 100.0  # 10 Float64 + 100 UInt8 + 50 Float16
        finally
            rewind!(pool)
        end
    end

    # After all iterations: no leaks
    @test pool.float64.n_active == 0
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0

    # Backing vectors: should not have grown
    @test length(pool.others[UInt8].vectors) == 1
    @test length(pool.others[Float16].vectors) == 1
end

@testset "17b. @with_pool stress: 500 iterations with multiple fallback types" begin
    # Warmup cycle to populate task-local pool (may already have entries from prior tests)
    @with_pool pool begin
        acquire!(pool, UInt8, 10)
        acquire!(pool, Float16, 20)
        acquire!(pool, Int16, 30)
    end

    pool = AdaptiveArrayPools.get_task_local_pool()
    u8_baseline = length(pool.others[UInt8].vectors)
    f16_baseline = length(pool.others[Float16].vectors)
    i16_baseline = length(pool.others[Int16].vectors)

    for _ in 1:500
        @with_pool pool begin
            acquire!(pool, UInt8, 10)
            acquire!(pool, Float16, 20)
            acquire!(pool, Int16, 30)
        end
    end

    pool = AdaptiveArrayPools.get_task_local_pool()
    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
    @test others_n_active(pool, Int16) == 0

    # Backing vectors should not have grown beyond warmup baseline
    @test length(pool.others[UInt8].vectors) == u8_baseline
    @test length(pool.others[Float16].vectors) == f16_baseline
    @test length(pool.others[Int16].vectors) == i16_baseline
end

# ==============================================================================
# 18. Memory Leak Canary: Total Pool Size Stability
# ==============================================================================

@testset "18. Total others pool size doesn't grow unbounded" begin
    pool = AdaptiveArrayPool()

    # Warmup
    checkpoint!(pool)
    acquire!(pool, UInt8, 1000)
    acquire!(pool, Float16, 1000)
    rewind!(pool)

    # Measure baseline memory footprint
    function total_backing_bytes(pool)
        total = 0
        for (T, tp) in pool.others
            for v in tp.vectors
                total += sizeof(v)
            end
        end
        total
    end

    baseline_bytes = total_backing_bytes(pool)

    # Run many cycles
    for _ in 1:500
        checkpoint!(pool)
        acquire!(pool, UInt8, 1000)
        acquire!(pool, Float16, 1000)
        rewind!(pool)
    end

    final_bytes = total_backing_bytes(pool)
    @test final_bytes == baseline_bytes  # No growth
end

# ==============================================================================
# 19. reset! and empty! Properly Handle Fallback Types
# ==============================================================================

@testset "19. reset! clears fallback n_active but preserves vectors" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    acquire!(pool, UInt8, 100)
    acquire!(pool, Float16, 200)
    # Don't rewind — simulate leaked state

    reset!(pool)

    @test others_n_active(pool, UInt8) == 0
    @test others_n_active(pool, Float16) == 0
    @test pool._current_depth == 1
    # Vectors should be preserved for reuse
    @test length(pool.others[UInt8].vectors) == 1
    @test length(pool.others[Float16].vectors) == 1
end

@testset "19b. empty! clears fallback types completely" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    acquire!(pool, UInt8, 100)
    acquire!(pool, Float16, 200)
    rewind!(pool)

    empty!(pool)

    @test isempty(pool.others)
    @test pool._current_depth == 1
end

# ==============================================================================
# 20. Edge Case: Acquire Zero-Length Array of Fallback Type
# ==============================================================================

@testset "20. Zero-length fallback array acquire/rewind" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    v = acquire!(pool, UInt8, 0)
    @test length(v) == 0
    @test others_n_active(pool, UInt8) == 1

    rewind!(pool)
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 21. Parametric Dual-Like Type: Basic Reclamation
# ==============================================================================
# ForwardDiff.Dual{Tag{f}, V, N} — each unique parameterization is a DIFFERENT
# concrete type, creating separate pool.others entries. This tests that pool
# correctly handles multiple parametric variants of the same "family" of types.

@testset "21. Dual-like parametric type: basic acquire/rewind" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    v = acquire!(pool, Dual_f1_11, 10)
    @test eltype(v) == Dual_f1_11
    @test others_n_active(pool, Dual_f1_11) == 1

    rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0
end

@testset "21b. Dual-like: different param variants are separate pool entries" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    # Three different Dual variants — each gets its own IdDict entry
    acquire!(pool, Dual_f1_11, 10)
    acquire!(pool, Dual_f1_4, 20)
    acquire!(pool, Dual_f2_11, 30)

    @test length(pool.others) == 3
    @test others_n_active(pool, Dual_f1_11) == 1
    @test others_n_active(pool, Dual_f1_4) == 1
    @test others_n_active(pool, Dual_f2_11) == 1

    rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0
    @test others_n_active(pool, Dual_f1_4) == 0
    @test others_n_active(pool, Dual_f2_11) == 0
end

# ==============================================================================
# 22. Parametric Dual-Like: Nested Scopes (Simulates ForwardDiff Chunk Processing)
# ==============================================================================
# ForwardDiff.gradient processes data in chunks, calling the function N times.
# Each call creates a @with_pool scope. The pool must correctly rewind Dual
# arrays created during each chunk evaluation.

@testset "22. Dual-like: simulated ForwardDiff chunk processing" begin
    pool = AdaptiveArrayPool()
    n_chunks = 11  # Like processing 121 elements in chunks of 11

    for chunk in 1:n_chunks
        checkpoint!(pool)
        # Simulates what happens inside cubic_interp when called with Dual data
        partials = acquire!(pool, Dual_f1_11, 44)   # like (4, 11) partials array
        workspace = acquire!(pool, Dual_f1_11, 11)   # temporary workspace
        @test others_n_active(pool, Dual_f1_11) == 2

        rewind!(pool)
        @test others_n_active(pool, Dual_f1_11) == 0
    end

    # After all chunks: zero leak
    @test others_n_active(pool, Dual_f1_11) == 0
    @test length(pool.others[Dual_f1_11].vectors) == 2  # reuses 2 backing vectors
end

@testset "22b. Dual-like: simulated nested @with_pool in chunk processing" begin
    pool = AdaptiveArrayPool()

    for chunk in 1:11
        # Outer scope: oneshot function
        checkpoint!(pool)
        partials = acquire!(pool, Dual_f1_11, 44)

        # Inner scope: solver function (nested @with_pool)
        checkpoint!(pool)
        m = acquire!(pool, Dual_f1_11, 11)
        @test others_n_active(pool, Dual_f1_11) == 2
        rewind!(pool)
        @test others_n_active(pool, Dual_f1_11) == 1  # only partials

        rewind!(pool)
        @test others_n_active(pool, Dual_f1_11) == 0
    end

    @test others_n_active(pool, Dual_f1_11) == 0
end

# ==============================================================================
# 23. Parametric Dual-Like: Lazy Mode (Macro-Generated Path)
# ==============================================================================

@testset "23. Dual-like: lazy checkpoint/rewind" begin
    pool = AdaptiveArrayPool()

    _lazy_checkpoint!(pool)
    acquire!(pool, Dual_f1_11, 44)
    acquire!(pool, Dual_f1_11, 11)
    @test others_n_active(pool, Dual_f1_11) == 2
    @test pool._touched_has_others[pool._current_depth] == true

    _lazy_rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0
end

@testset "23b. Dual-like: lazy mode with pre-existing Dual pool" begin
    pool = AdaptiveArrayPool()

    # Pre-populate Dual pool (simulates warmup call)
    checkpoint!(pool)
    acquire!(pool, Dual_f1_11, 10)
    rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0

    # Lazy scope — pre-existing Dual pool must be eagerly checkpointed
    _lazy_checkpoint!(pool)
    acquire!(pool, Dual_f1_11, 44)
    @test others_n_active(pool, Dual_f1_11) == 1

    _lazy_rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0
end

@testset "23c. Dual-like: lazy nested, Dual acquired only in inner scope" begin
    pool = AdaptiveArrayPool()

    # Outer: lazy, acquires Float64 only
    _lazy_checkpoint!(pool)
    acquire!(pool, Float64, 10)

    # Inner: lazy, acquires Dual (new type created inside nested lazy scope)
    _lazy_checkpoint!(pool)
    acquire!(pool, Dual_f1_11, 44)
    @test others_n_active(pool, Dual_f1_11) == 1

    _lazy_rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0

    _lazy_rewind!(pool)
    @test pool.float64.n_active == 0
end

# ==============================================================================
# 24. Parametric Dual-Like: Typed-Lazy (use_typed=true Macro Path)
# ==============================================================================
# This simulates the MOST LIKELY leak scenario:
# The macro extracts Float64 as the tracked type, but at runtime the data is
# Dual-typed. The typed-lazy path must correctly handle untracked Dual types.

@testset "24. Dual-like: typed-lazy with Dual as untracked type" begin
    pool = AdaptiveArrayPool()

    # Typed-lazy tracking Float64 — Dual is untracked (goes through has_others)
    _typed_lazy_checkpoint!(pool, Float64)
    acquire!(pool, Float64, 10)       # tracked type
    acquire!(pool, Dual_f1_11, 44)    # untracked Dual
    @test others_n_active(pool, Dual_f1_11) == 1
    @test pool._touched_has_others[pool._current_depth] == true

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test pool.float64.n_active == 0
    @test others_n_active(pool, Dual_f1_11) == 0
end

@testset "24b. Dual-like: typed-lazy nested, parent has Dual, child adds more" begin
    pool = AdaptiveArrayPool()

    # Pre-populate Dual pool
    checkpoint!(pool)
    acquire!(pool, Dual_f1_11, 10)
    rewind!(pool)

    # Parent: full checkpoint, acquires Dual
    checkpoint!(pool)
    acquire!(pool, Dual_f1_11, 5)
    @test others_n_active(pool, Dual_f1_11) == 1

    # Child: typed-lazy (Float64), helper acquires more Dual
    _typed_lazy_checkpoint!(pool, Float64)
    acquire!(pool, Float64, 10)
    acquire!(pool, Dual_f1_11, 44)
    @test others_n_active(pool, Dual_f1_11) == 2

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test others_n_active(pool, Dual_f1_11) == 1  # parent's preserved

    rewind!(pool)
    @test others_n_active(pool, Dual_f1_11) == 0
end

@testset "24c. Dual-like: typed-lazy with Dual as TRACKED type via eltype" begin
    pool = AdaptiveArrayPool()

    # Simulates: @with_pool pool function f(y::Vector{Dual})
    #   z = similar!(pool, y)  → macro extracts eltype(y) = Dual as tracked type
    # Since Dual is a fallback type, _tracked_mask_for_types(Dual) == UInt16(0)
    # The _can_use_typed_path check becomes:
    #   touched_mask & ~0 == 0 → always true IF no has_others set
    # But _checkpoint!(pool, Dual) does checkpoint the Dual pool.

    # This tests the typed path where the only tracked type is a fallback type
    checkpoint!(pool, Dual_f1_11)
    acquire!(pool, Dual_f1_11, 44)
    acquire!(pool, Dual_f1_11, 11)
    @test others_n_active(pool, Dual_f1_11) == 2

    rewind!(pool, Dual_f1_11)
    @test others_n_active(pool, Dual_f1_11) == 0
end

# ==============================================================================
# 25. Dual-Like: Stress Test — Simulates Full ForwardDiff.gradient Pipeline
# ==============================================================================
# Simulates: ForwardDiff.gradient(f, z) where f uses cubic_interp
# ForwardDiff calls f(z_dual) multiple times (one per chunk)
# Each call: @with_pool → acquire Dual arrays → rewind

@testset "25. Dual-like: full gradient simulation stress test" begin
    pool = AdaptiveArrayPool()
    n_chunks = 11

    function simulate_cubic_interp_dual!(pool)
        # Outer oneshot scope
        checkpoint!(pool)
        try
            partials = acquire!(pool, Dual_f1_11, 4 * 11)
            workspace = acquire!(pool, Float64, 10)  # spacing (Float64, not Dual)

            # Inner solver scope
            checkpoint!(pool)
            try
                m = acquire!(pool, Dual_f1_11, 11)
                # solve...
            finally
                rewind!(pool)
            end

            # Inner differentiation scope
            checkpoint!(pool)
            try
                line = acquire!(pool, Dual_f1_11, 11)
                dline = acquire!(pool, Dual_f1_11, 11)
                # compute derivatives...
            finally
                rewind!(pool)
            end

            return nothing
        finally
            rewind!(pool)
        end
    end

    for chunk in 1:n_chunks
        simulate_cubic_interp_dual!(pool)
    end

    # After all chunks: zero leak
    @test pool.float64.n_active == 0
    @test others_n_active(pool, Dual_f1_11) == 0

    # Checkpoint stack must be clean
    if haskey(pool.others, Dual_f1_11)
        dual_pool = pool.others[Dual_f1_11]
        @test length(dual_pool._checkpoint_depths) == 1  # sentinel only
        @test dual_pool._checkpoint_depths[1] == 0  # sentinel value
    end
end

@testset "25b. Dual-like: @with_pool stress with Dual types" begin
    # Warmup
    @with_pool pool begin
        acquire!(pool, Dual_f1_11, 44)
    end

    tl_pool = AdaptiveArrayPools.get_task_local_pool()
    baseline_vecs = length(tl_pool.others[Dual_f1_11].vectors)

    for _ in 1:200
        @with_pool pool begin
            acquire!(pool, Dual_f1_11, 44)
            acquire!(pool, Dual_f1_11, 11)
        end
    end

    tl_pool = AdaptiveArrayPools.get_task_local_pool()
    @test others_n_active(tl_pool, Dual_f1_11) == 0
    @test length(tl_pool.others[Dual_f1_11].vectors) <= baseline_vecs + 1
end

# ==============================================================================
# 26. Dual-Like: New Dual Type Created Mid-Scope (First-Touch Scenario)
# ==============================================================================
# When ForwardDiff first calls f(z_dual), the Dual type doesn't exist in pool.others yet.
# get_typed_pool! auto-checkpoints new types (pushes sentinel n_active=0).
# This tests that the auto-checkpoint + rewind cycle is correct for first-ever encounter.

@testset "26. Dual-like: first-touch auto-checkpoint in lazy mode" begin
    pool = AdaptiveArrayPool()
    # Pool has never seen Dual_f2_11 before
    @test !haskey(pool.others, Dual_f2_11)

    _lazy_checkpoint!(pool)
    # First-ever acquire of this Dual variant
    v = acquire!(pool, Dual_f2_11, 10)
    @test haskey(pool.others, Dual_f2_11)
    @test others_n_active(pool, Dual_f2_11) == 1

    _lazy_rewind!(pool)
    @test others_n_active(pool, Dual_f2_11) == 0
end

@testset "26b. Dual-like: first-touch in typed-lazy mode" begin
    pool = AdaptiveArrayPool()
    @test !haskey(pool.others, Dual_f2_11)

    _typed_lazy_checkpoint!(pool, Float64)
    v = acquire!(pool, Dual_f2_11, 10)
    @test others_n_active(pool, Dual_f2_11) == 1

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))
    @test others_n_active(pool, Dual_f2_11) == 0
end

@testset "26c. Dual-like: first-touch in typed (only) checkpoint — tracked fallback" begin
    pool = AdaptiveArrayPool()
    @test !haskey(pool.others, Dual_f2_11)

    # checkpoint!(pool, Dual_f2_11) creates the pool entry via get_typed_pool!
    # AND pushes checkpoint for it
    checkpoint!(pool, Dual_f2_11)
    v = acquire!(pool, Dual_f2_11, 10)
    @test others_n_active(pool, Dual_f2_11) == 1

    rewind!(pool, Dual_f2_11)
    @test others_n_active(pool, Dual_f2_11) == 0
end

# ==============================================================================
# 27. Dual-Like: Mixed Fixed + Multiple Dual Variants (Type Explosion)
# ==============================================================================

@testset "27. Type explosion: multiple Dual variants + fixed types" begin
    pool = AdaptiveArrayPool()

    checkpoint!(pool)
    acquire!(pool, Float64, 10)       # fixed slot
    acquire!(pool, Int32, 5)          # fixed slot
    acquire!(pool, Dual_f1_11, 44)    # fallback Dual variant 1
    acquire!(pool, Dual_f1_4, 16)     # fallback Dual variant 2
    acquire!(pool, Dual_f2_11, 33)    # fallback Dual variant 3
    acquire!(pool, UInt8, 20)         # fallback primitive

    @test pool.float64.n_active == 1
    @test pool.int32.n_active == 1
    @test others_n_active(pool, Dual_f1_11) == 1
    @test others_n_active(pool, Dual_f1_4) == 1
    @test others_n_active(pool, Dual_f2_11) == 1
    @test others_n_active(pool, UInt8) == 1

    rewind!(pool)

    @test pool.float64.n_active == 0
    @test pool.int32.n_active == 0
    @test others_n_active(pool, Dual_f1_11) == 0
    @test others_n_active(pool, Dual_f1_4) == 0
    @test others_n_active(pool, Dual_f2_11) == 0
    @test others_n_active(pool, UInt8) == 0
end

# ==============================================================================
# 28. Dual-Like: _acquire_impl! Bypass (Macro Transform Path)
# ==============================================================================
# When @with_pool transforms acquire! → _acquire_impl!, _record_type_touch!
# is bypassed. This tests that fallback types still get properly rewound
# even when type touch recording is skipped.

@testset "28. _acquire_impl! bypass: fallback type with typed checkpoint" begin
    pool = AdaptiveArrayPool()
    using AdaptiveArrayPools: _acquire_impl!

    # Simulate typed path: checkpoint specific type, use _acquire_impl! directly
    # checkpoint!(pool, Dual) creates the pool entry via get_typed_pool! which
    # now sets has_others=true at creation time
    checkpoint!(pool, Dual_f1_11)
    # _acquire_impl! does NOT call _record_type_touch!
    v = _acquire_impl!(pool, Dual_f1_11, 10)
    @test others_n_active(pool, Dual_f1_11) == 1

    rewind!(pool, Dual_f1_11)
    @test others_n_active(pool, Dual_f1_11) == 0
end

@testset "28b. _acquire_impl! bypass: typed-lazy fallback to lazy rewind" begin
    pool = AdaptiveArrayPool()
    using AdaptiveArrayPools: _acquire_impl!

    # Typed-lazy tracking Float64, Dual acquired via _acquire_impl! (no touch)
    _typed_lazy_checkpoint!(pool, Float64)

    # Tracked type via _acquire_impl!
    _acquire_impl!(pool, Float64, 10)

    # Untracked Dual via _acquire_impl! — NO touch recording!
    # But get_typed_pool! now sets _touched_has_others when creating new fallback type
    _acquire_impl!(pool, Dual_f1_11, 44)
    @test others_n_active(pool, Dual_f1_11) == 1

    # has_others should now be true (set by get_typed_pool! on first creation)
    @test pool._touched_has_others[pool._current_depth] == true

    _typed_lazy_rewind!(pool, _tracked_mask_for_types(Float64))

    # Fix: get_typed_pool! sets has_others → rewind iterates pool.others → n_active restored
    @test others_n_active(pool, Dual_f1_11) == 0
end

end  # top-level @testset
