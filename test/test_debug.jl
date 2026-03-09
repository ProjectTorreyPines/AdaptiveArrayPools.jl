import AdaptiveArrayPools: _validate_pool_return, _check_bitchunks_overlap, _eltype_may_contain_arrays, PoolRuntimeEscapeError

@testset "POOL_DEBUG and Safety Validation" begin

    # ==============================================================================
    # POOL_DEBUG flag toggle
    # ==============================================================================

    @testset "POOL_DEBUG flag" begin
        old_debug = POOL_DEBUG[]

        # Default is false
        POOL_DEBUG[] = false

        # When debug is off, no validation happens even if SubArray escapes
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            identity(v)  # compile-time suppressed; runtime LV<2 won't catch
        end
        @test result isa SubArray  # No error when debug is off

        POOL_DEBUG[] = old_debug
    end

    @testset "POOL_DEBUG with safety violation" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # Should throw error when returning SubArray with debug on
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            identity(v)  # compile-time suppressed; caught by runtime LV2
        end

        # Safe returns should work fine
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            sum(v)  # Safe: returning scalar
        end
        @test result == 10.0

        # Returning a copy is also safe
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 5)
            v .= 2.0
            collect(v)  # Safe: returning a copy
        end
        @test result == [2.0, 2.0, 2.0, 2.0, 2.0]

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # _validate_pool_return — direct tests
    # ==============================================================================

    @testset "_validate_pool_return" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Non-SubArray values pass validation
        _validate_pool_return(42, pool)
        _validate_pool_return([1, 2, 3], pool)
        _validate_pool_return("hello", pool)
        _validate_pool_return(nothing, pool)

        # SubArray not from pool passes validation
        external_vec = [1.0, 2.0, 3.0]
        external_view = view(external_vec, 1:2)
        _validate_pool_return(external_view, pool)

        # SubArray from pool fails validation (fixed slot)
        pool_view = acquire!(pool, Float64, 10)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(pool_view, pool)

        rewind!(pool)

        # Test with fallback type (others)
        checkpoint!(pool)
        pool_view_uint8 = acquire!(pool, UInt8, 10)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(pool_view_uint8, pool)
        rewind!(pool)

        # DisabledPool always passes
        _validate_pool_return(pool_view, DISABLED_CPU)
        _validate_pool_return(42, DISABLED_CPU)
    end

    @testset "_validate_pool_return with all fixed slots" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Test each fixed slot type
        v_f64 = acquire!(pool, Float64, 5)
        v_f32 = acquire!(pool, Float32, 5)
        v_i64 = acquire!(pool, Int64, 5)
        v_i32 = acquire!(pool, Int32, 5)
        v_c64 = acquire!(pool, ComplexF64, 5)
        v_c32 = acquire!(pool, ComplexF32, 5)
        v_bool = acquire!(pool, Bool, 5)

        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_f64, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_f32, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_i64, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_i32, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_c64, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_c32, pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_bool, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with N-D arrays" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # N-D ReshapedArray from pool should fail validation (pointer overlap check)
        mat = acquire!(pool, Float64, 10, 10)
        @test mat isa Base.ReshapedArray{Float64, 2}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(mat, pool)

        # 3D ReshapedArray should also fail
        tensor = acquire!(pool, Float64, 5, 5, 5)
        @test tensor isa Base.ReshapedArray{Float64, 3}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(tensor, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with unsafe_acquire!" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Raw Vector from unsafe_acquire! should fail validation
        v = unsafe_acquire!(pool, Float64, 100)
        @test v isa Vector{Float64}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v, pool)

        # Raw Matrix from unsafe_acquire! should fail validation
        mat = unsafe_acquire!(pool, Float64, 10, 10)
        @test mat isa Matrix{Float64}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(mat, pool)

        # Raw 3D Array from unsafe_acquire! should fail validation
        tensor = unsafe_acquire!(pool, Float64, 5, 5, 5)
        @test tensor isa Array{Float64, 3}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(tensor, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with view(unsafe_acquire!)" begin
        # Bug fix test: view() wrapped around unsafe_acquire! result
        # The parent Vector/Array is created by unsafe_wrap, not the pool's internal vector
        # This requires pointer overlap check, not identity check
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D: view(unsafe_acquire!(...), :) should fail validation
        v = unsafe_acquire!(pool, Float64, 100)
        v_view = view(v, :)
        @test v_view isa SubArray
        @test parent(v_view) === v  # Parent is unsafe_wrap'd Vector, not pool's internal vector
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_view, pool)

        # Partial view should also fail
        v_partial = view(v, 1:50)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_partial, pool)

        # 2D: view(unsafe_acquire!(...), :, :) should fail validation
        mat = unsafe_acquire!(pool, Float64, 10, 10)
        mat_view = view(mat, :, :)
        @test mat_view isa SubArray
        @test_throws PoolRuntimeEscapeError _validate_pool_return(mat_view, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return external arrays pass" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Acquire some memory to populate the pool
        _ = acquire!(pool, Float64, 100)

        # External N-D arrays should pass validation
        external_mat = zeros(Float64, 10, 10)
        external_view = view(external_mat, :, :)
        _validate_pool_return(external_view, pool)
        _validate_pool_return(external_mat, pool)

        # External 3D array should pass
        external_tensor = zeros(Float64, 5, 5, 5)
        _validate_pool_return(external_tensor, pool)

        rewind!(pool)
    end

    @testset "POOL_DEBUG with N-D arrays" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # N-D ReshapedArray should throw error when returned
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            mat = acquire!(pool, Float64, 10, 10)
            @skip_check_vars mat
            identity(mat)  # compile-time suppressed; caught by runtime LV2
        end

        # Raw Array from unsafe_acquire! should throw error when returned
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            mat = unsafe_acquire!(pool, Float64, 10, 10)
            @skip_check_vars mat
            identity(mat)  # compile-time suppressed; caught by runtime LV2
        end

        # Safe returns should work fine
        result = @with_pool pool begin
            mat = acquire!(pool, Float64, 10, 10)
            mat .= 1.0
            sum(mat)  # Safe: returning scalar
        end
        @test result == 100.0

        # Returning a copy is also safe
        result = @with_pool pool begin
            mat = acquire!(pool, Float64, 3, 3)
            mat .= 2.0
            collect(mat)  # Safe: returning a copy
        end
        @test result == fill(2.0, 3, 3)

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # BitArray overlap detection (_check_bitchunks_overlap)
    # ==============================================================================

    @testset "_check_bitchunks_overlap - direct BitArray validation" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D BitVector from pool - should detect overlap
        bv = acquire!(pool, Bit, 100)
        @test bv isa BitVector
        @test_throws PoolRuntimeEscapeError _check_bitchunks_overlap(bv, pool)

        # N-D BitArray from pool - should detect overlap (shares chunks with pool)
        ba = acquire!(pool, Bit, 10, 10)
        @test ba isa BitMatrix
        @test_throws PoolRuntimeEscapeError _check_bitchunks_overlap(ba, pool)

        # 3D BitArray from pool
        ba3 = acquire!(pool, Bit, 4, 5, 3)
        @test ba3 isa BitArray{3}
        @test_throws PoolRuntimeEscapeError _check_bitchunks_overlap(ba3, pool)

        rewind!(pool)
    end

    @testset "_check_bitchunks_overlap - external BitArray passes" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Populate pool with some BitVectors
        _ = acquire!(pool, Bit, 100)
        _ = acquire!(pool, Bit, 200)

        # External BitVector (not from pool) should pass validation
        external_bv = BitVector(undef, 50)
        _check_bitchunks_overlap(external_bv, pool)  # Should not throw

        # External BitMatrix should pass
        external_ba = BitArray(undef, 10, 10)
        _check_bitchunks_overlap(external_ba, pool)  # Should not throw

        # External 3D BitArray should pass
        external_ba3 = BitArray(undef, 5, 5, 5)
        _check_bitchunks_overlap(external_ba3, pool)  # Should not throw

        rewind!(pool)
    end

    @testset "_validate_pool_return with BitArray (via _check_bitchunks_overlap)" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Direct BitVector from pool fails validation
        bv = acquire!(pool, Bit, 100)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(bv, pool)

        # Direct BitMatrix from pool fails validation
        ba = acquire!(pool, Bit, 10, 10)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(ba, pool)

        # External BitArray passes validation
        external_bv = BitVector(undef, 50)
        _validate_pool_return(external_bv, pool)  # Should not throw

        rewind!(pool)
    end

    @testset "_validate_pool_return with SubArray{BitArray} parent" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Create a view of a pool BitVector
        bv = acquire!(pool, Bit, 100)
        bv_view = view(bv, 1:50)
        @test bv_view isa SubArray
        @test parent(bv_view) isa BitVector
        @test_throws PoolRuntimeEscapeError _validate_pool_return(bv_view, pool)

        # View of external BitVector should pass
        external_bv = BitVector(undef, 100)
        external_view = view(external_bv, 1:50)
        _validate_pool_return(external_view, pool)  # Should not throw

        rewind!(pool)
    end

    @testset "POOL_DEBUG with BitArray" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # BitVector from pool should throw error when returned with debug on
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            bv = acquire!(pool, Bit, 100)
            @skip_check_vars bv
            identity(bv)  # compile-time suppressed; caught by runtime LV2
        end

        # BitMatrix from pool should throw error when returned
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            ba = acquire!(pool, Bit, 10, 10)
            @skip_check_vars ba
            identity(ba)  # compile-time suppressed; caught by runtime LV2
        end

        # Safe returns should work fine
        result = @with_pool pool begin
            bv = acquire!(pool, Bit, 100)
            bv .= true
            count(bv)  # Safe: returning scalar
        end
        @test result == 100

        # Returning a copy is also safe
        result = @with_pool pool begin
            bv = acquire!(pool, Bit, 5)
            bv .= true
            copy(bv)  # Safe: returning a copy
        end
        @test result == trues(5)

        POOL_DEBUG[] = old_debug
    end

    # ==============================================================================
    # POOL_DEBUG with function definition forms
    # ==============================================================================

    # ==============================================================================
    # _validate_pool_return — recursive container inspection (Tuple, NamedTuple, Pair)
    # ==============================================================================

    @testset "_validate_pool_return with Tuple" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Pool array inside tuple → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return((42, v), pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return((v,), pool)

        # Nested tuple: pool array deep inside → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return((1, (2, v)), pool)

        # Safe tuple (no pool arrays) → passes
        _validate_pool_return((1, 2, 3), pool)
        _validate_pool_return((1, "hello", [1,2,3]), pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with NamedTuple" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Pool array inside NamedTuple → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return((data=v, n=10), pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return((result=42, buffer=v), pool)

        # Nested: NamedTuple containing tuple with pool array
        @test_throws PoolRuntimeEscapeError _validate_pool_return((meta=(v, 1),), pool)

        # Safe NamedTuple → passes
        _validate_pool_return((a=1, b="hello"), pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with Pair" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Pool array as Pair value → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return(:data => v, pool)

        # Pool array as Pair key (unusual but possible) → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v => :data, pool)

        # Safe Pair → passes
        _validate_pool_return(:a => 42, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return recursive with mixed containers" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)
        bv = acquire!(pool, Bit, 50)

        # Tuple containing NamedTuple with pool array
        @test_throws PoolRuntimeEscapeError _validate_pool_return((1, (data=v,)), pool)

        # Pair inside tuple
        @test_throws PoolRuntimeEscapeError _validate_pool_return((:key => v, 42), pool)

        # BitVector inside tuple
        @test_throws PoolRuntimeEscapeError _validate_pool_return((bv, 1), pool)

        # Multiple pool arrays in different container positions
        @test_throws PoolRuntimeEscapeError _validate_pool_return((v, bv), pool)

        # N-D ReshapedArray inside NamedTuple
        mat = acquire!(pool, Float64, 5, 5)
        @test_throws PoolRuntimeEscapeError _validate_pool_return((matrix=mat, size=(5,5)), pool)

        rewind!(pool)
    end

    # ==============================================================================
    # _validate_pool_return — Dict, Set, and Vector-of-arrays container inspection
    # ==============================================================================

    @testset "_eltype_may_contain_arrays guard" begin
        @test _eltype_may_contain_arrays(Float64) == false
        @test _eltype_may_contain_arrays(Int32) == false
        @test _eltype_may_contain_arrays(ComplexF64) == false
        @test _eltype_may_contain_arrays(String) == false
        @test _eltype_may_contain_arrays(Symbol) == false
        @test _eltype_may_contain_arrays(Char) == false
        @test _eltype_may_contain_arrays(Any) == true
        @test _eltype_may_contain_arrays(SubArray) == true
        @test _eltype_may_contain_arrays(AbstractArray) == true
        @test _eltype_may_contain_arrays(Vector{Float64}) == true
    end

    @testset "_validate_pool_return with Dict" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Pool array as Dict value → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Dict(:data => v), pool)

        # Pool array as Dict key (unusual but possible) → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Dict(v => :data), pool)

        # Multiple pool arrays in Dict values
        w = acquire!(pool, Int64, 5)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Dict(:a => v, :b => w), pool)

        # Safe Dict → passes
        _validate_pool_return(Dict(:a => 1, :b => 2), pool)
        _validate_pool_return(Dict{String, Float64}("x" => 1.0), pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with nested Dict" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Dict inside Tuple → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return((1, Dict(:data => v)), pool)

        # Dict inside NamedTuple → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return((result=Dict(:buf => v),), pool)

        # Nested Dict (Dict of Dict) → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Dict(:outer => Dict(:inner => v)), pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with Set" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Pool array inside Set → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Set([v]), pool)

        # Safe Set → passes
        _validate_pool_return(Set([1, 2, 3]), pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with Vector-of-arrays (element recursion)" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Vector{SubArray} — pool array as element → caught
        external_container = Any[v]
        @test_throws PoolRuntimeEscapeError _validate_pool_return(external_container, pool)

        # Multiple pool arrays in Vector
        w = acquire!(pool, Int64, 5)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[v, w], pool)

        # Nested: Vector inside Tuple
        @test_throws PoolRuntimeEscapeError _validate_pool_return((42, Any[v]), pool)

        # Safe Vector{Float64} — passes (eltype guard skips element iteration)
        _validate_pool_return([1.0, 2.0, 3.0], pool)
        _validate_pool_return(zeros(1000), pool)  # large but still fast (eltype guard)

        # Vector{Any} with safe values — passes
        _validate_pool_return(Any[1, "hello", :sym], pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return Vector-of-arrays with unsafe_acquire!" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # unsafe_acquire! Array inside Vector → caught
        raw = unsafe_acquire!(pool, Float64, 100)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[raw], pool)

        # BitVector inside Vector → caught
        bv = acquire!(pool, Bit, 50)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[bv], pool)

        # ReshapedArray inside Vector → caught
        mat = acquire!(pool, Float64, 5, 5)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[mat], pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return containers via @with_pool macro (LV2)" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 2

        # Tuple containing pool array — caught at runtime
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            identity((sum(v), v))  # compile-time suppressed; runtime LV2 catches v inside tuple
        end

        # NamedTuple containing pool array — caught at runtime
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            identity((data=v, n=10))  # compile-time suppressed; runtime LV2 catches v inside NamedTuple
        end

        # Safe containers pass
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            v .= 3.0
            (sum(v), length(v))
        end
        @test result == (30.0, 10)

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # @skip_check_vars — compile-time suppression does NOT suppress runtime LV2
    # ==============================================================================

    @testset "@skip_check_vars suppresses compile-time but runtime LV2 still catches" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 2

        # @skip_check_vars prevents compile-time PoolEscapeError,
        # but runtime _validate_pool_return at LV2 still catches the escape.
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            identity(v)  # compile-time suppressed; runtime LV2 catches
        end

        # Multiple vars: suppress some, escape detection still works for suppressed ones at runtime
        @test_throws PoolRuntimeEscapeError @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            @skip_check_vars v w
            identity(v)
        end

        # Safe return still works with @skip_check_vars
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            v .= 1.0
            sum(v)  # scalar — safe
        end
        @test result == 10.0

        POOL_SAFETY_LV[] = old_safety
    end

    @testset "@skip_check_vars with LV1 (no runtime escape check)" begin
        old_safety = POOL_SAFETY_LV[]
        POOL_SAFETY_LV[] = 1

        # At LV1, @skip_check_vars suppresses compile-time and runtime doesn't check escapes
        # (only structural invalidation), so the SubArray escapes silently.
        result = @with_pool pool begin
            v = acquire!(pool, Float64, 10)
            @skip_check_vars v
            identity(v)
        end
        @test result isa SubArray  # Escapes — no runtime check at LV1

        POOL_SAFETY_LV[] = old_safety
    end

    # ==============================================================================
    # POOL_DEBUG with function definition forms
    # ==============================================================================

    @testset "POOL_DEBUG with @with_pool function definition" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # Unsafe: function returns pool-backed SubArray
        @with_pool pool function _test_debug_func_unsafe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            @skip_check_vars v
            identity(v)  # compile-time suppressed; caught by runtime LV2
        end
        @test_throws PoolRuntimeEscapeError _test_debug_func_unsafe(10)

        # Safe: function returns scalar
        @with_pool pool function _test_debug_func_safe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            sum(v)
        end
        @test _test_debug_func_safe(10) == 10.0

        # Safe: function returns a copy
        @with_pool pool function _test_debug_func_copy(n)
            v = acquire!(pool, Float64, n)
            v .= 2.0
            collect(v)
        end
        @test _test_debug_func_copy(5) == fill(2.0, 5)

        # Unsafe: N-D ReshapedArray from function
        @with_pool pool function _test_debug_func_nd(m, n)
            mat = acquire!(pool, Float64, m, n)
            mat .= 1.0
            @skip_check_vars mat
            identity(mat)  # compile-time suppressed; caught by runtime LV2
        end
        @test_throws PoolRuntimeEscapeError _test_debug_func_nd(3, 4)

        # Unsafe: BitVector from function
        @with_pool pool function _test_debug_func_bit(n)
            bv = acquire!(pool, Bit, n)
            bv .= true
            @skip_check_vars bv
            identity(bv)  # compile-time suppressed; caught by runtime LV2
        end
        @test_throws PoolRuntimeEscapeError _test_debug_func_bit(100)

        POOL_DEBUG[] = old_debug
    end

    @testset "POOL_DEBUG with @maybe_with_pool function definition" begin
        old_debug = POOL_DEBUG[]
        old_maybe = MAYBE_POOLING[]
        POOL_DEBUG[] = true
        MAYBE_POOLING[] = true

        # Unsafe: function returns pool-backed array
        @maybe_with_pool pool function _test_maybe_debug_unsafe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            @skip_check_vars v
            identity(v)  # compile-time suppressed; caught by runtime LV2
        end
        @test_throws PoolRuntimeEscapeError _test_maybe_debug_unsafe(10)

        # Safe: function returns scalar
        @maybe_with_pool pool function _test_maybe_debug_safe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            sum(v)
        end
        @test _test_maybe_debug_safe(10) == 10.0

        # When pooling disabled, no validation needed (DisabledPool returns fresh arrays)
        MAYBE_POOLING[] = false
        @maybe_with_pool pool function _test_maybe_debug_disabled(n)
            v = zeros!(pool, n)
            @skip_check_vars v
            identity(v)  # compile-time suppressed; disabled pool returns fresh arrays
        end
        result = _test_maybe_debug_disabled(5)
        @test result == zeros(5)

        POOL_DEBUG[] = old_debug
        MAYBE_POOLING[] = old_maybe
    end

    @testset "POOL_DEBUG with @with_pool :cpu function definition" begin
        old_debug = POOL_DEBUG[]
        POOL_DEBUG[] = true

        # Unsafe: backend function returns pool-backed array
        @with_pool :cpu pool function _test_backend_debug_unsafe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            @skip_check_vars v
            identity(v)  # compile-time suppressed; caught by runtime LV2
        end
        @test_throws PoolRuntimeEscapeError _test_backend_debug_unsafe(10)

        # Safe: returns scalar
        @with_pool :cpu pool function _test_backend_debug_safe(n)
            v = acquire!(pool, Float64, n)
            v .= 1.0
            sum(v)
        end
        @test _test_backend_debug_safe(10) == 10.0

        POOL_DEBUG[] = old_debug
    end

end # POOL_DEBUG and Safety Validation
