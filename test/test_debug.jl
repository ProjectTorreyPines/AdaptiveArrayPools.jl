import AdaptiveArrayPools: _validate_pool_return, _check_bitchunks_overlap, _eltype_may_contain_arrays,
    PoolRuntimeEscapeError, _poison_value, _shorten_location,
    _make_pool, _lazy_checkpoint!, _lazy_rewind!
_test_leak(x) = x  # opaque to compile-time escape checker (only identity() is transparent)

@testset "Safety Validation" begin

    # ==============================================================================
    # _validate_pool_return — direct tests
    # ==============================================================================

    @testset "_validate_pool_return" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Non-Array values pass validation
        _validate_pool_return(42, pool)
        _validate_pool_return([1, 2, 3], pool)
        _validate_pool_return("hello", pool)
        _validate_pool_return(nothing, pool)

        # SubArray not from pool passes validation
        external_vec = [1.0, 2.0, 3.0]
        external_view = view(external_vec, 1:2)
        _validate_pool_return(external_view, pool)

        # Array from pool fails validation (fixed slot)
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

        # N-D Array from pool should fail validation (pointer overlap check)
        mat = acquire!(pool, Float64, 10, 10)
        @test mat isa Matrix{Float64}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(mat, pool)

        # 3D Array should also fail
        tensor = acquire!(pool, Float64, 5, 5, 5)
        @test tensor isa Array{Float64, 3}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(tensor, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with acquire! (all dimensionalities)" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # Vector from acquire! should fail validation
        v = acquire!(pool, Float64, 100)
        @test v isa Vector{Float64}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v, pool)

        # Matrix from acquire! should fail validation
        mat = acquire!(pool, Float64, 10, 10)
        @test mat isa Matrix{Float64}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(mat, pool)

        # 3D Array from acquire! should fail validation
        tensor = acquire!(pool, Float64, 5, 5, 5)
        @test tensor isa Array{Float64, 3}
        @test_throws PoolRuntimeEscapeError _validate_pool_return(tensor, pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with view(acquire!)" begin
        # Bug fix test: view() wrapped around acquire! result
        # The parent Array is from the pool's internal storage
        # This requires pointer overlap check, not identity check
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D: view(acquire!(...), :) should fail validation
        v = acquire!(pool, Float64, 100)
        v_view = view(v, :)
        @test v_view isa SubArray
        @test parent(v_view) === v  # Parent is unsafe_wrap'd Vector, not pool's internal vector
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_view, pool)

        # Partial view should also fail
        v_partial = view(v, 1:50)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(v_partial, pool)

        # 2D: view(acquire!(...), :, :) should fail validation
        mat = acquire!(pool, Float64, 10, 10)
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

    @testset "Pool{1} escape detection with N-D arrays" begin
        # N-D Array should throw error when returned
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            mat = acquire!(pool, Float64, 10, 10)
            _validate_pool_return(_test_leak(mat), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # Another acquire! Array should also throw error when returned
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            mat = acquire!(pool, Float64, 10, 10)
            _validate_pool_return(_test_leak(mat), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # Safe returns should work fine
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        result = try
            mat = acquire!(pool, Float64, 10, 10)
            mat .= 1.0
            sum(mat)  # Safe: returning scalar
        finally
            _lazy_rewind!(pool)
        end
        @test result == 100.0

        # Returning a copy is also safe
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        result = try
            mat = acquire!(pool, Float64, 3, 3)
            mat .= 2.0
            collect(mat)  # Safe: returning a copy
        finally
            _lazy_rewind!(pool)
        end
        @test result == fill(2.0, 3, 3)
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

    @testset "Pool{1} escape detection with BitArray" begin
        # BitVector from pool should throw error when returned
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            bv = acquire!(pool, Bit, 100)
            _validate_pool_return(_test_leak(bv), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # BitMatrix from pool should throw error when returned
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            ba = acquire!(pool, Bit, 10, 10)
            _validate_pool_return(_test_leak(ba), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # Safe returns should work fine
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        result = try
            bv = acquire!(pool, Bit, 100)
            bv .= true
            count(bv)  # Safe: returning scalar
        finally
            _lazy_rewind!(pool)
        end
        @test result == 100

        # Returning a copy is also safe
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        result = try
            bv = acquire!(pool, Bit, 5)
            bv .= true
            copy(bv)  # Safe: returning a copy
        finally
            _lazy_rewind!(pool)
        end
        @test result == trues(5)
    end

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
        _validate_pool_return((1, "hello", [1, 2, 3]), pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return with NamedTuple" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v = acquire!(pool, Float64, 10)

        # Pool array inside NamedTuple → caught
        @test_throws PoolRuntimeEscapeError _validate_pool_return((data = v, n = 10), pool)
        @test_throws PoolRuntimeEscapeError _validate_pool_return((result = 42, buffer = v), pool)

        # Nested: NamedTuple containing tuple with pool array
        @test_throws PoolRuntimeEscapeError _validate_pool_return((meta = (v, 1),), pool)

        # Safe NamedTuple → passes
        _validate_pool_return((a = 1, b = "hello"), pool)

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
        @test_throws PoolRuntimeEscapeError _validate_pool_return((1, (data = v,)), pool)

        # Pair inside tuple
        @test_throws PoolRuntimeEscapeError _validate_pool_return((:key => v, 42), pool)

        # BitVector inside tuple
        @test_throws PoolRuntimeEscapeError _validate_pool_return((bv, 1), pool)

        # Multiple pool arrays in different container positions
        @test_throws PoolRuntimeEscapeError _validate_pool_return((v, bv), pool)

        # N-D Array inside NamedTuple
        mat = acquire!(pool, Float64, 5, 5)
        @test_throws PoolRuntimeEscapeError _validate_pool_return((matrix = mat, size = (5, 5)), pool)

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
        @test_throws PoolRuntimeEscapeError _validate_pool_return((result = Dict(:buf => v),), pool)

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

        # Vector{Any} — pool array as element → caught
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

    @testset "_validate_pool_return Vector-of-arrays with acquire!" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # acquire! Array inside Vector → caught
        raw = acquire!(pool, Float64, 100)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[raw], pool)

        # BitVector inside Vector → caught
        bv = acquire!(pool, Bit, 50)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[bv], pool)

        # N-D Array inside Vector → caught
        mat = acquire!(pool, Float64, 5, 5)
        @test_throws PoolRuntimeEscapeError _validate_pool_return(Any[mat], pool)

        rewind!(pool)
    end

    @testset "_validate_pool_return containers via Pool{1} (direct validation)" begin
        # Tuple containing pool array — caught
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak((sum(v), v)), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # NamedTuple containing pool array — caught
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak((data = v, n = 10)), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # Safe containers pass
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        result = try
            v = acquire!(pool, Float64, 10)
            v .= 3.0
            (sum(v), length(v))
        finally
            _lazy_rewind!(pool)
        end
        @test result == (30.0, 10)
    end

    # ==============================================================================
    # Pool{1} escape detection through opaque function calls (direct validation)
    # ==============================================================================

    @testset "Pool{1} catches escapes through direct validation" begin
        # Opaque function call bypasses compile-time PoolEscapeError,
        # but direct _validate_pool_return on Pool{1} still catches the escape.
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            v = acquire!(pool, Float64, 10)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # Multiple vars: validation still catches escape
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        err = try
            v = acquire!(pool, Float64, 10)
            w = acquire!(pool, Float64, 5)
            _validate_pool_return(_test_leak(v), pool)
            nothing
        catch e
            e
        finally
            _lazy_rewind!(pool)
        end
        @test err isa PoolRuntimeEscapeError

        # Safe return works fine
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        result = try
            v = acquire!(pool, Float64, 10)
            v .= 1.0
            sum(v)  # scalar — safe
        finally
            _lazy_rewind!(pool)
        end
        @test result == 10.0
    end

    # ==============================================================================
    # Coverage: PoolRuntimeEscapeError showerror with return_site
    # ==============================================================================

    @testset "PoolRuntimeEscapeError showerror with return_site" begin
        # Construct error with both callsite and return_site to cover lines 169-180
        err = PoolRuntimeEscapeError(
            "Vector{Float64}",
            "Float64",
            "test.jl:10\nacquire!(pool, Float64, 10)",
            "test.jl:15\nreturn v"
        )
        msg = sprint(showerror, err)
        @test occursin("escapes at", msg)
        @test occursin("return v", msg)
        @test occursin("acquired at", msg)

        # Return site without expression text (no \n)
        err2 = PoolRuntimeEscapeError(
            "Vector{Float64}",
            "Float64",
            "test.jl:10",
            "test.jl:15"
        )
        msg2 = sprint(showerror, err2)
        @test occursin("escapes at", msg2)
    end

    # ==============================================================================
    # Coverage: PoolRuntimeEscapeError 3-arg showerror (backtrace suppression)
    # ==============================================================================

    @testset "PoolRuntimeEscapeError 3-arg showerror" begin
        err = PoolRuntimeEscapeError("Vector{Float64}", "Float64", nothing, nothing)
        msg2 = sprint(showerror, err)
        msg3 = sprint() do io
            showerror(io, err, nothing)
        end
        @test msg2 == msg3
    end

    # ==============================================================================
    # Coverage: _poison_value generic fallback (line 258)
    # ==============================================================================

    @testset "_poison_value generic fallback" begin
        # Rational is not AbstractFloat, Integer, or Complex → hits generic fallback
        @test _poison_value(Rational{Int}) == zero(Rational{Int})

        # Exercise through actual pool rewind at S=1 with a non-fixed-slot type
        pool = _make_pool(true)
        _lazy_checkpoint!(pool)
        v = acquire!(pool, Rational{Int}, 5)
        v .= 1 // 3
        _lazy_rewind!(pool)  # triggers _poison_fill! → _poison_value(Rational{Int}) → zero(Rational)
    end

    # ==============================================================================
    # Coverage: _shorten_location no-colon fallback (line 304)
    # ==============================================================================

    @testset "_shorten_location edge cases" begin
        # Location without colon → returned as-is
        @test _shorten_location("nocolon") == "nocolon"
        # Location with colon → shortened
        loc = _shorten_location("somefile.jl:42")
        @test occursin("42", loc)
    end

end # Safety Validation
