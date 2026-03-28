import AdaptiveArrayPools: _make_pool, _check_wrapper_mutation!

@testset "Runtime Structural Mutation Detection" begin

    # ==============================================================================
    # TypedPool: MemoryRef reallocation detection
    # ==============================================================================

    @testset "resize! reallocation detected (TypedPool)" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 1.0

        # Resize far beyond capacity → forces reallocation (new Memory)
        resize!(v, 10_000)

        # rewind! should emit a warning about structural mutation
        @test_logs (:warn, r"wrapper reallocation detected") rewind!(pool)
    end

    @testset "push! reallocation detected (TypedPool)" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        v .= 1.0

        # push! enough elements to force reallocation
        for _ in 1:10_000
            push!(v, 99.0)
        end

        @test_logs (:warn, r"wrapper reallocation detected") rewind!(pool)
    end

    @testset "no mutation → no warning (TypedPool)" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 100)
        v .= 42.0  # element mutation is fine

        # rewind! should NOT emit any mutation warning
        @test_logs rewind!(pool)
    end

    @testset "S=0 → no check (TypedPool)" begin
        pool = _make_pool(false)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)
        resize!(v, 10_000)

        # S=0: no invalidation, no mutation check, no warning
        @test_logs rewind!(pool)
    end

    # ==============================================================================
    # TypedPool: setfield!(:size) does NOT trigger warning (Check 2 removed)
    # When pointers match, wrapper and backing share Memory — size inflation
    # via setfield! is not a real-world scenario (internal API only).
    # ==============================================================================

    @testset "setfield!(:size) inflation — no warning (TypedPool, Check 2 removed)" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v = acquire!(pool, Float64, 10)

        tp = AdaptiveArrayPools.get_typed_pool!(pool, Float64)
        vec = tp.vectors[tp.n_active]
        vec_len = length(vec)

        wrappers_1d = tp.arr_wrappers[1]
        if wrappers_1d !== nothing && tp.n_active <= length(wrappers_1d)
            wrapper = wrappers_1d[tp.n_active]
            if wrapper !== nothing
                arr = wrapper::Array{Float64}
                # Inflate wrapper size — pointer still matches → no warning
                setfield!(arr, :size, (vec_len + 100,))
                @test_logs rewind!(pool)  # no warning expected
            else
                rewind!(pool)
            end
        else
            rewind!(pool)
        end
    end

    # ==============================================================================
    # N-D Array: pointer-based reallocation detection
    # ==============================================================================

    @testset "N-D wrapper pointer-based detection" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        mat = acquire!(pool, Float64, 10, 10)  # 100 elements
        mat .= 1.0

        # Get the 2D wrapper and break its pointer via setfield!(:ref)
        tp = AdaptiveArrayPools.get_typed_pool!(pool, Float64)
        wrappers_2d = length(tp.arr_wrappers) >= 2 ? tp.arr_wrappers[2] : nothing
        if wrappers_2d !== nothing && tp.n_active <= length(wrappers_2d)
            wrapper = wrappers_2d[tp.n_active]
            if wrapper !== nothing
                arr = wrapper::Array{Float64}
                # Create a fresh vector and break pointer identity
                fresh = Vector{Float64}(undef, 200)
                setfield!(arr, :ref, getfield(fresh, :ref))
                setfield!(arr, :size, (10, 10))  # keep original dims so prod != 0
                @test_logs (:warn, r"reallocation detected") rewind!(pool)
            else
                rewind!(pool)
            end
        else
            rewind!(pool)
        end
    end

    # ==============================================================================
    # BitTypedPool: chunks reallocation detection
    # ==============================================================================

    @testset "resize! detected (BitTypedPool)" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        bv = acquire!(pool, Bit, 64)
        bv .= true

        # Resize far beyond → wrapper length diverges from backing
        # (BitVector resize! grows chunks in-place, so chunks identity stays the same;
        #  detection is via wrapper.len > backing.len, not chunks reallocation)
        resize!(bv, 100_000)

        @test_logs (:warn, r"wrapper grew beyond backing") rewind!(pool)
    end

    @testset "no mutation → no warning (BitTypedPool)" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        bv = acquire!(pool, Bit, 100)
        bv .= false  # element mutation is fine

        @test_logs rewind!(pool)
    end

    # ==============================================================================
    # Multiple slots: only first mutation triggers warning
    # ==============================================================================

    @testset "multiple slots: one warning per rewind" begin
        pool = _make_pool(true)
        checkpoint!(pool)
        v1 = acquire!(pool, Float64, 10)
        v2 = acquire!(pool, Float64, 10)
        resize!(v1, 10_000)
        resize!(v2, 10_000)

        # Should only warn once (first detected mutation), not twice
        @test_logs (:warn, r"wrapper reallocation detected") rewind!(pool)
    end
end
