import AdaptiveArrayPools: _check_wrapper_mutation!

@testset "Runtime Structural Mutation Detection (Metal)" begin

    # ==============================================================================
    # MetalTypedPool: DataRef reallocation detection
    # ==============================================================================

    @testset "resize! reallocation detected (MetalTypedPool)" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        Metal.fill!(v, 1.0f0)

        # Resize far beyond capacity → forces new GPU buffer allocation (new DataRef)
        resize!(v, 100_000)

        # rewind! should emit a warning about structural mutation
        @test_logs (:warn, r"resize!.*reallocation") rewind!(pool)
    end

    @testset "no mutation → no warning (MetalTypedPool)" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 100)
        Metal.fill!(v, 42.0f0)  # element mutation is fine

        # rewind! should NOT emit any mutation warning
        @test_logs rewind!(pool)
    end

    @testset "R=0 → no check (MetalTypedPool)" begin
        pool = _make_metal_pool(0)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)
        resize!(v, 100_000)

        # R=0: no invalidation, no mutation check, no warning
        @test_logs rewind!(pool)
    end

    # ==============================================================================
    # MetalTypedPool: wrapper length exceeds backing vector
    # ==============================================================================

    @testset "wrapper length > backing detected (MetalTypedPool)" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, Float32, 10)

        # Get the TypedPool internals
        tp = pool.float32
        vec = tp.vectors[tp.n_active]
        vec_len = length(vec)

        # Look up cached 1D wrapper and artificially inflate its dims
        wrappers_1d = 1 <= length(tp.arr_wrappers) ? tp.arr_wrappers[1] : nothing
        if wrappers_1d !== nothing && tp.n_active <= length(wrappers_1d)
            wrapper = wrappers_1d[tp.n_active]
            if wrapper !== nothing
                mtl = wrapper::MtlArray
                # Artificially set wrapper dims larger than backing
                setfield!(mtl, :dims, (vec_len + 100,))
                @test_logs (:warn, r"grew beyond backing") rewind!(pool)
            else
                rewind!(pool)  # no wrapper cached yet, skip
            end
        else
            rewind!(pool)
        end
    end

    # ==============================================================================
    # N-D wrapper mutation detection
    # ==============================================================================

    @testset "N-D wrapper mutation detected (MetalTypedPool)" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        mat = acquire!(pool, Float32, 10, 10)  # 100 elements
        Metal.fill!(mat, 1.0f0)

        # Get the 2D wrapper and manually break its dims
        tp = pool.float32
        wrappers_2d = length(tp.arr_wrappers) >= 2 ? tp.arr_wrappers[2] : nothing
        if wrappers_2d !== nothing && tp.n_active <= length(wrappers_2d)
            wrapper = wrappers_2d[tp.n_active]
            if wrapper !== nothing
                mtl = wrapper::MtlArray
                # Artificially set wrapper dims to something huge
                setfield!(mtl, :dims, (1000, 1000))
                @test_logs (:warn, r"grew beyond backing") rewind!(pool)
            else
                rewind!(pool)
            end
        else
            rewind!(pool)
        end
    end

    # ==============================================================================
    # Multiple slots: only first mutation triggers warning
    # ==============================================================================

    @testset "multiple slots: one warning per rewind" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v1 = acquire!(pool, Float32, 10)
        v2 = acquire!(pool, Float32, 10)
        resize!(v1, 100_000)
        resize!(v2, 100_000)

        # Should only warn once (first detected mutation), not twice
        @test_logs (:warn, r"resize!.*reallocation") rewind!(pool)
    end

    # ==============================================================================
    # Fallback type (pool.others) mutation detection
    # ==============================================================================

    @testset "resize! reallocation detected (others type)" begin
        pool = _make_metal_pool(1)
        checkpoint!(pool)
        v = acquire!(pool, UInt8, 16)
        Metal.fill!(v, UInt8(1))

        # Resize far beyond capacity
        resize!(v, 100_000)

        @test_logs (:warn, r"resize!.*reallocation") rewind!(pool)
    end

end
