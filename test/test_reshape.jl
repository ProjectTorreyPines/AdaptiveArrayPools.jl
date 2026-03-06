# ==============================================================================
# Tests for reshape! — Pool-based zero-allocation array reshaping
# ==============================================================================

@testset "reshape!" begin

    # ==========================================================================
    # Basic reshape (cross-dimensional)
    # ==========================================================================

    @testset "Basic reshape (cross-dim)" begin
        pool = AdaptiveArrayPool()

        # 1D → 2D
        A = collect(1.0:12.0)
        checkpoint!(pool)
        B = reshape!(pool, A, 3, 4)
        @test size(B) == (3, 4)
        @test eltype(B) == Float64
        @test B[1, 1] == 1.0
        @test B[3, 4] == 12.0
        rewind!(pool)

        # 2D → 1D
        A2d = reshape(collect(1.0:12.0), 3, 4)
        checkpoint!(pool)
        C = reshape!(pool, A2d, 12)
        @test size(C) == (12,)
        @test C[1] == 1.0
        @test C[12] == 12.0
        rewind!(pool)

        # 1D → 3D
        checkpoint!(pool)
        D = reshape!(pool, A, 2, 3, 2)
        @test size(D) == (2, 3, 2)
        @test D[1, 1, 1] == 1.0
        @test D[2, 3, 2] == 12.0
        rewind!(pool)
    end

    # ==========================================================================
    # Same-dim reshape
    # ==========================================================================

    @testset "Same-dim reshape" begin
        pool = AdaptiveArrayPool()
        A2d = reshape(collect(1.0:12.0), 3, 4)

        checkpoint!(pool)
        E = reshape!(pool, A2d, 4, 3)
        @test size(E) == (4, 3)
        @test length(E) == 12
        rewind!(pool)
    end

    # ==========================================================================
    # Data preservation / memory sharing
    # ==========================================================================

    @testset "Data preservation / memory sharing" begin
        pool = AdaptiveArrayPool()
        A = collect(1.0:12.0)

        checkpoint!(pool)
        B = reshape!(pool, A, 3, 4)

        # Data identity
        @test vec(B) == A

        # Mutation in B visible in A
        B[1, 1] = 999.0
        @test A[1] == 999.0

        # Mutation in A visible in B
        A[12] = -1.0
        @test B[3, 4] == -1.0
        rewind!(pool)
    end

    # ==========================================================================
    # DimensionMismatch
    # ==========================================================================

    @testset "DimensionMismatch" begin
        pool = AdaptiveArrayPool()
        A = collect(1.0:12.0)
        @test_throws DimensionMismatch reshape!(pool, A, 5, 5)
        @test_throws DimensionMismatch reshape!(pool, A, (7, 3))
    end

    # ==========================================================================
    # Tuple and vararg syntax
    # ==========================================================================

    @testset "Tuple and vararg syntax" begin
        pool = AdaptiveArrayPool()
        A = collect(1.0:12.0)

        checkpoint!(pool)
        B_vararg = reshape!(pool, A, 3, 4)
        @test size(B_vararg) == (3, 4)
        rewind!(pool)

        checkpoint!(pool)
        B_tuple = reshape!(pool, A, (3, 4))
        @test size(B_tuple) == (3, 4)
        rewind!(pool)
    end

    # ==========================================================================
    # Multiple element types
    # ==========================================================================

    @testset "Multiple element types" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        A_f64 = collect(1.0:6.0)
        B_f64 = reshape!(pool, A_f64, 2, 3)
        @test eltype(B_f64) == Float64
        @test size(B_f64) == (2, 3)

        A_i32 = Int32.(1:6)
        B_i32 = reshape!(pool, A_i32, 3, 2)
        @test eltype(B_i32) == Int32
        @test size(B_i32) == (3, 2)

        A_bool = Bool[true, false, true, false]
        B_bool = reshape!(pool, A_bool, 2, 2)
        @test eltype(B_bool) == Bool
        @test size(B_bool) == (2, 2)

        rewind!(pool)
    end

    # ==========================================================================
    # checkpoint!/rewind! integration
    # ==========================================================================

    @testset "checkpoint!/rewind! integration" begin
        pool = AdaptiveArrayPool()
        A = collect(1.0:12.0)

        tp = get_typed_pool!(pool, Float64)
        n_before = tp.n_active

        checkpoint!(pool)
        B = reshape!(pool, A, 3, 4)
        @test size(B) == (3, 4)

        rewind!(pool)
        @test tp.n_active == n_before  # slot reclaimed
    end

    # ==========================================================================
    # @with_pool integration
    # ==========================================================================

    @testset "@with_pool integration" begin
        A = collect(1.0:12.0)
        result = @with_pool pool begin
            B = reshape!(pool, A, 3, 4)
            sum(B)
        end
        @test result == sum(1.0:12.0)
    end

    # ==========================================================================
    # External arrays (not from pool)
    # ==========================================================================

    @testset "External arrays" begin
        pool = AdaptiveArrayPool()
        A = rand(6)

        checkpoint!(pool)
        B = reshape!(pool, A, 2, 3)
        @test size(B) == (2, 3)
        @test B[1, 1] == A[1]
        rewind!(pool)
    end

    # ==========================================================================
    # Zero allocation (v1.11+ only)
    # ==========================================================================

    @static if VERSION >= v"1.11-"
        @testset "Zero allocation — cross-dim reshape" begin
            function _test_reshape_cross_dim()
                pool = AdaptiveArrayPool()
                A = collect(1.0:12.0)

                # Warmup (compile + cache)
                for _ in 1:3
                    checkpoint!(pool)
                    B = reshape!(pool, A, 3, 4)
                    _ = sum(B)
                    rewind!(pool)
                end

                alloc = @allocated begin
                    checkpoint!(pool)
                    B = reshape!(pool, A, 3, 4)
                    _ = sum(B)
                    rewind!(pool)
                end
                return alloc
            end

            _test_reshape_cross_dim()  # compile
            _test_reshape_cross_dim()  # compile again
            alloc = _test_reshape_cross_dim()
            println("  reshape! cross-dim: $alloc bytes")
            @test alloc == 0
        end

        @testset "Zero allocation — same-dim reshape" begin
            function _test_reshape_same_dim()
                pool = AdaptiveArrayPool()
                A = reshape(collect(1.0:12.0), 3, 4)

                for _ in 1:3
                    checkpoint!(pool)
                    B = reshape!(pool, A, 4, 3)
                    _ = sum(B)
                    rewind!(pool)
                end

                alloc = @allocated begin
                    checkpoint!(pool)
                    B = reshape!(pool, A, 4, 3)
                    _ = sum(B)
                    rewind!(pool)
                end
                return alloc
            end

            _test_reshape_same_dim()
            _test_reshape_same_dim()
            alloc = _test_reshape_same_dim()
            println("  reshape! same-dim: $alloc bytes")
            @test alloc == 0
        end

        @testset "Zero allocation — multiple reshapes in sequence" begin
            function _test_reshape_sequence()
                pool = AdaptiveArrayPool()
                A = collect(1.0:24.0)

                for _ in 1:3
                    checkpoint!(pool)
                    B = reshape!(pool, A, 4, 6)
                    C = reshape!(pool, A, 2, 3, 4)
                    _ = sum(B) + sum(C)
                    rewind!(pool)
                end

                alloc = @allocated begin
                    checkpoint!(pool)
                    B = reshape!(pool, A, 4, 6)
                    C = reshape!(pool, A, 2, 3, 4)
                    _ = sum(B) + sum(C)
                    rewind!(pool)
                end
                return alloc
            end

            _test_reshape_sequence()
            _test_reshape_sequence()
            alloc = _test_reshape_sequence()
            println("  reshape! sequence: $alloc bytes")
            @test alloc == 0
        end
    end

    # ==========================================================================
    # DisabledPool fallback
    # ==========================================================================

    @testset "DisabledPool fallback" begin
        A = collect(1.0:12.0)
        B = reshape!(DISABLED_CPU, A, 3, 4)
        @test size(B) == (3, 4)
        @test B[1, 1] == 1.0
        @test B[3, 4] == 12.0

        # Tuple syntax
        C = reshape!(DISABLED_CPU, A, (2, 6))
        @test size(C) == (2, 6)
    end

    # ==========================================================================
    # @with_pool function — realistic mixed operations
    # ==========================================================================

    @testset "@with_pool function — mixed acquire + reshape" begin
        src = collect(1.0:24.0)

        @with_pool pool function _test_reshape_mixed_ops(src)
            # 1) acquire! a temp buffer, copy and scale
            tmp = acquire!(pool, Float64, length(src))
            tmp .= src .* 2.0

            # 2) reshape! external array → matrix
            M = reshape!(pool, src, 4, 6)

            # 3) zeros! for column-sum accumulation
            col_sums = zeros!(pool, Float64, 6)
            for j in 1:6, i in 1:4
                col_sums[j] += M[i, j]
            end

            # 4) memory sharing: mutation through M visible in src
            old = src[1]
            M[1, 1] = -999.0
            shared = (src[1] == -999.0)
            src[1] = old  # restore

            # 5) another reshape! of same data → 3D
            M3 = reshape!(pool, src, 2, 3, 4)

            return (
                sum(tmp),                    # 2× sum of 1:24
                sum(col_sums),               # sum of 1:24
                shared,                      # memory sharing
                size(M), size(M3),           # shapes
                M3[1, 1, 1], M3[2, 3, 4],   # values in 3D view
            )
        end

        s_tmp, s_cols, mem_ok, sz2d, sz3d, v111, v234 = _test_reshape_mixed_ops(src)
        @test s_tmp == sum(1.0:24.0) * 2.0
        @test s_cols ≈ sum(1.0:24.0)
        @test mem_ok == true
        @test sz2d == (4, 6)
        @test sz3d == (2, 3, 4)
        @test v111 == 1.0
        @test v234 == 24.0

        # External data integrity: src must be unchanged after call
        @test src == collect(1.0:24.0)
        @test src isa Vector{Float64}
        @test size(src) == (24,)
    end

    # ==========================================================================
    # Zero allocation — @with_pool function (v1.11+)
    # ==========================================================================

    @static if VERSION >= v"1.11-"
        @testset "Zero allocation — @with_pool function (acquire + reshape + zeros!)" begin
            ext = collect(1.0:24.0)

            @with_pool pool function _test_reshape_func_alloc(data)
                tmp = acquire!(pool, Float64, length(data))
                tmp .= data
                M = reshape!(pool, data, 4, 6)
                buf = zeros!(pool, Float64, 6)
                for j in 1:6, i in 1:4
                    buf[j] += M[i, j]
                end
                return sum(tmp) + sum(buf)
            end

            # Warmup (compile + cache)
            for _ in 1:4; _test_reshape_func_alloc(ext); end

            alloc = @allocated _test_reshape_func_alloc(ext)
            println("  @with_pool function (acquire+reshape+zeros!): $alloc bytes")
            @test alloc == 0
        end
    end

    # ==========================================================================
    # @maybe_with_pool — pooling vs no-pooling proves zero-alloc
    # ==========================================================================

    @static if VERSION >= v"1.11-"
        @testset "@maybe_with_pool — pooling vs no-pooling allocation" begin
            ext = collect(1.0:12.0)

            @maybe_with_pool pool function _test_maybe_reshape_alloc(data)
                M = reshape!(pool, data, 3, 4)
                tmp = acquire!(pool, Float64, 12)
                tmp .= data .* 2.0
                return sum(M) + sum(tmp)
            end

            function _measure_maybe_reshape(data, enabled)
                MAYBE_POOLING_ENABLED[] = enabled
                for _ in 1:4; _test_maybe_reshape_alloc(data); end
                return @allocated _test_maybe_reshape_alloc(data)
            end

            expected = sum(1.0:12.0) * 3.0

            old_state = MAYBE_POOLING_ENABLED[]
            try
                # Compile both paths
                _measure_maybe_reshape(ext, true)
                _measure_maybe_reshape(ext, false)

                # Measure
                alloc_pooled   = _measure_maybe_reshape(ext, true)
                alloc_unpooled = _measure_maybe_reshape(ext, false)

                println("  @maybe_with_pool pooled:   $alloc_pooled bytes")
                println("  @maybe_with_pool unpooled: $alloc_unpooled bytes")

                # Pool: zero allocation
                @test alloc_pooled == 0
                # No pool: must allocate (reshape wrapper + Vector)
                @test alloc_unpooled > 0

                # Both paths produce correct results
                MAYBE_POOLING_ENABLED[] = true
                @test _test_maybe_reshape_alloc(ext) ≈ expected
                MAYBE_POOLING_ENABLED[] = false
                @test _test_maybe_reshape_alloc(ext) ≈ expected
            finally
                MAYBE_POOLING_ENABLED[] = old_state
            end
        end
    end

end
