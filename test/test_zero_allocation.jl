# ==============================================================================
# Comprehensive Zero-Allocation Tests (README.md pattern)
# ==============================================================================
#
# These tests verify that the pool achieves true zero-allocation in realistic
# usage patterns matching the README.md examples.
#
# Pattern:
# 1. Create explicit pool (shared across iterations)
# 2. Inner loop: @with_pool + multiple acquire!/unsafe_acquire! + in-place ops → scalar
# 3. Verify: loop has 0 bytes allocation after warmup

@testset "Zero-allocation Patterns" begin
    
    # ==============================================================================
    # Pattern 1: acquire! only (SubArray) - N-D matrices
    # ==============================================================================

    function _test_acquire_single()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                A = acquire!(pool, Float64, 10, 10)
                B = acquire!(pool, Float64, 10, 10)
                C = acquire!(pool, Float64, 10, 10)
                fill!(A, 1.0)
                fill!(B, 2.0)
                @. C = A + B
                sum(C)
            end
        end

        # Measure single iteration (cache hit)
        alloc = @allocated @with_pool pool begin
            A = acquire!(pool, Float64, 10, 10)
            B = acquire!(pool, Float64, 10, 10)
            C = acquire!(pool, Float64, 10, 10)
            fill!(A, 1.0)
            fill!(B, 2.0)
            @. C = A + B
            sum(C)
        end

        return alloc
    end

    @testset "README pattern: acquire! zero-allocation" begin
        # Compile
        _test_acquire_single()
        _test_acquire_single()

        alloc = _test_acquire_single()
        println("  acquire! (single iteration): $alloc bytes")
        @test alloc == 0
    end

    function _test_acquire_loop()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                A = acquire!(pool, Float64, 10, 10)
                B = acquire!(pool, Float64, 10, 10)
                C = acquire!(pool, Float64, 10, 10)
                fill!(A, 1.0)
                fill!(B, 2.0)
                @. C = A + B
                sum(C)
            end
        end

        # Measure loop
        total = 0.0
        alloc = @allocated for _ in 1:100
            total += @with_pool pool begin
                A = acquire!(pool, Float64, 10, 10)
                B = acquire!(pool, Float64, 10, 10)
                C = acquire!(pool, Float64, 10, 10)
                fill!(A, 1.0)
                fill!(B, 2.0)
                @. C = A + B
                sum(C)
            end
        end

        return alloc, total
    end

    # ==============================================================================
    # Pattern 2: unsafe_acquire! only (raw Array)
    # Note: With N-D array caching, unsafe_acquire! achieves zero allocation on cache hit.
    #       The Array objects are cached and reused, avoiding heap allocations.
    # ==============================================================================

    function _test_unsafe_single()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                A = unsafe_acquire!(pool, Float64, 10, 10)
                B = unsafe_acquire!(pool, Float64, 10, 10)
                C = unsafe_acquire!(pool, Float64, 10, 10)
                fill!(A, 1.0)
                fill!(B, 2.0)
                @. C = A + B
                sum(C)
            end
        end

        # Measure single iteration
        alloc = @allocated @with_pool pool begin
            A = unsafe_acquire!(pool, Float64, 10, 10)
            B = unsafe_acquire!(pool, Float64, 10, 10)
            C = unsafe_acquire!(pool, Float64, 10, 10)
            fill!(A, 1.0)
            fill!(B, 2.0)
            @. C = A + B
            sum(C)
        end

        return alloc
    end

    @testset "README pattern: unsafe_acquire! zero-allocation" begin
        # Compile
        _test_unsafe_single()
        _test_unsafe_single()

        alloc = _test_unsafe_single()
        println("  unsafe_acquire! (single iteration): $alloc bytes")

        # With N-D array caching, unsafe_acquire! achieves zero allocation
        @test alloc == 0
    end

    function _test_unsafe_loop()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                A = unsafe_acquire!(pool, Float64, 10, 10)
                B = unsafe_acquire!(pool, Float64, 10, 10)
                C = unsafe_acquire!(pool, Float64, 10, 10)
                fill!(A, 1.0)
                fill!(B, 2.0)
                @. C = A + B
                sum(C)
            end
        end

        # Measure loop
        total = 0.0
        alloc = @allocated for _ in 1:100
            total += @with_pool pool begin
                A = unsafe_acquire!(pool, Float64, 10, 10)
                B = unsafe_acquire!(pool, Float64, 10, 10)
                C = unsafe_acquire!(pool, Float64, 10, 10)
                fill!(A, 1.0)
                fill!(B, 2.0)
                @. C = A + B
                sum(C)
            end
        end

        return alloc, total
    end

    # ==============================================================================
    # Pattern 3: Mixed acquire! + unsafe_acquire!
    # ==============================================================================

    function _inner_mul!(C, A, B)
        @. C = A * B
    end

    function _test_mixed_loop()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                A = acquire!(pool, Float64, 10, 10)
                B = acquire!(pool, Float64, 10, 10)
                C = unsafe_acquire!(pool, Float64, 10, 10)
                fill!(A, 2.0)
                fill!(B, 3.0)
                _inner_mul!(C, A, B)
                sum(C)
            end
        end

        # Measure loop
        total = 0.0
        alloc = @allocated for _ in 1:100
            total += @with_pool pool begin
                A = acquire!(pool, Float64, 10, 10)
                B = acquire!(pool, Float64, 10, 10)
                C = unsafe_acquire!(pool, Float64, 10, 10)
                fill!(A, 2.0)
                fill!(B, 3.0)
                _inner_mul!(C, A, B)
                sum(C)
            end
        end

        return alloc, total
    end

    @testset "README pattern: mixed acquire!/unsafe_acquire! zero-allocation loop" begin
        # Compile
        _test_mixed_loop()
        _test_mixed_loop()

        alloc, total = _test_mixed_loop()
        println("  mixed acquire!/unsafe_acquire! loop (100 iterations): $alloc bytes")
        @test alloc == 0
        @test total == 100 * (10 * 10 * 6.0)  # 2.0 * 3.0 = 6.0
    end

    # ==============================================================================
    # Pattern 4: 1D + N-D mixed dimensions
    # ==============================================================================

    function _test_1d_nd_loop()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                v1 = acquire!(pool, Float64, 100)
                v2 = acquire!(pool, Float64, 100)
                m1 = acquire!(pool, Float64, 10, 10)
                m2 = acquire!(pool, Float64, 10, 10)
                fill!(v1, 1.0)
                fill!(v2, 2.0)
                fill!(m1, 3.0)
                fill!(m2, 4.0)
                sum(v1) + sum(v2) + sum(m1) + sum(m2)
            end
        end

        # Measure loop
        total = 0.0
        alloc = @allocated for _ in 1:100
            total += @with_pool pool begin
                v1 = acquire!(pool, Float64, 100)
                v2 = acquire!(pool, Float64, 100)
                m1 = acquire!(pool, Float64, 10, 10)
                m2 = acquire!(pool, Float64, 10, 10)
                fill!(v1, 1.0)
                fill!(v2, 2.0)
                fill!(m1, 3.0)
                fill!(m2, 4.0)
                sum(v1) + sum(v2) + sum(m1) + sum(m2)
            end
        end

        return alloc, total
    end

    @testset "README pattern: 1D + N-D mixed zero-allocation loop" begin
        # Compile
        _test_1d_nd_loop()
        _test_1d_nd_loop()

        alloc, total = _test_1d_nd_loop()
        println("  1D + N-D mixed loop (100 iterations): $alloc bytes")
        @test alloc == 0
        @test total == 100 * (100 * 1.0 + 100 * 2.0 + 100 * 3.0 + 100 * 4.0)
    end

    # ==============================================================================
    # Pattern 5: Multiple element types
    # ==============================================================================

    function _test_multi_type_loop()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                f64 = acquire!(pool, Float64, 10, 10)
                f32 = acquire!(pool, Float32, 10, 10)
                i64 = acquire!(pool, Int64, 10)
                i32 = acquire!(pool, Int32, 10)
                fill!(f64, 1.0)
                fill!(f32, 2.0f0)
                fill!(i64, Int64(3))
                fill!(i32, Int32(4))
                sum(f64) + sum(f32) + sum(i64) + sum(i32)
            end
        end

        # Measure loop
        total = 0.0
        alloc = @allocated for _ in 1:100
            total += @with_pool pool begin
                f64 = acquire!(pool, Float64, 10, 10)
                f32 = acquire!(pool, Float32, 10, 10)
                i64 = acquire!(pool, Int64, 10)
                i32 = acquire!(pool, Int32, 10)
                fill!(f64, 1.0)
                fill!(f32, 2.0f0)
                fill!(i64, Int64(3))
                fill!(i32, Int32(4))
                sum(f64) + sum(f32) + sum(i64) + sum(i32)
            end
        end

        return alloc, total
    end

    @testset "README pattern: multiple element types zero-allocation loop" begin
        # Compile
        _test_multi_type_loop()
        _test_multi_type_loop()

        alloc, total = _test_multi_type_loop()
        println("  multi-type loop (100 iterations): $alloc bytes")
        @test alloc == 0
        # f64: 100*1=100, f32: 100*2=200, i64: 10*3=30, i32: 10*4=40 → total=370
        @test total == 100 * (100 * 1.0 + 100 * 2.0 + 10 * 3.0 + 10 * 4.0)
    end

    # ==============================================================================
    # Pattern 6: 3D tensors
    # ==============================================================================

    function _test_3d_loop()
        pool = AdaptiveArrayPool()

        # Warmup
        for _ in 1:3
            @with_pool pool begin
                T1 = acquire!(pool, Float64, 5, 5, 5)
                T2 = acquire!(pool, Float64, 5, 5, 5)
                T3 = acquire!(pool, Float64, 5, 5, 5)
                fill!(T1, 1.0)
                fill!(T2, 2.0)
                @. T3 = T1 + T2
                sum(T3)
            end
        end

        # Measure loop
        total = 0.0
        alloc = @allocated for _ in 1:100
            total += @with_pool pool begin
                T1 = acquire!(pool, Float64, 5, 5, 5)
                T2 = acquire!(pool, Float64, 5, 5, 5)
                T3 = acquire!(pool, Float64, 5, 5, 5)
                fill!(T1, 1.0)
                fill!(T2, 2.0)
                @. T3 = T1 + T2
                sum(T3)
            end
        end

        return alloc, total
    end

    @testset "README pattern: 3D tensor zero-allocation loop" begin
        # Compile
        _test_3d_loop()
        _test_3d_loop()

        alloc, total = _test_3d_loop()
        println("  3D tensor loop (100 iterations): $alloc bytes")
        @test alloc == 0
        @test total == 100 * (125 * 3.0)  # 5^3 = 125 elements, each = 3.0
    end

    # ==============================================================================
    # Summary test: All patterns combined
    # ==============================================================================

    @testset "Zero-allocation summary" begin
        results = Dict{String, Int}()

        # Compile all
        _test_acquire_loop()
        _test_unsafe_loop()
        _test_mixed_loop()
        _test_1d_nd_loop()
        _test_multi_type_loop()
        _test_3d_loop()

        # Measure
        results["acquire!"], _ = _test_acquire_loop()
        results["unsafe_acquire!"], _ = _test_unsafe_loop()
        results["mixed"], _ = _test_mixed_loop()
        results["1D+N-D"], _ = _test_1d_nd_loop()
        results["multi-type"], _ = _test_multi_type_loop()
        results["3D tensor"], _ = _test_3d_loop()

        println("\n  === Zero-Allocation Summary (100 iterations each) ===")
        for (name, alloc) in sort(collect(results))
            status = alloc == 0 ? "✓" : "✗"
            println("    $status $name: $alloc bytes")
        end
        println()

        for (name, alloc) in results
            @test alloc == 0
        end
    end

end # Zero-allocation Patterns