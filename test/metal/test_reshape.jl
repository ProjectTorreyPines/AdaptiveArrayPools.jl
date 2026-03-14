# Metal Reshape Tests
# Tests for reshape! with MtlArray

@testset "Metal Reshape" begin

    @testset "Basic reshape 1D → 2D" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        result = @with_pool :metal p begin
            v = acquire!(p, Float32, 12)
            v .= 1.0f0
            A = reshape!(p, v, 3, 4)
            @test size(A) == (3, 4)
            @test A isa MtlArray{Float32, 2}
            sum(A)
        end
        @test result == 12.0f0
    end

    @testset "Reshape 1D → 3D" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        result = @with_pool :metal p begin
            v = acquire!(p, Float32, 24)
            v .= 2.0f0
            T = reshape!(p, v, 2, 3, 4)
            @test size(T) == (2, 3, 4)
            sum(T)
        end
        @test result == 48.0f0
    end

    @testset "Reshape with tuple dims" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @with_pool :metal p begin
            v = acquire!(p, Float32, 20)
            A = reshape!(p, v, (4, 5))
            @test size(A) == (4, 5)
        end
    end

    @testset "Same-dim reshape (no cross-dim)" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @with_pool :metal p begin
            A = acquire!(p, Float32, 3, 4)
            B = reshape!(p, A, 4, 3)
            @test size(B) == (4, 3)
            # Same dimensionality: in-place setfield!
            @test B === A
        end
    end

    @testset "DimensionMismatch on wrong element count" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        @with_pool :metal p begin
            v = acquire!(p, Float32, 10)
            @test_throws DimensionMismatch reshape!(p, v, 3, 4)
        end
    end

    @testset "Reshape reuse across scopes" begin
        pool = get_task_local_metal_pool()
        reset!(pool)

        # First scope: create reshape wrapper
        @with_pool :metal p begin
            v = acquire!(p, Float32, 12)
            A = reshape!(p, v, 3, 4)
            @test size(A) == (3, 4)
        end

        # Second scope: should reuse cached wrapper
        @with_pool :metal p begin
            v = acquire!(p, Float32, 12)
            A = reshape!(p, v, 3, 4)
            @test size(A) == (3, 4)
        end
    end

end
