@testset "Convenience Functions" begin

    @testset "zeros!" begin
        pool = AdaptiveArrayPool()

        @testset "with explicit type" begin
            v = zeros!(pool, Float64, 10)
            @test length(v) == 10
            @test eltype(v) == Float64
            @test all(v .== 0.0)

            v32 = zeros!(pool, Float32, 5)
            @test length(v32) == 5
            @test eltype(v32) == Float32
            @test all(v32 .== 0.0f0)

            vi = zeros!(pool, Int64, 8)
            @test length(vi) == 8
            @test eltype(vi) == Int64
            @test all(vi .== 0)
        end

        @testset "default type (Float64)" begin
            v = zeros!(pool, 10)
            @test length(v) == 10
            @test eltype(v) == Float64
            @test all(v .== 0.0)
        end

        @testset "multi-dimensional" begin
            m = zeros!(pool, Float64, 3, 4)
            @test size(m) == (3, 4)
            @test eltype(m) == Float64
            @test all(m .== 0.0)

            m32 = zeros!(pool, Float32, 2, 3, 4)
            @test size(m32) == (2, 3, 4)
            @test all(m32 .== 0.0f0)
        end

        @testset "tuple form" begin
            dims = (5, 6)
            m = zeros!(pool, dims)
            @test size(m) == dims
            @test eltype(m) == Float64
            @test all(m .== 0.0)

            m32 = zeros!(pool, Float32, dims)
            @test size(m32) == dims
            @test eltype(m32) == Float32
        end

        @testset "Nothing fallback" begin
            v = zeros!(nothing, Float64, 10)
            @test v isa Array{Float64}
            @test length(v) == 10
            @test all(v .== 0.0)

            v2 = zeros!(nothing, 5, 5)
            @test v2 isa Matrix{Float64}
            @test size(v2) == (5, 5)

            # NTuple fallbacks
            dims = (3, 4)
            v3 = zeros!(nothing, Float32, dims)
            @test v3 isa Array{Float32}
            @test size(v3) == dims

            v4 = zeros!(nothing, dims)
            @test v4 isa Array{Float64}
            @test size(v4) == dims
        end
    end

    @testset "ones!" begin
        pool = AdaptiveArrayPool()

        @testset "with explicit type" begin
            v = ones!(pool, Float64, 10)
            @test length(v) == 10
            @test eltype(v) == Float64
            @test all(v .== 1.0)

            v32 = ones!(pool, Float32, 5)
            @test length(v32) == 5
            @test eltype(v32) == Float32
            @test all(v32 .== 1.0f0)

            vi = ones!(pool, Int64, 8)
            @test length(vi) == 8
            @test eltype(vi) == Int64
            @test all(vi .== 1)
        end

        @testset "default type (Float64)" begin
            v = ones!(pool, 10)
            @test length(v) == 10
            @test eltype(v) == Float64
            @test all(v .== 1.0)
        end

        @testset "multi-dimensional" begin
            m = ones!(pool, Float64, 3, 4)
            @test size(m) == (3, 4)
            @test eltype(m) == Float64
            @test all(m .== 1.0)
        end

        @testset "tuple form" begin
            dims = (5, 6)
            m = ones!(pool, dims)
            @test size(m) == dims
            @test eltype(m) == Float64
            @test all(m .== 1.0)

            # NTuple with explicit type
            m32 = ones!(pool, Float32, dims)
            @test size(m32) == dims
            @test eltype(m32) == Float32
            @test all(m32 .== 1.0f0)
        end

        @testset "Nothing fallback" begin
            v = ones!(nothing, Float64, 10)
            @test v isa Array{Float64}
            @test length(v) == 10
            @test all(v .== 1.0)

            # Vararg without type
            v2 = ones!(nothing, 5, 5)
            @test v2 isa Matrix{Float64}
            @test size(v2) == (5, 5)

            # NTuple fallbacks
            dims = (3, 4)
            v3 = ones!(nothing, Float32, dims)
            @test v3 isa Array{Float32}
            @test size(v3) == dims

            v4 = ones!(nothing, dims)
            @test v4 isa Array{Float64}
            @test size(v4) == dims
        end
    end

    @testset "similar!" begin
        pool = AdaptiveArrayPool()
        template = rand(Float64, 10, 10)

        @testset "same type and size" begin
            v = similar!(pool, template)
            @test size(v) == size(template)
            @test eltype(v) == eltype(template)
        end

        @testset "different type" begin
            v = similar!(pool, template, Float32)
            @test size(v) == size(template)
            @test eltype(v) == Float32
        end

        @testset "different size" begin
            v = similar!(pool, template, 5, 5)
            @test size(v) == (5, 5)
            @test eltype(v) == eltype(template)
        end

        @testset "different type and size" begin
            v = similar!(pool, template, Int32, 3, 4)
            @test size(v) == (3, 4)
            @test eltype(v) == Int32
        end

        @testset "1D template" begin
            template1d = rand(20)
            v = similar!(pool, template1d)
            @test length(v) == 20
            @test eltype(v) == Float64
        end

        @testset "Nothing fallback" begin
            v = similar!(nothing, template)
            @test v isa Array{Float64}
            @test size(v) == size(template)

            v2 = similar!(nothing, template, Int64)
            @test v2 isa Array{Int64}
            @test size(v2) == size(template)

            # Vararg with different size (same type)
            v3 = similar!(nothing, template, 5, 5)
            @test v3 isa Array{Float64}
            @test size(v3) == (5, 5)

            # Vararg with different type and size
            v4 = similar!(nothing, template, Int32, 3, 4)
            @test v4 isa Array{Int32}
            @test size(v4) == (3, 4)
        end
    end

    @testset "Integration with @with_pool" begin
        @testset "zeros! in macro" begin
            result = @with_pool pool begin
                v = zeros!(pool, Float64, 100)
                v .+= 1.0
                sum(v)
            end
            @test result == 100.0
        end

        @testset "ones! in macro" begin
            result = @with_pool pool begin
                v = ones!(pool, Float64, 50)
                sum(v)
            end
            @test result == 50.0
        end

        @testset "similar! in macro" begin
            template = rand(10)
            result = @with_pool pool begin
                v = similar!(pool, template)
                v .= 2.0
                sum(v)
            end
            @test result == 20.0
        end

        @testset "mixed usage" begin
            result = @with_pool pool begin
                a = zeros!(pool, 10)
                b = ones!(pool, 10)
                c = acquire!(pool, Float64, 10)
                c .= a .+ b
                sum(c)
            end
            @test result == 10.0
        end
    end

    @testset "Pool state management" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        v1 = zeros!(pool, Float64, 10)
        v2 = ones!(pool, Float64, 10)
        @test pool.float64.n_active == 2

        rewind!(pool)
        @test pool.float64.n_active == 0
    end

end # Convenience Functions
