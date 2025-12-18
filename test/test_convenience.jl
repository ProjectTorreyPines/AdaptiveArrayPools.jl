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

        @testset "DisabledPool fallback" begin
            v = zeros!(DISABLED_CPU, Float64, 10)
            @test v isa Array{Float64}
            @test length(v) == 10
            @test all(v .== 0.0)

            v2 = zeros!(DISABLED_CPU, 5, 5)
            @test v2 isa Matrix{Float64}
            @test size(v2) == (5, 5)

            # NTuple fallbacks
            dims = (3, 4)
            v3 = zeros!(DISABLED_CPU, Float32, dims)
            @test v3 isa Array{Float32}
            @test size(v3) == dims

            v4 = zeros!(DISABLED_CPU, dims)
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

        @testset "DisabledPool fallback" begin
            v = ones!(DISABLED_CPU, Float64, 10)
            @test v isa Array{Float64}
            @test length(v) == 10
            @test all(v .== 1.0)

            # Vararg without type
            v2 = ones!(DISABLED_CPU, 5, 5)
            @test v2 isa Matrix{Float64}
            @test size(v2) == (5, 5)

            # NTuple fallbacks
            dims = (3, 4)
            v3 = ones!(DISABLED_CPU, Float32, dims)
            @test v3 isa Array{Float32}
            @test size(v3) == dims

            v4 = ones!(DISABLED_CPU, dims)
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

        @testset "DisabledPool fallback" begin
            v = similar!(DISABLED_CPU, template)
            @test v isa Array{Float64}
            @test size(v) == size(template)

            v2 = similar!(DISABLED_CPU, template, Int64)
            @test v2 isa Array{Int64}
            @test size(v2) == size(template)

            # Vararg with different size (same type)
            v3 = similar!(DISABLED_CPU, template, 5, 5)
            @test v3 isa Array{Float64}
            @test size(v3) == (5, 5)

            # Vararg with different type and size
            v4 = similar!(DISABLED_CPU, template, Int32, 3, 4)
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

    @testset "NTuple dispatch in @with_pool (size(x) form)" begin
        # These tests cover the _impl! NTuple overloads that are invoked when
        # convenience functions are called with size(x) inside @with_pool macro.
        # The macro transforms zeros!(pool, T, size(x)) → _zeros_impl!(pool, T, size(x))
        # which requires NTuple{N,Int} dispatch (not Vararg{Int,N}).

        @testset "zeros! with size(x)" begin
            x1d = rand(10)
            x2d = rand(5, 8)
            x3d = rand(2, 3, 4)

            # 1D with explicit type
            result = @with_pool pool begin
                v = zeros!(pool, Float64, size(x1d))
                @test length(v) == 10
                @test eltype(v) == Float64
                @test all(v .== 0.0)
                sum(v)
            end
            @test result == 0.0

            # 1D without type (default_eltype)
            result = @with_pool pool begin
                v = zeros!(pool, size(x1d))
                @test length(v) == 10
                @test eltype(v) == Float64
                sum(v)
            end
            @test result == 0.0

            # 2D with explicit type
            result = @with_pool pool begin
                m = zeros!(pool, Float32, size(x2d))
                @test size(m) == (5, 8)
                @test eltype(m) == Float32
                @test all(m .== 0.0f0)
                sum(m)
            end
            @test result == 0.0f0

            # 3D without type
            result = @with_pool pool begin
                t = zeros!(pool, size(x3d))
                @test size(t) == (2, 3, 4)
                @test eltype(t) == Float64
                sum(t)
            end
            @test result == 0.0
        end

        @testset "ones! with size(x)" begin
            x1d = rand(10)
            x2d = rand(5, 8)

            # 1D with explicit type
            result = @with_pool pool begin
                v = ones!(pool, Float64, size(x1d))
                @test length(v) == 10
                @test all(v .== 1.0)
                sum(v)
            end
            @test result == 10.0

            # 1D without type
            result = @with_pool pool begin
                v = ones!(pool, size(x1d))
                @test length(v) == 10
                @test eltype(v) == Float64
                sum(v)
            end
            @test result == 10.0

            # 2D with explicit type
            result = @with_pool pool begin
                m = ones!(pool, Float32, size(x2d))
                @test size(m) == (5, 8)
                @test eltype(m) == Float32
                sum(m)
            end
            @test result == 40.0f0
        end

        @testset "unsafe_zeros! with size(x)" begin
            x1d = rand(10)
            x2d = rand(5, 8)

            # 1D with explicit type
            result = @with_pool pool begin
                v = unsafe_zeros!(pool, Float64, size(x1d))
                @test v isa Array{Float64,1}
                @test length(v) == 10
                @test all(v .== 0.0)
                sum(v)
            end
            @test result == 0.0

            # 1D without type
            result = @with_pool pool begin
                v = unsafe_zeros!(pool, size(x1d))
                @test v isa Array{Float64,1}
                @test eltype(v) == Float64
                sum(v)
            end
            @test result == 0.0

            # 2D with explicit type
            result = @with_pool pool begin
                m = unsafe_zeros!(pool, Float32, size(x2d))
                @test m isa Array{Float32,2}
                @test size(m) == (5, 8)
                sum(m)
            end
            @test result == 0.0f0
        end

        @testset "unsafe_ones! with size(x)" begin
            x1d = rand(10)
            x2d = rand(5, 8)

            # 1D with explicit type
            result = @with_pool pool begin
                v = unsafe_ones!(pool, Float64, size(x1d))
                @test v isa Array{Float64,1}
                @test length(v) == 10
                @test all(v .== 1.0)
                sum(v)
            end
            @test result == 10.0

            # 1D without type
            result = @with_pool pool begin
                v = unsafe_ones!(pool, size(x1d))
                @test v isa Array{Float64,1}
                @test eltype(v) == Float64
                sum(v)
            end
            @test result == 10.0

            # 2D with explicit type
            result = @with_pool pool begin
                m = unsafe_ones!(pool, Float32, size(x2d))
                @test m isa Array{Float32,2}
                @test size(m) == (5, 8)
                sum(m)
            end
            @test result == 40.0f0
        end

        @testset "mixed size(x) usage" begin
            # Realistic scenario: using size() of input arrays
            input_vec = rand(100)
            input_mat = rand(10, 20)

            result = @with_pool pool begin
                # Create working arrays matching input sizes
                temp_vec = zeros!(pool, Float64, size(input_vec))
                temp_mat = ones!(pool, size(input_mat))

                # Use them
                temp_vec .= input_vec .* 2
                temp_mat .= temp_mat .+ 1.0

                sum(temp_vec) + sum(temp_mat)
            end
            @test result ≈ 2 * sum(input_vec) + 400.0
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

    @testset "unsafe_zeros!" begin
        pool = AdaptiveArrayPool()

        @testset "returns raw array (not view)" begin
            v = unsafe_zeros!(pool, Float64, 10)
            @test v isa Array{Float64,1}
            @test !(v isa SubArray)
            @test length(v) == 10
            @test all(v .== 0.0)
        end

        @testset "default type (Float64)" begin
            v = unsafe_zeros!(pool, 10)
            @test v isa Array{Float64,1}
            @test !(v isa SubArray)
            @test eltype(v) == Float64
            @test all(v .== 0.0)
        end

        @testset "multi-dimensional" begin
            m = unsafe_zeros!(pool, Float64, 3, 4)
            @test m isa Array{Float64,2}
            @test !(m isa SubArray)
            @test size(m) == (3, 4)
            @test all(m .== 0.0)
        end

        @testset "tuple form" begin
            dims = (5, 6)
            m = unsafe_zeros!(pool, dims)
            @test size(m) == dims
            @test !(m isa SubArray)

            m32 = unsafe_zeros!(pool, Float32, dims)
            @test size(m32) == dims
            @test eltype(m32) == Float32
        end

        @testset "DisabledPool fallback" begin
            v = unsafe_zeros!(DISABLED_CPU, Float64, 10)
            @test v isa Array{Float64}
            @test all(v .== 0.0)
        end
    end

    @testset "unsafe_ones!" begin
        pool = AdaptiveArrayPool()

        @testset "returns raw array (not view)" begin
            v = unsafe_ones!(pool, Float64, 10)
            @test v isa Array{Float64,1}
            @test !(v isa SubArray)
            @test length(v) == 10
            @test all(v .== 1.0)
        end

        @testset "default type (Float64)" begin
            v = unsafe_ones!(pool, 10)
            @test v isa Array{Float64,1}
            @test !(v isa SubArray)
            @test eltype(v) == Float64
            @test all(v .== 1.0)
        end

        @testset "multi-dimensional" begin
            m = unsafe_ones!(pool, Float64, 3, 4)
            @test m isa Array{Float64,2}
            @test !(m isa SubArray)
            @test size(m) == (3, 4)
            @test all(m .== 1.0)
        end

        @testset "tuple form" begin
            dims = (5, 6)
            m = unsafe_ones!(pool, dims)
            @test size(m) == dims
            @test !(m isa SubArray)

            m32 = unsafe_ones!(pool, Float32, dims)
            @test size(m32) == dims
            @test eltype(m32) == Float32
            @test all(m32 .== 1.0f0)
        end

        @testset "DisabledPool fallback" begin
            v = unsafe_ones!(DISABLED_CPU, Float64, 10)
            @test v isa Array{Float64}
            @test all(v .== 1.0)
        end
    end

    @testset "unsafe_similar!" begin
        pool = AdaptiveArrayPool()
        template = rand(Float64, 10, 10)

        @testset "returns raw array (not view)" begin
            v = unsafe_similar!(pool, template)
            @test v isa Array{Float64,2}
            @test !(v isa SubArray)
            @test size(v) == size(template)
        end

        @testset "different type" begin
            v = unsafe_similar!(pool, template, Float32)
            @test v isa Array{Float32,2}
            @test !(v isa SubArray)
            @test size(v) == size(template)
        end

        @testset "different size" begin
            v = unsafe_similar!(pool, template, 5, 5)
            @test v isa Array{Float64,2}
            @test !(v isa SubArray)
            @test size(v) == (5, 5)
        end

        @testset "different type and size" begin
            v = unsafe_similar!(pool, template, Int32, 3, 4)
            @test v isa Array{Int32,2}
            @test !(v isa SubArray)
            @test size(v) == (3, 4)
        end

        @testset "DisabledPool fallback" begin
            v = unsafe_similar!(DISABLED_CPU, template)
            @test v isa Array{Float64}
            @test size(v) == size(template)

            v2 = unsafe_similar!(DISABLED_CPU, template, Int64)
            @test v2 isa Array{Int64}
        end
    end

    @testset "Integration unsafe functions with @with_pool" begin
        @testset "unsafe_zeros! in macro" begin
            result = @with_pool pool begin
                v = unsafe_zeros!(pool, Float64, 100)
                @test v isa Array{Float64,1}
                @test !(v isa SubArray)
                v .+= 1.0
                sum(v)
            end
            @test result == 100.0
        end

        @testset "unsafe_ones! in macro" begin
            result = @with_pool pool begin
                v = unsafe_ones!(pool, Float64, 50)
                @test v isa Array{Float64,1}
                @test !(v isa SubArray)
                sum(v)
            end
            @test result == 50.0
        end

        @testset "unsafe_similar! in macro" begin
            template = rand(10)
            result = @with_pool pool begin
                v = unsafe_similar!(pool, template)
                @test v isa Array{Float64,1}
                @test !(v isa SubArray)
                v .= 2.0
                sum(v)
            end
            @test result == 20.0
        end
    end

end # Convenience Functions
