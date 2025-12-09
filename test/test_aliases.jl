using Test
using AdaptiveArrayPools

@testset "API Aliases" begin

    @testset "acquire_view! is alias for acquire!" begin
        # Verify they are the same function
        @test acquire_view! === acquire!

        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D - returns SubArray
        v = acquire_view!(pool, Float64, 100)
        @test v isa SubArray{Float64, 1}
        @test length(v) == 100

        # 2D - returns ReshapedArray
        m = acquire_view!(pool, Float64, 10, 10)
        @test m isa Base.ReshapedArray{Float64, 2}
        @test size(m) == (10, 10)

        # Tuple support
        t = acquire_view!(pool, Float64, (5, 5, 5))
        @test t isa Base.ReshapedArray{Float64, 3}
        @test size(t) == (5, 5, 5)

        # Similar-style
        ref = rand(3, 4)
        s = acquire_view!(pool, ref)
        @test size(s) == size(ref)
        @test eltype(s) == eltype(ref)

        rewind!(pool)

        # nothing fallback
        v_nothing = acquire_view!(nothing, Float64, 10)
        @test v_nothing isa Vector{Float64}

        m_nothing = acquire_view!(nothing, Float64, 5, 5)
        @test m_nothing isa Matrix{Float64}
    end

    @testset "acquire_array! is alias for unsafe_acquire!" begin
        # Verify they are the same function
        @test acquire_array! === unsafe_acquire!

        pool = AdaptiveArrayPool()
        checkpoint!(pool)

        # 1D - returns Vector
        v = acquire_array!(pool, Float64, 100)
        @test v isa Vector{Float64}
        @test length(v) == 100

        # 2D - returns Matrix
        m = acquire_array!(pool, Float64, 10, 10)
        @test m isa Matrix{Float64}
        @test size(m) == (10, 10)

        # 3D - returns Array{T,3}
        t = acquire_array!(pool, Float64, 5, 5, 5)
        @test t isa Array{Float64, 3}
        @test size(t) == (5, 5, 5)

        # Tuple support
        arr = acquire_array!(pool, Float64, (2, 3, 4))
        @test arr isa Array{Float64, 3}
        @test size(arr) == (2, 3, 4)

        # Similar-style
        ref = rand(3, 4)
        s = acquire_array!(pool, ref)
        @test size(s) == size(ref)
        @test s isa Matrix{Float64}

        rewind!(pool)

        # nothing fallback
        v_nothing = acquire_array!(nothing, Float64, 10)
        @test v_nothing isa Vector{Float64}

        m_nothing = acquire_array!(nothing, Float64, 5, 5)
        @test m_nothing isa Matrix{Float64}
    end

    @testset "Symmetric naming usage" begin
        # Demonstrate symmetric naming pattern
        pool = AdaptiveArrayPool()

        @with_pool pool begin
            # View-style (compiler may optimize)
            view_mat = acquire_view!(pool, Float64, 10, 10)
            view_mat .= 1.0

            # Array-style (for type-unspecified paths)
            array_mat = acquire_array!(pool, Float64, 10, 10)
            array_mat .= 2.0

            @test sum(view_mat) == 100.0
            @test sum(array_mat) == 200.0
        end
    end

    @testset "Similar-style _impl! via macro (runtime coverage)" begin
        # These tests exercise the _acquire_impl!(pool, x::AbstractArray) and
        # _unsafe_acquire_impl!(pool, x::AbstractArray) methods which are only
        # called through macro transformation (not public API).

        ref_mat = rand(5, 6)
        ref_vec = rand(10)
        ref_int = rand(Int32, 3, 4)

        @testset "acquire!(pool, x) via @with_pool" begin
            pool = AdaptiveArrayPool()

            result = @with_pool pool begin
                # Similar-style acquire - macro transforms to _acquire_impl!(pool, ref_mat)
                mat = acquire!(pool, ref_mat)
                @test size(mat) == size(ref_mat)
                @test eltype(mat) == eltype(ref_mat)
                @test mat isa Base.ReshapedArray{Float64, 2}

                vec = acquire!(pool, ref_vec)
                @test size(vec) == size(ref_vec)
                @test vec isa SubArray{Float64, 1}

                int_mat = acquire!(pool, ref_int)
                @test eltype(int_mat) == Int32
                @test size(int_mat) == (3, 4)

                sum(mat) + sum(vec) + sum(int_mat)
            end
            @test result isa Float64
        end

        @testset "unsafe_acquire!(pool, x) via @with_pool" begin
            pool = AdaptiveArrayPool()

            result = @with_pool pool begin
                # Similar-style unsafe_acquire - macro transforms to _unsafe_acquire_impl!(pool, ref_mat)
                mat = unsafe_acquire!(pool, ref_mat)
                @test size(mat) == size(ref_mat)
                @test mat isa Matrix{Float64}

                vec = unsafe_acquire!(pool, ref_vec)
                @test size(vec) == size(ref_vec)
                @test vec isa Vector{Float64}

                sum(mat) + sum(vec)
            end
            @test result isa Float64
        end

        @testset "acquire_view!/acquire_array! aliases via @with_pool" begin
            pool = AdaptiveArrayPool()

            @with_pool pool begin
                # acquire_view! is alias for acquire!
                v1 = acquire_view!(pool, ref_mat)
                @test size(v1) == size(ref_mat)

                # acquire_array! is alias for unsafe_acquire!
                v2 = acquire_array!(pool, ref_vec)
                @test size(v2) == size(ref_vec)
            end
        end
    end

end
