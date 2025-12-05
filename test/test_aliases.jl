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

end
