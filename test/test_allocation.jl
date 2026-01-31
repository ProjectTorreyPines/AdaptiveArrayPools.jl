@with_pool pool function foo()
	float64_vec = acquire!(pool, Float64, 10)
	float32_vec = acquire!(pool, Float32, 10)

	float64_mat = acquire!(pool, Float64, 10, 10)
	float32_mat = acquire!(pool, Float32, 10, 10)

	bv = acquire!(pool, Bit, 100)
	ba2 = acquire!(pool, Bit, 10, 10)
	ba3 = acquire!(pool, Bit, 5, 5, 4)

	tt1 = trues!(pool, 256)
	tt2 = ones!(pool, Bit, 10, 20)
	ff1 = falses!(pool, 100, 5)
	ff2 = zeros!(pool, Bit, 100)

	C = similar!(pool, tt1)
end


@testset "zero allocation on reuse" begin

    alloc1 = @allocated foo()
    alloc2 = @allocated foo()
    alloc3 = @allocated foo()

    @test alloc1 > 0 # First call allocates
    @test alloc2 == 0 # Subsequent calls reuse cached arrays
    @test alloc3 == 0 # Further calls also zero allocation
end