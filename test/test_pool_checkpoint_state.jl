# PoolCheckpointState extraction: checkpoint bookkeeping lives in a concrete,
# type-parameter-free struct; property forwarding keeps tp.n_active etc. working.
using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: TypedPool, BitTypedPool, PoolCheckpointState

@testset "PoolCheckpointState: extraction & forwarding" begin
    tp = TypedPool{Float64}()
    st = tp.state
    @test st isa PoolCheckpointState
    @test st.n_active == 0
    @test st._checkpoint_n_active == [0]
    @test st._checkpoint_depths == [0]

    # forwarding reads the SAME objects (identity, not copies)
    @test tp._checkpoint_n_active === st._checkpoint_n_active
    @test tp._checkpoint_depths === st._checkpoint_depths

    # forwarded write goes to the state
    tp.n_active = 3
    @test st.n_active == 3
    @test tp.n_active == 3
    tp.n_active = 0

    # fallback setproperty! keeps default auto-convert semantics
    tp._am_peak_n_active = Int32(2)
    @test tp._am_peak_n_active === 2
    tp._am_peak_n_active = 0

    # non-forwarded fields still direct
    @test tp.vectors isa Vector{Vector{Float64}}
    @test tp._am_peak_n_active == 0

    btp = BitTypedPool()
    @test btp.state isa PoolCheckpointState
    @test btp._checkpoint_depths === btp.state._checkpoint_depths
    btp.n_active = 2
    @test btp.state.n_active == 2

    # `state` is const — rebinding must throw
    @test_throws Exception (tp.state = PoolCheckpointState())
end

@testset "PoolCheckpointState: end-to-end through public API" begin
    pool = AdaptiveArrayPool()
    v = acquire!(pool, Float64, 8)
    @test pool.float64.state.n_active == 1
    reset!(pool)
    @test pool.float64.state.n_active == 0
    @test pool.float64.state._checkpoint_depths == [0]
end
