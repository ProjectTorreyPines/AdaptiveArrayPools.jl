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

using AdaptiveArrayPools: _cp_state, _checkpoint_state_core!, _rewind_state_core!

@testset "state cores: checkpoint/rewind on bare PoolCheckpointState" begin
    st = PoolCheckpointState()
    _checkpoint_state_core!(st, 2)
    @test st._checkpoint_depths == [0, 2]
    _checkpoint_state_core!(st, 2)                 # same-depth guard: no double push
    @test st._checkpoint_depths == [0, 2]
    st.n_active = 5
    @test _rewind_state_core!(st, 2) == 5          # returns pre-rewind n_active
    @test st.n_active == 0                         # Case A restore
    @test st._checkpoint_depths == [0]

    # Case B: no checkpoint at depth → restore from stack top
    st.n_active = 7
    @test _rewind_state_core!(st, 3) == 7
    @test st.n_active == 0                         # sentinel top

    # orphan cleanup: stale deeper entries popped first
    _checkpoint_state_core!(st, 4)
    st.n_active = 9
    @test _rewind_state_core!(st, 2) == 9          # pops orphan depth-4 entry, Case B
    @test st._checkpoint_depths == [0]
    @test st.n_active == 0

    # _cp_state mapping
    tp = TypedPool{Int32}()
    @test _cp_state(tp) === tp.state
    btp = BitTypedPool()
    @test _cp_state(btp) === btp.state
end
