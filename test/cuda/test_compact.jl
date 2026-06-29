# CUDA compact! Tests
# Parity with CPU/Metal test_compact.jl: shrink an over-allocated device buffer in place.
#
# Like Metal, CUDA has NO uncached views — acquire_view! == acquire! (returns a cached
# CuArray wrapper) — so the CPU view-extent bug is structurally absent. We verify
# slot_extents-based compaction shrinks the device buffer and re-syncs the held wrapper
# (the DataRef swap + refcount path). Uses Float32/Int32 (aligned_sizeof == sizeof for
# these, so the capacity helper below is exact).

@testset "CUDA compact!" begin

    # Device capacity (elements). For Float32/Int32, _aligned_sizeof == sizeof.
    _ccap(v) = Int(getfield(v, :maxsize) ÷ sizeof(eltype(v)))
    Summary = @NamedTuple{slots_compacted::Int, bytes_reclaimed::Int, gc_triggered::Bool}
    ZERO = (; slots_compacted = 0, bytes_reclaimed = 0, gc_triggered = false)

    @testset "inactive: compact!(pool) shrinks the device buffer" begin
        pool = CuAdaptiveArrayPool{0}()
        checkpoint!(pool); a = acquire!(pool, Float32, 1_000_000); fill!(a, 0.0f0); rewind!(pool)
        checkpoint!(pool); b = acquire!(pool, Float32, 100); fill!(b, 0.0f0); rewind!(pool)
        tp = pool.float32
        @test tp.n_active == 0
        @test _ccap(tp.vectors[1]) >= 1_000_000

        s = compact!(pool)

        @test s.slots_compacted == 1
        @test s.bytes_reclaimed > 0
        @test s.gc_triggered == false
        @test _ccap(tp.vectors[1]) < 1000                 # device buffer shrunk in place
    end

    @testset "active=true shrinks a HELD slot in place; data intact + writable" begin
        pool = CuAdaptiveArrayPool{0}()
        checkpoint!(pool); a = acquire!(pool, Float32, 1_000_000); fill!(a, 0.0f0); rewind!(pool)
        checkpoint!(pool)
        b = acquire!(pool, Float32, 100)                  # slot 1 ACTIVE (b held)
        copyto!(b, Float32.(1:100))
        tp = pool.float32
        @test tp.n_active == 1
        cap0 = _ccap(tp.vectors[1])
        @test cap0 >= 1_000_000

        @test compact!(pool; active = false).slots_compacted == 0   # leaves held slot alone
        @test _ccap(tp.vectors[1]) == cap0

        s = compact!(pool; active = true)
        @test s.slots_compacted == 1
        @test _ccap(tp.vectors[1]) < 1000
        @test Array(b) == Float32.(1:100)                 # held wrapper re-synced: data intact
        copyto!(b, fill(9.0f0, 100))                      # ...and still writable into the new buffer
        @test Array(b)[1] == 9.0f0
        rewind!(pool)
    end

    @testset "no-op + force_gc + type stability + DisabledPool" begin
        pool = CuAdaptiveArrayPool{0}()
        @test compact!(pool) == ZERO
        @test compact!(pool; force_gc = true).gc_triggered == true
        @test only(Base.return_types(compact!, (typeof(pool),))) == Summary
        @test (@inferred compact!(pool)) isa Summary
        @test compact!(ext.DISABLED_CUDA) == ZERO
        @test compact!(ext.DISABLED_CUDA, Float32) == ZERO
    end

    @testset "per-type + varargs (CUDA fixed types)" begin
        pool = CuAdaptiveArrayPool{0}()
        checkpoint!(pool); acquire!(pool, Float32, 1_000_000); rewind!(pool)
        checkpoint!(pool); acquire!(pool, Float32, 100); rewind!(pool)
        checkpoint!(pool); acquire!(pool, Int32, 1_000_000); rewind!(pool)
        checkpoint!(pool); acquire!(pool, Int32, 100); rewind!(pool)

        s = compact!(pool, Float32, Int32)
        @test s.slots_compacted == 2
        @test _ccap(pool.float32.vectors[1]) < 1000
        @test _ccap(pool.int32.vectors[1]) < 1000
        @test only(Base.return_types(compact!, (typeof(pool), Type{Float32}, Type{Int32}))) == Summary
        @test compact!(ext.DISABLED_CUDA, Float32, Int32) == ZERO
    end

end
