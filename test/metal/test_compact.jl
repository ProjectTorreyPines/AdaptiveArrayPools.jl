# Metal compact! Tests
# Parity with CPU test_compact.jl: shrink an over-allocated device buffer in place.
#
# Metal has NO uncached views — acquire_view! == acquire! (returns a cached MtlArray
# wrapper) — so the CPU view-extent bug is structurally absent here. We still verify
# slot_extents-based compaction shrinks the device buffer and re-syncs the held
# wrapper (the DataRef swap + refcount path) on real hardware. Uses Float32/Int32
# (Metal's fixed types; no Float64 on Metal).

@testset "Metal compact!" begin

    # Device capacity (elements) of a Metal backing buffer = maxsize bytes / sizeof(T).
    _mcap(v) = Int(getfield(v, :maxsize) ÷ sizeof(eltype(v)))
    Summary = @NamedTuple{slots_compacted::Int, bytes_reclaimed::Int, gc_triggered::Bool}
    ZERO = (; slots_compacted = 0, bytes_reclaimed = 0, gc_triggered = false)

    @testset "inactive: compact!(pool) shrinks the device buffer" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        # slot 1: small last-use (100) over a large high-water device buffer (1M)
        checkpoint!(pool); a = acquire!(pool, Float32, 1_000_000); fill!(a, 0.0f0); rewind!(pool)
        checkpoint!(pool); b = acquire!(pool, Float32, 100); fill!(b, 0.0f0); rewind!(pool)
        tp = pool.float32
        @test tp.n_active == 0
        @test _mcap(tp.vectors[1]) >= 1_000_000

        s = compact!(pool)

        @test s.slots_compacted == 1
        @test s.bytes_reclaimed > 0
        @test s.gc_triggered == false
        @test _mcap(tp.vectors[1]) < 1000                 # device buffer shrunk in place
    end

    @testset "active=true shrinks a HELD slot in place; data intact + writable" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool); a = acquire!(pool, Float32, 1_000_000); fill!(a, 0.0f0); rewind!(pool)
        checkpoint!(pool)
        b = acquire!(pool, Float32, 100)                  # slot 1 ACTIVE (b held)
        copyto!(b, Float32.(1:100))
        tp = pool.float32
        @test tp.n_active == 1
        cap0 = _mcap(tp.vectors[1])
        @test cap0 >= 1_000_000

        # default (active=false) leaves the held slot untouched
        @test compact!(pool; active = false).slots_compacted == 0
        @test _mcap(tp.vectors[1]) == cap0

        # active=true shrinks the device buffer the user is still holding
        s = compact!(pool; active = true)
        @test s.slots_compacted == 1
        @test _mcap(tp.vectors[1]) < 1000
        @test Array(b) == Float32.(1:100)                 # held wrapper re-synced: data intact
        copyto!(b, fill(9.0f0, 100))                      # ...and still writable into the new buffer
        @test Array(b)[1] == 9.0f0
        rewind!(pool)
    end

    @testset "no-op + force_gc + type stability + DisabledPool" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        @test compact!(pool) == ZERO
        @test compact!(pool; force_gc = true).gc_triggered == true
        @test only(Base.return_types(compact!, (typeof(pool),))) == Summary
        @test (@inferred compact!(pool)) isa Summary
        @test compact!(ext.DISABLED_METAL) == ZERO
        @test compact!(ext.DISABLED_METAL, Float32) == ZERO
    end

    @testset "per-type + varargs (Metal fixed types)" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool); acquire!(pool, Float32, 1_000_000); rewind!(pool)
        checkpoint!(pool); acquire!(pool, Float32, 100); rewind!(pool)
        checkpoint!(pool); acquire!(pool, Int32, 1_000_000); rewind!(pool)
        checkpoint!(pool); acquire!(pool, Int32, 100); rewind!(pool)

        s = compact!(pool, Float32, Int32)
        @test s.slots_compacted == 2
        @test _mcap(pool.float32.vectors[1]) < 1000
        @test _mcap(pool.int32.vectors[1]) < 1000
        @test only(Base.return_types(compact!, (typeof(pool), Type{Float32}, Type{Int32}))) == Summary
        @test compact!(ext.DISABLED_METAL, Float32, Int32) == ZERO
    end

    @testset "active=true: per-slot copy decision (active preserved, inactive shrinks)" begin
        # Parity with CPU `_compact_slot!`'s `copy_live`: the active-vs-inactive copy
        # decision is PER SLOT (slot ≤ n_active ⇒ active ⇒ copy live data; else inactive ⇒
        # SKIP the GPU→GPU copy of dead data). Build two bloated slots, hold ONLY slot 1,
        # compact both: both shrink, slot 1's held data survives, and slot 2 (inactive,
        # copy skipped) still re-acquires correctly. (Inactive contents are dead → never
        # asserted — the point is correctness despite the skipped copy.)
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool)                               # round 1: grow two slots to 1M
        acquire!(pool, Float32, 1_000_000)
        acquire!(pool, Float32, 1_000_000)
        rewind!(pool)
        checkpoint!(pool)                               # round 2: reuse both at 100 → bloated
        acquire!(pool, Float32, 100)
        acquire!(pool, Float32, 100)
        rewind!(pool)
        checkpoint!(pool)                               # round 3: hold ONLY slot 1 active
        h1 = acquire!(pool, Float32, 100)
        copyto!(h1, Float32.(1:100))
        tp = pool.float32
        @test tp.n_active == 1                          # slot 1 active, slot 2 inactive
        @test _mcap(tp.vectors[1]) >= 1_000_000
        @test _mcap(tp.vectors[2]) >= 1_000_000

        s = compact!(pool; active = true)

        @test s.slots_compacted == 2                    # BOTH slots shrink
        @test _mcap(tp.vectors[1]) < 1000               # active slot 1 shrunk
        @test _mcap(tp.vectors[2]) < 1000               # inactive slot 2 shrunk
        @test Array(h1) == Float32.(1:100)              # ACTIVE slot: live data preserved (copy_live)
        rewind!(pool)

        checkpoint!(pool)                               # compacted INACTIVE slot re-acquires OK
        acquire!(pool, Float32, 120)                    # slot 1
        r2 = acquire!(pool, Float32, 120)               # slot 2 (was inactive, copy-skipped)
        copyto!(r2, fill(5.0f0, 120))
        @test length(r2) == 120
        @test Array(r2) == fill(5.0f0, 120)             # usable despite the skipped copy
        rewind!(pool)
    end

end
