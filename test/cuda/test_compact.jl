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

    @testset "active=true preserves a live reshape! alias (cross-slot, different dim)" begin
        # `reshape!(pool, a, dims)` with M≠N stores the reshaped CuArray wrapper in a SEPARATE
        # placeholder slot while SHARING `a`'s device buffer (same DataRef). An active compact!
        # that shrinks `a`'s slot must re-point that cross-slot alias too — else `a` follows the
        # new buffer while the reshape view is stranded on the freed buffer. Regression for the
        # per-slot-only wrapper re-sync in the CUDA `_compact_slot!`.
        pool = CuAdaptiveArrayPool{0}()
        checkpoint!(pool); big = acquire!(pool, Float32, 200_000); rewind!(pool)   # slot 1 high-water
        checkpoint!(pool)
        a = acquire!(pool, Float32, 100); fill!(a, 1.0f0)     # reuse slot 1: bloated (used=100)
        b = reshape!(pool, a, 10, 10)                          # 2-D alias of `a` (placeholder slot 2)
        tp = pool.float32
        @test getfield(a, :data).rc === getfield(b, :data).rc # one shared device buffer before compact
        @test _ccap(tp.vectors[1]) >= 200_000

        s = compact!(pool; active = true, min_bytes = 0)      # shrink slot 1's device buffer in place
        @test s.slots_compacted == 1

        # The alias must SURVIVE: still one buffer, and a write through `a` is visible in `b`.
        @test getfield(a, :data).rc === getfield(b, :data).rc # ← reshape view follows the new buffer
        fill!(a, 42.0f0)
        @test all(==(42.0f0), Array(b))                        # data flows across the alias post-compact
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

    @testset "active=true: per-slot copy decision (active preserved, inactive shrinks)" begin
        # Parity with CPU `_compact_slot!`'s `copy_live`: the active-vs-inactive copy
        # decision is PER SLOT (slot ≤ n_active ⇒ active ⇒ copy live data; else inactive ⇒
        # SKIP the GPU→GPU copy of dead data). Build two bloated slots, hold ONLY slot 1,
        # compact both: both shrink, slot 1's held data survives, and slot 2 (inactive,
        # copy skipped) still re-acquires correctly. (Inactive contents are dead → never
        # asserted — the point is correctness despite the skipped copy.)
        pool = CuAdaptiveArrayPool{0}()
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
        @test _ccap(tp.vectors[1]) >= 1_000_000
        @test _ccap(tp.vectors[2]) >= 1_000_000

        s = compact!(pool; active = true)

        @test s.slots_compacted == 2                    # BOTH slots shrink
        @test _ccap(tp.vectors[1]) < 1000               # active slot 1 shrunk
        @test _ccap(tp.vectors[2]) < 1000               # inactive slot 2 shrunk
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

    @testset "RUNTIME_CHECK=1: compact! leaves poisoned inactive slots untouched" begin
        # Parity with CPU: under S=1, releasing a slot poisons it AND shrinks its logical
        # length to 0 (device buffer retained) so an escaped view reads NaN/typemax. compact!
        # must NOT swap fresh device storage into such a slot — that would undo the poison.
        pool = CuAdaptiveArrayPool{1}()
        checkpoint!(pool); a = acquire!(pool, Float32, 1_000_000); fill!(a, 0.0f0); rewind!(pool)
        checkpoint!(pool); b = acquire!(pool, Float32, 100); fill!(b, 0.0f0); rewind!(pool)
        tp = pool.float32
        @test tp.n_active == 0
        @test length(tp.vectors[1]) == 0             # invalidated: logical length 0
        cap0 = _ccap(tp.vectors[1])                   # retained device capacity (~1M)
        @test cap0 >= 1_000_000

        @test compact!(pool).slots_compacted == 0    # poisoned slot skipped
        @test _ccap(tp.vectors[1]) == cap0            # device capacity (and poison) preserved
        @test compact!(pool; active = false).slots_compacted == 0
    end

end
