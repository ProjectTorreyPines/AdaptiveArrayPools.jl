# Metal Auto-Trim Tests
# ======================
# Parity with the CPU suite (test/test_auto_trim.jl) for the Metal backend — the second
# action of the unified auto_manage engine. Design: docs/plans/DESIGN_auto_trim.md
#
# A `MetalTypedPool` records the peak `n_active` reached since the last auto-trim in
# `_am_peak_n_active` (one hot-path `max` in `_metal_claim_slot!`, gated by AUTO_MANAGE).
# Every `trim_interval`, the base-module Timer sets the pool's `@atomic _trim_requested`;
# the owner services it at a `@with_pool :metal` entry — trimming each type's slot tail
# down to that recent peak (a type unused for the period trims to 0), then resetting it.
#
# The service layer (`_trim_to!`, `_auto_trim!`, `_run_auto_manage!`) is generic over
# `AbstractTypedPool`/`AbstractArrayPool`, so this backend needs only the two data-layer
# fields + the claim-path `max`. Uses Float32 (Metal's primary type; no Float64 on Metal).

import AdaptiveArrayPools as AAP

AAP.disable_auto_manage!()   # stop the __init__-started timer for deterministic tests

@testset "Metal auto_trim" begin

    # Scoped INSIDE the testset (local scope) so the helper doesn't define a top-level
    # `Main` method colliding with the identically-named helper in the CPU/CUDA auto-trim
    # files when the full suite loads several backends together.
    _clear_registry!() = lock(() -> empty!(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)
    _registry_len() = lock(() -> length(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)

    # ── Data layer ───────────────────────────────────────────────────────────
    @testset "MetalTypedPool._am_peak_n_active (default 0) + pool._trim_requested (default false)" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        @test pool.float32._am_peak_n_active == 0
        @test (@atomic pool._trim_requested) == false
        @atomic pool._trim_requested = true
        @test (@atomic pool._trim_requested) == true
    end

    @testset "_metal_claim_slot! records the peak n_active in _am_peak_n_active" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool)
        acquire!(pool, Float32, 4); acquire!(pool, Float32, 4); acquire!(pool, Float32, 4)  # n_active → 3
        rewind!(pool)
        @test pool.float32._am_peak_n_active == 3          # peak captured (not reset by rewind)
        # a narrower second period after a manual reset stays at the new peak
        pool.float32._am_peak_n_active = 0
        checkpoint!(pool); acquire!(pool, Float32, 4); rewind!(pool)
        @test pool.float32._am_peak_n_active == 1
    end

    # ── _trim_to! primitive (generic, dispatches on MetalTypedPool) ───────────
    @testset "_trim_to!(tp, keep) truncates vectors/wrappers/slot_extents; slot regrows" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool)
        acquire!(pool, Float32, 100); acquire!(pool, Float32, 100); acquire!(pool, Float32, 100)
        rewind!(pool)
        tp = pool.float32
        @test length(tp.vectors) == 3 && length(tp.slot_extents) == 3

        AAP._trim_to!(tp, 1)
        @test length(tp.vectors) == 1
        @test length(tp.slot_extents) == 1

        checkpoint!(pool); a = acquire!(pool, Float32, 60); fill!(a, 7.0f0); rewind!(pool)  # regrow slot 1
        @test length(a) == 60 && all(==(7.0f0), Array(a))
        AAP._trim_to!(tp, 0)                               # trim everything
        @test length(tp.vectors) == 0
    end

    # ── Service: auto-trim to the recent peak ────────────────────────────────
    @testset "_run_auto_manage! trims each type's tail to its recent peak, then resets" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool)                                  # wide: 3 concurrent
        acquire!(pool, Float32, 100); acquire!(pool, Float32, 100); acquire!(pool, Float32, 100)
        rewind!(pool)
        pool.float32._am_peak_n_active = 0                 # start a fresh observation period
        checkpoint!(pool); acquire!(pool, Float32, 100); rewind!(pool)   # narrowed: peak = 1
        @test pool.float32._am_peak_n_active == 1
        @test length(pool.float32.vectors) == 3            # tail still retained

        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)

        @test length(pool.float32.vectors) == 1            # trimmed to the recent peak
        @test pool.float32._am_peak_n_active == 0          # reset for the next period
        @test (@atomic pool._trim_requested) == false      # flag consumed
    end

    @testset "a type unused for the period trims to 0" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool); acquire!(pool, Int32, 1000); acquire!(pool, Int32, 1000); rewind!(pool)
        pool.int32._am_peak_n_active = 0                   # new period, then NO Int32 use
        @test length(pool.int32.vectors) == 2
        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)
        @test length(pool.int32.vectors) == 0              # unused type fully dropped
    end

    @testset "no-op when _trim_requested is clear" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool); acquire!(pool, Float32, 100); acquire!(pool, Float32, 100); rewind!(pool)
        pool.float32._am_peak_n_active = 0
        AAP._run_auto_manage!(pool)                        # no flag set
        @test length(pool.float32.vectors) == 2            # untouched
    end

    # ── Overhead note ────────────────────────────────────────────────────────
    # The hot-path `max` writes one `Int` field and is byte-identical to the CPU claim
    # path, whose zero-allocation is proven in test/test_auto_trim.jl. We do NOT re-assert
    # `@allocated == 0` here: Metal `acquire!` carries inherent CPU-side kernel-launch
    # overhead (~56 B/call; see test/metal/test_allocation.jl) that swamps the `max` and
    # would make a strict-zero assertion test Metal's acquire overhead, not the `max`.
    @testset "peak tracking is live through real acquire! (functional guard)" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool); acquire!(pool, Float32, 8); acquire!(pool, Float32, 8); rewind!(pool)
        @test pool.float32._am_peak_n_active >= 2          # the max is tracking the peak
    end

    # ── Safety (R=1): auto-trim drops the slot ref WITHOUT a buffer swap ──────
    # Unlike `compact!` (which allocates a smaller device buffer and swaps the slot's
    # DataRef via `_update_metal_wrapper_data!`), `_trim_to!` only drops the slot
    # reference. The captured MtlArray — and any escaped view sharing its DataRef — is
    # left byte-for-byte untouched, so R=1 poison on a dropped slot survives.
    @testset "R=1: auto-trim leaves the dropped slot's MtlArray DataRef intact (no swap)" begin
        pool = MetalAdaptiveArrayPool{1, METAL_STORAGE}()   # runtime checks ON
        checkpoint!(pool); a = acquire!(pool, Float32, 100); fill!(a, 1.0f0); rewind!(pool)  # released
        tp = pool.float32
        @test length(tp.vectors) == 1
        v = tp.vectors[1]                                   # the backing an escaped view would share
        data_before = getfield(v, :data)                    # its DataRef (compact! would swap this)
        pool.float32._am_peak_n_active = 0                  # unused this period → trim to 0
        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)
        @test length(tp.vectors) == 0                       # slot reference dropped
        @test getfield(v, :data) === data_before            # buffer NOT swapped → escaped view intact
    end

    # ── Cadence: the sweep now flags _trim_requested on a Metal pool ──────────
    # Proves the `hasfield(typeof(p), :_trim_requested)` guard in the sweep (added in the
    # CPU phase) now passes for Metal pools — i.e. the field is wired into this backend.
    @testset "registry sweep flags _trim_requested on a registered Metal pool" begin
        _clear_registry!()
        AAP.disable_auto_manage!()
        p = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        AAP.register_auto_manage!(p)
        @test _registry_len() == 1
        AAP.enable_auto_manage!(compact_interval = 0.1, trim_interval = 0.1)   # K=1: trim every sweep
        AAP._auto_manage_sweep!(nothing)                    # one sweep body → trim cadence hits
        @test (@atomic p._trim_requested) == true
        AAP.disable_auto_manage!()
        _clear_registry!()
    end
end

AAP.disable_auto_manage!()   # leave the timer stopped for later Metal test files
