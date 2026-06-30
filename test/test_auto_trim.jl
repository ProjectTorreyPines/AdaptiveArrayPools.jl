using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!
import AdaptiveArrayPools as AAP

# ==============================================================================
# Auto-trim — periodic working-set tail reclamation (the second action of the
# unified auto_manage engine). Design: docs/plans/DESIGN_auto_trim.md
#
# A `TypedPool` records the peak `n_active` reached since the last auto-trim in
# `_am_peak_n_active` (one hot-path `max` in `_claim_slot!`, gated by AUTO_MANAGE).
# Every `trim_interval`, the Timer sets the pool's `@atomic _trim_requested`; the
# owner services it at a `@with_pool` entry — trimming each type's slot tail down
# to that recent peak (a type unused for the period trims to 0), then resetting it.
#
# AUTO_MANAGE defaults on; we stop the __init__ timer up front for determinism.
# ==============================================================================

AAP.disable_auto_manage!()

@testset "auto_trim" begin

    _clear_registry!() = lock(() -> empty!(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)

    # ── Data layer ───────────────────────────────────────────────────────────
    @testset "TypedPool._am_peak_n_active (default 0) + pool._trim_requested (default false)" begin
        pool = AdaptiveArrayPool{0}()
        @test pool.float64._am_peak_n_active == 0
        @test (@atomic pool._trim_requested) == false
        @atomic pool._trim_requested = true
        @test (@atomic pool._trim_requested) == true
    end

    @testset "_claim_slot! records the peak n_active in _am_peak_n_active" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool)
        acquire!(pool, Float64, 4); acquire!(pool, Float64, 4); acquire!(pool, Float64, 4)  # n_active → 3
        rewind!(pool)
        @test pool.float64._am_peak_n_active == 3          # peak captured (not reset by rewind)
        # a narrower second period after a manual reset stays at the new peak
        pool.float64._am_peak_n_active = 0
        checkpoint!(pool); acquire!(pool, Float64, 4); rewind!(pool)
        @test pool.float64._am_peak_n_active == 1
    end

    # ── _trim_to! primitive ──────────────────────────────────────────────────
    @testset "_trim_to!(tp, keep) truncates vectors/wrappers/slot_extents; slot regrows" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool)
        acquire!(pool, Float64, 100); acquire!(pool, Float64, 100); acquire!(pool, Float64, 100)
        rewind!(pool)
        tp = pool.float64
        @test length(tp.vectors) == 3 && length(tp.slot_extents) == 3

        AAP._trim_to!(tp, 1)
        @test length(tp.vectors) == 1
        @test length(tp.slot_extents) == 1

        checkpoint!(pool); a = acquire!(pool, Float64, 60); a .= 7.0; rewind!(pool)  # regrow slot 1
        @test length(a) == 60 && all(a .== 7.0)
        AAP._trim_to!(tp, 0)                               # trim everything
        @test length(tp.vectors) == 0
    end

    # ── Service: auto-trim to the recent peak ────────────────────────────────
    @testset "_run_auto_manage! trims each type's tail to its recent peak, then resets" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool)                                  # wide: 3 concurrent
        acquire!(pool, Float64, 100); acquire!(pool, Float64, 100); acquire!(pool, Float64, 100)
        rewind!(pool)
        pool.float64._am_peak_n_active = 0                 # start a fresh observation period
        checkpoint!(pool); acquire!(pool, Float64, 100); rewind!(pool)   # narrowed: peak = 1
        @test pool.float64._am_peak_n_active == 1
        @test length(pool.float64.vectors) == 3            # tail still retained

        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)

        @test length(pool.float64.vectors) == 1            # trimmed to the recent peak
        @test pool.float64._am_peak_n_active == 0          # reset for the next period
        @test (@atomic pool._trim_requested) == false      # flag consumed
    end

    @testset "a type unused for the period trims to 0" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool); acquire!(pool, Int64, 1000); acquire!(pool, Int64, 1000); rewind!(pool)
        pool.int64._am_peak_n_active = 0                   # new period, then NO Int64 use
        @test length(pool.int64.vectors) == 2
        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)
        @test length(pool.int64.vectors) == 0              # unused type fully dropped
    end

    @testset "no-op when _trim_requested is clear" begin
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool); acquire!(pool, Float64, 100); acquire!(pool, Float64, 100); rewind!(pool)
        pool.float64._am_peak_n_active = 0
        AAP._run_auto_manage!(pool)                        # no flag set
        @test length(pool.float64.vectors) == 2            # untouched
    end

    # ── Overhead + safety ────────────────────────────────────────────────────
    @testset "the working-set `max` keeps acquire! zero-alloc" begin
        pool = AdaptiveArrayPool{0}()
        g(p) = (checkpoint!(p); acquire!(p, Float64, 8); acquire!(p, Float64, 8); rewind!(p); nothing)
        g(pool); g(pool)                               # warmup
        @test @allocated(g(pool)) == 0                 # the max adds no allocation
        @test pool.float64._am_peak_n_active >= 2      # …and it is tracking the peak
    end

    @testset "RUNTIME_CHECK=1: auto-trim drops the slot ref, preserving poison (no buffer swap)" begin
        pool = AdaptiveArrayPool{1}()                  # runtime checks ON
        checkpoint!(pool); a = acquire!(pool, Float64, 100); a .= 1.0; rewind!(pool)  # released → poisoned
        tp = pool.float64
        @test length(tp.vectors) == 1
        backing = tp.vectors[1]                        # the poisoned backing an escaped view would hold
        mem_before = getfield(backing, :ref).mem
        pool.float64._am_peak_n_active = 0             # unused this period → trim to 0
        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)
        @test length(tp.vectors) == 0                  # slot reference dropped
        @test getfield(backing, :ref).mem === mem_before  # buffer NOT swapped → poison intact (unlike compact!)
    end

    # ── Config + timer cadence ───────────────────────────────────────────────
    @testset "enable_auto_manage! accepts trim_interval; timer flags trim every period" begin
        _clear_registry!()
        AAP.disable_auto_manage!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_manage!(pool)
        # interval 0.1s, trim every 0.2s (K=2 ticks): within ~0.5s, _trim_requested fires
        AAP.enable_auto_manage!(interval = 0.1, trim_interval = 0.2)
        sleep(0.55)
        @test (@atomic pool._trim_requested) == true
        AAP.disable_auto_manage!()
        _clear_registry!()
    end

    @testset "trim_interval = Inf disables auto-trim (compact-only)" begin
        _clear_registry!()
        AAP.disable_auto_manage!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_manage!(pool)
        AAP.enable_auto_manage!(interval = 0.1, trim_interval = Inf)
        sleep(0.35)
        @test (@atomic pool._trim_requested) == false      # never flagged for trim
        AAP.disable_auto_manage!()
        _clear_registry!()
    end

    @testset "auto-trim reclaims a non-fixed-slot (`others`) type" begin
        # `_auto_trim!` trims the fixed slots AND every fallback type in `pool.others`.
        # UInt8 is not a fixed slot (FIXED_SLOT_FIELDS), so this exercises the `others` loop.
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool); acquire!(pool, UInt8, 100); acquire!(pool, UInt8, 100); rewind!(pool)
        tp = AAP.get_typed_pool!(pool, UInt8)
        @test length(tp.vectors) == 2
        tp._am_peak_n_active = 0                            # unused this period → trim the tail to 0
        @atomic pool._trim_requested = true
        AAP._run_auto_manage!(pool)
        @test length(tp.vectors) == 0                       # the `others`-type slots were reclaimed
    end
end

AAP.disable_auto_manage!()
