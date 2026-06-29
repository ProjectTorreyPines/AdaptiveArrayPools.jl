using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!
import AdaptiveArrayPools as AAP

# ==============================================================================
# Auto-compact — timer-driven background capacity compaction.
# Design: docs/plans/DESIGN_auto_compact.md
#
# Architecture: ONE global Timer sweeps a Vector{WeakRef} registry, setting each
# live pool's `@atomic _compact_requested` flag. Each pool's owner task runs the
# full compact! at its next `_current_depth == 1` scope-exit safepoint.
#
# Phase 1 covers the data layer (flag field + registry + config + run), with no
# timer and no macro hook yet — every piece is exercised directly.
# ==============================================================================

# Allocated backing capacity (Memory length), mirroring test_compact.jl.
_cap(v) = length(getfield(v, :ref).mem)

# An inactive, bloated Float64 slot (cap = `big` high-water, last extent = `small`).
function _inactive_bloated(big::Int, small::Int)
    pool = AdaptiveArrayPool{0}()
    checkpoint!(pool); a = acquire!(pool, Float64, big); a .= 0.0; rewind!(pool)
    checkpoint!(pool); b = acquire!(pool, Float64, small); b .= 0.0; rewind!(pool)
    return pool
end

_clear_registry!() = lock(() -> empty!(AAP._AUTO_COMPACT_REGISTRY), AAP._AUTO_COMPACT_LOCK)
_registry_len() = lock(() -> length(AAP._AUTO_COMPACT_REGISTRY), AAP._AUTO_COMPACT_LOCK)

@testset "auto_compact — Phase 1: flag field + registry + run" begin

    @testset "pool carries an atomic _compact_requested flag (default false)" begin
        pool = AdaptiveArrayPool{0}()
        @test (@atomic pool._compact_requested) == false
        @atomic pool._compact_requested = true
        @test (@atomic pool._compact_requested) == true
    end

    @testset "registry: sweep flags every live pool and prunes dead refs" begin
        _clear_registry!()
        p1 = AdaptiveArrayPool{0}()
        p2 = AdaptiveArrayPool{0}()
        AAP.register_auto_compact!(p1)
        AAP.register_auto_compact!(p2)
        # inject a guaranteed-dead entry to exercise pruning
        lock(() -> push!(AAP._AUTO_COMPACT_REGISTRY, WeakRef(nothing)), AAP._AUTO_COMPACT_LOCK)
        @test _registry_len() == 3

        AAP._auto_compact_sweep!(nothing)            # the timer callback body

        @test (@atomic p1._compact_requested) == true
        @test (@atomic p2._compact_requested) == true
        @test _registry_len() == 2                   # dead WeakRef pruned
        _clear_registry!()
    end

    @testset "_run_auto_compact! resets the flag and compacts inactive bloat" begin
        pool = _inactive_bloated(1_000_000, 100)
        @atomic pool._compact_requested = true
        @test _cap(pool.float64.vectors[1]) >= 1_000_000

        AAP._run_auto_compact!(pool)

        @test (@atomic pool._compact_requested) == false   # reset (before the work)
        @test _cap(pool.float64.vectors[1]) == 150          # shrunk to ceil(1.5 * 100)
    end

    @testset "_run_auto_compact! uses the global config defaults" begin
        @test AAP._AUTO_COMPACT_CONFIG.factor == 10
        @test AAP._AUTO_COMPACT_CONFIG.shrink_to == 1.5
        @test AAP._AUTO_COMPACT_CONFIG.min_bytes == 2^20
        @test AAP._AUTO_COMPACT_CONFIG.active == true
    end
end

@testset "auto_compact — Phase 2: timer lifecycle + enable/disable" begin
    # In the default test build the `auto_compact` Preference is off, so
    # `enable_auto_compact!` emits a guidance @warn — asserted via @test_logs (which
    # also captures it, keeping output clean). The timer still starts, so the
    # sweep→flag mechanism is exercisable against a manually-registered pool.

    @testset "enable! sets config and starts a timer that flags registered pools" begin
        _clear_registry!()
        AAP.disable_auto_compact!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_compact!(pool)

        @test_logs (:warn,) match_mode = :any AAP.enable_auto_compact!(
            interval = 0.1, factor = 20, shrink_to = 2.0, min_bytes = 4096, active = false,
        )
        @test AAP.auto_compact_enabled() == true
        @test AAP._AUTO_COMPACT_CONFIG.factor == 20.0
        @test AAP._AUTO_COMPACT_CONFIG.shrink_to == 2.0
        @test AAP._AUTO_COMPACT_CONFIG.min_bytes == 4096
        @test AAP._AUTO_COMPACT_CONFIG.active == false

        sleep(0.35)                                    # ~3 ticks @ 0.1s
        @test (@atomic pool._compact_requested) == true

        AAP.disable_auto_compact!()
        @test AAP.auto_compact_enabled() == false
        # restore config defaults for the other testsets
        AAP._AUTO_COMPACT_CONFIG.factor = 10.0
        AAP._AUTO_COMPACT_CONFIG.shrink_to = 1.5
        AAP._AUTO_COMPACT_CONFIG.min_bytes = 2^20
        AAP._AUTO_COMPACT_CONFIG.active = true
        _clear_registry!()
    end

    @testset "disable! stops the timer and is idempotent" begin
        _clear_registry!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_compact!(pool)
        @test_logs (:warn,) match_mode = :any AAP.enable_auto_compact!(interval = 0.1)
        AAP.disable_auto_compact!()
        @test AAP.auto_compact_enabled() == false
        @atomic pool._compact_requested = false
        sleep(0.3)
        @test (@atomic pool._compact_requested) == false    # no ticks after disable
        AAP.disable_auto_compact!()                          # idempotent: must not throw
        @test AAP.auto_compact_enabled() == false
        _clear_registry!()
    end

    @testset "enable! replaces an existing timer (closes the old one)" begin
        AAP.disable_auto_compact!()
        @test_logs (:warn,) match_mode = :any AAP.enable_auto_compact!(interval = 0.1)
        t1 = AAP._AUTO_COMPACT_TIMER[]
        @test_logs (:warn,) match_mode = :any AAP.enable_auto_compact!(interval = 0.2)
        t2 = AAP._AUTO_COMPACT_TIMER[]
        @test t1 !== t2
        @test !isopen(t1)                                    # old timer closed
        @test isopen(t2)
        AAP.disable_auto_compact!()
    end

    @testset "a throwing sweep is caught — the timer survives" begin
        _clear_registry!()
        bogus = Ref(42)                                      # heap object, NOT a pool
        lock(() -> push!(AAP._AUTO_COMPACT_REGISTRY, WeakRef(bogus)), AAP._AUTO_COMPACT_LOCK)
        # the `::AdaptiveArrayPool` assertion in the sweep throws on the bogus entry;
        # the _safe wrapper must swallow it so the Timer would keep running.
        @test (AAP._safe_auto_compact_sweep!(nothing); true)
        @test bogus[] == 42                                  # keep `bogus` alive past the sweep
        _clear_registry!()
    end
end
