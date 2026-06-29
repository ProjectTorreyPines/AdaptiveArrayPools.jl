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
