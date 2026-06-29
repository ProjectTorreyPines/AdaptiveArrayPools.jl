using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!
import AdaptiveArrayPools as AAP

# ==============================================================================
# Auto-compact — timer-driven background capacity compaction.
# Design: docs/plans/DESIGN_auto_compact.md
#
# ONE global Timer sweeps a Vector{WeakRef} registry, setting each live pool's
# `@atomic _compact_requested` flag. Each pool's owner task runs the full compact!
# at its next `_current_depth == 1` scope-exit safepoint (the macro hook).
#
# AUTO_COMPACT defaults to `true`, so __init__ auto-starts the timer and the
# @with_pool hook is live. We stop that timer up front so each testset controls it
# explicitly and the suite stays deterministic.
# ==============================================================================

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

AAP.disable_auto_compact!()   # stop the __init__-started timer for deterministic tests

@testset "auto_compact" begin

    # ── Data layer: flag field + registry + run ──────────────────────────────
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

    # ── Timer lifecycle (enable/disable). AUTO_COMPACT is on → enable does NOT warn ─
    @testset "enable! sets config and starts a timer that flags registered pools" begin
        _clear_registry!()
        AAP.disable_auto_compact!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_compact!(pool)

        AAP.enable_auto_compact!(
            interval = 0.1, factor = 20, shrink_to = 2.0, min_bytes = 4096, active = false,
        )
        @test AAP.auto_compact_enabled() == true
        @test AAP._AUTO_COMPACT_CONFIG.factor == 20.0
        @test AAP._AUTO_COMPACT_CONFIG.shrink_to == 2.0
        @test AAP._AUTO_COMPACT_CONFIG.min_bytes == 4096
        @test AAP._AUTO_COMPACT_CONFIG.active == false

        sleep(0.35)                                  # ~3 ticks @ 0.1s
        @test (@atomic pool._compact_requested) == true

        AAP.disable_auto_compact!()
        @test AAP.auto_compact_enabled() == false
        AAP._AUTO_COMPACT_CONFIG.factor = 10.0       # restore defaults
        AAP._AUTO_COMPACT_CONFIG.shrink_to = 1.5
        AAP._AUTO_COMPACT_CONFIG.min_bytes = 2^20
        AAP._AUTO_COMPACT_CONFIG.active = true
        _clear_registry!()
    end

    @testset "disable! stops the timer and is idempotent" begin
        _clear_registry!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_compact!(pool)
        AAP.enable_auto_compact!(interval = 0.1)
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
        AAP.enable_auto_compact!(interval = 0.1)
        t1 = AAP._AUTO_COMPACT_TIMER[]
        AAP.enable_auto_compact!(interval = 0.2)
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
        @test (AAP._safe_auto_compact_sweep!(nothing); true) # _safe wrapper swallows the throw
        @test bogus[] == 42                                  # keep `bogus` alive past the sweep
        _clear_registry!()
    end

    # ── Scope-exit hook ──────────────────────────────────────────────────────
    @testset "_maybe_auto_compact! fires only at _current_depth == 1" begin
        pool = _inactive_bloated(1_000_000, 100)
        checkpoint!(pool)                                    # depth → 2
        @atomic pool._compact_requested = true
        AAP._maybe_auto_compact!(pool)                       # depth 2 ≠ 1 → must NOT act
        @test (@atomic pool._compact_requested) == true      # flag preserved for the outer exit
        @test _cap(pool.float64.vectors[1]) >= 1_000_000     # not compacted
        rewind!(pool)                                        # depth → 1
        AAP._maybe_auto_compact!(pool)                       # depth 1 → fires
        @test (@atomic pool._compact_requested) == false     # reset
        @test _cap(pool.float64.vectors[1]) == 150           # compacted
    end

    @testset "_maybe_auto_compact! is a no-op when the flag is clear" begin
        pool = _inactive_bloated(1_000_000, 100)
        @test (@atomic pool._compact_requested) == false
        AAP._maybe_auto_compact!(pool)
        @test _cap(pool.float64.vectors[1]) >= 1_000_000     # untouched (no request)
    end

    @testset "_maybe_auto_compact! no-ops on non-CPU pools (GPU fallback)" begin
        @test AAP._maybe_auto_compact!(42) === nothing       # ::Any fallback, no throw
        @test AAP._maybe_auto_compact!("not a pool") === nothing
    end

    @testset "macro wires the gated scope-exit hook into @with_pool" begin
        s = string(
            @macroexpand @with_pool pool begin
                x = acquire!(pool, Float64, 4)
                sum(x)
            end
        )
        @test occursin("_maybe_auto_compact!", s)            # hook emitted…
        @test occursin("AUTO_COMPACT", s)                    # …gated by the compile-time const
    end

    # ── Default-on integration (AUTO_COMPACT === true) ───────────────────────
    @testset "AUTO_COMPACT is on by default" begin
        @test AAP.AUTO_COMPACT == true
    end

    @testset "__init__ auto-starts the timer when AUTO_COMPACT is on" begin
        AAP.disable_auto_compact!()
        @test AAP.auto_compact_enabled() == false
        AAP.__init__()                                       # re-run module init
        @test AAP.auto_compact_enabled() == true             # auto-started
        AAP.disable_auto_compact!()
    end

    @testset "get_task_local_pool auto-registers the pool (gate on)" begin
        _clear_registry!()
        fetch(Threads.@spawn get_task_local_pool())          # fresh task → creates + registers
        @test _registry_len() >= 1
        _clear_registry!()
    end

    @testset "end-to-end: bloat auto-compacts at a @with_pool boundary" begin
        AAP.disable_auto_compact!()
        _clear_registry!()
        pool = get_task_local_pool()
        empty!(pool)                                         # clean slate
        AAP.register_auto_compact!(pool)
        @with_pool p begin                                   # grow slot to 1M high-water
            x = acquire!(p, Float64, 1_000_000); x .= 0.0
        end
        @with_pool p begin                                   # reuse small → bloated, inactive
            x = acquire!(p, Float64, 100); x .= 0.0
        end
        cap0 = _cap(pool.float64.vectors[1])
        @test cap0 >= 1_000_000

        # Set the request flag directly (deterministic — the timer→flag link is covered by
        # the "enable! flags registered pools" test). This exercises the unique end-to-end
        # piece: the MACRO-emitted hook firing at a real @with_pool exit and running compact!.
        @atomic pool._compact_requested = true
        @with_pool p begin                                   # scope exit at depth 1 → hook fires
            acquire!(p, Float64, 4)
        end

        @test (@atomic pool._compact_requested) == false     # hook consumed the request
        @test _cap(pool.float64.vectors[1]) < cap0           # auto-compacted, no manual compact!
        empty!(pool)
        _clear_registry!()
    end
end

AAP.disable_auto_compact!()   # leave the timer stopped for later test files
