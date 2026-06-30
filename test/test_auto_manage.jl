using Test
using Base.CoreLogging: with_logger, NullLogger   # from Base — no `Logging` dep needed
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!
import AdaptiveArrayPools as AAP

# ==============================================================================
# Auto-compact — timer-driven background capacity compaction.
# Design: docs/plans/DESIGN_auto_manage.md
#
# ONE global Timer sweeps a Vector{WeakRef} registry, setting each live pool's
# `@atomic _compact_requested` flag. Each pool's owner task runs the full compact!
# at its next `@with_pool` entry (the `_current_depth == 1` safepoint; the macro hook).
#
# AUTO_MANAGE defaults to `true`, so __init__ auto-starts the timer and the
# @with_pool hook is live. We stop that timer up front so each testset controls it
# explicitly and the suite stays deterministic.
# ==============================================================================

AAP.disable_auto_manage!()   # stop the __init__-started timer for deterministic tests

@testset "auto_manage" begin

    # Helpers are scoped INSIDE this testset (a local scope) so they don't define top-level
    # methods in `Main` that collide with identically-named helpers in other test files when
    # the full suite loads them together — e.g. `_cap` is also defined at the top of
    # test_compact.jl (otherwise: "WARNING: Method definition _cap(Any) ... overwritten").
    _cap(v) = length(getfield(v, :ref).mem)
    _clear_registry!() = lock(() -> empty!(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)
    _registry_len() = lock(() -> length(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)
    # An inactive, bloated Float64 slot (cap = `big` high-water, last extent = `small`).
    function _inactive_bloated(big::Int, small::Int)
        pool = AdaptiveArrayPool{0}()
        checkpoint!(pool); a = acquire!(pool, Float64, big); a .= 0.0; rewind!(pool)
        checkpoint!(pool); b = acquire!(pool, Float64, small); b .= 0.0; rewind!(pool)
        return pool
    end

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
        AAP.register_auto_manage!(p1)
        AAP.register_auto_manage!(p2)
        lock(() -> push!(AAP._AUTO_MANAGE_REGISTRY, WeakRef(nothing)), AAP._AUTO_MANAGE_LOCK)
        @test _registry_len() == 3

        AAP._auto_manage_sweep!(nothing)            # the timer callback body

        @test (@atomic p1._compact_requested) == true
        @test (@atomic p2._compact_requested) == true
        @test _registry_len() == 2                   # dead WeakRef pruned
        _clear_registry!()
    end

    @testset "register_auto_manage! skips flagless pools (no sweep poison)" begin
        # A pool without the `_compact_requested` flag (DisabledPool) must NOT be registered:
        # otherwise the sweep would throw on it and starve every later pool in the registry.
        _clear_registry!()
        AAP.register_auto_manage!(DISABLED_CPU)     # DisabledPool: no @atomic flag
        @test _registry_len() == 0                   # silently skipped, not registered
        # And a real pool registered alongside is still swept normally (no poison).
        p = AdaptiveArrayPool{0}()
        AAP.register_auto_manage!(p)
        AAP._auto_manage_sweep!(nothing)
        @test (@atomic p._compact_requested) == true
        _clear_registry!()
    end

    @testset "_run_auto_manage! resets the flag and compacts inactive bloat" begin
        pool = _inactive_bloated(1_000_000, 100)
        @atomic pool._compact_requested = true
        @test _cap(pool.float64.vectors[1]) >= 1_000_000

        AAP._run_auto_manage!(pool)

        @test (@atomic pool._compact_requested) == false   # reset (before the work)
        @test _cap(pool.float64.vectors[1]) == 150          # shrunk to ceil(1.5 * 100)
    end

    @testset "_run_auto_manage! uses the global config defaults" begin
        @test AAP._AUTO_MANAGE_CONFIG.compact_bloat_factor == 10
        @test AAP._AUTO_MANAGE_CONFIG.compact_target_ratio == 1.5
        @test AAP._AUTO_MANAGE_CONFIG.compact_min_bytes == 2^20
    end

    # ── Timer lifecycle (enable/disable). AUTO_MANAGE is on → enable does NOT warn ─
    @testset "enable! sets config and starts a timer that flags registered pools" begin
        _clear_registry!()
        AAP.disable_auto_manage!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_manage!(pool)

        AAP.enable_auto_manage!(
            compact_interval = 0.1, compact_bloat_factor = 20,
            compact_target_ratio = 2.0, compact_min_bytes = 4096,
        )
        @test AAP.auto_manage_enabled() == true
        @test AAP._AUTO_MANAGE_CONFIG.compact_bloat_factor == 20.0
        @test AAP._AUTO_MANAGE_CONFIG.compact_target_ratio == 2.0
        @test AAP._AUTO_MANAGE_CONFIG.compact_min_bytes == 4096

        sleep(0.35)                                  # ~3 ticks @ 0.1s
        @test (@atomic pool._compact_requested) == true

        AAP.disable_auto_manage!()
        @test AAP.auto_manage_enabled() == false
        AAP._AUTO_MANAGE_CONFIG.compact_bloat_factor = 10.0   # restore defaults
        AAP._AUTO_MANAGE_CONFIG.compact_target_ratio = 1.5
        AAP._AUTO_MANAGE_CONFIG.compact_min_bytes = 2^20
        _clear_registry!()
    end

    @testset "disable! stops the timer and is idempotent" begin
        _clear_registry!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_manage!(pool)
        AAP.enable_auto_manage!(compact_interval = 0.1)
        AAP.disable_auto_manage!()
        @test AAP.auto_manage_enabled() == false
        @atomic pool._compact_requested = false
        sleep(0.3)
        @test (@atomic pool._compact_requested) == false    # no ticks after disable
        AAP.disable_auto_manage!()                          # idempotent: must not throw
        @test AAP.auto_manage_enabled() == false
        _clear_registry!()
    end

    @testset "enable! replaces an existing timer (closes the old one)" begin
        AAP.disable_auto_manage!()
        AAP.enable_auto_manage!(compact_interval = 0.1)
        t1 = AAP._AUTO_MANAGE_TIMER[]
        AAP.enable_auto_manage!(compact_interval = 0.2)
        t2 = AAP._AUTO_MANAGE_TIMER[]
        @test t1 !== t2
        @test !isopen(t1)                                    # old timer closed
        @test isopen(t2)
        AAP.disable_auto_manage!()
    end

    @testset "a throwing sweep is caught — the timer survives" begin
        _clear_registry!()
        bogus = Ref(42)                                      # heap object, NOT a pool
        lock(() -> push!(AAP._AUTO_MANAGE_REGISTRY, WeakRef(bogus)), AAP._AUTO_MANAGE_LOCK)
        # The _safe wrapper catches the sweep's throw (a non-pool in the registry) and @warns
        # so the timer survives. Silence that EXPECTED warning (NullLogger) so it doesn't read
        # as a real failure in the suite log; we still assert the wrapper does not rethrow.
        with_logger(NullLogger()) do
            @test (AAP._safe_auto_manage_sweep!(nothing); true)  # _safe wrapper swallows the throw
        end
        @test bogus[] == 42                                  # keep `bogus` alive past the sweep
        _clear_registry!()
    end

    @testset "_run_auto_manage! swallows per-action failures (trim + compact catch)" begin
        # The owner-side service must never surface a maintenance failure at the `@with_pool`
        # boundary: each action is wrapped in try/catch + @warn. Force both to throw by parking a
        # non-pool object in the others collections (mirrors the throwing-sweep test above), with
        # both flags set. `_auto_trim!` iterates `values(others)` (backend-generic); CPU `compact!`
        # iterates the `_others_values` cache — a real type registers in both, so we mirror that.
        pool = AdaptiveArrayPool{0}()
        pool.others[String] = "not a typed pool"             # _auto_trim! iterates values(others)
        push!(pool._others_values, "not a typed pool")       # compact! iterates _others_values
        @atomic pool._trim_requested = true
        @atomic pool._compact_requested = true
        with_logger(NullLogger()) do                          # silence the two EXPECTED @warns
            @test (AAP._run_auto_manage!(pool); true)         # both catches fire; does NOT rethrow
        end
        @test (@atomic pool._trim_requested) == false         # flags still consumed (reset before work)
        @test (@atomic pool._compact_requested) == false
    end

    # ── Scope-entry hook ─────────────────────────────────────────────────────
    @testset "_maybe_auto_manage! fires only at _current_depth == 1" begin
        pool = _inactive_bloated(1_000_000, 100)
        checkpoint!(pool)                                    # depth → 2
        @atomic pool._compact_requested = true
        AAP._maybe_auto_manage!(pool)                       # depth 2 ≠ 1 → must NOT act
        @test (@atomic pool._compact_requested) == true      # flag preserved for the outer exit
        @test _cap(pool.float64.vectors[1]) >= 1_000_000     # not compacted
        rewind!(pool)                                        # depth → 1
        AAP._maybe_auto_manage!(pool)                       # depth 1 → fires
        @test (@atomic pool._compact_requested) == false     # reset
        @test _cap(pool.float64.vectors[1]) == 150           # compacted
    end

    @testset "_maybe_auto_manage! is a no-op when the flag is clear" begin
        pool = _inactive_bloated(1_000_000, 100)
        @test (@atomic pool._compact_requested) == false
        AAP._maybe_auto_manage!(pool)
        @test _cap(pool.float64.vectors[1]) >= 1_000_000     # untouched (no request)
    end

    @testset "_maybe_auto_manage! no-ops on non-CPU pools (GPU fallback)" begin
        @test AAP._maybe_auto_manage!(42) === nothing       # ::Any fallback, no throw
        @test AAP._maybe_auto_manage!("not a pool") === nothing
    end

    @testset "macro wires the gated scope-entry hook into @with_pool" begin
        s = string(
            @macroexpand @with_pool pool begin
                x = acquire!(pool, Float64, 4)
                sum(x)
            end
        )
        @test occursin("_maybe_auto_manage!", s)            # hook emitted…
        @test occursin("AUTO_MANAGE", s)                    # …gated by the compile-time const
    end

    # ── Default-on integration (AUTO_MANAGE === true) ───────────────────────
    @testset "AUTO_MANAGE is on by default" begin
        @test AAP.AUTO_MANAGE == true
    end

    @testset "__init__ auto-starts the timer when AUTO_MANAGE is on" begin
        AAP.disable_auto_manage!()
        @test AAP.auto_manage_enabled() == false
        AAP.__init__()                                       # re-run module init
        @test AAP.auto_manage_enabled() == true             # auto-started
        AAP.disable_auto_manage!()
    end

    @testset "get_task_local_pool auto-registers the pool (gate on)" begin
        _clear_registry!()
        fetch(Threads.@spawn get_task_local_pool())          # fresh task → creates + registers
        @test _registry_len() >= 1
        _clear_registry!()
    end

    @testset "end-to-end: bloat auto-manages at a @with_pool boundary" begin
        AAP.disable_auto_manage!()
        _clear_registry!()
        pool = get_task_local_pool()
        empty!(pool)                                         # clean slate
        AAP.register_auto_manage!(pool)
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
        @with_pool p begin                                   # scope ENTRY at depth 1 → hook fires
            acquire!(p, Float64, 4)
        end

        @test (@atomic pool._compact_requested) == false     # hook consumed the request
        @test _cap(pool.float64.vectors[1]) < cap0           # auto-managed, no manual compact!
        empty!(pool)
        _clear_registry!()
    end

    @testset "request is serviced for scopes that early return / break" begin
        # The scope-ENTRY hook services a pending flag when the NEXT @with_pool is entered at
        # depth 1, regardless of how the previous scope exited — so a scope that early-returns
        # or breaks (which skip the normal exit path) still gets compacted at its entry.
        pool = get_task_local_pool()
        AAP.disable_auto_manage!()

        empty!(pool)                                         # early RETURN
        @with_pool p begin
            x = acquire!(p, Float64, 1_000_000); x .= 0.0
        end
        @with_pool p begin
            x = acquire!(p, Float64, 100); x .= 0.0
        end
        cap0 = _cap(pool.float64.vectors[1])
        @atomic pool._compact_requested = true
        ret() = @with_pool p begin
            acquire!(p, Float64, 4)
            return 42
        end
        @test ret() == 42
        @test (@atomic pool._compact_requested) == false     # serviced at the next scope ENTRY
        @test _cap(pool.float64.vectors[1]) < cap0

        empty!(pool)                                         # early BREAK
        @with_pool p begin
            x = acquire!(p, Float64, 1_000_000); x .= 0.0
        end
        @with_pool p begin
            x = acquire!(p, Float64, 100); x .= 0.0
        end
        cap1 = _cap(pool.float64.vectors[1])
        @atomic pool._compact_requested = true
        for _ in 1:3
            @with_pool p begin
                acquire!(p, Float64, 4)
                break
            end
        end
        @test (@atomic pool._compact_requested) == false     # serviced at the scope ENTRY
        @test _cap(pool.float64.vectors[1]) < cap1
        empty!(pool)
    end

    @testset "@safe_with_pool fires the hook during exception unwind (and preserves it)" begin
        pool = get_task_local_pool()
        AAP.disable_auto_manage!()
        empty!(pool)
        @with_pool p begin
            x = acquire!(p, Float64, 1_000_000); x .= 0.0
        end
        @with_pool p begin
            x = acquire!(p, Float64, 100); x .= 0.0
        end
        cap0 = _cap(pool.float64.vectors[1])
        @atomic pool._compact_requested = true
        threw = false
        try
            @safe_with_pool p begin
                acquire!(p, Float64, 4)
                error("boom")
            end
        catch e
            threw = true
            @test e isa ErrorException                       # original exception NOT masked
        end
        @test threw
        @test (@atomic pool._compact_requested) == false     # hook fired at scope ENTRY (before the throw)
        @test _cap(pool.float64.vectors[1]) < cap0
        empty!(pool)
    end
end

AAP.disable_auto_manage!()   # leave the timer stopped for later test files
