# Metal Auto-Compact Tests
# =========================
# Parity with the CPU suite (test/test_auto_manage.jl) for the Metal backend.
#
# The single global Timer lives in the BASE module and runs on a CPU thread. It sweeps
# the shared `Vector{WeakRef}` registry and sets each live pool's `@atomic _compact_requested`
# flag — CPU and Metal pools alike. Each pool's owner task runs the full `compact!` at its
# next scope entry (the `_current_depth == 1` safepoint): a CPU pool at `@with_pool`, a Metal
# pool at `@with_pool :metal`. The two are independent — one sweep flags both, and each
# services (and clears) only its own flag at its own boundary.
#
# AUTO_MANAGE defaults on, so `__init__` auto-starts the timer; we stop it up front so each
# testset is deterministic. Uses Float32 (Metal's primary type; no Float64 on Metal).

import AdaptiveArrayPools as AAP

AAP.disable_auto_manage!()   # stop the __init__-started timer for deterministic tests

@testset "Metal auto_manage" begin

    # Helpers are scoped INSIDE this testset (a local scope) so they don't define top-level
    # methods in `Main` that collide with the identically-named helpers in the CPU/CUDA
    # auto-manage test files when the full suite loads several backends' files together
    # (otherwise: "WARNING: Method definition _clear_registry!() ... overwritten").
    _mcap(v) = Int(getfield(v, :maxsize) ÷ sizeof(eltype(v)))     # Metal device capacity (elements)
    _cpucap(v) = length(getfield(v, :ref).mem)                   # CPU backing capacity (coexistence test)
    _clear_registry!() = lock(() -> empty!(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)
    _registry_len() = lock(() -> length(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)
    # An inactive, bloated Float32 Metal slot (cap = `big` high-water, last extent = `small`).
    function _metal_inactive_bloated(big::Int, small::Int)
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        checkpoint!(pool); a = acquire!(pool, Float32, big); fill!(a, 0.0f0); rewind!(pool)
        checkpoint!(pool); b = acquire!(pool, Float32, small); fill!(b, 0.0f0); rewind!(pool)
        return pool
    end

    # ── Data layer: flag field + registry + run ──────────────────────────────
    @testset "Metal pool carries an atomic _compact_requested flag (default false)" begin
        pool = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        @test (@atomic pool._compact_requested) == false
        @atomic pool._compact_requested = true
        @test (@atomic pool._compact_requested) == true
    end

    @testset "registry sweep flags a registered Metal pool" begin
        _clear_registry!()
        p = MetalAdaptiveArrayPool{0, METAL_STORAGE}()
        AAP.register_auto_manage!(p)
        @test _registry_len() == 1
        AAP._auto_manage_sweep!(nothing)               # the timer callback body
        @test (@atomic p._compact_requested) == true
        _clear_registry!()
    end

    @testset "_run_auto_manage! resets the flag and compacts inactive Metal bloat" begin
        pool = _metal_inactive_bloated(1_000_000, 100)
        @atomic pool._compact_requested = true
        @test _mcap(pool.float32.vectors[1]) >= 1_000_000

        AAP._run_auto_manage!(pool)

        @test (@atomic pool._compact_requested) == false   # reset (before the work)
        @test _mcap(pool.float32.vectors[1]) < 1000        # device buffer shrunk in place
    end

    # ── Scope-entry hook (dispatches on MetalAdaptiveArrayPool) ───────────────
    @testset "_maybe_auto_manage!(::MetalAdaptiveArrayPool) fires only at depth 1" begin
        pool = _metal_inactive_bloated(1_000_000, 100)
        checkpoint!(pool)                                   # depth → 2
        @atomic pool._compact_requested = true
        AAP._maybe_auto_manage!(pool)                      # depth 2 ≠ 1 → must NOT act
        @test (@atomic pool._compact_requested) == true     # flag preserved for the outer exit
        @test _mcap(pool.float32.vectors[1]) >= 1_000_000   # not compacted
        rewind!(pool)                                       # depth → 1
        AAP._maybe_auto_manage!(pool)                      # depth 1 → fires
        @test (@atomic pool._compact_requested) == false    # reset
        @test _mcap(pool.float32.vectors[1]) < 1000         # compacted
    end

    @testset "_maybe_auto_manage! is a no-op when the Metal flag is clear" begin
        pool = _metal_inactive_bloated(1_000_000, 100)
        @test (@atomic pool._compact_requested) == false
        AAP._maybe_auto_manage!(pool)
        @test _mcap(pool.float32.vectors[1]) >= 1_000_000   # untouched (no request)
    end

    @testset "macro wires the gated scope-entry hook into @with_pool :metal" begin
        s = string(
            @macroexpand @with_pool :metal pool begin
                x = acquire!(pool, Float32, 4)
                sum(x)
            end
        )
        @test occursin("_maybe_auto_manage!", s)           # hook emitted…
        @test occursin("AUTO_MANAGE", s)                   # …gated by the compile-time const
    end

    # ── Auto-registration via the Metal task-local slow path ──────────────────
    @testset "get_task_local_metal_pool auto-registers the pool (gate on)" begin
        _clear_registry!()
        fetch(Threads.@spawn get_task_local_metal_pool())   # fresh task → creates + registers
        @test _registry_len() >= 1
        _clear_registry!()
    end

    @testset "end-to-end: bloat auto-manages at a @with_pool :metal boundary" begin
        AAP.disable_auto_manage!()
        _clear_registry!()
        pool = get_task_local_metal_pool()
        empty!(pool)                                        # clean slate
        AAP.register_auto_manage!(pool)
        @with_pool :metal p begin                           # grow slot to 1M high-water
            x = acquire!(p, Float32, 1_000_000); fill!(x, 0.0f0)
        end
        @with_pool :metal p begin                           # reuse small → bloated, inactive
            x = acquire!(p, Float32, 100); fill!(x, 0.0f0)
        end
        cap0 = _mcap(pool.float32.vectors[1])
        @test cap0 >= 1_000_000

        # Set the request flag directly (deterministic). This exercises the unique end-to-end
        # piece: the MACRO-emitted hook firing at a real @with_pool :metal entry and dispatching
        # to _maybe_auto_manage!(::MetalAdaptiveArrayPool).
        @atomic pool._compact_requested = true
        @with_pool :metal p begin                           # scope ENTRY at depth 1 → hook fires
            acquire!(p, Float32, 4)
        end

        @test (@atomic pool._compact_requested) == false    # hook consumed the request
        @test _mcap(pool.float32.vectors[1]) < cap0         # auto-managed, no manual compact!
        empty!(pool)
        _clear_registry!()
    end

    # ── Coexistence: a CPU pool and a Metal pool, one Timer, independent compaction ──
    # This pins the behavior asked about: the single CPU-side Timer flags BOTH a CPU and a
    # Metal pool in one sweep (structurally like two tasks each holding a pool — the registry
    # tracks pools, not tasks). Each then compacts independently at its OWN scope boundary;
    # servicing one never touches the other's flag or buffers.
    @testset "CPU + Metal coexist: one sweep flags both; each compacts independently" begin
        AAP.disable_auto_manage!()
        _clear_registry!()

        # CPU pool, inactive-bloated Float64.
        cpu = AdaptiveArrayPool{0}()
        checkpoint!(cpu); a = acquire!(cpu, Float64, 1_000_000); a .= 0.0; rewind!(cpu)
        checkpoint!(cpu); b = acquire!(cpu, Float64, 100); b .= 0.0; rewind!(cpu)
        # Metal pool, inactive-bloated Float32.
        met = _metal_inactive_bloated(1_000_000, 100)

        AAP.register_auto_manage!(cpu)
        AAP.register_auto_manage!(met)
        @test _registry_len() == 2

        cpu_cap0 = _cpucap(cpu.float64.vectors[1])
        met_cap0 = _mcap(met.float32.vectors[1])
        @test cpu_cap0 >= 1_000_000
        @test met_cap0 >= 1_000_000

        # ONE sweep (the single CPU-side Timer body) flags BOTH pools.
        AAP._auto_manage_sweep!(nothing)
        @test (@atomic cpu._compact_requested) == true
        @test (@atomic met._compact_requested) == true

        # Service the CPU pool at ITS boundary → only the CPU pool compacts; Metal untouched.
        AAP._maybe_auto_manage!(cpu)
        @test (@atomic cpu._compact_requested) == false          # CPU flag consumed
        @test _cpucap(cpu.float64.vectors[1]) < cpu_cap0         # CPU compacted
        @test (@atomic met._compact_requested) == true           # Metal flag still pending
        @test _mcap(met.float32.vectors[1]) >= 1_000_000         # Metal NOT compacted yet

        # Service the Metal pool at ITS boundary → now the Metal pool compacts, on its own.
        AAP._maybe_auto_manage!(met)
        @test (@atomic met._compact_requested) == false          # Metal flag consumed
        @test _mcap(met.float32.vectors[1]) < 1000               # Metal compacted independently

        _clear_registry!()
    end
end

AAP.disable_auto_manage!()   # leave the timer stopped for later Metal test files
