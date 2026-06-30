# CUDA Auto-Compact Tests
# ========================
# Parity with the CPU suite (test/test_auto_compact.jl) and the Metal mirror, for CUDA.
#
# The single global Timer lives in the BASE module and runs on a CPU thread. It sweeps
# the shared `Vector{WeakRef}` registry and sets each live pool's `@atomic _compact_requested`
# flag — CPU and CUDA pools alike. Each pool's owner task runs the full `compact!` at its
# next scope entry (the `_current_depth == 1` safepoint): a CPU pool at `@with_pool`, a CUDA
# pool at `@with_pool :cuda`. The two are independent — one sweep flags both, and each
# services (and clears) only its own flag at its own boundary.
#
# AUTO_COMPACT defaults on, so `__init__` auto-starts the timer; we stop it up front so each
# testset is deterministic. Uses Float32 (GPU's primary type).

import AdaptiveArrayPools as AAP

# Device capacity (elements) of a CUDA backing buffer.
_ccap(v) = Int(getfield(v, :maxsize) ÷ sizeof(eltype(v)))
# CPU backing capacity (Memory length) — for the coexistence test.
_cpucap(v) = length(getfield(v, :ref).mem)

_clear_registry!() = lock(() -> empty!(AAP._AUTO_COMPACT_REGISTRY), AAP._AUTO_COMPACT_LOCK)
_registry_len() = lock(() -> length(AAP._AUTO_COMPACT_REGISTRY), AAP._AUTO_COMPACT_LOCK)

# An inactive, bloated Float32 CUDA slot (cap = `big` high-water, last extent = `small`).
function _cuda_inactive_bloated(big::Int, small::Int)
    pool = CuAdaptiveArrayPool{0}()
    checkpoint!(pool); a = acquire!(pool, Float32, big); fill!(a, 0.0f0); rewind!(pool)
    checkpoint!(pool); b = acquire!(pool, Float32, small); fill!(b, 0.0f0); rewind!(pool)
    return pool
end

AAP.disable_auto_compact!()   # stop the __init__-started timer for deterministic tests

@testset "CUDA auto_compact" begin

    # ── Data layer: flag field + registry + run ──────────────────────────────
    @testset "CUDA pool carries an atomic _compact_requested flag (default false)" begin
        pool = CuAdaptiveArrayPool{0}()
        @test (@atomic pool._compact_requested) == false
        @atomic pool._compact_requested = true
        @test (@atomic pool._compact_requested) == true
    end

    @testset "registry sweep flags a registered CUDA pool" begin
        _clear_registry!()
        p = CuAdaptiveArrayPool{0}()
        AAP.register_auto_compact!(p)
        @test _registry_len() == 1
        AAP._auto_compact_sweep!(nothing)               # the timer callback body
        @test (@atomic p._compact_requested) == true
        _clear_registry!()
    end

    @testset "_run_auto_compact! resets the flag and compacts inactive CUDA bloat" begin
        pool = _cuda_inactive_bloated(1_000_000, 100)
        @atomic pool._compact_requested = true
        @test _ccap(pool.float32.vectors[1]) >= 1_000_000

        AAP._run_auto_compact!(pool)

        @test (@atomic pool._compact_requested) == false   # reset (before the work)
        @test _ccap(pool.float32.vectors[1]) < 1000        # device buffer shrunk in place
    end

    # ── Scope-entry hook (dispatches on CuAdaptiveArrayPool) ──────────────────
    @testset "_maybe_auto_compact!(::CuAdaptiveArrayPool) fires only at depth 1" begin
        pool = _cuda_inactive_bloated(1_000_000, 100)
        checkpoint!(pool)                                   # depth → 2
        @atomic pool._compact_requested = true
        AAP._maybe_auto_compact!(pool)                      # depth 2 ≠ 1 → must NOT act
        @test (@atomic pool._compact_requested) == true     # flag preserved for the outer exit
        @test _ccap(pool.float32.vectors[1]) >= 1_000_000   # not compacted
        rewind!(pool)                                       # depth → 1
        AAP._maybe_auto_compact!(pool)                      # depth 1 → fires
        @test (@atomic pool._compact_requested) == false    # reset
        @test _ccap(pool.float32.vectors[1]) < 1000         # compacted
    end

    @testset "_maybe_auto_compact! is a no-op when the CUDA flag is clear" begin
        pool = _cuda_inactive_bloated(1_000_000, 100)
        @test (@atomic pool._compact_requested) == false
        AAP._maybe_auto_compact!(pool)
        @test _ccap(pool.float32.vectors[1]) >= 1_000_000   # untouched (no request)
    end

    @testset "macro wires the gated scope-entry hook into @with_pool :cuda" begin
        s = string(
            @macroexpand @with_pool :cuda pool begin
                x = acquire!(pool, Float32, 4)
                sum(x)
            end
        )
        @test occursin("_maybe_auto_compact!", s)           # hook emitted…
        @test occursin("AUTO_COMPACT", s)                   # …gated by the compile-time const
    end

    # ── Auto-registration via the CUDA task-local slow path ───────────────────
    @testset "get_task_local_cuda_pool auto-registers the pool (gate on)" begin
        _clear_registry!()
        fetch(Threads.@spawn get_task_local_cuda_pool())   # fresh task → creates + registers
        @test _registry_len() >= 1
        _clear_registry!()
    end

    @testset "end-to-end: bloat auto-compacts at a @with_pool :cuda boundary" begin
        AAP.disable_auto_compact!()
        _clear_registry!()
        pool = get_task_local_cuda_pool()
        empty!(pool)                                        # clean slate
        AAP.register_auto_compact!(pool)
        @with_pool :cuda p begin                            # grow slot to 1M high-water
            x = acquire!(p, Float32, 1_000_000); fill!(x, 0.0f0)
        end
        @with_pool :cuda p begin                            # reuse small → bloated, inactive
            x = acquire!(p, Float32, 100); fill!(x, 0.0f0)
        end
        cap0 = _ccap(pool.float32.vectors[1])
        @test cap0 >= 1_000_000

        # Set the request flag directly (deterministic). This exercises the unique end-to-end
        # piece: the MACRO-emitted hook firing at a real @with_pool :cuda entry and dispatching
        # to _maybe_auto_compact!(::CuAdaptiveArrayPool).
        @atomic pool._compact_requested = true
        @with_pool :cuda p begin                            # scope ENTRY at depth 1 → hook fires
            acquire!(p, Float32, 4)
        end

        @test (@atomic pool._compact_requested) == false    # hook consumed the request
        @test _ccap(pool.float32.vectors[1]) < cap0         # auto-compacted, no manual compact!
        empty!(pool)
        _clear_registry!()
    end

    # ── Coexistence: a CPU pool and a CUDA pool, one Timer, independent compaction ──
    # This pins the cross-backend behavior: the single CPU-side Timer flags BOTH a CPU and a
    # CUDA pool in one sweep (structurally like two tasks each holding a pool — the registry
    # tracks pools, not tasks). Each then compacts independently at its OWN scope boundary;
    # servicing one never touches the other's flag or buffers.
    @testset "CPU + CUDA coexist: one sweep flags both; each compacts independently" begin
        AAP.disable_auto_compact!()
        _clear_registry!()

        # CPU pool, inactive-bloated Float64.
        cpu = AdaptiveArrayPool{0}()
        checkpoint!(cpu); a = acquire!(cpu, Float64, 1_000_000); a .= 0.0; rewind!(cpu)
        checkpoint!(cpu); b = acquire!(cpu, Float64, 100); b .= 0.0; rewind!(cpu)
        # CUDA pool, inactive-bloated Float32.
        cu = _cuda_inactive_bloated(1_000_000, 100)

        AAP.register_auto_compact!(cpu)
        AAP.register_auto_compact!(cu)
        @test _registry_len() == 2

        cpu_cap0 = _cpucap(cpu.float64.vectors[1])
        cu_cap0 = _ccap(cu.float32.vectors[1])
        @test cpu_cap0 >= 1_000_000
        @test cu_cap0 >= 1_000_000

        # ONE sweep (the single CPU-side Timer body) flags BOTH pools.
        AAP._auto_compact_sweep!(nothing)
        @test (@atomic cpu._compact_requested) == true
        @test (@atomic cu._compact_requested) == true

        # Service the CPU pool at ITS boundary → only the CPU pool compacts; CUDA untouched.
        AAP._maybe_auto_compact!(cpu)
        @test (@atomic cpu._compact_requested) == false          # CPU flag consumed
        @test _cpucap(cpu.float64.vectors[1]) < cpu_cap0         # CPU compacted
        @test (@atomic cu._compact_requested) == true            # CUDA flag still pending
        @test _ccap(cu.float32.vectors[1]) >= 1_000_000          # CUDA NOT compacted yet

        # Service the CUDA pool at ITS boundary → now the CUDA pool compacts, on its own.
        AAP._maybe_auto_compact!(cu)
        @test (@atomic cu._compact_requested) == false           # CUDA flag consumed
        @test _ccap(cu.float32.vectors[1]) < 1000                # CUDA compacted independently

        _clear_registry!()
    end
end

AAP.disable_auto_compact!()   # leave the timer stopped for later CUDA test files
