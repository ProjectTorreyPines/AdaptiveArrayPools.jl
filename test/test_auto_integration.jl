using Test
using AdaptiveArrayPools
using AdaptiveArrayPools: checkpoint!, rewind!
import AdaptiveArrayPools as AAP

# ==============================================================================
# Auto-manage END-TO-END integration + real-time tests.
#
# The unit suites (test_auto_manage.jl / test_auto_trim.jl) cover the mechanism in
# isolation — set a flag, call the service; the depth-1 safepoint no-op. These tests
# close the remaining gap by running the REAL background Timer against a real,
# pool-heavy computation:
#
#   (A) DIFFERENTIAL INTEGRITY — the safety guarantee. The same computation, run with
#       auto-manage OFF and then ON (real Timer firing repeatedly mid-loop), must
#       produce BIT-IDENTICAL results. Auto-compact/auto-trim reclaim memory between
#       scopes; they must never change a single computed value. If they ever did, this
#       test fails — which is the "never corrupt a simulation" contract, asserted.
#
#   (B) REAL-TIME CADENCE — over real elapsed seconds, the Timer flags compaction every
#       tick and auto-trim every K-th tick (K = round(trim_interval/interval)).
#
# AUTO_MANAGE defaults on; we stop the __init__ timer up front for determinism and
# re-enable it explicitly (short intervals) inside the testsets that need it.
# ==============================================================================

AAP.disable_auto_manage!()

@testset "auto_manage integration + real-time" begin

    _cap(v) = length(getfield(v, :ref).mem)                  # CPU backing capacity (elements)
    _clear_registry!() = lock(() -> empty!(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)

    # Sized (~100 µs/step) so the ON run spans ~0.4 s wall-time — at the 0.01 s Timer
    # interval that is ~40 sweeps (≈20 auto-trims) firing DURING the computation.
    N_STEPS = 4000

    # A deterministic, pool-heavy "simulation". Each step varies BOTH the array size
    # (periodic 1.6 MB slot → auto-compact bloat) and the concurrency width 1..5 (slot-count
    # peak → auto-trim tail), across a nested depth-2 scope. Every value is freshly computed
    # from arithmetic only, so the output must be identical whether or not auto-manage
    # reclaims memory between steps. Returns the per-step reduction vector.
    function run_sim!(n_steps)
        out = Vector{Float64}(undef, n_steps)
        for it in 1:n_steps
            width = (it % 5) + 1                              # 1..5 concurrent arrays
            big = (it % 11 == 0) ? 200_000 : 256             # periodic compactable bloat
            @with_pool p begin                               # depth-1 entry → auto-manage safepoint
                acc = 0.0
                @with_pool q begin                           # depth-2: hold `width` arrays at once
                    arrs = Vector{Vector{Float64}}(undef, width)
                    for k in 1:width
                        m = (k == 1) ? big : (64 + 7k)
                        a = acquire!(q, Float64, m)
                        @inbounds for i in 1:m
                            a[i] = sin(0.01 * (it + k)) + (i % 13)
                        end
                        arrs[k] = a
                    end
                    @inbounds for k in 1:width               # cross-array reduction (would break on aliasing)
                        acc += sum(arrs[k]) - first(arrs[k]) * k
                    end
                end
                @inbounds out[it] = acc + it
            end
        end
        return out
    end

    # ── A. Differential integrity: auto-manage never changes results ─────────
    @testset "A. real Timer firing mid-computation produces bit-identical results" begin
        AAP.disable_auto_manage!(); _clear_registry!()
        pool = get_task_local_pool()

        # Reference: auto-manage OFF — pool accumulates to the peak, nothing reclaimed.
        empty!(pool)
        ref = run_sim!(N_STEPS)
        slots_off = length(pool.float64.vectors)             # retained peak width (no trim ran)
        big_off = maximum(_cap, pool.float64.vectors)        # bloated big slot (no compact ran)
        @test slots_off >= 5                                  # the sim really reached width-5…
        @test big_off >= 200_000                              # …and really created 1.6 MB bloat

        # Test: auto-manage ON with the REAL Timer at a short interval — it fires many times
        # during the ~N_STEPS loop (interval 0.01 s ⇒ tens of sweeps over the run's wall-time).
        empty!(pool)
        AAP.register_auto_manage!(pool)
        AAP.enable_auto_manage!(interval = 0.01, trim_interval = 0.02)
        test = run_sim!(N_STEPS)
        AAP.disable_auto_manage!()

        @test test == ref                                     # ★ SAFETY: identical values, no corruption

        # Non-vacuity: the reclamation path actually shrinks the pool and the next acquire
        # self-heals correctly. Reset the recent peak so the next auto-trim drops the whole
        # tail deterministically, then service one flagged scope entry.
        pool.float64._am_peak_n_active = 0
        @atomic pool._trim_requested = true
        @atomic pool._compact_requested = true
        @with_pool p begin
            v = acquire!(p, Float64, 8); v .= 3.0
            @test all(==(3.0), v)                             # self-heal after reclamation is correct
        end
        @test length(pool.float64.vectors) < slots_off        # ★ auto-trim reclaimed the slot tail
        @test maximum(_cap, pool.float64.vectors) < big_off   # ★ auto-compact reclaimed the byte bloat

        empty!(pool); _clear_registry!()
    end

    # ── B. Real-time cadence over real elapsed seconds ───────────────────────
    # Observe the raw flag cadence WITHOUT entering @with_pool (so nothing services/clears
    # the flags): compact is flagged every tick, auto-trim only every K-th tick.
    @testset "B. over real seconds: compact every tick, trim every K-th tick" begin
        AAP.disable_auto_manage!(); _clear_registry!()
        pool = AdaptiveArrayPool{0}()
        AAP.register_auto_manage!(pool)
        # interval 0.2 s, trim every 0.6 s ⇒ K = 3 ticks. Timer first fires at t = interval.
        AAP.enable_auto_manage!(interval = 0.2, trim_interval = 0.6)

        sleep(0.35)                                           # ~1 tick elapsed (tick @ 0.2 s)
        @test (@atomic pool._compact_requested) == true       # compact flagged after 1 tick
        @test (@atomic pool._trim_requested) == false          # trim NOT yet (needs 3 ticks)

        sleep(0.75)                                           # ~total 1.1 s ⇒ ≥3 ticks (trim @ tick 3 = 0.6 s)
        @test (@atomic pool._trim_requested) == true           # trim flagged after K ticks

        AAP.disable_auto_manage!(); _clear_registry!()
    end
end

AAP.disable_auto_manage!()   # leave the timer stopped for later test files
