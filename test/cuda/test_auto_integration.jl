# CUDA Auto-manage END-TO-END integration + real-time tests.
# ========================================================
# Parity with the CPU suite (test/test_auto_integration.jl) and the Metal mirror, for CUDA.
#
#   (A) DIFFERENTIAL INTEGRITY — the same pool-heavy GPU computation, run with auto-manage
#       OFF then ON (real Timer firing repeatedly mid-loop), must produce BIT-IDENTICAL
#       results. Auto-compact/auto-trim reclaim device buffers between scopes; they must
#       never change a computed value.
#   (B) REAL-TIME CADENCE — over real elapsed seconds, compact is flagged every tick and
#       auto-trim every K-th tick, on a CUDA pool.
#
# GPU adaptation vs. the CPU sim: no element-wise writes (that is GPU scalar indexing).
# Each slot is `fill!`ed with a small INTEGER-valued Float32 and kept under 2^24 elements*value
# so `sum` is an exact integer regardless of the parallel reduction order — bit-reproducible
# run-to-run, while `fill!` writing the whole buffer keeps the test sensitive to any aliasing.

import AdaptiveArrayPools as AAP

AAP.disable_auto_manage!()

@testset "CUDA auto_manage integration + real-time" begin

    _ccap(v) = Int(getfield(v, :maxsize) ÷ sizeof(eltype(v)))   # CUDA device capacity (elements)
    _clear_registry!() = lock(() -> empty!(AAP._AUTO_MANAGE_REGISTRY), AAP._AUTO_MANAGE_LOCK)

    # GPU kernel-launch bound; spans enough wall-time for many 0.01 s ticks.
    N_STEPS = 300

    # Deterministic GPU "simulation": varies array size (periodic 1.6 MB slot → compact bloat)
    # and concurrency width 1..5 (slot-count peak → auto-trim tail) across a nested depth-2
    # scope. Fills with integer-valued Float32 (exact sums) and reduces on-device.
    function run_sim_cuda!(n_steps)
        out = Vector{Float64}(undef, n_steps)
        for it in 1:n_steps
            width = (it % 5) + 1
            big = (it % 11 == 0) ? 200_000 : 256
            @with_pool :cuda p begin
                acc = 0.0
                @with_pool :cuda q begin
                    arrs = Vector{Any}(undef, width)
                    for k in 1:width
                        m = (k == 1) ? big : (64 + 7k)
                        a = acquire!(q, Float32, m)
                        fill!(a, Float32((it + k) % 7 + 1))      # integer fill → exact sum
                        arrs[k] = a
                    end
                    for k in 1:width
                        acc += Float64(sum(arrs[k]))             # on-device reduction, exact
                    end
                end
                @inbounds out[it] = acc + it
            end
        end
        return out
    end

    # ── A. Differential integrity ────────────────────────────────────────────
    @testset "A. real Timer firing mid-computation produces bit-identical results" begin
        AAP.disable_auto_manage!(); _clear_registry!()
        pool = get_task_local_cuda_pool()

        empty!(pool)
        ref = run_sim_cuda!(N_STEPS)
        slots_off = length(pool.float32.vectors)
        big_off = maximum(_ccap, pool.float32.vectors)
        @test slots_off >= 5
        @test big_off >= 200_000

        empty!(pool)
        AAP.register_auto_manage!(pool)
        AAP.enable_auto_manage!(interval = 0.01, trim_interval = 0.02)
        test = run_sim_cuda!(N_STEPS)
        AAP.disable_auto_manage!()

        @test test == ref                                         # ★ SAFETY: identical values, no corruption

        # Deterministic reclamation + self-heal after the run.
        pool.float32._am_peak_n_active = 0
        @atomic pool._trim_requested = true
        @atomic pool._compact_requested = true
        @with_pool :cuda p begin
            v = acquire!(p, Float32, 8); fill!(v, 3.0f0)
            @test all(==(3.0f0), Array(v))                        # self-heal after reclamation correct
        end
        @test length(pool.float32.vectors) < slots_off            # ★ auto-trim reclaimed the tail
        @test maximum(_ccap, pool.float32.vectors) < big_off      # ★ auto-compact reclaimed the bloat

        empty!(pool); _clear_registry!()
    end

    # ── B. Real-time cadence over real elapsed seconds (CUDA pool) ────────────
    @testset "B. over real seconds: compact every tick, trim every K-th tick" begin
        AAP.disable_auto_manage!(); _clear_registry!()
        pool = CuAdaptiveArrayPool{0}()
        AAP.register_auto_manage!(pool)
        AAP.enable_auto_manage!(interval = 0.2, trim_interval = 0.6)   # K = 3

        sleep(0.35)                                               # ~1 tick
        @test (@atomic pool._compact_requested) == true
        @test (@atomic pool._trim_requested) == false

        sleep(0.75)                                               # ~total 1.1 s ⇒ ≥3 ticks
        @test (@atomic pool._trim_requested) == true

        AAP.disable_auto_manage!(); _clear_registry!()
    end
end

AAP.disable_auto_manage!()
