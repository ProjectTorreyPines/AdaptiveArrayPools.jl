# ==============================================================================
# Tests for rand! / randn! pool constructors (random-array convenience functions)
# ==============================================================================
# Covers:
#   - basic random properties (shape, eltype, value range / membership)
#   - EXACT seed-equivalence with Base/Random (our rand!/randn! must consume the
#     RNG stream identically to rand()/randn(), incl. consecutive-draw and
#     @with_pool RNG-neutrality)
#   - @with_pool integration (macro transform, size(x) form, rewind reuse)
#   - zero-allocation for the 0-alloc forms (Float64 rand!/randn!, range rand!)
#   - escape detection, DisabledPool fallback, error handling, namespace re-export
#
# NOTE: seed-equivalence checks compare INSIDE the @with_pool scope and return a
# Bool — the pool array never crosses the scope boundary (avoids tangling with
# escape detection / slot-reuse aliasing).

using Random
using AdaptiveArrayPools: _extract_acquired_vars

@testset "Random Constructors (rand!/randn!)" begin

    # =========================================================================
    # 1. Namespace: re-export and binding identity (no conflict with Random)
    # =========================================================================
    @testset "re-export & binding identity" begin
        @test :rand! in names(AdaptiveArrayPools)
        @test :randn! in names(AdaptiveArrayPools)
        # Re-exported the SAME binding as Random → no conflict when both loaded
        @test AdaptiveArrayPools.rand! === Random.rand!
        @test AdaptiveArrayPools.randn! === Random.randn!
    end

    # =========================================================================
    # 2. rand! — basic properties (shape, eltype, value range)
    # =========================================================================
    @testset "rand! properties" begin
        pool = AdaptiveArrayPool()

        @testset "explicit type" begin
            v = rand!(pool, Float64, 1000)
            @test v isa Array{Float64, 1}
            @test length(v) == 1000
            @test all(x -> 0.0 <= x < 1.0, v)

            v32 = rand!(pool, Float32, 500)
            @test eltype(v32) == Float32
            @test all(x -> 0.0f0 <= x < 1.0f0, v32)
        end

        @testset "default type (Float64)" begin
            v = rand!(pool, 1000)
            @test v isa Array{Float64, 1}
            @test all(x -> 0.0 <= x < 1.0, v)
        end

        @testset "multi-dimensional" begin
            m = rand!(pool, Float64, 30, 40)
            @test size(m) == (30, 40)
            @test all(x -> 0.0 <= x < 1.0, m)

            t = rand!(pool, Float32, 4, 5, 6)
            @test size(t) == (4, 5, 6)
        end

        @testset "tuple form" begin
            m = rand!(pool, (8, 9))
            @test size(m) == (8, 9)
            @test eltype(m) == Float64

            m32 = rand!(pool, Float32, (3, 4))
            @test size(m32) == (3, 4)
            @test eltype(m32) == Float32
        end

        @testset "returns Array not view" begin
            v = rand!(pool, Float64, 10)
            @test v isa Array{Float64, 1}
            @test !(v isa SubArray)
        end
    end

    # =========================================================================
    # 3. rand! — collection/range sampling form
    # =========================================================================
    @testset "rand! collection form" begin
        pool = AdaptiveArrayPool()

        @testset "integer range (dice)" begin
            Random.seed!(101)
            v = rand!(pool, 1:6, 1000)
            @test eltype(v) == Int
            @test length(v) == 1000
            @test all(x -> x in 1:6, v)
            @test Set(v) == Set(1:6)   # all faces appear in 1000 draws
        end

        @testset "multi-dim range" begin
            m = rand!(pool, 1:6, 20, 30)
            @test size(m) == (20, 30)
            @test all(x -> x in 1:6, m)
        end

        @testset "non-integer collection" begin
            chars = ('a', 'b', 'c')
            v = rand!(pool, chars, 200)
            @test eltype(v) == Char
            @test all(x -> x in chars, v)

            floats = [1.5, 2.5, 3.5]
            vf = rand!(pool, floats, 200)
            @test eltype(vf) == Float64
            @test all(x -> x in floats, vf)
        end

        @testset "tuple-dims collection form" begin
            m = rand!(pool, 1:6, (5, 5))
            @test size(m) == (5, 5)
            @test all(x -> x in 1:6, m)
        end

        @testset "returns Array not view" begin
            v = rand!(pool, 1:6, 10)
            @test v isa Array{Int, 1}
            @test !(v isa SubArray)
        end
    end

    # =========================================================================
    # 4. randn! — basic properties
    # =========================================================================
    @testset "randn! properties" begin
        pool = AdaptiveArrayPool()

        @testset "explicit type" begin
            v = randn!(pool, Float64, 1000)
            @test v isa Array{Float64, 1}
            @test length(v) == 1000
            @test all(isfinite, v)

            v32 = randn!(pool, Float32, 500)
            @test eltype(v32) == Float32
            @test all(isfinite, v32)
        end

        @testset "default type (Float64)" begin
            v = randn!(pool, 1000)
            @test eltype(v) == Float64
            @test all(isfinite, v)
        end

        @testset "multi-dim & tuple" begin
            m = randn!(pool, Float64, 10, 20)
            @test size(m) == (10, 20)

            t = randn!(pool, (4, 5))
            @test size(t) == (4, 5)
            @test eltype(t) == Float64
        end

        @testset "returns Array not view" begin
            v = randn!(pool, 10)
            @test v isa Array{Float64, 1}
            @test !(v isa SubArray)
        end
    end

    @testset "randn! distribution sanity (seeded)" begin
        pool = AdaptiveArrayPool()
        Random.seed!(20240601)
        v = randn!(pool, Float64, 100_000)
        m = sum(v) / length(v)
        s = sqrt(sum(abs2, v .- m) / (length(v) - 1))
        @test abs(m) < 0.05        # mean ≈ 0
        @test abs(s - 1.0) < 0.05  # std  ≈ 1
    end

    # =========================================================================
    # 5. Seed-equivalence with plain Base/Random (the headline property)
    # =========================================================================
    @testset "seed equivalence with plain rand/randn" begin

        @testset "rand!(pool, T, n) == rand(T, n)" begin
            for T in (Float64, Float32, Int)
                Random.seed!(424242); ref = rand(T, 64)
                pool = AdaptiveArrayPool()
                Random.seed!(424242); got = copy(rand!(pool, T, 64))
                @test ref == got
            end
        end

        @testset "default eltype rand!(pool, n) == rand(Float64, n)" begin
            Random.seed!(1); ref = rand(Float64, 50)
            pool = AdaptiveArrayPool()
            Random.seed!(1); got = copy(rand!(pool, 50))
            @test ref == got
        end

        @testset "multi-dim equivalence" begin
            Random.seed!(99); ref = rand(Float64, 7, 11)
            pool = AdaptiveArrayPool()
            Random.seed!(99); got = copy(rand!(pool, Float64, 7, 11))
            @test ref == got
        end

        @testset "collection form rand!(pool, S, n) == rand(S, n)" begin
            Random.seed!(7); ref = rand(1:6, 100)
            pool = AdaptiveArrayPool()
            Random.seed!(7); got = copy(rand!(pool, 1:6, 100))
            @test ref == got
        end

        @testset "randn!(pool, n) == randn(Float64, n)" begin
            Random.seed!(2024); ref = randn(Float64, 64)
            pool = AdaptiveArrayPool()
            Random.seed!(2024); got = copy(randn!(pool, 64))
            @test ref == got
        end

        @testset "consecutive draws: no RNG over-consumption" begin
            # If rand! filled more than n (e.g. the whole backing buffer), the
            # SECOND draw would desync from the plain stream. This catches that.
            Random.seed!(555)
            r1 = rand(Float64, 5); r2 = rand(Float64, 9)
            pool = AdaptiveArrayPool()
            Random.seed!(555)
            ok = @with_pool pool begin
                a = rand!(pool, 5)
                chk1 = (a == r1)
                b = rand!(pool, 9)
                chk2 = (b == r2)
                chk1 && chk2
            end
            @test ok
        end

        @testset "@with_pool machinery does not touch the RNG" begin
            # Acquiring / zeroing other arrays and entering the scope must not
            # consume any random numbers.
            Random.seed!(13)
            ref = rand(Float64, 6)
            pool = AdaptiveArrayPool()
            Random.seed!(13)
            ok = @with_pool pool begin
                z = zeros!(pool, 100)            # fill zeros — no RNG
                w = acquire!(pool, Float64, 50)  # acquire — no RNG
                fill!(w, 1.0)
                x = rand!(pool, 6)
                x == ref
            end
            @test ok
        end

        @testset "scope enter/exit (checkpoint/rewind) is RNG-neutral" begin
            Random.seed!(31)
            ref1 = rand(Float64, 5); ref2 = rand(Float64, 5)
            pool = AdaptiveArrayPool()
            Random.seed!(31)
            ok1 = @with_pool pool begin
                rand!(pool, 5) == ref1
            end
            ok2 = @with_pool pool begin
                rand!(pool, 5) == ref2
            end
            @test ok1
            @test ok2
        end

        @testset "reproducibility & seed sensitivity" begin
            pool = AdaptiveArrayPool()
            Random.seed!(123); a = copy(rand!(pool, 32))
            Random.seed!(123); b = copy(rand!(pool, 32))
            @test a == b                 # same seed → same sequence
            Random.seed!(456); c = copy(rand!(pool, 32))
            @test a != c                 # different seed → different sequence
        end
    end

    # =========================================================================
    # 6. @with_pool integration
    # =========================================================================
    @testset "integration with @with_pool" begin

        @testset "rand!/randn! in macro return Array" begin
            result = @with_pool pool begin
                v = rand!(pool, Float64, 100)
                @test v isa Array{Float64, 1}
                @test !(v isa SubArray)
                g = randn!(pool, 100)
                @test g isa Array{Float64, 1}
                length(v) + length(g)
            end
            @test result == 200
        end

        @testset "collection form in macro" begin
            result = @with_pool pool begin
                v = rand!(pool, 1:6, 100)
                @test all(x -> x in 1:6, v)
                sum(v)
            end
            @test 100 <= result <= 600
        end

        @testset "size(x) form (macro NTuple transform)" begin
            x1 = rand(10); x2 = rand(5, 8)
            @with_pool pool begin
                v = rand!(pool, Float64, size(x1))
                @test size(v) == (10,)
                m = rand!(pool, size(x2))
                @test size(m) == (5, 8)
                d = rand!(pool, 1:6, size(x1))
                @test size(d) == (10,)
                @test all(y -> y in 1:6, d)
                nothing
            end
        end

        @testset "rewind reuse gives fresh, seed-equivalent values" begin
            pool = AdaptiveArrayPool()
            Random.seed!(0)
            got = Vector{Float64}[]
            for _ in 1:5
                @with_pool pool begin
                    x = rand!(pool, 4)
                    push!(got, copy(x))
                    nothing
                end
            end
            @test allunique(got)               # each iteration fresh (not stuck)
            Random.seed!(0)
            ref = [rand(Float64, 4) for _ in 1:5]
            @test got == ref                   # rewind-reuse is seed-equivalent
        end

        @testset "pool state: n_active increments" begin
            pool = AdaptiveArrayPool()
            checkpoint!(pool)
            rand!(pool, Float64, 10)
            randn!(pool, Float64, 10)
            @test pool.float64.n_active == 2
            rewind!(pool)
            @test pool.float64.n_active == 0
        end
    end

    # =========================================================================
    # 7. Zero-allocation (function barrier + warmup), for the 0-alloc forms
    # =========================================================================
    @testset "zero allocation" begin
        function _alloc_rand()
            pool = AdaptiveArrayPool()
            for _ in 1:3
                @with_pool pool begin
                    a = rand!(pool, Float64, 100)
                    b = randn!(pool, Float64, 100)
                    c = rand!(pool, 1:6, 100)
                    sum(a) + sum(b) + sum(c)
                end
            end
            @allocated @with_pool pool begin
                a = rand!(pool, Float64, 100)
                b = randn!(pool, Float64, 100)
                c = rand!(pool, 1:6, 100)
                sum(a) + sum(b) + sum(c)
            end
        end
        _alloc_rand(); _alloc_rand()
        alloc = _alloc_rand()
        println("  rand!/randn!/rand!(1:6) (single iter): $alloc bytes")
        @test alloc == 0
    end

    # =========================================================================
    # 8. Escape detection: vars assigned from rand!/randn! are tracked
    # =========================================================================
    @testset "escape detection tracks rand!/randn!" begin
        vars = _extract_acquired_vars(
            quote
                r = rand!(pool, 5)
                g = randn!(pool, Float64, 5)
                d = rand!(pool, 1:6, 5)
            end, :pool
        )
        @test :r in vars
        @test :g in vars
        @test :d in vars
    end

    # =========================================================================
    # 9. DisabledPool fallback (allocating, plain Array)
    # =========================================================================
    @testset "DisabledPool fallback" begin
        v = rand!(DISABLED_CPU, Float64, 10)
        @test v isa Array{Float64, 1}
        @test all(x -> 0.0 <= x < 1.0, v)

        v2 = rand!(DISABLED_CPU, 5, 5)
        @test v2 isa Matrix{Float64}
        @test size(v2) == (5, 5)

        vd = rand!(DISABLED_CPU, 1:6, 20)
        @test vd isa Array{Int, 1}
        @test all(x -> x in 1:6, vd)

        g = randn!(DISABLED_CPU, Float64, 10)
        @test g isa Array{Float64, 1}
        @test all(isfinite, g)

        v3 = rand!(DISABLED_CPU, Float32, (3, 4))   # NTuple fallback
        @test v3 isa Array{Float32, 2}
        @test size(v3) == (3, 4)
    end

    @testset "DisabledPool seed-equivalence" begin
        Random.seed!(77); ref = rand(Float64, 16)
        Random.seed!(77); got = rand!(DISABLED_CPU, Float64, 16)
        @test ref == got
    end

    # =========================================================================
    # 10. Error handling
    # =========================================================================
    @testset "randn! requires a float eltype" begin
        pool = AdaptiveArrayPool()
        @test_throws Exception randn!(pool, Int, 5)
    end

    # =========================================================================
    # 11. Seed-equivalence across more eltypes (regression guard for the
    #     headline invariant — Float16/Complex consume the RNG via different
    #     per-type codepaths than Float64).
    # =========================================================================
    @testset "seed equivalence: extra eltypes" begin
        @testset "rand! across eltypes" begin
            for T in (Float16, ComplexF64, ComplexF32, Bool)
                Random.seed!(11); ref = rand(T, 48)
                pool = AdaptiveArrayPool()
                Random.seed!(11); got = copy(rand!(pool, T, 48))
                @test eltype(got) == T
                @test ref == got
            end
        end
        @testset "randn! across float/complex eltypes" begin
            for T in (Float32, Float16, ComplexF64, ComplexF32)
                Random.seed!(2024); ref = randn(T, 64)
                pool = AdaptiveArrayPool()
                Random.seed!(2024); got = copy(randn!(pool, T, 64))
                @test eltype(got) == T
                @test ref == got
            end
        end
    end

    # =========================================================================
    # 12. Collection-form seed-equivalence: multi-dim, tuple-dims, non-Int
    #     collections — each is a distinct dispatch path from the 1-D vararg form.
    # =========================================================================
    @testset "seed equivalence: collection-form shapes" begin
        pool = AdaptiveArrayPool()

        Random.seed!(7); ref = rand(1:6, 7, 11)            # multi-dim vararg
        Random.seed!(7); got = copy(rand!(pool, 1:6, 7, 11))
        @test ref == got

        Random.seed!(8); ref2 = rand(1:6, (5, 5))          # tuple-dims (separate method)
        Random.seed!(8); got2 = copy(rand!(pool, 1:6, (5, 5)))
        @test ref2 == got2

        chars = ('a', 'b', 'c', 'd')                       # non-Int collection
        Random.seed!(9); ref3 = rand(chars, 40)
        Random.seed!(9); got3 = copy(rand!(pool, chars, 40))
        @test ref3 == got3

        vals = [1.5, 2.5, 3.5]                              # Vector collection
        Random.seed!(10); ref4 = rand(vals, 40)
        Random.seed!(10); got4 = copy(rand!(pool, vals, 40))
        @test ref4 == got4
    end

    # =========================================================================
    # 13. DisabledPool seed-equivalence: collection + randn! fallbacks
    #     (distinct methods from the typed rand! fallback).
    # =========================================================================
    @testset "DisabledPool seed equivalence: collection & randn!" begin
        Random.seed!(77); ref = rand(1:6, 50)
        Random.seed!(77); got = rand!(DISABLED_CPU, 1:6, 50)
        @test ref == got

        Random.seed!(88); ref2 = rand(1:6, (5, 5))
        Random.seed!(88); got2 = rand!(DISABLED_CPU, 1:6, (5, 5))
        @test ref2 == got2

        Random.seed!(99); ref3 = randn(Float64, 16)
        Random.seed!(99); got3 = randn!(DISABLED_CPU, Float64, 16)
        @test ref3 == got3
    end

    # =========================================================================
    # 14. Empty (n=0) boundary: correct empty Array + ZERO RNG consumption.
    # =========================================================================
    @testset "empty (n=0) request" begin
        pool = AdaptiveArrayPool()
        @test length(rand!(pool, Float64, 0)) == 0
        @test length(randn!(pool, 0)) == 0
        @test length(rand!(pool, 1:6, 0)) == 0
        @test eltype(rand!(pool, 1:6, 0)) == Int

        # An empty draw must consume zero random numbers (RNG-neutral).
        Random.seed!(1); ref = rand(Float64, 4)
        pool2 = AdaptiveArrayPool()
        Random.seed!(1)
        ok = @with_pool pool2 begin
            rand!(pool2, Float64, 0)             # empty: draws nothing
            rand!(pool2, Float64, 4) == ref
        end
        @test ok
    end

    # =========================================================================
    # 15. Collection-form type registration & reclamation.
    #     The Int touch is self-recorded inside `_rand_impl!` (NOT by the macro,
    #     which spuriously registers the default Float64). Guard that path:
    #     int64 must activate (not float64), and reclaim across @with_pool scopes.
    # =========================================================================
    @testset "collection-form Int registration & reclamation" begin
        pool = AdaptiveArrayPool()
        checkpoint!(pool)
        rand!(pool, 1:6, 10)
        @test pool.int64.n_active == 1       # Int touch recorded
        @test pool.float64.n_active == 0     # NOT the spurious default type
        rewind!(pool)
        @test pool.int64.n_active == 0

        # Many @with_pool iterations (typed-lazy path) must not leak the Int pool;
        # a regression in the self-record would grow int64.n_active across scopes.
        tlp = get_task_local_pool()
        base = tlp.int64.n_active
        for _ in 1:500
            @with_pool p begin
                d = rand!(p, 1:6, 64)
                sum(d)
            end
        end
        @test tlp.int64.n_active == base     # no residual Int slots after scopes
    end

    # =========================================================================
    # 16. Disambiguation: a bare Int-tuple is DIMS, not a sample collection.
    #     rand!(pool, (8,9)) → 8x9 array; rand!(pool, (8,9), k) → sample from {8,9}.
    # =========================================================================
    @testset "tuple-as-dims vs tuple-as-collection" begin
        m = rand!(AdaptiveArrayPool(), (8, 9))         # dims → 8x9 matrix
        @test size(m) == (8, 9)
        @test eltype(m) == Float64

        v = rand!(AdaptiveArrayPool(), (8, 9), 100)     # collection {8,9} → 100 samples
        @test length(v) == 100
        @test all(x -> x in (8, 9), v)
    end

    # =========================================================================
    # 17. Type stability — every form must @inferred to a concrete Array type
    #     (type stability is what makes the zero-allocation path possible).
    # =========================================================================
    @testset "type stability (@inferred)" begin
        pool = AdaptiveArrayPool()
        @test (@inferred rand!(pool, Float64, 10)) isa Vector{Float64}
        @test (@inferred rand!(pool, 10)) isa Vector{Float64}
        @test (@inferred rand!(pool, Float64, 3, 4)) isa Matrix{Float64}
        @test (@inferred rand!(pool, Float32, 10)) isa Vector{Float32}
        @test (@inferred rand!(pool, 1:6, 10)) isa Vector{Int}        # eltype(S) inferred
        @test (@inferred rand!(pool, 1:6, (5, 5))) isa Matrix{Int}
        @test (@inferred randn!(pool, Float64, 10)) isa Vector{Float64}
        @test (@inferred randn!(pool, 10)) isa Vector{Float64}
        @test (@inferred randn!(pool, Float32, 3, 4)) isa Matrix{Float32}
    end

    # =========================================================================
    # 18. @maybe_with_pool: when MAYBE_POOLING[]=false, rand!/randn! must revert
    #     to plain rand/randn (plain Array, seed-equivalent) — safe with no pool.
    # =========================================================================
    @testset "@maybe_with_pool reverts to rand when MAYBE_POOLING[]=false" begin
        old = AdaptiveArrayPools.MAYBE_POOLING[]
        AdaptiveArrayPools.MAYBE_POOLING[] = false
        try
            Random.seed!(5); ref = rand(Float64, 8)
            Random.seed!(5)
            got = @maybe_with_pool mp begin
                copy(rand!(mp, Float64, 8))
            end
            @test got isa Vector{Float64}       # plain Array, not pool-backed
            @test got == ref                    # seed-equivalent to plain rand

            Random.seed!(6); refd = rand(1:6, 10)
            Random.seed!(6)
            gotd = @maybe_with_pool mp begin
                copy(rand!(mp, 1:6, 10))
            end
            @test gotd isa Vector{Int}
            @test gotd == refd

            Random.seed!(7); refn = randn(Float64, 8)
            Random.seed!(7)
            gotn = @maybe_with_pool mp begin
                copy(randn!(mp, 8))
            end
            @test gotn isa Vector{Float64}
            @test gotn == refn
        finally
            AdaptiveArrayPools.MAYBE_POOLING[] = old
        end

        # Sanity: with pooling ENABLED (default), the same call still works.
        Random.seed!(5); ref = rand(Float64, 8)
        Random.seed!(5)
        ok = @maybe_with_pool mp begin
            rand!(mp, Float64, 8) == ref
        end
        @test ok
    end
end
