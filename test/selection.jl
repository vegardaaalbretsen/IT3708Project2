# test/selection.jl
using Test
using StableRNGs
using HomeCareGA  # assumes you expose/select these; otherwise prefix with HomeCareGA.

@info ("Testing selection")
@testset verbose=true "Selection" begin
    rng = StableRNG(123)

    # Simple population: 6 individuals, fitness is deterministic
    pop = Vector{Vector{Int}}([
        [1,2,3,-1,4,5],      # two routes
        [3,2,1,-1,5,4],
        [1,3,5,-1,2,4],
        [5,4,2,-1,1,3],
    ])
    fit = [10.0, 5.0, 7.0, 1.0]
    μ = length(pop)

    @testset "TournamentSelector basics" begin
        sel = TournamentSelector(3)
        winners, wfit = select(sel, pop, fit)

        @test length(winners) == μ
        @test length(wfit) == μ
        @test all(f -> f in fit, wfit)
        @test all(w -> w in pop, winners)
    end

    @testset "RouletteWheelSelector basics" begin
        sel = RouletteWheelSelector()
        winners, wfit = select(sel, pop, fit)

        @test length(winners) == μ
        @test length(wfit) == μ
        @test all(f -> f in fit, wfit)
        @test all(w -> w in pop, winners)
    end

    @testset "ElitistSelector keeps best parents and best children" begin
        # children are worse except two good ones
        children = Vector{Vector{Int}}([
            [2,3,1,-1,4,6,5],
            [3,1,2,-1,5,6,4],
            [1,5,3,-1,2,6,4],
            [6,4,5,-1,1,3,2],
        ])

        child_fit = Float64[20.0, 30.0, 4.0, 2.0]

        E = 2
        surv = ElitistSelector(num_elites=E)
        nextpop, nextfit = select(surv, pop, fit, children, child_fit)

        @test length(nextpop) == μ
        @test length(nextfit) == μ

        # elites from parents must be present: best 2 in pop fitness are 1 and 3
        elite_idx = sortperm(fit)[1:E]
        elites = pop[elite_idx]
        @test all(e -> e in nextpop, elites)

        # remaining slots should be best children (μ-E smallest)
        child_idx = sortperm(child_fit)[1:(μ - E)]
        best_children = children[child_idx]
        # They should all be in nextpop (may also overlap with elites)
        @test all(c -> c in nextpop, best_children)

        # Fitness alignment sanity: every nextfit value comes from either parent fit or child fit
        @test all(f -> (f in fit) || (f in child_fit), nextfit)
    end

    @testset "ElitistSelector clamps num_elites" begin
        children = pop
        child_fit = fit .+ 100.0

        # num_elites > μ should behave like μ
        surv = ElitistSelector(num_elites=μ+10)
        nextpop, nextfit = select(surv, pop, fit, children, child_fit)

        @test nextpop == pop[sortperm(fit)]  # all parents kept, ordered by fitness
        @test nextfit == fit[sortperm(fit)]
    end

    @testset "GeneralizedCrowdingSelector phi=0 is deterministic with similarity pairing" begin
        gc_pop = Vector{Vector{Int}}([
            [1,2,3,-1,4,5],    # p1
            [6,5,4,-1,3,2],    # p2
            [1,3,2,-1,4,5],    # p3
            [6,4,5,-1,2,3],    # p4
        ])
        gc_fit = Float64[10.0, 20.0, 30.0, 40.0]

        # c1 is closer to p2, c2 is closer to p1 -> requires cross pairing for first family.
        gc_children = Vector{Vector{Int}}([
            [6,5,4,-1,2,3],    # c1 (better than p2, worse than p1)
            [1,2,3,-1,5,4],    # c2 (better than p1)
            [1,3,2,-1,5,4],    # c3 (better than p3)
            [6,4,5,-1,3,2],    # c4 (better than p4)
        ])
        gc_child_fit = Float64[15.0, 5.0, 25.0, 35.0]

        surv = GeneralizedCrowdingSelector(phi=0.0)
        nextpop, nextfit = select(surv, gc_pop, gc_fit, gc_children, gc_child_fit, StableRNG(1))

        @test nextpop == [gc_children[2], gc_children[1], gc_children[3], gc_children[4]]
        @test nextfit == [gc_child_fit[2], gc_child_fit[1], gc_child_fit[3], gc_child_fit[4]]
    end

    @testset "GeneralizedCrowdingSelector phi=1 matches probabilistic crowding rate" begin
        parent = [[1,2,3,-1,4]]
        child = [[1,3,2,-1,4]]
        parent_fit = [100.0]
        child_fit = [50.0]

        surv = GeneralizedCrowdingSelector(phi=1.0)
        rng_rate = StableRNG(2026)
        trials = 4000
        child_wins = 0

        for _ in 1:trials
            nextpop, _ = select(surv, parent, parent_fit, child, child_fit, rng_rate)
            child_wins += (nextpop[1] == child[1])
        end

        expected = parent_fit[1] / (parent_fit[1] + child_fit[1])
        observed = child_wins / trials
        @test abs(observed - expected) < 0.05
    end

    @testset "GeneralizedCrowdingSelector rejects phi < 0" begin
        surv = GeneralizedCrowdingSelector(phi=-0.1)
        @test_throws ArgumentError select(
            surv,
            [[1,2,3,-1,4]],
            [10.0],
            [[1,3,2,-1,4]],
            [9.0],
            StableRNG(7)
        )
    end
end
