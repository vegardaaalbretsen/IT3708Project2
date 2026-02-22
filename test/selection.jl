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
end