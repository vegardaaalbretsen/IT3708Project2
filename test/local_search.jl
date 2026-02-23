using Test
using StableRNGs
using HomeCareGA

@info("Testing local search")
@testset verbose=true "Local Search" begin
    function line_instance(N::Int)
        travel = Matrix{Float64}(undef, N + 1, N + 1)
        @inbounds for i in 0:N, j in 0:N
            travel[i + 1, j + 1] = abs(i - j)
        end

        return HCInstance(
            name="line_$N",
            N=N,
            travel=travel,
            start_time=zeros(N),
            end_time=fill(10_000.0, N),
            care_time=zeros(N),
            demand=ones(N),
            capacity_nurse=10_000.0,
            depot_return_time=10_000.0
        )
    end

    weights = FitnessWeights(
        w_travel=1.0,
        w_wait=0.0,
        w_capacity=1000.0,
        w_return=1000.0,
        w_late=2000.0,
        w_early=0.0
    )
    schedule = PenaltySchedule(min_scale=1.0, max_scale=1.0, power=1.0, mag_scale=0.0)

    @testset "TwoOptLocalSearch preserves representation and does not worsen fitness" begin
        inst = line_instance(6)
        x = Int[1, 3, 2, 4, SPLIT, 5, 6]

        cfg = GAConfig(
            p_c=0.0,
            p_m=0.0,
            p_ls=1.0,
            selector=TournamentSelector(1),
            crossover=O1XCrossover(),
            mutator=SwapMutator(),
            local_search=TwoOptLocalSearch(),
            survivor=ElitistSelector(num_elites=0),
            pop_size=1,
            max_generations=2,
            fitness_weights=weights,
            penalty_schedule=schedule,
            keep_history=false,
            verbose=false
        )

        f_before = fitness(x, inst; weights=weights, schedule=schedule, generation=1, max_generations=2)
        y = improve(TwoOptLocalSearch(), x, inst, cfg, 1, 2)
        f_after = fitness(y, inst; weights=weights, schedule=schedule, generation=1, max_generations=2)

        @test length(y) == length(x)
        @test count(==(SPLIT), y) == count(==(SPLIT), x)
        @test sort(filter(!=(SPLIT), y)) == sort(filter(!=(SPLIT), x))
        @test f_after <= f_before + 1e-9
    end

    @testset "GA with local search improves vs no local search (deterministic toy case)" begin
        inst = line_instance(4)
        initial_population = [Int[1, 3, 2, 4]]

        base_cfg = GAConfig(
            p_c=0.0,
            p_m=0.0,
            p_ls=0.0,
            selector=TournamentSelector(1),
            crossover=O1XCrossover(),
            mutator=SwapMutator(),
            local_search=nothing,
            survivor=ElitistSelector(num_elites=0),
            pop_size=1,
            max_generations=2,
            fitness_weights=weights,
            penalty_schedule=schedule,
            keep_history=true,
            verbose=false
        )

        ls_cfg = GAConfig(
            p_c=0.0,
            p_m=0.0,
            p_ls=1.0,
            selector=TournamentSelector(1),
            crossover=O1XCrossover(),
            mutator=SwapMutator(),
            local_search=TwoOptLocalSearch(),
            survivor=ElitistSelector(num_elites=0),
            pop_size=1,
            max_generations=2,
            fitness_weights=weights,
            penalty_schedule=schedule,
            keep_history=true,
            verbose=false
        )

        res_no_ls = HomeCareGA.run(
            inst,
            base_cfg;
            rng=StableRNG(123),
            initial_population=initial_population,
            max_generations=2
        )
        res_ls = HomeCareGA.run(
            inst,
            ls_cfg;
            rng=StableRNG(123),
            initial_population=initial_population,
            max_generations=2
        )

        @test res_ls.best_fitness <= res_no_ls.best_fitness + 1e-9
    end
end
