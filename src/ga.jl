function initialize_population(inst::Instance, cfg::GAConfig, rng::AbstractRNG)
    pop = Vector{Candidate}()
    attempts = 0
    max_attempts = max(cfg.population_size * 60, 300)

    while length(pop) < cfg.population_size && attempts < max_attempts
        c = construct_solution(inst, rng; randomized = true)
        if c !== nothing
            push!(pop, c)
        end
        attempts += 1
    end

    if isempty(pop)
        error("Could not construct an initial feasible population.")
    end

    while length(pop) < cfg.population_size
        base = copy_candidate(pop[rand(rng, eachindex(pop))])
        push!(pop, mutate(inst, base, rng))
    end

    return pop
end

function tournament_select(pop::Vector{Candidate}, rng::AbstractRNG, k::Int)::Candidate
    best = pop[rand(rng, eachindex(pop))]
    for _ in 2:k
        c = pop[rand(rng, eachindex(pop))]
        if c.fitness < best.fitness
            best = c
        end
    end
    return best
end

function current_best(pop::Vector{Candidate})
    best = pop[1]
    for i in 2:length(pop)
        c = pop[i]
        if c.feasible
            if !best.feasible || (c.total_travel < best.total_travel)
                best = c
            end
        elseif !best.feasible && (c.fitness < best.fitness)
            best = c
        end
    end
    return best
end

@inline function _median_of_sorted(vals::Vector{Float64})::Float64
    n = length(vals)
    if n == 0
        return NaN
    end
    mid = n ÷ 2
    if isodd(n)
        return vals[mid + 1]
    end
    return (vals[mid] + vals[mid + 1]) / 2
end

function _generation_metrics(population::Vector{Candidate}, generation::Int)
    n = length(population)
    n == 0 && error("Population is empty while collecting metrics.")

    feasible_travels = Float64[]
    sizehint!(feasible_travels, n)
    feasible_count = 0
    for c in population
        if c.feasible
            feasible_count += 1
            push!(feasible_travels, c.total_travel)
        end
    end
    feasible_ratio = feasible_count / n

    best_travel = NaN
    median_travel = NaN
    worst_travel = NaN
    if !isempty(feasible_travels)
        sort!(feasible_travels)
        best_travel = feasible_travels[1]
        median_travel = _median_of_sorted(feasible_travels)
        worst_travel = feasible_travels[end]
    end

    return (
        generation = generation,
        best_total_travel = best_travel,
        median_total_travel = median_travel,
        worst_total_travel = worst_travel,
        feasible_ratio = feasible_ratio,
    )
end

function run_ga(inst::Instance, cfg::GAConfig, rng::AbstractRNG)
    population = initialize_population(inst, cfg, rng)
    best = copy_candidate(current_best(population))
    metrics_history = NamedTuple[_generation_metrics(population, 0)]
    start_time = time()

    for gen in 1:cfg.generations
        sort!(population, by = c -> c.fitness)
        elite_count = min(cfg.elitism, length(population))
        offspring_count = cfg.population_size - elite_count

        next_pop = Vector{Candidate}(undef, cfg.population_size)
        for i in 1:elite_count
            next_pop[i] = copy_candidate(population[i])
        end

        # Generate offspring using deterministic per-index RNG seeds.
        if offspring_count > 0
            seed_base = rand(rng, UInt)
            for i in 1:offspring_count
                local_seed = seed_base + UInt(gen) * UInt(1_000_003) + UInt(i) * UInt(97)
                local_rng = MersenneTwister(local_seed)

                p1 = tournament_select(population, local_rng, cfg.tournament_size)
                child = copy_candidate(p1)

                if rand(local_rng) < cfg.crossover_rate
                    p2 = tournament_select(population, local_rng, cfg.tournament_size)
                    child = crossover(inst, p1, p2, local_rng)
                end

                if rand(local_rng) < cfg.mutation_rate
                    child = mutate(inst, child, local_rng)
                end

                next_pop[elite_count + i] = child
            end
        end

        population = next_pop
        push!(
            metrics_history,
            _generation_metrics(population, gen),
        )
        generation_best = current_best(population)

        if generation_best.feasible && (!best.feasible || generation_best.total_travel < best.total_travel)
            best = copy_candidate(generation_best)
        elseif !best.feasible && generation_best.fitness < best.fitness
            best = copy_candidate(generation_best)
        end

        if (gen % cfg.log_every == 0) || (gen == 1)
            @printf(
                "Gen %4d | best travel: %.2f | feasible: %s | routes: %d\n",
                gen,
                best.total_travel,
                string(best.feasible),
                length(best.routes),
            )
        end

        if (time() - start_time) >= cfg.time_limit_sec
            @printf("Stopped due to time limit (%.1f s)\n", cfg.time_limit_sec)
            break
        end
    end

    return best, metrics_history
end
