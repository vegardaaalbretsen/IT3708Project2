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
    feasible = filter(c -> c.feasible, pop)
    if !isempty(feasible)
        sort!(feasible, by = c -> c.total_travel)
        return feasible[1]
    end
    sort!(pop, by = c -> c.fitness)
    return pop[1]
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

function _generation_metrics(
    inst::Instance,
    population::Vector{Candidate},
    generation::Int;
    crossover_applied::Int = 0,
    crossover_improved::Int = 0,
    mutation_applied::Int = 0,
    mutation_improved::Int = 0,
)
    n = length(population)
    n == 0 && error("Population is empty while collecting metrics.")

    feasible = Candidate[]
    sizehint!(feasible, n)
    for c in population
        if c.feasible
            push!(feasible, c)
        end
    end
    feasible_ratio = length(feasible) / n

    best_travel = NaN
    median_travel = NaN
    worst_travel = NaN
    if !isempty(feasible)
        travels = Float64[c.total_travel for c in feasible]
        sort!(travels)
        best_travel = travels[1]
        median_travel = _median_of_sorted(travels)
        worst_travel = travels[end]
    end

    avg_lateness = sum(c.total_lateness for c in population) / n

    used_routes = Int[length(c.routes) for c in population]
    avg_used_routes = sum(used_routes) / n
    min_used_routes = minimum(used_routes)
    max_used_routes = maximum(used_routes)

    total_routes = 0
    total_visits = 0
    sum_route_duration = 0.0
    sum_route_travel = 0.0
    sum_capacity_slack = 0.0
    min_capacity_slack = Inf
    sum_capacity_utilization = 0.0
    sum_depot_slack = 0.0
    min_depot_slack = Inf
    sum_wait_time = 0.0
    sum_timewindow_slack = 0.0
    min_timewindow_slack = Inf

    inv_capacity = 1.0 / float(inst.capacity_nurse)

    for c in population
        for route in c.routes
            r = route_eval(inst, route)
            total_routes += 1

            sum_route_duration += r.duration
            sum_route_travel += r.travel

            cap_slack = float(inst.capacity_nurse - r.demand)
            sum_capacity_slack += cap_slack
            min_capacity_slack = min(min_capacity_slack, cap_slack)
            sum_capacity_utilization += r.demand * inv_capacity

            depot_slack = inst.return_time - r.duration
            sum_depot_slack += depot_slack
            min_depot_slack = min(min_depot_slack, depot_slack)

            for v in r.visits
                total_visits += 1
                sum_wait_time += (v.start - v.arrival)
                tw_slack = inst.patients[v.patient].end_time - v.finish
                sum_timewindow_slack += tw_slack
                min_timewindow_slack = min(min_timewindow_slack, tw_slack)
            end
        end
    end

    avg_route_duration = total_routes > 0 ? (sum_route_duration / total_routes) : NaN
    avg_route_travel = total_routes > 0 ? (sum_route_travel / total_routes) : NaN
    avg_capacity_slack = total_routes > 0 ? (sum_capacity_slack / total_routes) : NaN
    min_capacity_slack = isfinite(min_capacity_slack) ? min_capacity_slack : NaN
    avg_capacity_utilization = total_routes > 0 ? (sum_capacity_utilization / total_routes) : NaN
    avg_depot_slack = total_routes > 0 ? (sum_depot_slack / total_routes) : NaN
    min_depot_slack = isfinite(min_depot_slack) ? min_depot_slack : NaN
    avg_wait_time = total_visits > 0 ? (sum_wait_time / total_visits) : NaN
    avg_timewindow_slack = total_visits > 0 ? (sum_timewindow_slack / total_visits) : NaN
    min_timewindow_slack = isfinite(min_timewindow_slack) ? min_timewindow_slack : NaN

    edge_counts = Dict{Tuple{Int, Int}, Int}()
    total_edges = 0
    for c in population
        for route in c.routes
            if isempty(route)
                continue
            end
            prev = 0
            for pid in route
                edge = (prev, pid)
                edge_counts[edge] = get(edge_counts, edge, 0) + 1
                total_edges += 1
                prev = pid
            end
            edge = (prev, 0)
            edge_counts[edge] = get(edge_counts, edge, 0) + 1
            total_edges += 1
        end
    end

    unique_edges = length(edge_counts)
    edge_diversity = total_edges == 0 ? 0.0 : (unique_edges / total_edges)
    edge_entropy = 0.0
    if unique_edges > 1 && total_edges > 0
        inv_total = 1.0 / total_edges
        h = 0.0
        for cnt in values(edge_counts)
            p = cnt * inv_total
            h -= p * log(p)
        end
        edge_entropy = h / log(unique_edges)
    end

    crossover_success_rate = crossover_applied > 0 ? (crossover_improved / crossover_applied) : NaN
    mutation_success_rate = mutation_applied > 0 ? (mutation_improved / mutation_applied) : NaN

    return (
        generation = generation,
        best_total_travel = best_travel,
        median_total_travel = median_travel,
        worst_total_travel = worst_travel,
        feasible_ratio = feasible_ratio,
        avg_lateness = avg_lateness,
        avg_used_routes = avg_used_routes,
        min_used_routes = min_used_routes,
        max_used_routes = max_used_routes,
        avg_route_duration = avg_route_duration,
        avg_route_travel = avg_route_travel,
        avg_capacity_utilization = avg_capacity_utilization,
        avg_capacity_slack = avg_capacity_slack,
        min_capacity_slack = min_capacity_slack,
        avg_depot_slack = avg_depot_slack,
        min_depot_slack = min_depot_slack,
        avg_wait_time = avg_wait_time,
        avg_timewindow_slack = avg_timewindow_slack,
        min_timewindow_slack = min_timewindow_slack,
        edge_diversity = edge_diversity,
        edge_entropy = edge_entropy,
        crossover_success_rate = crossover_success_rate,
        mutation_success_rate = mutation_success_rate,
    )
end

function run_ga(inst::Instance, cfg::GAConfig, rng::AbstractRNG)
    population = initialize_population(inst, cfg, rng)
    best = copy_candidate(current_best(population))
    metrics_history = NamedTuple[_generation_metrics(inst, population, 0)]
    start_time = time()

    for gen in 1:cfg.generations
        sort!(population, by = c -> c.fitness)
        elite_count = min(cfg.elitism, length(population))
        offspring_count = cfg.population_size - elite_count
        crossover_applied = 0
        crossover_improved = 0
        mutation_applied = 0
        mutation_improved = 0

        next_pop = Vector{Candidate}(undef, cfg.population_size)
        for i in 1:elite_count
            next_pop[i] = copy_candidate(population[i])
        end

        # Generate offspring in parallel using deterministic per-index RNG seeds.
        if offspring_count > 0
            offspring = Vector{Candidate}(undef, offspring_count)
            seed_base = rand(rng, UInt)
            crossover_applied_flags = falses(offspring_count)
            crossover_improved_flags = falses(offspring_count)
            mutation_applied_flags = falses(offspring_count)
            mutation_improved_flags = falses(offspring_count)

            # Phase 1: Generate offspring via crossover and mutation operators
            Threads.@threads for i in 1:offspring_count
                # Deterministic unique seed per thread and offspring index (avoid thread race conditions on rng and keeep reproducibility)
                local_seed = seed_base + UInt(gen) * UInt(1_000_003) + UInt(i) * UInt(97)
                local_rng = MersenneTwister(local_seed)

                p1 = tournament_select(population, local_rng, cfg.tournament_size)
                child = copy_candidate(p1)

                if rand(local_rng) < cfg.crossover_rate
                    crossover_applied_flags[i] = true
                    p2 = tournament_select(population, local_rng, cfg.tournament_size)
                    child = crossover(inst, p1, p2, local_rng)
                    if child.fitness + 1e-9 < p1.fitness
                        crossover_improved_flags[i] = true
                    end
                end

                if rand(local_rng) < cfg.mutation_rate
                    mutation_applied_flags[i] = true
                    fitness_before_mutation = child.fitness
                    child = mutate(inst, child, local_rng)
                    if child.fitness + 1e-9 < fitness_before_mutation
                        mutation_improved_flags[i] = true
                    end
                end

                offspring[i] = child
            end

            crossover_applied = count(identity, crossover_applied_flags)
            crossover_improved = count(identity, crossover_improved_flags)
            mutation_applied = count(identity, mutation_applied_flags)
            mutation_improved = count(identity, mutation_improved_flags)

            # Phase 2: Repair all offspring and re-evaluate
            for i in 1:offspring_count
                repair_routes!(inst, offspring[i].routes, rng)
                offspring[i] = evaluate_candidate(inst, offspring[i].routes)
            end

            # Phase 3: Apply local search to selected offspring
            for i in 1:offspring_count
                if rand(rng) < cfg.local_search_rate
                    offspring[i] = local_search(inst, offspring[i], rng)
                end
            end

            for i in 1:offspring_count
                next_pop[elite_count + i] = offspring[i]
            end
        end

        population = next_pop

        # --- Collect metrics and update best solution ---
        push!(
            metrics_history,
            _generation_metrics(
                inst,
                population,
                gen;
                crossover_applied = crossover_applied,
                crossover_improved = crossover_improved,
                mutation_applied = mutation_applied,
                mutation_improved = mutation_improved,
            ),
        )
        generation_best = current_best(population)

        if generation_best.feasible && (!best.feasible || generation_best.total_travel < best.total_travel)
            best = copy_candidate(generation_best)
        elseif !best.feasible && generation_best.fitness < best.fitness
            best = copy_candidate(generation_best)
        end

        # --- Logging and time limit check ---
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
