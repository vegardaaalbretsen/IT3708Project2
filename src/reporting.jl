function format_route_sequence(inst::Instance, route_info::RouteEval)::String
    if isempty(route_info.visits)
        return "D (0.00) -> D (0.00)"
    end

    parts = String["D (0.00)"]
    for v in route_info.visits
        p = inst.patients[v.patient]
        push!(
            parts,
            @sprintf(
                "%d (%.2f-%.2f) [%.0f-%.0f]",
                v.patient,
                v.start,
                v.finish,
                p.start_time,
                p.end_time,
            ),
        )
    end
    push!(parts, @sprintf("D (%.2f)", route_info.duration))
    return join(parts, " -> ")
end

function write_instance_header(io::IO, inst::Instance, used_nurses::Int)
    available_nurses = inst.nbr_nurses - used_nurses
    @printf(io, "Instance: %s\n", inst.name)
    @printf(io, "Nurse capacity: %d\n", inst.capacity_nurse)
    @printf(io, "Used nurses: %d | Available nurses: %d (of %d)\n", used_nurses, available_nurses, inst.nbr_nurses)
    @printf(io, "Capacity delta per route: + means under capacity, - means over capacity\n")
    @printf(io, "Depot return time: %.2f\n", inst.return_time)
end

function solution_report(inst::Instance, best::Candidate)::String
    io = IOBuffer()
    used_nurses = length(best.routes)
    write_instance_header(io, inst, used_nurses)
    @printf(io, "--------------------------------------------------------------------------------\n")
    @printf(io, "Nurse    Route duration    Covered demand    Cap Δ(+/-)    Patient sequence\n")
    @printf(io, "--------------------------------------------------------------------------------\n")

    for nurse_idx in 1:used_nurses
        route = best.routes[nurse_idx]
        r = route_eval(inst, route)
        cap_delta = inst.capacity_nurse - r.demand
        seq = format_route_sequence(inst, r)
        @printf(io, "N%-6d %-16.2f %-16d %+12d    %s\n", nurse_idx, r.duration, r.demand, cap_delta, seq)
    end

    @printf(io, "--------------------------------------------------------------------------------\n")
    @printf(io, "Objective value (total travel time): %.2f\n", best.total_travel)
    if inst.benchmark > 0
        gap_pct = 100.0 * (best.total_travel - inst.benchmark) / inst.benchmark
        @printf(io, "Benchmark: %.2f | Gap: %.2f%%\n", inst.benchmark, gap_pct)
    end
    return String(take!(io))
end

function config_report(config::GAConfig)::String
    io = IOBuffer()
    @printf(io, "GA Configuration:\n")
    @printf(io, "Population size: %d\n", config.population_size)
    @printf(io, "Generations: %d\n", config.generations)
    @printf(io, "Tournament size: %d\n", config.tournament_size)
    @printf(io, "Elitism: %d\n", config.elitism)
    @printf(io, "Crossover rate: %.2f\n", config.crossover_rate)
    @printf(io, "Mutation rate: %.2f\n", config.mutation_rate)
    @printf(io, "Time limit (sec): %.2f\n", config.time_limit_sec)
    return String(take!(io))
end

function terminal_summary(inst::Instance, best::Candidate)::String
    io = IOBuffer()
    used_nurses = length(best.routes)
    write_instance_header(io, inst, used_nurses)
    @printf(io, "\n")
    return String(take!(io))
end
