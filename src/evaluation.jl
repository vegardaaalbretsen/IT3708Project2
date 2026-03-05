@inline patient_count(inst::Instance) = length(inst.patients)

function route_eval(inst::Instance, route::Vector{Int})::RouteEval
    node = 0
    t = 0.0
    travel = 0.0
    load = 0
    lateness = 0.0
    visits = Vector{VisitInfo}(undef, 0)

    for pid in route
        p = inst.patients[pid]
        tr = inst.travel_times[node + 1, pid + 1]
        travel += tr
        arrival = t + tr
        start_visit = max(arrival, p.start_time)
        finish = start_visit + p.care_time
        if finish > p.end_time
            lateness += finish - p.end_time
        end
        push!(visits, VisitInfo(pid, arrival, start_visit, finish))
        t = finish
        load += p.demand
        if load > inst.capacity_nurse
            lateness += (load - inst.capacity_nurse)
        end
        node = pid
    end

    tr_back = inst.travel_times[node + 1, 1]
    travel += tr_back
    t += tr_back
    if t > inst.return_time
        lateness += (t - inst.return_time)
    end

    feasible = lateness <= 1e-9
    return RouteEval(feasible, travel, t, load, visits, lateness)
end

function deepcopy_routes(routes::Vector{Vector{Int}})
    return [copy(r) for r in routes]
end

function normalize_routes!(routes::Vector{Vector{Int}})
    filter!(!isempty, routes)
    return routes
end

function evaluate_candidate(inst::Instance, routes::Vector{Vector{Int}})::Candidate
    n = patient_count(inst)
    counts = zeros(Int, n)
    total_travel = 0.0
    total_lateness = 0.0
    invalid_genes = 0

    for route in routes
        r = route_eval(inst, route)
        total_travel += r.travel
        total_lateness += r.lateness
        for pid in route
            if 1 <= pid <= n
                counts[pid] += 1
            else
                invalid_genes += 1
                total_lateness += 1_000.0
            end
        end
    end

    duplicates = 0
    missing = 0
    for c in counts
        if c == 0
            missing += 1
        elseif c > 1
            duplicates += (c - 1)
        end
    end

    extra_routes = max(0, length(routes) - inst.nbr_nurses)
    penalty = 1_000_000.0 * (duplicates + missing + extra_routes)
    fitness = total_travel + penalty + (10_000.0 * total_lateness)
    feasible = (invalid_genes == 0) && (duplicates == 0) && (missing == 0) && (extra_routes == 0) && (total_lateness <= 1e-9)

    return Candidate(
        deepcopy_routes(routes),
        total_travel,
        fitness,
        feasible,
    )
end

function copy_candidate(c::Candidate)::Candidate
    return Candidate(
        deepcopy_routes(c.routes),
        c.total_travel,
        c.fitness,
        c.feasible,
    )
end
