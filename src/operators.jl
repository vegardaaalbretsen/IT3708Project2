function construct_solution(inst::Instance, rng::AbstractRNG; randomized::Bool = true)
    n = patient_count(inst)
    unassigned = collect(1:n)
    routes = Vector{Vector{Int}}()

    while !isempty(unassigned)
        if length(routes) >= inst.nbr_nurses
            return nothing
        end

        sort!(
            unassigned;
            by = pid -> inst.patients[pid].end_time + (randomized ? rand(rng) * 20.0 : 0.0),
        )
        k = min(length(unassigned), randomized ? 8 : 1)
        seed = unassigned[randomized ? rand(rng, 1:k) : 1]
        route = [seed]
        deleteat!(unassigned, findfirst(==(seed), unassigned))

        if !route_eval(inst, route).feasible
            return nothing
        end

        improved = true
        while improved && !isempty(unassigned)
            improved = false
            base = route_eval(inst, route).travel
            best_pid = 0
            best_pos = 1
            best_score = Inf

            candidates = copy(unassigned)
            if randomized && length(candidates) > 35
                shuffle!(rng, candidates)
                resize!(candidates, 35)
            end

            for pid in candidates
                for pos in 1:(length(route) + 1)
                    cand = copy(route)
                    insert!(cand, pos, pid)
                    r = route_eval(inst, cand)
                    if r.feasible
                        score = (r.travel - base) + (randomized ? rand(rng) * 0.5 : 0.0)
                        if score < best_score
                            best_score = score
                            best_pid = pid
                            best_pos = pos
                        end
                    end
                end
            end

            if best_pid != 0
                insert!(route, best_pos, best_pid)
                deleteat!(unassigned, findfirst(==(best_pid), unassigned))
                improved = true
            end
        end

        push!(routes, route)
    end

    repair_routes!(inst, routes, rng)
    candidate = evaluate_candidate(inst, routes)
    return candidate.feasible ? candidate : nothing
end

function crossover(inst::Instance, a::Candidate, b::Candidate, rng::AbstractRNG)::Candidate
    routes = Vector{Vector{Int}}()
    n = patient_count(inst)
    used = falses(n)

    idx = collect(eachindex(a.routes))
    shuffle!(rng, idx)
    keep_count = isempty(idx) ? 0 : clamp(round(Int, length(idx) * (0.3 + 0.3 * rand(rng))), 1, length(idx))

    for i in 1:keep_count
        route = copy(a.routes[idx[i]])
        push!(routes, route)
        for pid in route
            used[pid] = true
        end
    end

    order = Int[]
    for route in b.routes
        append!(order, route)
    end
    for route in a.routes
        append!(order, route)
    end

    for pid in order
        if used[pid]
            continue
        end
        if !insert_best_feasible!(inst, routes, pid, rng; stochastic = true)
            force_insert!(inst, routes, pid, rng)
        end
        used[pid] = true
    end

    missing = Int[]
    for pid in 1:n
        if !used[pid]
            push!(missing, pid)
        end
    end
    for pid in missing
        if !insert_best_feasible!(inst, routes, pid, rng; stochastic = true)
            force_insert!(inst, routes, pid, rng)
        end
    end

    repair_routes!(inst, routes, rng)
    return evaluate_candidate(inst, routes)
end

function two_opt_route(inst::Instance, route::Vector{Int}; max_checks::Int = 120)
    if length(route) < 4
        return route
    end
    best_route = copy(route)
    best = route_eval(inst, best_route)
    checks = 0

    for i in 1:(length(route) - 2)
        for j in (i + 1):length(route)
            cand = copy(best_route)
            cand[i:j] = reverse(cand[i:j])
            r = route_eval(inst, cand)
            checks += 1
            if r.feasible && r.travel + 1e-9 < best.travel
                best_route = cand
                best = r
            end
            if checks >= max_checks
                return best_route
            end
        end
    end
    return best_route
end

function try_relocate_improve!(inst::Instance, routes::Vector{Vector{Int}}, rng::AbstractRNG; attempts::Int = 40)
    if length(routes) < 2
        return
    end
    for _ in 1:attempts
        donor_idx = rand(rng, eachindex(routes))
        receiver_idx = rand(rng, eachindex(routes))
        if donor_idx == receiver_idx || isempty(routes[donor_idx])
            continue
        end

        donor = routes[donor_idx]
        receiver = routes[receiver_idx]

        i = rand(rng, eachindex(donor))
        pid = donor[i]
        base = route_eval(inst, donor).travel + route_eval(inst, receiver).travel

        donor_after = copy(donor)
        deleteat!(donor_after, i)
        donor_eval = route_eval(inst, donor_after)
        if !donor_eval.feasible
            continue
        end

        best_pos = 0
        best_new = Inf
        for pos in 1:(length(receiver) + 1)
            recv_after = copy(receiver)
            insert!(recv_after, pos, pid)
            recv_eval = route_eval(inst, recv_after)
            if recv_eval.feasible
                new_total = donor_eval.travel + recv_eval.travel
                if new_total + 1e-9 < best_new
                    best_new = new_total
                    best_pos = pos
                end
            end
        end

        if best_pos != 0 && best_new + 1e-9 < base
            deleteat!(routes[donor_idx], i)
            insert!(routes[receiver_idx], best_pos, pid)
            normalize_routes!(routes)
        end
    end
end

function mutate(inst::Instance, parent::Candidate, rng::AbstractRNG)::Candidate
    routes = deepcopy_routes(parent.routes)
    flat = Int[]
    for route in routes
        append!(flat, route)
    end
    if isempty(flat)
        return evaluate_candidate(inst, routes)
    end

    k = rand(rng, 2:min(10, length(flat)))
    shuffle!(rng, flat)
    removed = flat[1:k]

    for pid in removed
        remove_patient!(routes, pid)
    end
    normalize_routes!(routes)

    shuffle!(rng, removed)
    for pid in removed
        if !insert_best_feasible!(inst, routes, pid, rng; stochastic = true)
            force_insert!(inst, routes, pid, rng)
        end
    end

    if !isempty(routes) && rand(rng) < 0.6
        idx = rand(rng, eachindex(routes))
        routes[idx] = two_opt_route(inst, routes[idx])
    end
    if rand(rng) < 0.5
        try_relocate_improve!(inst, routes, rng)
    end

    repair_routes!(inst, routes, rng)
    return evaluate_candidate(inst, routes)
end
