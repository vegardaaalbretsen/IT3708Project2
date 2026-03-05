# Utility for flattening patient order across a route collection.
@inline function append_route_patients!(dest::Vector{Int}, routes::Vector{Vector{Int}})
    for route in routes
        append!(dest, route)
    end
    return dest
end

function flatten_routes(routes::Vector{Vector{Int}})
    patients = Int[]
    append_route_patients!(patients, routes)
    return patients
end

"""
Construct one initial feasible candidate with a randomized greedy insertion heuristic.

The method repeatedly starts a new route from an urgent patient (low time-window end),
then inserts additional unassigned patients where feasible insertion cost is lowest.
If route feasibility cannot be maintained within the available nurse count, `nothing`
is returned. Final routes are repaired and evaluated before returning.
"""
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

        while !isempty(unassigned)
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
            else
                break
            end
        end

        push!(routes, route)
    end

    repair_routes!(inst, routes, rng)
    candidate = evaluate_candidate(inst, routes)
    return candidate.feasible ? candidate : nothing
end

"""
Combine two parent candidates into one child.

The operator keeps a random subset of routes from parent `a`, then inserts remaining
patients in a best-feasible order using both parents as insertion order hints. Missing
or conflicting assignments are repaired, and the final child is returned as an
evaluated `Candidate`.
"""
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
    append_route_patients!(order, b.routes)
    append_route_patients!(order, a.routes)

    for pid in order
        if used[pid]
            continue
        end
        insert_or_force!(inst, routes, pid, rng)
        used[pid] = true
    end

    for pid in 1:n
        if !used[pid]
            insert_or_force!(inst, routes, pid, rng)
        end
    end

    repair_routes!(inst, routes, rng)
    return evaluate_candidate(inst, routes)
end

"""
Apply bounded 2-opt improvement on a single route.

Tries segment reversals and keeps the best feasible travel reduction found, stopping
after `max_checks` candidate checks. Returns the improved route (or original route
if no improvement is found).
"""
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

"""
Try inter-route relocate moves to reduce combined travel.

Randomly samples donor/receiver route pairs and moves one patient when a feasible
relocation strictly improves the combined travel of the two routes. Operates in place.
"""
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

"""
Mutate a candidate by partial destroy-and-repair plus local improvement.

Removes a random subset of patients, reinserts them with feasible-biased insertion,
optionally applies 2-opt and relocate improvement, then repairs route consistency and
returns a newly evaluated `Candidate`.
"""
function mutate(inst::Instance, parent::Candidate, rng::AbstractRNG)::Candidate
    routes = deepcopy_routes(parent.routes)
    flat = flatten_routes(routes)
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
        insert_or_force!(inst, routes, pid, rng)
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
