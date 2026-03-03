function remove_patient!(routes::Vector{Vector{Int}}, pid::Int)
    for route in routes
        idx = findfirst(==(pid), route)
        if idx !== nothing
            deleteat!(route, idx)
            return true
        end
    end
    return false
end

function insert_best_feasible!(
    inst::Instance,
    routes::Vector{Vector{Int}},
    pid::Int,
    rng::AbstractRNG;
    stochastic::Bool = false,
)::Bool
    best_route = 0
    best_pos = 0
    best_new_route = false
    best_score = Inf

    for ridx in eachindex(routes)
        route = routes[ridx]
        base = route_eval(inst, route).travel
        for pos in 1:(length(route) + 1)
            cand = copy(route)
            insert!(cand, pos, pid)
            r = route_eval(inst, cand)
            if r.feasible
                score = (r.travel - base)
                if stochastic
                    score += rand(rng) * 0.5
                end
                if score < best_score
                    best_score = score
                    best_route = ridx
                    best_pos = pos
                    best_new_route = false
                end
            end
        end
    end

    if length(routes) < inst.nbr_nurses
        solo = route_eval(inst, [pid])
        if solo.feasible
            score = solo.travel + (stochastic ? rand(rng) * 0.5 : 0.0)
            if score < best_score
                best_score = score
                best_route = 0
                best_pos = 1
                best_new_route = true
            end
        end
    end

    if best_score < Inf
        if best_new_route
            push!(routes, [pid])
        else
            insert!(routes[best_route], best_pos, pid)
        end
        return true
    end
    return false
end

function force_insert!(inst::Instance, routes::Vector{Vector{Int}}, pid::Int, rng::AbstractRNG)
    best_route = 0
    best_pos = 1
    best_score = Inf

    for ridx in eachindex(routes)
        route = routes[ridx]
        base = route_eval(inst, route)
        for pos in 1:(length(route) + 1)
            cand = copy(route)
            insert!(cand, pos, pid)
            r = route_eval(inst, cand)
            # Low score is good; no hard feasibility requirement in force mode.
            score = (r.travel - base.travel) + (10_000.0 * r.lateness) + rand(rng)
            if score < best_score
                best_score = score
                best_route = ridx
                best_pos = pos
            end
        end
    end

    if best_route == 0
        if length(routes) < inst.nbr_nurses
            push!(routes, [pid])
            return
        end
        # Last-resort fallback: append to shortest route.
        ridx = argmin(length.(routes))
        push!(routes[ridx], pid)
        return
    end

    insert!(routes[best_route], best_pos, pid)
end

function deduplicate_and_missing!(inst::Instance, routes::Vector{Vector{Int}})
    n = patient_count(inst)
    seen = falses(n)
    missing = Int[]

    for route in routes
        i = 1
        while i <= length(route)
            pid = route[i]
            if !(1 <= pid <= n) || seen[pid]
                deleteat!(route, i)
            else
                seen[pid] = true
                i += 1
            end
        end
    end

    for pid in 1:n
        if !seen[pid]
            push!(missing, pid)
        end
    end

    normalize_routes!(routes)
    return missing
end

function repair_routes!(
    inst::Instance,
    routes::Vector{Vector{Int}},
    rng::AbstractRNG;
    max_iter::Int = 1_000,
)
    missing = deduplicate_and_missing!(inst, routes)
    shuffle!(rng, missing)

    for pid in missing
        if !insert_best_feasible!(inst, routes, pid, rng; stochastic = true)
            force_insert!(inst, routes, pid, rng)
        end
    end

    iter = 0
    while iter < max_iter
        bad_idx = 0
        for ridx in eachindex(routes)
            if !route_eval(inst, routes[ridx]).feasible
                bad_idx = ridx
                break
            end
        end
        if bad_idx == 0
            break
        end

        route = routes[bad_idx]
        if isempty(route)
            deleteat!(routes, bad_idx)
            continue
        end

        # Remove the patient whose removal most improves route feasibility.
        best_i = 1
        best_score = Inf
        for i in eachindex(route)
            tmp = copy(route)
            deleteat!(tmp, i)
            r = route_eval(inst, tmp)
            score = (10_000.0 * r.lateness) + r.travel
            if score < best_score
                best_score = score
                best_i = i
            end
        end

        pid = route[best_i]
        deleteat!(route, best_i)
        normalize_routes!(routes)

        if !insert_best_feasible!(inst, routes, pid, rng; stochastic = true)
            force_insert!(inst, routes, pid, rng)
        end

        iter += 1
    end

    normalize_routes!(routes)
    return routes
end
