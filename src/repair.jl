"""
Repair a potentially inconsistent/infeasible route set in-place.

Workflow:
1) Remove invalid/duplicate assignments and collect missing patients.
2) Reinsert missing patients with feasible-first insertion, else force insertion.
3) While infeasible routes remain, remove the most problematic visit from one bad
   route and reinsert it elsewhere.
4) Normalize and return repaired routes.
"""
function repair_routes!(
    inst::Instance,
    routes::Vector{Vector{Int}},
    rng::AbstractRNG;
    max_iter::Int = 1_000,
)
    missing = deduplicate_and_missing!(inst, routes)
    shuffle!(rng, missing)

    for pid in missing
        insert_or_force!(inst, routes, pid, rng)
    end

    iter = 0
    while iter < max_iter
        bad_idx = first_infeasible_route_idx(inst, routes)
        if bad_idx == 0
            break
        end

        route = routes[bad_idx]
        if isempty(route)
            deleteat!(routes, bad_idx)
            continue
        end

        # Remove the patient whose removal most improves route feasibility.
        best_i = best_removal_idx(inst, route)
        pid = route[best_i]
        deleteat!(route, best_i)
        normalize_routes!(routes)

        insert_or_force!(inst, routes, pid, rng)

        iter += 1
    end

    normalize_routes!(routes)
    return routes
end

"""
Remove the first occurrence of `pid` from the route set.

Returns `true` if the patient was found and removed, otherwise `false`.
"""
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




"""
Insert one patient in a feasible position, or force insertion if no feasible option exists.

Attempts to insert the patient at the least-cost feasible position. If no feasible
insertion is possible, falls back to force insertion with soft penalties.
"""
@inline function insert_or_force!(
    inst::Instance,
    routes::Vector{Vector{Int}},
    pid::Int,
    rng::AbstractRNG,
)
    if !insert_best_feasible!(inst, routes, pid, rng; stochastic = true)
        force_insert!(inst, routes, pid, rng)
    end
    return nothing
end



# ---------- Internal Helpers ----------

"""
Insert one patient in the least-cost feasible position.

Searches all insertion positions across existing routes and optionally a new route
if nurse capacity allows it. Uses travel increase (plus optional random tie-break)
as score. Returns `true` if insertion succeeded.
"""
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


"""
Insert one patient even when no strictly feasible insertion exists.

Chooses the position with lowest soft-penalized score (travel delta + lateness
penalty + noise). Falls back to opening a new route or appending to the shortest
route when needed.
"""
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

"""
Clean route assignments and report missing patients.

Removes invalid patient ids and duplicate visits in-place, then returns a vector
of patient ids that are not assigned to any route.
"""
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


@inline function first_infeasible_route_idx(inst::Instance, routes::Vector{Vector{Int}})::Int
    for ridx in eachindex(routes)
        if !route_eval(inst, routes[ridx]).feasible
            return ridx
        end
    end
    return 0
end

function best_removal_idx(inst::Instance, route::Vector{Int})::Int
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
    return best_i
end