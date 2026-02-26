"""
Local search operators:
    2-opt (route-wise)
"""

abstract type LocalSearch end

struct TwoOptLocalSearch <: LocalSearch end

function improve(::LocalSearch,
                 individual::Vector{Int},
                 instance::Fitness.HCInstance,
                 ga_config,
                 generation::Int,
                 max_generations::Int)::Vector{Int}
    throw(ArgumentError("No improve method defined for local search operator."))
end

@inline function _full_fitness(individual::Vector{Int},
                               instance::Fitness.HCInstance,
                               ga_config,
                               generation::Int,
                               max_generations::Int)::Float64
    return Fitness.fitness(
        individual,
        instance;
        weights=ga_config.fitness_weights,
        schedule=ga_config.penalty_schedule,
        generation=generation,
        max_generations=max_generations
    )
end

function _route_ranges(individual::Vector{Int})
    ranges = Tuple{Int, Int}[]
    start_idx = 1
    n = length(individual)

    @inbounds for i in 1:n
        if individual[i] == Fitness.SPLIT
            if start_idx <= i - 1
                push!(ranges, (start_idx, i - 1))
            end
            start_idx = i + 1
        end
    end

    if start_idx <= n
        push!(ranges, (start_idx, n))
    end

    return ranges
end

@inline function _relocate_move(individual::Vector{Int}, from_idx::Int, to_idx::Int)::Vector{Int}
    candidate = copy(individual)
    gene = candidate[from_idx]
    deleteat!(candidate, from_idx)
    insert_idx = to_idx > from_idx ? to_idx - 1 : to_idx
    insert!(candidate, insert_idx, gene)
    return candidate
end

function _inter_route_relocate_first_improvement(
    current::Vector{Int},
    current_fit::Float64,
    instance::Fitness.HCInstance,
    ga_config,
    generation::Int,
    max_generations::Int
)::Tuple{Vector{Int}, Float64, Bool}
    routes = _route_ranges(current)
    num_active_routes = length(routes)
    num_active_routes <= 1 && return current, current_fit, false

    @inbounds for src_route_idx in 1:num_active_routes
        src_start, src_end = routes[src_route_idx]
        src_len = src_end - src_start + 1

        # Do not collapse below configured minimum number of active routes.
        if src_len == 1 && num_active_routes <= ga_config.min_active_routes
            continue
        end

        for from_idx in src_start:src_end
            for dst_route_idx in 1:num_active_routes
                dst_route_idx == src_route_idx && continue
                dst_start, dst_end = routes[dst_route_idx]

                # Lightweight candidate set: route start and route end.
                for to_idx in (dst_start, dst_end + 1)
                    candidate = _relocate_move(current, from_idx, to_idx)
                    cand_fit = _full_fitness(candidate, instance, ga_config, generation, max_generations)
                    if cand_fit < current_fit - 1e-9
                        return candidate, cand_fit, true
                    end
                end
            end
        end
    end

    return current, current_fit, false
end

"""
    improve(::TwoOptLocalSearch, individual, instance, ga_config, generation, max_generations)

Run a lightweight 2-opt local-search pass:
- Operates route-by-route (between `-1` separators)
- One first-improvement pass per route
- One lightweight inter-route relocate first-improvement pass
- Uses full penalized GA fitness as objective
"""
function improve(::TwoOptLocalSearch,
                 individual::Vector{Int},
                 instance::Fitness.HCInstance,
                 ga_config,
                 generation::Int,
                 max_generations::Int)::Vector{Int}
    current = copy(individual)
    current_fit = _full_fitness(current, instance, ga_config, generation, max_generations)

    for (route_start, route_end) in _route_ranges(current)
        # Need at least 4 customers in a route for meaningful 2-opt in this setup.
        if route_end - route_start + 1 < 4
            continue
        end

        improved = false
        for i in route_start:(route_end - 2)
            for j in (i + 1):route_end
                candidate = copy(current)
                reverse!(candidate, i, j)

                cand_fit = _full_fitness(candidate, instance, ga_config, generation, max_generations)
                if cand_fit < current_fit - 1e-9
                    current = candidate
                    current_fit = cand_fit
                    improved = true
                    break
                end
            end
            improved && break
        end
    end

    current, current_fit, _ = _inter_route_relocate_first_improvement(
        current, current_fit, instance, ga_config, generation, max_generations
    )

    return current
end
