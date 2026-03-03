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

"""
    improve(::TwoOptLocalSearch, individual, instance, ga_config, generation, max_generations)

Run a lightweight 2-opt local-search pass:
- Operates route-by-route (between `-1` separators)
- One first-improvement pass per route
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

    return current
end
