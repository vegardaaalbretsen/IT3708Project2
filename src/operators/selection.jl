using Random: AbstractRNG, rand
using StatsBase: sample, Weights
"""
Parent selection:
    Tournament

Survivor selection:
    Elitist generational replacement (minimizing):
        Keeps `num_elites` best from the *current* population, and fills the rest with
        the best from the *children*.

        Assumes:
        - length(population) == length(children) == μ
        - fitness vectors aligned with their populations

    Generalized crowding replacement (minimizing):
        Pairs offspring against the most similar parent in each mating pair.
        The child replaces the parent with probability controlled by `phi`.
        - phi = 0.0 -> deterministic crowding
        - phi = 1.0 -> probabilistic crowding
"""


abstract type ParentSelector end
abstract type SurvivorSelector end


struct TournamentSelector <: ParentSelector
    k # Tournament size
end

struct RouletteWheelSelector <: ParentSelector end

Base.@kwdef struct ElitistSelector <: SurvivorSelector 
    num_elites::Int
end

Base.@kwdef struct GeneralizedCrowdingSelector <: SurvivorSelector
    phi::Float64 = 1.0
end


"""
Tournament selection (minimizing).
"""
function select(
        method::TournamentSelector,
        population::Vector{Vector{Int}},
        fitnesses::Vector{Float64}
)::Tuple{Vector{Vector{Int}}, Vector{Float64}}
    
    pop_size = length(population)
    num_winners = pop_size
    @assert length(fitnesses) == pop_size
    
    winners = Vector{Vector{Int}}(undef, num_winners)
    winner_fits = Vector{Float64}(undef, num_winners)
    
    for i in 1:pop_size
        # Randomly pick k indices of individuals
        tournament_indices = sample(1:pop_size, method.k, replace=false)
        
        # Get index of the tournament winner (min fitness)
        winner_idx = tournament_indices[argmin(fitnesses[tournament_indices])]
        
        winners[i] = population[winner_idx]
        winner_fits[i] = fitnesses[winner_idx]
    end
    return winners, winner_fits
end


"""
Roulette wheel selection (minimizing).
"""
function select(
        method::RouletteWheelSelector,
        parents::Vector{Vector{Int}},
        fitnesses::Vector{Float64}
)::Tuple{Vector{Vector{Int}}, Vector{Float64}}
    max_fit = maximum(fitnesses)
    min_fit = minimum(fitnesses)
    pop_size = length(parents)
    @assert length(fitnesses) == pop_size
    
    epsilon = (max_fit - min_fit) * 0.01 + 1e-6
    offset_fitnesses = (max_fit .+ epsilon) .- fitnesses # Inverts fitnesses (e.g. min fit -> max fit and vice versa)

    probabilities = Weights(offset_fitnesses)

    idx_pool = 1:length(parents)
    selected_indices = sample(idx_pool, probabilities, length(parents))
    
    return parents[selected_indices], fitnesses[selected_indices]
end

function select(method::ElitistSelector,
                population::Vector{Vector{Int}},
                pop_fitness::Vector{Float64},
                children::Vector{Vector{Int}},
                child_fitness::Vector{Float64})
    μ = length(population)
    @assert length(pop_fitness) == μ
    @assert length(children) == μ
    @assert length(child_fitness) == μ

    E = clamp(method.num_elites, 0, μ)

    # Indices of best (min fitness)
    elite_idx = sortperm(pop_fitness)[1:E]
    fill_idx  = sortperm(child_fitness)[1:(μ - E)]

    next_pop = Vector{Vector{Int}}(undef, μ)
    next_fit = Vector{Float64}(undef, μ)

    # Copy elites
    for (j, i) in enumerate(elite_idx)
        next_pop[j] = population[i]
        next_fit[j] = pop_fitness[i]
    end

    # Fill remaining with best children
    offset = E
    for (j, i) in enumerate(fill_idx)
        next_pop[offset + j] = children[i]
        next_fit[offset + j] = child_fitness[i]
    end

    return next_pop, next_fit
end

function select(method::SurvivorSelector,
                population::Vector{Vector{Int}},
                pop_fitness::Vector{Float64},
                children::Vector{Vector{Int}},
                child_fitness::Vector{Float64},
                rng::AbstractRNG)
    return select(method, population, pop_fitness, children, child_fitness)
end

@inline function _hamming_distance(a::Vector{Int}, b::Vector{Int})::Int
    @assert length(a) == length(b)
    d = 0
    @inbounds for i in eachindex(a)
        d += (a[i] != b[i])
    end
    return d
end

function _gc_child_probability(
    method::GeneralizedCrowdingSelector,
    parent_fit::Float64,
    child_fit::Float64
)::Float64
    phi = method.phi
    phi >= 0.0 || throw(ArgumentError("GeneralizedCrowdingSelector.phi must be >= 0.0."))

    child_fit == parent_fit && return 0.5

    # Keep denominator positive even if fitness values are <= 0.
    min_fit = min(parent_fit, child_fit)
    shift = min_fit <= 0.0 ? (1.0 - min_fit) : 0.0
    p = parent_fit + shift
    c = child_fit + shift

    if child_fit < parent_fit
        # Child better (minimization): scale weaker side (child) by phi.
        denom = p + phi * c
        return denom == 0.0 ? 0.5 : p / denom
    else
        # Child worse (minimization): scale weaker side (parent) by phi.
        denom = phi * p + c
        return denom == 0.0 ? 0.5 : (phi * p) / denom
    end
end

@inline function _gc_pick(
    method::GeneralizedCrowdingSelector,
    parent::Vector{Int},
    parent_fit::Float64,
    child::Vector{Int},
    child_fit::Float64,
    rng::AbstractRNG
)::Tuple{Vector{Int}, Float64}
    p_child = _gc_child_probability(method, parent_fit, child_fit)
    if rand(rng) < p_child
        return child, child_fit
    end
    return parent, parent_fit
end

function select(
    method::GeneralizedCrowdingSelector,
    population::Vector{Vector{Int}},
    pop_fitness::Vector{Float64},
    children::Vector{Vector{Int}},
    child_fitness::Vector{Float64},
    rng::AbstractRNG
)
    μ = length(population)
    @assert length(pop_fitness) == μ
    @assert length(children) == μ
    @assert length(child_fitness) == μ

    next_pop = Vector{Vector{Int}}(undef, μ)
    next_fit = Vector{Float64}(undef, μ)

    i = 1
    while i <= μ
        if i == μ
            # Odd population size: final parent-child duel.
            next_pop[i], next_fit[i] = _gc_pick(
                method,
                population[i],
                pop_fitness[i],
                children[i],
                child_fitness[i],
                rng
            )
            i += 1
            continue
        end

        p1 = population[i]
        p2 = population[i + 1]
        c1 = children[i]
        c2 = children[i + 1]

        d_same = _hamming_distance(p1, c1) + _hamming_distance(p2, c2)
        d_cross = _hamming_distance(p1, c2) + _hamming_distance(p2, c1)

        if d_cross < d_same
            c1, c2 = c2, c1
            child_fitness_i = child_fitness[i + 1]
            child_fitness_j = child_fitness[i]
        else
            child_fitness_i = child_fitness[i]
            child_fitness_j = child_fitness[i + 1]
        end

        next_pop[i], next_fit[i] = _gc_pick(
            method,
            p1,
            pop_fitness[i],
            c1,
            child_fitness_i,
            rng
        )
        next_pop[i + 1], next_fit[i + 1] = _gc_pick(
            method,
            p2,
            pop_fitness[i + 1],
            c2,
            child_fitness_j,
            rng
        )

        i += 2
    end

    return next_pop, next_fit
end
