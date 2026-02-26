using StatsBase: sample, Weights
using Random: AbstractRNG
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


"""
Tournament selection (minimizing).
"""
function select(
        method::TournamentSelector,
        population::Vector{Vector{Int}},
        fitnesses::Vector{Float64};
        rng::AbstractRNG = Random.GLOBAL_RNG
)::Tuple{Vector{Vector{Int}}, Vector{Float64}}
    
    pop_size = length(population)
    num_winners = pop_size
    @assert length(fitnesses) == pop_size
    
    winners = Vector{Vector{Int}}(undef, num_winners)
    winner_fits = Vector{Float64}(undef, num_winners)
    
    for i in 1:pop_size
        # Randomly pick k indices of individuals
        tournament_indices = sample(rng, 1:pop_size, method.k, replace=false)
        
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
        fitnesses::Vector{Float64};
        rng::AbstractRNG = Random.GLOBAL_RNG
)::Tuple{Vector{Vector{Int}}, Vector{Float64}}
    max_fit = maximum(fitnesses)
    min_fit = minimum(fitnesses)
    pop_size = length(parents)
    @assert length(fitnesses) == pop_size
    
    epsilon = (max_fit - min_fit) * 0.01 + 1e-6
    offset_fitnesses = (max_fit .+ epsilon) .- fitnesses # Inverts fitnesses (e.g. min fit -> max fit and vice versa)

    probabilities = Weights(offset_fitnesses)

    idx_pool = 1:length(parents)
    selected_indices = sample(rng, idx_pool, probabilities, length(parents))
    
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
