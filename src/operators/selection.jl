# TODO

"""
Parent selection:
    Tournament

Survivor selection:
    
"""


abstract type ParentSelector end
abstract type SurvivorSelector end


struct TournamentSelector <: ParentSelector
    k # Tournament size
end

struct RouletteWheelSelector <: ParentSelector end


"""
Tournament selection (minimizing).
"""
function select(
        method::TournamentSelector,
        population::Vector{BitVector},
        fitnesses::Vector{Float64}
)::Tuple{Vector{Vector{Int8}}, Float64}
    
    pop_size = length(parents)
    num_winners = pop_size
    
    winners = Vector{BitVector}(undef, num_winners)
    winner_fits = Vector{Float64}(undef, num_winners)
    
    for i in 1:num_winners
        # Randomly pick k indices of individuals
        tournament_indices = sample(1:pop_size, method.k, replace=false)
        
        # Get index of the tournament winner (min fitness)
        winner_idx = tournament_indices[argmin(fitnesses[tournament_indices])]
        
        winners[i] = parents[winner_idx]
        winner_fits[i] = fitnesses[winner_idx]
    end
    return winners, winner_fits
end


"""
Roulette wheel selection (minimizing).
"""
function select(
        method::RouletteWheelSelector,
        parents::Vector{BitVector},
        fitnesses::Vector{Float64}
)::Tuple{Vector{Vector{Int8}}, Float64}
    max_fit = maximum(fitnesses)
    min_fit = minimum(fitnesses)
    
    epsilon = (max_fit - min_fit) * 0.01 + 1e-6
    offset_fitnesses = (max_fit .+ epsilon) .- fitnesses # Inverts fitnesses (e.g. min fit -> max fit and vice versa)

    probabilities = Weights(offset_fitnesses)

    idx_pool = 1:length(parents)
    selected_indices = sample(idx_pool, probabilities, length(parents))
    
    return parents[selected_indices], fitnesses[selected_indices]
end