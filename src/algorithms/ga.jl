# Individual (chromosome): Vector{Int}
#   [4,2,7,-1,1,6,-1,3,5]  -> routes: [4,2,7], [1,6], [3,5]
#
# Population: Vector{Vector{Int}}
#   [
#     [4,2,7,-1,1,6,-1,3,5],
#     [2,5,-1,4,1,3,-1,7,6]
#   ]

include("ga_config.jl")
   
# TODO implement this
function GA(
        fitness_fn::Function,
        population::Vector{Vector{Int}},
        max_generations::Int,
        ga_config::GAConfig
)

    # n = length(population)

    # fitnesses = Vector{Vector{Float}}

    # for g in 1:max_generations
        # fitnesses = fitness_fn(pop, params(?))