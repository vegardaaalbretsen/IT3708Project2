"""
    default_config(...)

Default hyperparameters for the genetic algorithm.
These defaults are used both programmatically and as command-line defaults in run.jl.
This is the single source of truth for GA configuration defaults.
"""
function default_config(;
    population_size::Int = 80,
    generations::Int = 2_000,
    tournament_size::Int = 3,
    elitism::Int = 3,
    crossover_rate::Float64 = 0.6,
    mutation_rate::Float64 = 0.7,
    local_search_rate::Float64 = 0.7,
    time_limit_sec::Float64 = 120.0,
    log_every::Int = 25,
)
    return GAConfig(
        population_size,
        generations,
        tournament_size,
        elitism,
        crossover_rate,
        mutation_rate,
        local_search_rate,
        time_limit_sec,
        log_every,
    )
end
