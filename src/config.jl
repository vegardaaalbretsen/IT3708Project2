function default_config(;
    population_size::Int = 80,
    generations::Int = 1_000,
    tournament_size::Int = 4,
    elitism::Int = 4,
    crossover_rate::Float64 = 0.9,
    mutation_rate::Float64 = 0.35,
    time_limit_sec::Float64 = 60.0,
    log_every::Int = 25,
)
    return GAConfig(
        population_size,
        generations,
        tournament_size,
        elitism,
        crossover_rate,
        mutation_rate,
        time_limit_sec,
        log_every,
    )
end
