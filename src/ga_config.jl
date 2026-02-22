# @kwdef makes it possible to use the field names when using the struct, instead of just having positional args.
Base.@kwdef struct GAConfig
    p_c::Float64 # Probability of crossover
    p_m::Float64 # Probability of mutation

    # Core operators
    selector::ParentSelector
    crossover::Crossover
    mutator::Mutator
    survivor::SurvivorSelector

    # Run configuration
    generator::Union{Nothing, Generator} = nothing
    pop_size::Int = 100
    max_generations::Int = 200

    # Fitness configuration
    fitness_weights::Fitness.FitnessWeights = Fitness.FitnessWeights()
    penalty_schedule::Fitness.PenaltySchedule = Fitness.PenaltySchedule()

    # Runtime behavior
    keep_history::Bool = true
    verbose::Bool = false
    log_every::Int = 25

    # Optional solution output (uses src/algorithms/logging.jl)
    solution_output_file::Union{Nothing, String} = nothing
    instance_json_file::Union{Nothing, String} = nothing
end
