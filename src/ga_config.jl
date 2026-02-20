# @kwdef makes it possible to use the field names when using the struct, instead of just having positional args.
Base.@kwdef struct GAConfig
    p_c::Float64 # Probability of crossover
    p_m::Float64 # Probability of mutation

    #generator::PopulationGenerator
    selector::ParentSelector
    crossover::Crossover
    mutator::Mutator
    survivor::SurvivorSelector
end