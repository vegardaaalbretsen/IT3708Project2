module HomeCareGA
using Random
using Random: AbstractRNG, rand
# Individual (chromosome): Vector{Int}
#   [4,2,7,-1,1,6,-1,3,5]  -> routes: [4,2,7], [1,6], [3,5]
#
# Population: Vector{Vector{Int}}
#   [
#     [4,2,7,-1,1,6,-1,3,5],
#     [2,5,-1,4,1,3,-1,7,6]
#   ]

include("operators/crossover.jl")
include("operators/selection.jl")
include("operators/mutation.jl")
include("utils/chromosome_utils.jl")
include("operators/generators.jl")
include("fitness.jl")
include("operators/local_search.jl")

module Algorithms
include("algorithms/logging.jl")
end

include("ga_config.jl")

export GA, GARunResult, GAConfig
export SwapMutator, SwapAnyMutator, mutate # Add more mutators
export O1XCrossover, crossover # Add more crossovers
export RandomGenerator, SweepTWGenerator, generate, canonicalize_chromosome,
       active_route_count, enforce_min_active_routes # Generators / chromosome utils
export LocalSearch, TwoOptLocalSearch, improve # Local search
export TournamentSelector, RouletteWheelSelector, ElitistSelector,
       GeneralizedCrowdingSelector, select # Add more selectors
using .Fitness

export HCInstance, FitnessWeights, PenaltySchedule, FitnessBreakdown,
       load_instance, fitness, fitness_breakdown, SPLIT

Base.@kwdef struct GARunResult
    best_individual::Vector{Int}
    best_fitness::Float64
    best_generation::Int
    history::Vector{Float64}
    entropy_history::Vector{Float64} = Float64[]
    population::Vector{Vector{Int}}
    fitnesses::Vector{Float64}
end

function _validate_run_config(ga_config::GAConfig, max_generations::Int)
    0.0 <= ga_config.p_c <= 1.0 || throw(ArgumentError("ga_config.p_c must be in [0, 1]."))
    0.0 <= ga_config.p_m <= 1.0 || throw(ArgumentError("ga_config.p_m must be in [0, 1]."))
    0.0 <= ga_config.p_ls <= 1.0 || throw(ArgumentError("ga_config.p_ls must be in [0, 1]."))
    ga_config.pop_size > 0 || throw(ArgumentError("ga_config.pop_size must be > 0."))
    ga_config.min_active_routes > 0 || throw(ArgumentError("ga_config.min_active_routes must be > 0."))
    max_generations > 0 || throw(ArgumentError("max_generations must be > 0."))
    ga_config.log_every >= 0 || throw(ArgumentError("ga_config.log_every must be >= 0."))
    if !isnothing(ga_config.no_improvement_patience)
        ga_config.no_improvement_patience > 0 || throw(ArgumentError(
            "ga_config.no_improvement_patience must be > 0 when set."
        ))
    end
    if ga_config.survivor isa GeneralizedCrowdingSelector
        ga_config.survivor.phi >= 0.0 || throw(ArgumentError("GeneralizedCrowdingSelector.phi must be >= 0.0."))
    end
    return nothing
end

function _initialize_population(
    ga_config::GAConfig,
    initial_population::Union{Nothing, Vector{Vector{Int}}},
    rng::AbstractRNG
)::Vector{Vector{Int}}
    if !isnothing(initial_population)
        isempty(initial_population) && throw(ArgumentError("initial_population cannot be empty."))
        return [enforce_min_active_routes(copy(ind), ga_config.min_active_routes) for ind in initial_population]
    end

    isnothing(ga_config.generator) && throw(ArgumentError(
        "ga_config.generator is required when initial_population is not provided."
    ))
    population = generate(ga_config.generator; pop_size=ga_config.pop_size, rng=rng)
    return [enforce_min_active_routes(ind, ga_config.min_active_routes) for ind in population]
end

function _evaluate_population(
    population::Vector{Vector{Int}},
    instance::HCInstance,
    ga_config::GAConfig,
    generation::Int,
    max_generations::Int
)::Vector{Float64}
    scores = Vector{Float64}(undef, length(population))
    Threads.@threads :dynamic for i in eachindex(population)
        @inbounds scores[i] = fitness(
            population[i],
            instance;
            weights = ga_config.fitness_weights,
            schedule = ga_config.penalty_schedule,
            generation = generation,
            max_generations = max_generations
        )
    end
    return scores
end

function _population_entropy(population::Vector{Vector{Int}})::Float64
    pop_size = length(population)
    pop_size == 0 && return 0.0

    chrom_length = length(population[1])
    chrom_length == 0 && return 0.0

    total_entropy = 0.0

    for pos in 1:chrom_length
        counts = Dict{Int, Int}()
        @inbounds for individual in population
            gene = individual[pos]
            counts[gene] = get(counts, gene, 0) + 1
        end

        k = length(counts)
        if k <= 1
            continue
        end

        h = 0.0
        @inbounds for count in values(counts)
            p = count / pop_size
            h -= p * log(p)
        end

        # Normalize to [0,1] at each position to make values easier to compare.
        total_entropy += h / log(k)
    end

    return total_entropy / chrom_length
end

function _make_children(
    parents::Vector{Vector{Int}},
    ga_config::GAConfig,
    rng::AbstractRNG
)::Vector{Vector{Int}}
    pop_size = length(parents)
    children = Vector{Vector{Int}}(undef, pop_size)

    i = 1
    while i <= pop_size
        p1 = parents[i]
        p2 = parents[i == pop_size ? 1 : i + 1]

        child1 = copy(p1)
        child2 = copy(p2)

        if rand(rng) < ga_config.p_c
            child1, child2 = crossover(ga_config.crossover, p1, p2, rng)
        end

        if rand(rng) < ga_config.p_m
            child1 = mutate(mutator=ga_config.mutator, individual=child1, rng=rng)
        end
        if rand(rng) < ga_config.p_m
            child2 = mutate(mutator=ga_config.mutator, individual=child2, rng=rng)
        end

        child1 = enforce_min_active_routes(child1, ga_config.min_active_routes)
        child2 = enforce_min_active_routes(child2, ga_config.min_active_routes)

        children[i] = child1
        if i + 1 <= pop_size
            children[i + 1] = child2
        end

        i += 2
    end

    return children
end

function _maybe_log_solution(
    best_individual::Vector{Int},
    best_fitness::Float64,
    ga_config::GAConfig,
    instance_json_file::Union{Nothing, AbstractString}
)
    if isnothing(ga_config.solution_output_file)
        return nothing
    end

    if isnothing(instance_json_file)
        @warn "Skipping solution logging because no instance_json_file was provided."
        return nothing
    end

    Algorithms.log_solution(
        best_individual,
        String(instance_json_file),
        best_fitness,
        ga_config.solution_output_file
    )
    return nothing
end

function run(
    instance::HCInstance,
    ga_config::GAConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    initial_population::Union{Nothing, Vector{Vector{Int}}} = nothing,
    max_generations::Int = ga_config.max_generations,
    instance_json_file::Union{Nothing, AbstractString} = ga_config.instance_json_file
)
    _validate_run_config(ga_config, max_generations)
    population = _initialize_population(ga_config, initial_population, rng)

    history = ga_config.keep_history ? Vector{Float64}(undef, max_generations) : Float64[]
    entropy_history = ga_config.keep_history ? Vector{Float64}(undef, max_generations) : Float64[]
    best_individual = copy(population[1])
    best_fitness = Inf
    best_generation = 1
    generations_without_improvement = 0
    generations_ran = 0

    population_fitness = Float64[]

    for generation in 1:max_generations
        generations_ran = generation
        population_fitness = _evaluate_population(
            population,
            instance,
            ga_config,
            generation,
            max_generations
        )

        generation_best_idx = argmin(population_fitness)
        generation_best_fit = population_fitness[generation_best_idx]

        if generation == 1 || generation_best_fit < best_fitness
            best_fitness = generation_best_fit
            best_individual = copy(population[generation_best_idx])
            best_generation = generation
            generations_without_improvement = 0
        else
            generations_without_improvement += 1
        end

        if ga_config.keep_history
            history[generation] = generation_best_fit
            entropy_history[generation] = _population_entropy(population)
        end

        if ga_config.verbose &&
           (generation == 1 ||
            generation == max_generations ||
            (ga_config.log_every > 0 && generation % ga_config.log_every == 0))
            @info "GA generation" generation=generation generation_best=generation_best_fit best_overall=best_fitness
        end

        if !isnothing(ga_config.no_improvement_patience) &&
           generations_without_improvement >= ga_config.no_improvement_patience
            if ga_config.verbose
                @info "Early stopping (no improvement)" generation=generation patience=ga_config.no_improvement_patience
            end
            break
        end

        parents, parent_fitness = select(ga_config.selector, population, population_fitness)
        children = _make_children(parents, ga_config, rng)

        if !isnothing(ga_config.local_search) && ga_config.p_ls > 0.0
            @inbounds for i in eachindex(children)
                if rand(rng) < ga_config.p_ls
                    children[i] = improve(
                        ga_config.local_search,
                        children[i],
                        instance,
                        ga_config,
                        generation,
                        max_generations
                    )
                end
            end
        end

        child_fitness = _evaluate_population(
            children,
            instance,
            ga_config,
            generation,
            max_generations
        )

        if ga_config.survivor isa GeneralizedCrowdingSelector
            population, population_fitness = select(
                ga_config.survivor,
                parents,
                parent_fitness,
                children,
                child_fitness,
                rng
            )
        else
            population, population_fitness = select(
                ga_config.survivor,
                population,
                population_fitness,
                children,
                child_fitness,
                rng
            )
        end
    end

    if ga_config.keep_history
        resize!(history, generations_ran)
        resize!(entropy_history, generations_ran)
    end

    _maybe_log_solution(best_individual, best_fitness, ga_config, instance_json_file)

    return GARunResult(
        best_individual=best_individual,
        best_fitness=best_fitness,
        best_generation=best_generation,
        history=history,
        entropy_history=entropy_history,
        population=population,
        fitnesses=population_fitness
    )
end

function run(
    instance_json_file::AbstractString,
    ga_config::GAConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    initial_population::Union{Nothing, Vector{Vector{Int}}} = nothing,
    max_generations::Int = ga_config.max_generations
)
    instance = load_instance(instance_json_file)
    return run(
        instance,
        ga_config;
        rng=rng,
        initial_population=initial_population,
        max_generations=max_generations,
        instance_json_file=instance_json_file
    )
end

function GA(
    population::Vector{Vector{Int}},
    max_generations::Int,
    ga_config::GAConfig,
    instance::HCInstance;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    instance_json_file::Union{Nothing, AbstractString} = ga_config.instance_json_file
)
    return run(
        instance,
        ga_config;
        rng=rng,
        initial_population=population,
        max_generations=max_generations,
        instance_json_file=instance_json_file
    )
end

function GA(
    population::Vector{Vector{Int}},
    max_generations::Int,
    ga_config::GAConfig,
    instance_json_file::AbstractString;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    return run(
        instance_json_file,
        ga_config;
        rng=rng,
        initial_population=population,
        max_generations=max_generations
    )
end

GA(instance::HCInstance, ga_config::GAConfig; kwargs...) = run(instance, ga_config; kwargs...)
GA(instance_json_file::AbstractString, ga_config::GAConfig; kwargs...) = run(instance_json_file, ga_config; kwargs...)

end
