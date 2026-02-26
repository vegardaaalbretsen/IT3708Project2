#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)

using HomeCareGA
using StableRNGs
using JSON3
include(joinpath(@__DIR__, "src", "algorithms", "plotting.jl"))

# -----------------------------------------------------------------------------
# Edit these variables for your experiment setup.
# Then run:
#   julia --project=. run_ga.jl
# -----------------------------------------------------------------------------

const INSTANCE_FILE = "train/train_1.json"
const OUTPUT_FILE = "results/logs/best_solution.txt"   # Set to `nothing` to disable file output
const ROUTES_PLOT_FILE = "results/plots/routes/best_routes.png"  # Set to `nothing` to disable saving
const FITNESS_ENTROPY_PLOT_FILE = "results/plots/entropy/fitness_entropy.png" # Set to `nothing` to disable saving
const SHOW_PLOTS = false                             # Set true for interactive display

const RNG_SEED = 42
const MAX_NURSES = 5                               # `nothing` => use `nbr_nurses` from instance JSON (upper bound)
const POP_SIZE = 250
const MAX_GENERATIONS = 1000

const P_C = 0.85
const P_M = 0.10
const P_LS = 0.15
const PARENT_SELECTION = :tournament              # :tournament or :roulette
const TOURNAMENT_K = 3                            # Used only if PARENT_SELECTION = :tournament
const NUM_ELITES = 10
const MUTATOR = :swap_any                         # :swap_any lets GA move -1 separators and change how many nurses are active
const GENERATOR = :sweep_tw                       # :random or :sweep_tw
const USE_LOCAL_SEARCH = true

const O1X_MIN_FRAC = 0.07
const O1X_MAX_FRAC = 0.15

const VERBOSE = true
const LOG_EVERY = 250
const KEEP_HISTORY = true

# Fitness and penalty settings
const WEIGHTS = FitnessWeights(
    w_travel = 1.0,
    w_wait = 0.0,
    w_capacity = 1000.0,
    w_return = 1000.0,
    w_late = 2000.0,
    w_early = 0.0,
)

const PENALTY_SCHEDULE = PenaltySchedule(
    min_scale = 1.0,
    max_scale = 10.0,
    power = 1.0,
    mag_scale = 0.0,
)

function _resolve_path(path::AbstractString)::String
    return isabspath(path) ? String(path) : normpath(joinpath(@__DIR__, path))
end

function _resolve_optional_path(path::Union{Nothing, AbstractString})::Union{Nothing, String}
    return isnothing(path) ? nothing : _resolve_path(path)
end

function _instance_name_prefix(instance_path::AbstractString)::String
    # Example: ".../train_1.json" -> "train_1_"
    return splitext(basename(instance_path))[1] * "_"
end

function _with_prefix(path::Union{Nothing, AbstractString}, prefix::AbstractString)::Union{Nothing, String}
    if isnothing(path)
        return nothing
    end
    full = _resolve_path(path)
    return joinpath(dirname(full), prefix * basename(full))
end

function _build_selector()
    if PARENT_SELECTION == :tournament
        TOURNAMENT_K > 0 || error("TOURNAMENT_K must be > 0.")
        return TournamentSelector(TOURNAMENT_K)
    elseif PARENT_SELECTION == :roulette
        return RouletteWheelSelector()
    else
        error("PARENT_SELECTION must be :tournament or :roulette.")
    end
end

function _build_mutator()
    if MUTATOR == :swap
        return SwapMutator()
    elseif MUTATOR == :swap_any
        return SwapAnyMutator()
    else
        error("MUTATOR must be :swap or :swap_any.")
    end
end

function _build_generator(instance::HCInstance, instance_path::AbstractString, max_nurses::Int)
    if GENERATOR == :random
        return RandomGenerator(num_jobs=instance.N, num_routes=max_nurses)
    elseif GENERATOR == :sweep_tw
        return SweepTWGenerator(instance_path; num_routes=max_nurses)
    else
        error("GENERATOR must be :random or :sweep_tw.")
    end
end

function _max_nurses_from_instance(instance_path::AbstractString)::Int
    data = JSON3.read(read(instance_path, String))
    haskey(data, :nbr_nurses) || error("Instance JSON is missing 'nbr_nurses': $instance_path")
    n = Int(data["nbr_nurses"])
    n > 0 || error("'nbr_nurses' must be > 0 in: $instance_path")
    return n
end

function _validate()
    if !isnothing(MAX_NURSES)
        MAX_NURSES > 0 || error("MAX_NURSES must be > 0 when set.")
    end
    POP_SIZE > 0 || error("POP_SIZE must be > 0.")
    MAX_GENERATIONS > 0 || error("MAX_GENERATIONS must be > 0.")
    NUM_ELITES >= 0 || error("NUM_ELITES must be >= 0.")
    0.0 <= P_C <= 1.0 || error("P_C must be in [0, 1].")
    0.0 <= P_M <= 1.0 || error("P_M must be in [0, 1].")
    0.0 <= P_LS <= 1.0 || error("P_LS must be in [0, 1].")
    0.0 < O1X_MIN_FRAC <= O1X_MAX_FRAC <= 1.0 || error("Require 0 < O1X_MIN_FRAC <= O1X_MAX_FRAC <= 1.")
    (GENERATOR == :random || GENERATOR == :sweep_tw) || error("GENERATOR must be :random or :sweep_tw.")
    LOG_EVERY >= 0 || error("LOG_EVERY must be >= 0.")
end

function main()
    _validate()

    instance_path = _resolve_path(INSTANCE_FILE)
    isfile(instance_path) || error("Instance file does not exist: $instance_path")
    max_nurses = isnothing(MAX_NURSES) ? _max_nurses_from_instance(instance_path) : MAX_NURSES
    file_prefix = _instance_name_prefix(instance_path)

    output_path = _with_prefix(OUTPUT_FILE, file_prefix)
    routes_plot_path = _with_prefix(ROUTES_PLOT_FILE, file_prefix)
    fitness_entropy_plot_path = _with_prefix(FITNESS_ENTROPY_PLOT_FILE, file_prefix)

    if !isnothing(output_path)
        mkpath(dirname(output_path))
    end
    if !isnothing(routes_plot_path)
        mkpath(dirname(routes_plot_path))
    end
    if !isnothing(fitness_entropy_plot_path)
        mkpath(dirname(fitness_entropy_plot_path))
    end

    instance = load_instance(instance_path)
    config = GAConfig(
        p_c = P_C,
        p_m = P_M,
        p_ls = P_LS,
        selector = _build_selector(),
        crossover = O1XCrossover(min_frac=O1X_MIN_FRAC, max_frac=O1X_MAX_FRAC),
        mutator = _build_mutator(),
        local_search = USE_LOCAL_SEARCH ? TwoOptLocalSearch() : nothing,
        survivor = ElitistSelector(num_elites=NUM_ELITES),
        generator = _build_generator(instance, instance_path, max_nurses),
        pop_size = POP_SIZE,
        max_generations = MAX_GENERATIONS,
        fitness_weights = WEIGHTS,
        penalty_schedule = PENALTY_SCHEDULE,
        keep_history = KEEP_HISTORY,
        verbose = VERBOSE,
        log_every = LOG_EVERY,
        solution_output_file = output_path,
        instance_json_file = instance_path,
    )

    result = GA(instance_path, config; rng=StableRNG(RNG_SEED))

    if !isnothing(routes_plot_path)
        route_plt = plot_routes_stream(result.best_individual, instance_path; show_plot=SHOW_PLOTS)
        savefig(route_plt, routes_plot_path)
    elseif SHOW_PLOTS
        plot_routes_stream(result.best_individual, instance_path; show_plot=true)
    end

    if KEEP_HISTORY && !isempty(result.history) && !isempty(result.entropy_history)
        if !isnothing(fitness_entropy_plot_path)
            plot_fitness_entropy(
                result.history,
                result.entropy_history;
                output_file=fitness_entropy_plot_path,
                show_plot=SHOW_PLOTS
            )
        elseif SHOW_PLOTS
            plot_fitness_entropy(result.history, result.entropy_history; show_plot=true)
        end
    end

    println("Run complete")
    println("  Instance:        ", instance_path)
    println("  Best fitness:    ", result.best_fitness)
    println("  Best generation: ", result.best_generation, "/", MAX_GENERATIONS)
    println("  Best individual: ", result.best_individual)
    println("  Max nurses:      ", max_nurses)
    println("  Mutator:         ", MUTATOR)
    println("  Generator:       ", GENERATOR)
    println("  Local search:    ", USE_LOCAL_SEARCH ? "2-opt (p_ls=$(P_LS))" : "disabled")
    if !isnothing(output_path)
        println("  Solution file:   ", output_path)
    end
    if !isnothing(routes_plot_path)
        println("  Routes plot:     ", routes_plot_path)
    end
    if !isnothing(fitness_entropy_plot_path) && KEEP_HISTORY
        println("  Fit/Entropy plot:", fitness_entropy_plot_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
