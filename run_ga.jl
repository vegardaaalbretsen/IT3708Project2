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
const OUTPUT_FILE = "results/best_solution2.txt"   # Set to `nothing` to disable file output

const RNG_SEED = 42
const MAX_NURSES = nothing                         # `nothing` => use `nbr_nurses` from instance JSON (upper bound)
const POP_SIZE = 1000
const MAX_GENERATIONS = 10000

const P_C = 0.90
const P_M = 0.06
const P_LS = 0.2
const PARENT_SELECTION = :tournament              # :tournament or :roulette
const TOURNAMENT_K = 3                            # Used only if PARENT_SELECTION = :tournament
const NUM_ELITES = 10
const MUTATOR = :swap_any                         # :swap_any lets GA move -1 separators and change how many nurses are active
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
    LOG_EVERY >= 0 || error("LOG_EVERY must be >= 0.")
end

function main()
    _validate()

    instance_path = _resolve_path(INSTANCE_FILE)
    isfile(instance_path) || error("Instance file does not exist: $instance_path")
    max_nurses = isnothing(MAX_NURSES) ? _max_nurses_from_instance(instance_path) : MAX_NURSES

    output_path = isnothing(OUTPUT_FILE) ? nothing : _resolve_path(OUTPUT_FILE)
    if !isnothing(output_path)
        mkpath(dirname(output_path))
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
        generator = RandomGenerator(num_jobs=instance.N, num_routes=max_nurses),
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

    plt = plot_routes_stream(result.best_individual, instance_path)
    savefig(plt, joinpath(@__DIR__, "results", "best_routes.png"))

    println("Run complete")
    println("  Instance:        ", instance_path)
    println("  Best fitness:    ", result.best_fitness)
    println("  Best generation: ", result.best_generation, "/", MAX_GENERATIONS)
    println("  Best individual: ", result.best_individual)
    println("  Max nurses:      ", max_nurses)
    println("  Mutator:         ", MUTATOR)
    println("  Local search:    ", USE_LOCAL_SEARCH ? "2-opt (p_ls=$(P_LS))" : "disabled")
    if !isnothing(output_path)
        println("  Solution file:   ", output_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
