#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)

using HomeCareGA
using StableRNGs
using JSON3
using Printf

# -----------------------------------------------------------------------------
# Grid-search setup
# Run with:
#   julia --project=. run_grid_search.jl
# -----------------------------------------------------------------------------

const INSTANCE_FILE = "train/train_1.json"
const OUTPUT_CSV = "results/grid_search_results.csv"

# Logging via src/algorithms/logging.jl
const LOG_EACH_RUN_SOLUTION = false
const RUN_SOLUTIONS_DIR = "results/grid_run_solutions"
const LOG_BEST_SOLUTION = true
const BEST_SOLUTION_FILE = "results/grid_best_solution.txt"

const RNG_SEED = 42
const MAX_NURSES = nothing
const POP_SIZE = 1000
const MAX_GENERATIONS = 10000
const NUM_ELITES = 10

const PARENT_SELECTION = :tournament   # :tournament or :roulette
const TOURNAMENT_K = 3
const MUTATOR = :swap_any              # :swap or :swap_any
const USE_LOCAL_SEARCH = true

const O1X_MIN_FRAC = 0.07
const O1X_MAX_FRAC = 0.30

# Sweep values (edit these)
const P_M_VALUES = [0.03, 0.07, 0.12]
const P_C_VALUES = [0.80, 0.90, 0.98]
const P_LS_VALUES = [0.05, 0.10, 0.20]

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
    max_scale = 5.0,
    power = 1.0,
    mag_scale = 0.0,
)

function _resolve_path(path::AbstractString)::String
    return isabspath(path) ? String(path) : normpath(joinpath(@__DIR__, path))
end

function _max_nurses_from_instance(instance_path::AbstractString)::Int
    data = JSON3.read(read(instance_path, String))
    haskey(data, :nbr_nurses) || error("Instance JSON is missing 'nbr_nurses': $instance_path")
    n = Int(data["nbr_nurses"])
    n > 0 || error("'nbr_nurses' must be > 0 in: $instance_path")
    return n
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

function _validate_prob_grid(values::Vector{Float64}, name::AbstractString)
    isempty(values) && error("$name cannot be empty.")
    for v in values
        0.0 <= v <= 1.0 || error("$name contains value outside [0, 1]: $v")
    end
    return nothing
end

function _write_results_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "rank,p_c,p_m,p_ls,best_fitness,best_generation,elapsed_seconds")
        for (rank, row) in enumerate(rows)
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%d,%.3f\n",
                rank,
                row.p_c,
                row.p_m,
                row.p_ls,
                row.best_fitness,
                row.best_generation,
                row.elapsed_seconds
            )
        end
    end
end

@inline function _fmt_prob(p::Real)::String
    return replace(@sprintf("%.3f", Float64(p)), "." => "p")
end

function main()
    _validate_prob_grid(Float64.(P_C_VALUES), "P_C_VALUES")
    _validate_prob_grid(Float64.(P_M_VALUES), "P_M_VALUES")
    _validate_prob_grid(Float64.(P_LS_VALUES), "P_LS_VALUES")

    instance_path = _resolve_path(INSTANCE_FILE)
    isfile(instance_path) || error("Instance file does not exist: $instance_path")

    max_nurses = isnothing(MAX_NURSES) ? _max_nurses_from_instance(instance_path) : MAX_NURSES
    max_nurses > 0 || error("MAX_NURSES must be > 0 when set.")

    output_csv_path = _resolve_path(OUTPUT_CSV)
    run_solutions_dir = _resolve_path(RUN_SOLUTIONS_DIR)
    best_solution_path = _resolve_path(BEST_SOLUTION_FILE)

    if LOG_EACH_RUN_SOLUTION
        mkpath(run_solutions_dir)
    end

    instance = load_instance(instance_path)

    combos = [(p_c, p_m, p_ls) for p_c in P_C_VALUES, p_m in P_M_VALUES, p_ls in P_LS_VALUES]
    flat_combos = vec(combos)

    println("Grid search start")
    println("  Instance:         ", instance_path)
    println("  Total runs:       ", length(flat_combos))
    println("  Pop size:         ", POP_SIZE)
    println("  Generations:      ", MAX_GENERATIONS)
    println("  Max nurses:       ", max_nurses)

    rows = NamedTuple[]
    best_solution = nothing
    best_fitness = Inf
    best_pc = NaN
    best_pm = NaN
    best_pls = NaN

    for (i, (p_c, p_m, p_ls)) in enumerate(flat_combos)
        run_output_file = nothing
        if LOG_EACH_RUN_SOLUTION
            run_name = @sprintf(
                "run_%03d_pc%s_pm%s_pls%s.txt",
                i,
                _fmt_prob(p_c),
                _fmt_prob(p_m),
                _fmt_prob(p_ls),
            )
            run_output_file = joinpath(run_solutions_dir, run_name)
        end

        config = GAConfig(
            p_c = p_c,
            p_m = p_m,
            p_ls = p_ls,
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
            keep_history = false,
            verbose = false,
            log_every = 0,
            solution_output_file = run_output_file,
            instance_json_file = instance_path,
        )

        local result
        elapsed_seconds = @elapsed result = GA(instance_path, config; rng=StableRNG(RNG_SEED + i - 1))

        row = (
            p_c = Float64(p_c),
            p_m = Float64(p_m),
            p_ls = Float64(p_ls),
            best_fitness = result.best_fitness,
            best_generation = result.best_generation,
            elapsed_seconds = elapsed_seconds,
        )
        push!(rows, row)

        if result.best_fitness < best_fitness
            best_fitness = result.best_fitness
            best_solution = copy(result.best_individual)
            best_pc = Float64(p_c)
            best_pm = Float64(p_m)
            best_pls = Float64(p_ls)
        end

        println(@sprintf(
            "[%d/%d] p_c=%.3f p_m=%.3f p_ls=%.3f -> best=%.4f (gen %d, %.2fs)",
            i,
            length(flat_combos),
            row.p_c,
            row.p_m,
            row.p_ls,
            row.best_fitness,
            row.best_generation,
            row.elapsed_seconds,
        ))
    end

    sorted_rows = sort(rows; by=r -> r.best_fitness)
    _write_results_csv(output_csv_path, sorted_rows)

    if LOG_BEST_SOLUTION && !isnothing(best_solution)
        mkpath(dirname(best_solution_path))
        HomeCareGA.Algorithms.log_solution(best_solution, instance_path, best_fitness, best_solution_path)
        println(@sprintf(
            "Best-solution log: %s (p_c=%.3f, p_m=%.3f, p_ls=%.3f)",
            best_solution_path,
            best_pc,
            best_pm,
            best_pls,
        ))
    end

    println("\nTop combinations (best fitness first):")
    top_k = min(10, length(sorted_rows))
    for rank in 1:top_k
        r = sorted_rows[rank]
        println(@sprintf(
            "#%d p_c=%.3f p_m=%.3f p_ls=%.3f best=%.4f gen=%d time=%.2fs",
            rank,
            r.p_c,
            r.p_m,
            r.p_ls,
            r.best_fitness,
            r.best_generation,
            r.elapsed_seconds,
        ))
    end

    println("\nSaved CSV: ", output_csv_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
