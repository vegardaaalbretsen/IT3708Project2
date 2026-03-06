#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "HomeCareGA.jl"))
using .HomeCareGA

function parse_args(args::Vector{String})
    # Defaults are sourced from config.jl::default_config()
    # Only GA config parameters are taken from defaults; others are required or have defaults here
    opts = Dict{String, String}(
        "instance" => "resources/train_0.json",
        "seed" => "42",
        "output-dir" => "results",
        "run-all" => "false",
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            println(
                """
                Usage:
                  julia --project=. run.jl [options]

                Options:
                  --instance <path>       JSON instance path (default: resources/train_0.json)
                  --run-all               Solve all resources/train_*.json
                  --seed <int>            Random seed
                  --population <int>      Population size
                  --generations <int>     Max generations
                  --tournament <int>      Tournament size
                  --elitism <int>         Number of elite survivors
                  --crossover <float>     Crossover probability
                  --mutation <float>      Mutation probability
                  --local-search <float>  Local search probability
                  --time-limit <float>    Time limit in seconds per instance
                  --log-every <int>       Logging interval in generations
                  --output-dir <path>     Output directory
                  --help, -h              Show this message
                """,
            )
            exit(0)
        elseif a == "--run-all"
            opts["run-all"] = "true"
            i += 1
            continue
        elseif startswith(a, "--")
            key = replace(a, "--" => "")
            if i == length(args)
                error("Missing value for $a")
            end
            opts[key] = args[i + 1]
            i += 2
            continue
        else
            error("Unknown argument: $a")
        end
    end
    return opts
end

function build_config(opts::Dict{String, String})
    # Start with all defaults from config.jl
    cfg = default_config()
    
    # Override the parameters that are specified in opts, otherwise keep defaults
    population_size = haskey(opts, "population") ? parse(Int, opts["population"]) : cfg.population_size
    generations = haskey(opts, "generations") ? parse(Int, opts["generations"]) : cfg.generations
    tournament_size = haskey(opts, "tournament") ? parse(Int, opts["tournament"]) : cfg.tournament_size
    elitism = haskey(opts, "elitism") ? parse(Int, opts["elitism"]) : cfg.elitism
    crossover_rate = haskey(opts, "crossover") ? parse(Float64, opts["crossover"]) : cfg.crossover_rate
    mutation_rate = haskey(opts, "mutation") ? parse(Float64, opts["mutation"]) : cfg.mutation_rate
    local_search_rate = haskey(opts, "local-search") ? parse(Float64, opts["local-search"]) : cfg.local_search_rate
    time_limit_sec = haskey(opts, "time-limit") ? parse(Float64, opts["time-limit"]) : cfg.time_limit_sec
    log_every = haskey(opts, "log-every") ? parse(Int, opts["log-every"]) : cfg.log_every
    
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

function main()
    opts = parse_args(ARGS)
    cfg = build_config(opts)
    seed = parse(Int, opts["seed"])
    outdir = opts["output-dir"]

    if lowercase(opts["run-all"]) == "true"
        files = sort(
            filter(
                p -> endswith(p, ".json"),
                readdir(joinpath(@__DIR__, "resources"); join = true),
            ),
        )
        for p in files
            solve_instance(p; seed = seed, config = cfg, output_dir = outdir)
        end
    else
        solve_instance(opts["instance"]; seed = seed, config = cfg, output_dir = outdir)
    end
end

main()
