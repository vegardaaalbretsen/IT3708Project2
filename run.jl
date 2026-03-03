#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "HomeCareGA.jl"))
using .HomeCareGA

function parse_args(args::Vector{String})
    opts = Dict{String, String}(
        "instance" => "resources/train_0.json",
        "seed" => "42",
        "population" => "80",
        "generations" => "1000",
        "tournament" => "4",
        "elitism" => "4",
        "crossover" => "0.9",
        "mutation" => "0.35",
        "time-limit" => "60.0",
        "log-every" => "25",
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
    return default_config(
        population_size = parse(Int, opts["population"]),
        generations = parse(Int, opts["generations"]),
        tournament_size = parse(Int, opts["tournament"]),
        elitism = parse(Int, opts["elitism"]),
        crossover_rate = parse(Float64, opts["crossover"]),
        mutation_rate = parse(Float64, opts["mutation"]),
        time_limit_sec = parse(Float64, opts["time-limit"]),
        log_every = parse(Int, opts["log-every"]),
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
