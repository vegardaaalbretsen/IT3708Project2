function write_text_file(path::String, content::String)
    open(path, "w") do io
        write(io, content)
    end
end

function write_metrics_csv(path::String, metrics_history)
    if isempty(metrics_history)
        return
    end

    open(path, "w") do io
        write(io, "generation,best_total_travel,median_total_travel,worst_total_travel,feasible_ratio\n")
        for m in metrics_history
            @printf(
                io,
                "%d,%.6f,%.6f,%.6f,%.6f\n",
                m.generation,
                m.best_total_travel,
                m.median_total_travel,
                m.worst_total_travel,
                m.feasible_ratio,
            )
        end
    end
end

function solve_instance(
    instance_path::String;
    seed::Int = 42,
    config::GAConfig = default_config(),
    output_dir::String = "results",
)
    rng = MersenneTwister(seed)
    inst = load_instance(instance_path)
    mkpath(output_dir)
    instance_output_dir = joinpath(output_dir, inst.name)
    mkpath(instance_output_dir)

    println("Running GA on $(inst.name) with seed $seed")
    best, metrics_history = run_ga(inst, config, rng)
    if !best.feasible
        @warn "Best solution is infeasible. Fitness=$(best.fitness)"
    end

    report = solution_report(inst, best)
    txt_path = joinpath(instance_output_dir, "best_solution.txt")
    write_text_file(txt_path, report)

    cnfg_report = config_report(config)
    cnfg_path = joinpath(instance_output_dir, "best_config.txt")
    write_text_file(cnfg_path, cnfg_report)

    png_path = joinpath(instance_output_dir, "best_solution.png")
    plot_solution(inst, best, png_path)

    metrics_path = joinpath(instance_output_dir, "metrics.csv")
    write_metrics_csv(metrics_path, metrics_history)
    spread_plot_path = joinpath(instance_output_dir, "fitness_spread.png")
    plot_fitness_spread(metrics_history, spread_plot_path; instance_name = inst.name)

    print(terminal_summary(inst, best))

    return best
end
