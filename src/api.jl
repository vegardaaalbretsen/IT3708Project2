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
    open(txt_path, "w") do io
        write(io, report)
    end

    cnfg_report = config_report(config)
    cnfg_path = joinpath(instance_output_dir, "best_config.txt")
    open(cnfg_path, "w") do io
        write(io, cnfg_report)
    end

    png_path = joinpath(instance_output_dir, "best_solution.png")
    plot_solution(inst, best, png_path)

    metrics_path = joinpath(instance_output_dir, "metrics.csv")
    if !isempty(metrics_history)
        fields = propertynames(metrics_history[1])
        open(metrics_path, "w") do io
            write(io, join(string.(fields), ",") * "\n")
            for m in metrics_history
                values = String[]
                for field in fields
                    v = getproperty(m, field)
                    if v isa Integer
                        push!(values, string(v))
                    elseif v isa AbstractFloat
                        if isnan(v)
                            push!(values, "NaN")
                        else
                            push!(values, @sprintf("%.6f", v))
                        end
                    else
                        push!(values, string(v))
                    end
                end
                write(io, join(values, ",") * "\n")
            end
        end
    end
    spread_plot_path = joinpath(instance_output_dir, "fitness_spread.png")
    plot_fitness_spread(metrics_history, spread_plot_path; instance_name = inst.name)

    print(terminal_summary(inst, best))

    return best
end
