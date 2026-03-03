function plot_solution(inst::Instance, best::Candidate, output_path::String)
    p = plot(
        legend = false,
        axis = nothing,
        grid = false,
        framestyle = :none,
        size = (1920, 1080),
        dpi = 220,
        aspect_ratio = :equal,
    )

    n_routes = max(length(best.routes), 1)
    colors = palette(:tab20, n_routes)

    for (i, route) in enumerate(best.routes)
        rx = Float64[inst.depot_x]
        ry = Float64[inst.depot_y]
        for pid in route
            push!(rx, inst.patients[pid].x)
            push!(ry, inst.patients[pid].y)
        end
        push!(rx, inst.depot_x)
        push!(ry, inst.depot_y)
        plot!(p, rx, ry; color = colors[i], linewidth = 2.5, alpha = 0.95, label = "")
    end

    xs = [pt.x for pt in inst.patients]
    ys = [pt.y for pt in inst.patients]
    scatter!(
        p,
        xs,
        ys;
        color = :black,
        marker = :rect,
        ms = 4.2,
        markeralpha = 1.0,
        markerstrokewidth = 0.2,
        markerstrokecolor = :white,
        label = "",
    )
    scatter!(
        p,
        [inst.depot_x],
        [inst.depot_y];
        color = :black,
        marker = :circle,
        ms = 14,
        markeralpha = 1.0,
        markerstrokewidth = 0.4,
        markerstrokecolor = :white,
        label = "",
    )

    savefig(p, output_path)
end

function plot_fitness_spread(metrics_history, output_path::String; instance_name::String = "")
    gens = Int[m.generation for m in metrics_history]
    best_vals = Float64[m.best_total_travel for m in metrics_history]
    median_vals = Float64[m.median_total_travel for m in metrics_history]
    worst_vals = Float64[m.worst_total_travel for m in metrics_history]

    title_txt = isempty(instance_name) ? "Fitness Spread per Generation" : "Fitness Spread: $(instance_name)"

    p = plot(
        gens,
        best_vals;
        label = "Best",
        color = :forestgreen,
        linewidth = 3.0,
        xlabel = "Generation",
        ylabel = "Total travel time",
        title = title_txt,
        size = (1920, 1080),
        dpi = 240,
        legend = :topright,
        background_color = :white,
        foreground_color_grid = RGBA(0.7, 0.7, 0.7, 0.25),
        grid = true,
        framestyle = :box,
    )

    plot!(
        p,
        gens,
        median_vals;
        label = "Median",
        color = :dodgerblue3,
        linewidth = 2.5,
        linestyle = :dash,
    )

    plot!(
        p,
        gens,
        worst_vals;
        label = "Worst",
        color = :firebrick3,
        linewidth = 2.5,
        linestyle = :dot,
    )

    savefig(p, output_path)
end
