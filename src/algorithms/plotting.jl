using Plots
using Colors
using JSON3
using HomeCareGA.Fitness: SPLIT

@inline function _quantile_sorted(sorted_vals::Vector{Float64}, q::Float64)::Float64
    n = length(sorted_vals)
    n > 0 || error("Cannot compute quantile of empty vector.")
    qc = clamp(q, 0.0, 1.0)
    idx = 1 + floor(Int, (n - 1) * qc)
    return sorted_vals[idx]
end

function _bounded_ylim(
    values::Vector{Float64};
    lower_q::Float64=0.0,
    upper_q::Float64=0.98,
    pad_frac::Float64=0.04
)
    finite_vals = [v for v in values if isfinite(v)]
    isempty(finite_vals) && error("All values are non-finite.")

    sort!(finite_vals)
    y_lo = _quantile_sorted(finite_vals, lower_q)
    y_hi = _quantile_sorted(finite_vals, upper_q)

    if y_hi <= y_lo
        y_hi = finite_vals[end]
        if y_hi <= y_lo
            y_hi = y_lo + 1.0
        end
    end

    span = y_hi - y_lo
    pad = max(1e-9, span * max(0.0, pad_frac))
    return (y_lo - pad, y_hi + pad)
end

"""
    plot_routes_stream(solution::Vector{Int}, json_file::String)

Plots the solution routes on a map.
Automatically extracts patient coordinates and depot from JSON file.

Example:
    plot_routes_stream(chromosome, "train/train_0.json")
"""
function plot_routes_stream(
    solution::Vector{Int},
    json_file::String;
    show_plot::Bool=true
)
    # Load and parse JSON instance
    instance = JSON3.read(read(json_file, String))
    
    # Extract patient coordinates
    patients = Dict{Int, Tuple{Int, Int}}()
    for (patient_id_str, patient_data) in instance["patients"]
        patient_id = parse(Int, String(patient_id_str))
        patients[patient_id] = (patient_data["x_coord"], patient_data["y_coord"])
    end
    
    # Extract depot coordinates
    depot = (instance["depot"]["x_coord"], instance["depot"]["y_coord"])

    n_routes = count(==(SPLIT), solution) + 1
    # Generate vibrant colors avoiding white/black—use HSV with fixed saturation & value
    colors = [convert(RGB, HSV((i-1) * 360 / n_routes, 0.8, 0.9)) for i in 1:n_routes]

    plt = plot(
        legend = false,
        axis = nothing,
        grid = false,
        framestyle = :none,
        size = (1280, 960),
        dpi = 300
    )

    xs = Int[depot[1]]
    ys = Int[depot[2]]
    nurse = 1
    visited_ids = Int[]
    invalid_ids = Int[]

    for p in solution
        if p == SPLIT
            push!(xs, depot[1]); push!(ys, depot[2])
            plot!(plt, xs, ys; lw=1.5, color = colors[nurse])

            nurse += 1
            xs = Int[depot[1]]
            ys = Int[depot[2]]
        else
            if haskey(patients, p)
                (x, y) = patients[p]
                push!(xs, x); push!(ys, y)
                push!(visited_ids, p)
            else
                push!(invalid_ids, p)
            end
        end
    end

    if length(xs) > 1
        push!(xs, depot[1]); push!(ys, depot[2])
        plot!(plt, xs, ys; lw=1.5, color = colors[nurse])
    end

    # --- depot
    scatter!(
        plt,
        [depot[1]], [depot[2]];
        marker = :circle,
        markersize = 12,
        markercolor = :black
    )

    # --- patient nodes
    patient_ids = sort(collect(keys(patients)))
    all_x = [patients[id][1] for id in patient_ids]
    all_y = [patients[id][2] for id in patient_ids]

    scatter!(
        plt,
        all_x, all_y;
        marker = :rect,
        markersize = 4,
        markercolor = :black
    )

    if show_plot
        display(plt)
    end
    return plt
end

"""
    plot_fitness_entropy(fitness_history::Vector{Float64}, entropy_history::Vector{Float64};
                         output_file::Union{Nothing,AbstractString}=nothing,
                         show_plot::Bool=true)

Plot best-fitness history and population-entropy history over generations.
If `output_file` is provided, the figure is saved to that path.
"""
function plot_fitness_entropy(
    fitness_history::Vector{Float64},
    entropy_history::Vector{Float64};
    output_file::Union{Nothing,AbstractString}=nothing,
    show_plot::Bool=true,
    fitness_ylims::Union{Nothing,Tuple{Float64,Float64}}=nothing,
    fitness_ymax::Union{Nothing,Float64}=nothing,
    bounded_fitness_axis::Bool=true,
    fitness_lower_quantile::Float64=0.0,
    fitness_upper_quantile::Float64=0.98,
    fitness_axis_pad_frac::Float64=0.04
)
    n_fit = length(fitness_history)
    n_ent = length(entropy_history)
    n_fit > 0 || throw(ArgumentError("fitness_history is empty. Enable keep_history in GAConfig."))
    n_ent > 0 || throw(ArgumentError("entropy_history is empty. Enable keep_history in GAConfig."))
    n_fit == n_ent || throw(ArgumentError("fitness_history and entropy_history must have equal length."))

    gens = 1:n_fit
    finite_fit = [v for v in fitness_history if isfinite(v)]
    isempty(finite_fit) && throw(ArgumentError("fitness_history has no finite values."))

    applied_fitness_ylims = if !isnothing(fitness_ylims)
        y0, y1 = fitness_ylims
        y0 < y1 || throw(ArgumentError("fitness_ylims must satisfy ymin < ymax."))
        (y0, y1)
    elseif !isnothing(fitness_ymax)
        ymax_cap = fitness_ymax
        isfinite(ymax_cap) || throw(ArgumentError("fitness_ymax must be finite."))
        data_max = maximum(finite_fit)
        ymax = min(ymax_cap, data_max)
        ymin_data = minimum(finite_fit)
        if ymax <= ymin_data
            pad = max(1e-9, abs(ymax) * max(fitness_axis_pad_frac, 0.04), 1.0)
            (ymax - pad, ymax)
        else
            pad = max(1e-9, (ymax - ymin_data) * fitness_axis_pad_frac)
            (ymin_data - pad, ymax)
        end
    elseif bounded_fitness_axis
        _bounded_ylim(
            fitness_history;
            lower_q=fitness_lower_quantile,
            upper_q=fitness_upper_quantile,
            pad_frac=fitness_axis_pad_frac
        )
    else
        :auto
    end

    p1 = plot(
        gens,
        fitness_history;
        xlabel="Generation",
        ylabel="Best fitness",
        title="Best Fitness per Generation",
        label="Best fitness",
        lw=2,
        ylims=applied_fitness_ylims
    )

    if applied_fitness_ylims != :auto
        ymin, ymax = applied_fitness_ylims
        clipped_idx = findall(v -> v > ymax, fitness_history)
        if !isempty(clipped_idx)
            scatter!(
                p1,
                clipped_idx,
                fill(ymax, length(clipped_idx));
                markershape=:utriangle,
                markersize=4,
                markercolor=:red,
                label="Clipped high ($(length(clipped_idx)))"
            )
        end
    end

    p2 = plot(
        gens,
        entropy_history;
        xlabel="Generation",
        ylabel="Entropy (0-1)",
        title="Population Entropy per Generation",
        label="Entropy",
        lw=2,
        color=:darkgreen
    )

    plt = plot(p1, p2; layout=(2,1), size=(1200, 800), dpi=200)

    if !isnothing(output_file)
        savefig(plt, String(output_file))
    end

    if show_plot
        display(plt)
    end
    return plt
end
