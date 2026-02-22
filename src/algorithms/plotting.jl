using Plots
using Colors
using JSON3

const SPLIT = -1

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

    display(plt)
    return plt
end

"""
    plot_fitness_entropy(fitness_history::Vector{Float64}, entropy_history::Vector{Float64};
                         output_file::Union{Nothing,AbstractString}=nothing)

Plot best-fitness history and population-entropy history over generations.
If `output_file` is provided, the figure is saved to that path.
"""
function plot_fitness_entropy(
    fitness_history::Vector{Float64},
    entropy_history::Vector{Float64};
    output_file::Union{Nothing,AbstractString}=nothing
)
    n_fit = length(fitness_history)
    n_ent = length(entropy_history)
    n_fit > 0 || throw(ArgumentError("fitness_history is empty. Enable keep_history in GAConfig."))
    n_ent > 0 || throw(ArgumentError("entropy_history is empty. Enable keep_history in GAConfig."))
    n_fit == n_ent || throw(ArgumentError("fitness_history and entropy_history must have equal length."))

    gens = 1:n_fit

    p1 = plot(
        gens,
        fitness_history;
        xlabel="Generation",
        ylabel="Best fitness",
        title="Best Fitness per Generation",
        label="Best fitness",
        lw=2
    )

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

    display(plt)
    return plt
end
