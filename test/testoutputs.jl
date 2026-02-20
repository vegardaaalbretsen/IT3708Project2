using Plots
using JSON3

# include plotting-funksjonen
include("../src/algorithms/outputs.jl")

json_file = joinpath(@__DIR__, "..", "train", "train_0.json")
data = JSON3.read(read(json_file, String))

# Extract patient IDs from data
patient_ids = [parse(Int, String(id)) for id in keys(data["patients"])]

chromosome = Int[]
for (idx, id) in enumerate(patient_ids)
    push!(chromosome, id)
    if idx % 4 == 0 && idx != length(patient_ids)
        push!(chromosome, -1)
    end
end

# Plot the routes
plot_routes_stream(chromosome, json_file)

# Log the solution to file
output_file = joinpath(@__DIR__, "solution_output.txt")
objective_value = data["benchmark"]
log_solution(chromosome, json_file, objective_value, output_file)
