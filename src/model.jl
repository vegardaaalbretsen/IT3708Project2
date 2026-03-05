"""
One home-care patient with demand, time window, service time, and coordinates.
"""
struct Patient
    id::Int
    demand::Int
    start_time::Float64
    end_time::Float64
    care_time::Float64
    x::Float64
    y::Float64
end

"""
Full problem instance (depot, resources, patients, and travel-time matrix).
"""
struct Instance
    name::String
    nbr_nurses::Int
    capacity_nurse::Int
    benchmark::Float64
    return_time::Float64
    depot_x::Float64
    depot_y::Float64
    patients::Vector{Patient}
    travel_times::Matrix{Float64}
end

"""
Timing details for one patient visit inside a route simulation.
"""
struct VisitInfo
    patient::Int
    arrival::Float64
    start::Float64
    finish::Float64
end

"""
Evaluation result for a single route.

Includes feasibility, travel/duration totals, served demand, per-visit timing, and
aggregated lateness/violation amount.
"""
struct RouteEval
    feasible::Bool
    travel::Float64
    duration::Float64
    demand::Int
    visits::Vector{VisitInfo}
    lateness::Float64
end

"""
A GA individual: route encoding plus objective/constraint diagnostics.

`fitness` is the penalized objective used by selection, while `total_travel` is the
raw objective value for feasible comparisons.
"""
mutable struct Candidate
    routes::Vector{Vector{Int}}
    total_travel::Float64
    fitness::Float64
    feasible::Bool
    total_lateness::Float64
    missing_patients::Int
    duplicate_visits::Int
    extra_routes::Int
    invalid_genes::Int
end

"""
Genetic algorithm hyperparameters and runtime controls.
"""
struct GAConfig
    population_size::Int
    generations::Int
    tournament_size::Int
    elitism::Int
    crossover_rate::Float64
    mutation_rate::Float64
    time_limit_sec::Float64
    log_every::Int
end
