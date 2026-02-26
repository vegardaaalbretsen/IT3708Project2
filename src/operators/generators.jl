using JSON3

abstract type Generator end

"""
Population generators:
    Random
    Sweep + time-window aware
"""
Base.@kwdef struct RandomGenerator <: Generator
    num_jobs::Int      # Number of patients/jobs to visit
    num_routes::Int    # Number of nurses/routes
    min_active_routes::Int = 1
end

"""
Sweep-based generator with simple time-window awareness.

Construction:
    SweepTWGenerator("path/to/instance.json"; num_routes=nothing)
"""
Base.@kwdef struct SweepTWGenerator <: Generator
    num_jobs::Int
    num_routes::Int
    min_active_routes::Int = 1
    depot_x::Float64
    depot_y::Float64
    patient_x::Vector{Float64}
    patient_y::Vector{Float64}
    start_time::Vector{Float64}
    end_time::Vector{Float64}
    shuffle_route_sizes::Bool = true
    allow_empty_routes::Bool = false
end

function SweepTWGenerator(
    instance_json_path::AbstractString;
    num_routes::Union{Nothing, Int}=nothing,
    min_active_routes::Int=1,
    shuffle_route_sizes::Bool=true,
    allow_empty_routes::Bool=false
)
    data = JSON3.read(read(instance_json_path, String))

    haskey(data, :patients) || error("Instance JSON missing 'patients': $instance_json_path")
    haskey(data, :depot) || error("Instance JSON missing 'depot': $instance_json_path")

    n = length(data["patients"])
    r = isnothing(num_routes) ? Int(data["nbr_nurses"]) : num_routes
    r > 0 || error("num_routes must be > 0.")
    min_active_routes >= 1 || error("min_active_routes must be >= 1.")
    min_active_routes <= r || error("min_active_routes cannot exceed num_routes.")
    min_active_routes <= n || error("min_active_routes cannot exceed num_jobs.")

    depot_x = Float64(data["depot"]["x_coord"])
    depot_y = Float64(data["depot"]["y_coord"])

    patient_x = Vector{Float64}(undef, n)
    patient_y = Vector{Float64}(undef, n)
    start_time = Vector{Float64}(undef, n)
    end_time = Vector{Float64}(undef, n)

    for (pid_str, pdata) in data["patients"]
        pid = parse(Int, String(pid_str))
        patient_x[pid] = Float64(pdata["x_coord"])
        patient_y[pid] = Float64(pdata["y_coord"])
        start_time[pid] = Float64(pdata["start_time"])
        end_time[pid] = Float64(pdata["end_time"])
    end

    return SweepTWGenerator(
        num_jobs=n,
        num_routes=r,
        min_active_routes=min_active_routes,
        depot_x=depot_x,
        depot_y=depot_y,
        patient_x=patient_x,
        patient_y=patient_y,
        start_time=start_time,
        end_time=end_time,
        shuffle_route_sizes=shuffle_route_sizes,
        allow_empty_routes=allow_empty_routes
    )
end

@inline function _balanced_route_sizes(num_jobs::Int, num_routes::Int)::Vector{Int}
    sizes = fill(div(num_jobs, num_routes), num_routes)
    @inbounds for i in 1:rem(num_jobs, num_routes)
        sizes[i] += 1
    end
    return sizes
end

@inline function _random_route_sizes(
    num_jobs::Int,
    num_routes::Int,
    rng::AbstractRNG
)::Vector{Int}
    num_routes == 1 && return [num_jobs]

    # "Stars and bars": sample separator positions uniformly among all placements.
    # This allows empty routes, making num_routes an upper bound on active routes.
    total_slots = num_jobs + num_routes - 1
    bars = sort!(randperm(rng, total_slots)[1:(num_routes - 1)])

    sizes = Vector{Int}(undef, num_routes)
    prev = 0
    @inbounds for r in 1:(num_routes - 1)
        b = bars[r]
        sizes[r] = b - prev - 1
        prev = b
    end
    sizes[end] = total_slots - prev
    return sizes
end

function _validate_sweep_generator(method::SweepTWGenerator)
    method.num_jobs > 0 || error("SweepTWGenerator.num_jobs must be > 0.")
    method.num_routes > 0 || error("SweepTWGenerator.num_routes must be > 0.")
    method.min_active_routes >= 1 || error("SweepTWGenerator.min_active_routes must be >= 1.")
    method.min_active_routes <= method.num_routes || error("SweepTWGenerator.min_active_routes cannot exceed num_routes.")
    method.min_active_routes <= method.num_jobs || error("SweepTWGenerator.min_active_routes cannot exceed num_jobs.")
    length(method.patient_x) == method.num_jobs || error("patient_x length must equal num_jobs.")
    length(method.patient_y) == method.num_jobs || error("patient_y length must equal num_jobs.")
    length(method.start_time) == method.num_jobs || error("start_time length must equal num_jobs.")
    length(method.end_time) == method.num_jobs || error("end_time length must equal num_jobs.")
    return nothing
end

"""
Generate a random population where each individual is a permutation of jobs (1 to num_jobs)
with separators (-1) dividing the different routes.

Chromosome length = num_jobs + (num_routes - 1) separators
"""
function generate(
    method::RandomGenerator;
    pop_size::Int,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Vector{Vector{Int}}
    method.min_active_routes >= 1 || error("RandomGenerator.min_active_routes must be >= 1.")
    method.min_active_routes <= method.num_routes || error("RandomGenerator.min_active_routes cannot exceed num_routes.")
    method.min_active_routes <= method.num_jobs || error("RandomGenerator.min_active_routes cannot exceed num_jobs.")

    population = Vector{Vector{Int}}(undef, pop_size)
    num_separators = method.num_routes - 1

    @inbounds for i in 1:pop_size
        jobs = shuffle(rng, 1:method.num_jobs)
        separators = fill(-1, num_separators)
        chromosome = vcat(jobs, separators)
        shuffle!(rng, chromosome)
        population[i] = enforce_min_active_routes(chromosome, method.min_active_routes)
    end

    return population
end

"""
Generate a sweep-based, time-window aware population.

Workflow per individual:
1. Circular sweep ordering around the depot (with random angle offset for diversity)
2. Route-length splitting
   - balanced + shuffled (default)
   - random with possible empty routes if `allow_empty_routes=true`
3. Earliest due-time ordering inside each route
4. Canonicalization for route-order symmetry reduction
"""
function generate(
    method::SweepTWGenerator;
    pop_size::Int,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Vector{Vector{Int}}
    _validate_sweep_generator(method)
    population = Vector{Vector{Int}}(undef, pop_size)

    ids = collect(1:method.num_jobs)
    angles = [atan(method.patient_y[i] - method.depot_y, method.patient_x[i] - method.depot_x) for i in 1:method.num_jobs]
    base_sizes = _balanced_route_sizes(method.num_jobs, method.num_routes)

    @inbounds for i in 1:pop_size
        shift = rand(rng) * (2pi)
        sort!(ids; by=pid -> mod2pi(angles[pid] + shift))

        route_sizes = if method.allow_empty_routes
            _random_route_sizes(method.num_jobs, method.num_routes, rng)
        else
            sizes = copy(base_sizes)
            if method.shuffle_route_sizes
                shuffle!(rng, sizes)
            end
            sizes
        end

        chromosome = Int[]
        sizehint!(chromosome, method.num_jobs + method.num_routes - 1)
        pos = 1

        for r in 1:method.num_routes
            L = route_sizes[r]
            if L > 0
                route = ids[pos:(pos + L - 1)]
                sort!(route; by=pid -> (method.end_time[pid], method.start_time[pid]))
                append!(chromosome, route)
                pos += L
            end
            if r < method.num_routes
                push!(chromosome, -1)
            end
        end

        population[i] = enforce_min_active_routes(chromosome, method.min_active_routes)
    end

    return population
end
