using JSON3
using HomeCareGA.Fitness: SPLIT


@inline function _fmt_num(x::Real)::String
    xf = Float64(x)
    if isapprox(xf, round(xf); atol=1e-9)
        return string(Int(round(xf)))
    end
    s = string(round(xf; digits=2))
    s = replace(s, r"0+$" => "")
    s = replace(s, r"\.$" => "")
    return s
end

function _fmt_duration(x::Real)::String
    v = round(Float64(x); digits=2)
    neg = v < 0
    av = abs(v)
    i = floor(Int, av)
    d = round(Int, (av - i) * 100)
    if d == 100
        i += 1
        d = 0
    end
    s = string(i, ".", lpad(string(d), 2, '0'))
    return neg ? "-" * s : s
end

function _parse_routes(solution::Vector{Int})::Vector{Vector{Int}}
    routes = Vector{Int}[]
    current_route = Int[]

    for p in solution
        if p == SPLIT
            push!(routes, current_route)
            current_route = Int[]
        else
            push!(current_route, p)
        end
    end

    push!(routes, current_route)
    return routes
end

function _simulate_route(route::Vector{Int}, instance)
    if isempty(route)
        return (total_demand=0.0, route_duration=0.0, route_travel=0.0, stops=NamedTuple[])
    end

    travel_times = instance["travel_times"]
    patients = instance["patients"]

    cur_time = 0.0
    cur_node = 0
    route_travel = 0.0
    total_demand = 0.0
    stops = NamedTuple[]

    for pid in route
        pdata = patients[string(pid)]

        travel_t = Float64(travel_times[cur_node + 1][pid + 1])
        arrival = cur_time + travel_t
        route_travel += travel_t

        tw_start = Float64(pdata["start_time"])
        tw_end = Float64(pdata["end_time"])
        care_t = Float64(pdata["care_time"])

        visit_start = max(arrival, tw_start)
        visit_end = visit_start + care_t

        push!(stops, (
            pid=pid,
            visit_start=visit_start,
            visit_end=visit_end,
            tw_start=tw_start,
            tw_end=tw_end
        ))

        cur_time = visit_end
        cur_node = pid
        total_demand += Float64(pdata["demand"])
    end

    back_to_depot = Float64(travel_times[cur_node + 1][1])
    cur_time += back_to_depot
    route_travel += back_to_depot
    return (total_demand=total_demand, route_duration=cur_time, route_travel=route_travel, stops=stops)
end

"""
    log_solution(solution::Vector{Int}, json_file::String, objective_value::Float64, output_file::String)

Logs the solution in formatted text output and saves to file.
Parses all necessary data (patients, depot, capacity) from the JSON instance file.

Example:
    log_solution(chromosome, "train/train_0.json", 827.3, "results/solution_0.txt")
"""
function log_solution(
    solution::Vector{Int},
    json_file::String,
    _objective_value::Float64,
    output_file::String
)
    instance = JSON3.read(read(json_file, String))
    instance_name = instance["instance_name"]
    nurse_capacity = instance["capacity_nurse"]
    available_nurses = haskey(instance, :nbr_nurses) ? Int(instance["nbr_nurses"]) : (count(==(SPLIT), solution) + 1)
    depot_return_time = instance["depot"]["return_time"]
    routes = [r for r in _parse_routes(solution) if !isempty(r)]
    active_nurses = length(routes)

    open(output_file, "w") do io
        println(io, "Instance: $instance_name")
        println(io, "Nurse capacity: ", _fmt_num(nurse_capacity))
        println(io, "Available nurses: ", available_nurses)
        println(io, "Active nurses: ", active_nurses)
        println(io, "Depot return time: ", _fmt_num(depot_return_time))
        println(io, "-" ^ 140)

        println(io, "Nurse           Route duration    Covered demand    Patient sequence")

        total_travel = 0.0
        for (nurse_idx, route) in enumerate(routes)
            sim = _simulate_route(route, instance)
            total_travel += sim.route_travel

            seq_parts = String["D (0)"]
            for stop in sim.stops
                push!(
                    seq_parts,
                    "$(stop.pid) (" * _fmt_num(stop.visit_start) * "-" * _fmt_num(stop.visit_end) * ")" *
                    " [" * _fmt_num(stop.tw_start) * "-" * _fmt_num(stop.tw_end) * "]"
                )
            end
            push!(seq_parts, "D (" * _fmt_duration(sim.route_duration) * ")")
            seq = join(seq_parts, " → ")

            nurse_label = "Nurse $(nurse_idx) (N$(nurse_idx))"
            println(
                io,
                rpad(nurse_label, 15), " ",
                rpad(_fmt_duration(sim.route_duration), 17), " ",
                rpad(_fmt_num(sim.total_demand), 16), " ",
                seq
            )
        end

        println(io, "-" ^ 140)
        println(io, "Objective value (total travel time): ", _fmt_duration(total_travel))
    end
end
