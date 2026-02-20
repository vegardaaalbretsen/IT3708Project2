using JSON3

const SPLIT = -1

# Helper function to calculate total route time using travel_times from instance
function calculate_route_duration(route::Vector{Int}, instance)
    if isempty(route)
        return 0.0
    end
    
    # Extract travel times matrix, patient info
    travel_times = instance["travel_times"]
    patients_info = Dict{Int, Tuple{Int, Int, Int}}()  # start_time, end_time, care_time
    
    for (patient_id_str, patient_data) in instance["patients"]
        patient_id = parse(Int, String(patient_id_str))
        patients_info[patient_id] = (patient_data["start_time"], patient_data["end_time"], patient_data["care_time"])
    end
    
    # Calculate total time: travel + wait + care times
    # Node 0 = depot, Node 1-N = patients
    current_time = 0.0  # Start at depot at time 0
    current_node = 0
    
    for patient_id in route
        # Travel to patient
        travel_time = travel_times[current_node + 1][patient_id + 1]  # Julia 1-indexed
        current_time += travel_time
        
        # Wait if we arrive before start_time
        start_time, end_time, care_time = patients_info[patient_id]
        if current_time < start_time
            current_time = start_time
        end
        
        # Care time at patient
        current_time += care_time
        
        current_node = patient_id
    end
    
    # Return to depot
    return_travel = travel_times[current_node + 1][0 + 1]
    current_time += return_travel
    
    return round(current_time, digits=2)
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
    objective_value::Float64,
    output_file::String
)
    
    # Load and parse JSON instance
    instance = JSON3.read(read(json_file, String))
    
    instance_name = instance["instance_name"]
    nurse_capacity = instance["capacity_nurse"]
    depot_return_time = instance["depot"]["return_time"]
    depot_coords = (instance["depot"]["x_coord"], instance["depot"]["y_coord"])
    
    # Build patient dicts: patient_id -> (x, y) and patient_id -> (start_time, end_time, care_time)
    patients_coords = Dict{Int, Tuple{Int, Int}}()
    patients_info = Dict{Int, Tuple{Int, Int, Int}}()  # start, end, care_time
    
    for (patient_id_str, patient_data) in instance["patients"]
        patient_id = parse(Int, String(patient_id_str))
        patients_coords[patient_id] = (patient_data["x_coord"], patient_data["y_coord"])
        patients_info[patient_id] = (patient_data["start_time"], patient_data["end_time"], patient_data["care_time"])
    end
    
    # Open file for writing
    open(output_file, "w") do io
        # Header info
        println(io, "Instance: $instance_name")
        println(io, "Nurse capacity: $nurse_capacity")
        println(io, "Depot return time: $depot_return_time")
        println(io, "-" ^ 110)
        
        # Parse routes from solution
        routes = Vector{Int}[]
        current_route = Int[]
        
        for p in solution
            if p == SPLIT
                if !isempty(current_route)
                    push!(routes, current_route)
                    current_route = Int[]
                end
            else
                push!(current_route, p)
            end
        end
        
        if !isempty(current_route)
            push!(routes, current_route)
        end
        
        # Print each nurse route
        for (nurse_idx, route) in enumerate(routes)
            # Calculate total demand and route duration
            total_demand = 0
            route_duration = calculate_route_duration(route, instance)
            
            for pid in route
                if haskey(instance["patients"], string(pid))
                    patient = instance["patients"][string(pid)]
                    total_demand += patient["demand"]
                end
            end
            
            # Build route string with time windows
            route_str = "D (0)"
            
            for patient_id in route
                if haskey(patients_info, patient_id)
                    start_time, end_time, care_time = patients_info[patient_id]
                    route_str *= " → $patient_id ($start_time-$end_time) [$start_time-$end_time]"
                else
                    route_str *= " → $patient_id"
                end
            end
            
            route_str *= " → D ($route_duration)"
            
            # Print nurse line
            patient_count = length(route)
            
            println(io, "Nurse $nurse_idx    $total_demand    $patient_count    $route_str")
            println(io, "    :            :          :                                :")
        end
        
        println(io, "-" ^ 110)
        println(io, "Objective value (total duration): $objective_value")
    end
end
