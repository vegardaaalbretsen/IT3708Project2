using HomeCareGA.Fitness: HCInstance


"""
    optimal_split(inst::HCInstance, perm::Vector{Int}) -> (routes, cost)

Compute the optimal segmentation ("split") of a giant-tour permutation into
feasible nurse routes.

# Description
Given a permutation of patients (optionally with -1 route delimiters),
this function removes any existing delimiters and interprets the sequence as
a giant route. It then finds the minimum-travel-cost way to partition the
sequence into feasible routes.

Each constructed route:
- Respects nurse capacity
- Respects patient time windows
- Returns to the depot before depot_return_time
- NB! Does not restrict the number of nurses 

The dynamic programing split procedure variables:

- dp[j] = minimum travel cost to serve the first j-1 patients
- pred[j] stores the predecessor index (route start) used for route splitting

If no feasible route split exists, the function returns (nothing, Inf).

# Returns
- result::Vector{Int}:
    The optimal route split encoded using -1 as route delimiters.
    Returns 'nothing' if there is no feasible split for the permutation.
- dp[n+1]::Float64:
    Total travel cost of the final split solution.
    Returns Inf if infeasible.

"""

function optimal_split(inst::HCInstance, perm::Vector{Int})
    
    # Strip out -1 delimiters to get only permutation (grand tour rperesentation)
    clean_perm = filter(x -> x != -1, perm)
    n = length(clean_perm) # Use the cleaned sequence length
    
    # DP[j] = best cost to serve the first j patients (Dynamic Programming)
    dp = fill(Inf, n + 1)
    # Predecessor
    pred = fill(0, n + 1) # pred[11] = 6 -> the route covering 11th patient in permutation started at index 6
    
    dp[1] = 0.0  # 0 patients served
    

    for i in 1:n
        if dp[i] == Inf
            continue
        end
        
        load = 0.0 # how much of nurse capacity is used
        time = 0.0
        travel_cost = 0.0
        prev_node_id = 0  # depot
        
        for j in i:n
            patient_id = clean_perm[j]
            
            # ---- Capacity ----
            load += inst.demand[patient_id]
            if load > inst.capacity_nurse
                break
            end
            
            # ---- Travel ----
            ttime = inst.travel[prev_node_id + 1, patient_id + 1]
            travel_cost += ttime
            
            arrival = time + ttime
            start_service = max(arrival, inst.start_time[patient_id])
            finish = start_service + inst.care_time[patient_id]
            
            # ---- Time window ----
            if finish > inst.end_time[patient_id]
                break
            end
            
            time = finish
            prev_node_id = patient_id
            
            # ---- Depot return feasibility ----
            return_time = time + inst.travel[prev_node_id+1, 1]
            if return_time > inst.depot_return_time
                break
            end
            
            route_cost = travel_cost + inst.travel[prev_node_id + 1, 1]
            
            if dp[i] + route_cost < dp[j + 1]
                dp[j + 1] = dp[i] + route_cost
                pred[j + 1] = i
            end
        end
    end
    
    if dp[n + 1] == Inf
        return nothing, Inf
    end
    
    # ---- Reconstruct routes with -1 delimiters ----
    result = Int[]
    idx = n + 1
    
    while idx > 1
        start_idx = pred[idx]
        
        # Traverse the route segment backwards until hitting start index (where the -1 delim shold be)
        for k in (idx - 1):-1:start_idx
            push!(result, clean_perm[k])
        end
        
        idx = start_idx
        
        # Inject the delimiter if there are more routes before this one
        if idx > 1
            push!(result, -1)
        end
    end
    
    # Reverse the array since we built it from the back to the front
    reverse!(result)
    
    # Return routes with -1 dummy encoding and the total travel time for the individual
    return result, dp[n+1]
end