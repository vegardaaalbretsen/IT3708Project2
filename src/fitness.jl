# src/fitness.jl
module Fitness

using JSON3
using Random: AbstractRNG

export SPLIT, HCInstance, PenaltySchedule, FitnessWeights, FitnessBreakdown,
       load_instance, fitness, fitness_breakdown

const SPLIT = -1

# ----------------------------
# Parsed, type-stable instance
# ----------------------------

Base.@kwdef struct HCInstance
    name::String
    N::Int

    # travel_times[node+1, node+1], where node 0 = depot, node 1..N = patients
    travel::Matrix{Float64}

    # patient indexed 1..N
    start_time::Vector{Float64}
    end_time::Vector{Float64}
    care_time::Vector{Float64}
    demand::Vector{Float64}

    # constraints
    capacity_nurse::Float64
    depot_return_time::Float64   # end of day / must return by
end

# Convenience constructor: copy an instance and override selected fields.
function HCInstance(inst::HCInstance;
    name::String = inst.name,
    N::Int = inst.N,
    travel::Matrix{Float64} = inst.travel,
    start_time::Vector{Float64} = inst.start_time,
    end_time::Vector{Float64} = inst.end_time,
    care_time::Vector{Float64} = inst.care_time,
    demand::Vector{Float64} = inst.demand,
    capacity_nurse::Float64 = inst.capacity_nurse,
    depot_return_time::Float64 = inst.depot_return_time
)
    return HCInstance(
        name=name, N=N, travel=travel,
        start_time=start_time, end_time=end_time, care_time=care_time, demand=demand,
        capacity_nurse=capacity_nurse, depot_return_time=depot_return_time
    )
end

"""
Load and parse JSON instance into a type-stable struct.

Assumes JSON:
- instance["travel_times"] is (N+1)x(N+1)
- instance["patients"] is a dict keyed by patient id strings ("1","2",...)
- depot has "return_time"
- capacity in "capacity_nurse"
"""
function load_instance(path::AbstractString)::HCInstance
    inst = JSON3.read(read(path, String))
    name = String(inst["instance_name"])
    cap  = Float64(inst["capacity_nurse"])
    rT   = Float64(inst["depot"]["return_time"])

    # travel times -> Matrix{Float64}
    tt = inst["travel_times"]
    nrows = length(tt)
    ncols = length(tt[1])
    travel = Matrix{Float64}(undef, nrows, ncols)
    @inbounds for i in 1:nrows
        row = tt[i]
        for j in 1:ncols
            travel[i, j] = Float64(row[j])
        end
    end

    # N is patients count (nodes: 0..N => matrix size N+1)
    N = nrows - 1

    st = Vector{Float64}(undef, N)
    en = Vector{Float64}(undef, N)
    ct = Vector{Float64}(undef, N)
    dm = Vector{Float64}(undef, N)

    # patients dict keys are strings; fill arrays by id
    for (pid_str, pdata) in inst["patients"]
        pid = parse(Int, String(pid_str))
        st[pid] = Float64(pdata["start_time"])
        en[pid] = Float64(pdata["end_time"])
        ct[pid] = Float64(pdata["care_time"])
        dm[pid] = Float64(pdata["demand"])
    end

    return HCInstance(
        name=name, N=N, travel=travel,
        start_time=st, end_time=en, care_time=ct, demand=dm,
        capacity_nurse=cap, depot_return_time=rT
    )
end

# ----------------------------
# Dynamic penalty machinery
# ----------------------------

Base.@kwdef struct FitnessWeights
    # objective
    w_travel::Float64 = 1.0
    w_wait::Float64   = 0.0

    # penalties (base weights)
    w_capacity::Float64 = 1000.0
    w_return::Float64   = 1000.0
    w_late::Float64     = 2000.0

    # optional (usually 0; early is typically handled as waiting)
    w_early::Float64    = 0.0
end

"""
PenaltySchedule scales penalties across generations:
scale(g) = min_scale + (max_scale-min_scale) * (progress^power)
where progress in [0,1].
Also optionally amplifies by violation magnitude smoothly.
"""
Base.@kwdef struct PenaltySchedule
    min_scale::Float64 = 1.0
    max_scale::Float64 = 10.0
    power::Float64 = 1.0
    mag_scale::Float64 = 0.0  # 0 disables magnitude amplification
end

Base.@kwdef struct FitnessBreakdown
    travel::Float64 = 0.0
    wait::Float64 = 0.0
    care::Float64 = 0.0

    cap_violation::Float64 = 0.0
    return_overtime::Float64 = 0.0
    late::Float64 = 0.0
    early::Float64 = 0.0

    total::Float64 = 0.0
end

@inline function penalty_scale(s::PenaltySchedule, gen::Int, max_gen::Int)
    if max_gen <= 1
        return s.max_scale
    end
    p = (gen - 1) / (max_gen - 1)
    p = ifelse(p < 0.0, 0.0, ifelse(p > 1.0, 1.0, p))
    return s.min_scale + (s.max_scale - s.min_scale) * (p ^ s.power)
end

@inline function mag_amp(s::PenaltySchedule, v::Float64)
    # smooth >1 multiplier for large violations
    return 1.0 + s.mag_scale * log1p(v)
end

# ----------------------------
# Fitness evaluation
# ----------------------------

"""
Fitness for a chromosome with routes split by -1.

Assumptions:
- time starts at 0 at depot for each nurse route (consistent with your route-duration code)
- hard feasibility is NOT required; violations are penalized (soft constraints)
- nodes: depot=0, patients=1..N, travel index = node+1
"""
function fitness(chrom::Vector{Int},
                 inst::HCInstance;
                 weights::FitnessWeights=FitnessWeights(),
                 schedule::PenaltySchedule=PenaltySchedule(),
                 generation::Int=1,
                 max_generations::Int=1)::Float64
    return fitness_breakdown(chrom, inst;
        weights=weights, schedule=schedule,
        generation=generation, max_generations=max_generations
    ).total
end

function fitness_breakdown(chrom::Vector{Int},
                           inst::HCInstance;
                           weights::FitnessWeights=FitnessWeights(),
                           schedule::PenaltySchedule=PenaltySchedule(),
                           generation::Int=1,
                           max_generations::Int=1)::FitnessBreakdown

    w = weights
    scale = penalty_scale(schedule, generation, max_generations)

    travel = 0.0
    wait = 0.0
    care = 0.0

    cap_v = 0.0
    overtime = 0.0
    late = 0.0
    early = 0.0

    # per-route state
    cur_time = 0.0
    cur_node = 0          # depot
    cur_load = 0.0

    T = inst.travel

    @inbounds for g in chrom
        if g == SPLIT
            # close route: return to depot
            travel_t = T[cur_node + 1, 0 + 1]
            travel += travel_t
            cur_time += travel_t

            if cur_time > inst.depot_return_time
                overtime += (cur_time - inst.depot_return_time)
            end

            # reset for next nurse
            cur_time = 0.0
            cur_node = 0
            cur_load = 0.0
            continue
        end

        pid = g  # patient id 1..N
        # travel to patient
        travel_t = T[cur_node + 1, pid + 1]
        travel += travel_t
        cur_time += travel_t

        st = inst.start_time[pid]
        en = inst.end_time[pid]

        # wait if early
        if cur_time < st
            wt = st - cur_time
            wait += wt
            cur_time = st
            # if you want to treat early as violation, accumulate:
            early += wt
        elseif cur_time > en
            late += (cur_time - en)
        end

        # care
        ct = inst.care_time[pid]
        care += ct
        cur_time += ct

        # capacity
        cur_load += inst.demand[pid]
        if cur_load > inst.capacity_nurse
            cap_v += (cur_load - inst.capacity_nurse)
        end

        cur_node = pid
    end

    # close last route if chromosome doesn't end with -1
    if cur_node != 0
        travel_t = T[cur_node + 1, 0 + 1]
        travel += travel_t
        cur_time += travel_t
        if cur_time > inst.depot_return_time
            overtime += (cur_time - inst.depot_return_time)
        end
    end

    # penalties (dynamic)
    pen_cap   = scale * w.w_capacity * cap_v * mag_amp(schedule, cap_v)
    pen_ret   = scale * w.w_return   * overtime * mag_amp(schedule, overtime)
    pen_late  = scale * w.w_late     * late * mag_amp(schedule, late)
    pen_early = scale * w.w_early    * early * mag_amp(schedule, early)

    total = w.w_travel * travel + w.w_wait * wait + pen_cap + pen_ret + pen_late + pen_early

    return FitnessBreakdown(
        travel=travel, wait=wait, care=care,
        cap_violation=cap_v, return_overtime=overtime, late=late, early=early,
        total=total
    )
end

end # module