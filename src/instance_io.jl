function load_instance(path::String)::Instance
    raw = JSON3.read(read(path, String))
    patients_raw = raw.patients
    n = length(patients_raw)
    patients = Vector{Patient}(undef, n)

    for i in 1:n
        p = patients_raw[string(i)]
        patients[i] = Patient(
            i,
            Int(p.demand),
            Float64(p.start_time),
            Float64(p.end_time),
            Float64(p.care_time),
            Float64(p.x_coord),
            Float64(p.y_coord),
        )
    end

    tt_raw = raw.travel_times
    travel_times = Matrix{Float64}(undef, n + 1, n + 1)
    for i in 1:(n + 1)
        row = tt_raw[i]
        for j in 1:(n + 1)
            travel_times[i, j] = Float64(row[j])
        end
    end

    return Instance(
        String(raw.instance_name),
        Int(raw.nbr_nurses),
        Int(raw.capacity_nurse),
        Float64(raw.benchmark),
        Float64(raw.depot.return_time),
        Float64(raw.depot.x_coord),
        Float64(raw.depot.y_coord),
        patients,
        travel_times,
    )
end
