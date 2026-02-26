using HomeCareGA
using StableRNGs
using JSON3
using Statistics
using Printf
using Random
using Dates

const TRAIN_DIR = joinpath(pwd(), "train")
const INSTANCE_FILES = sort(filter(f -> endswith(f, ".json"), readdir(TRAIN_DIR, join=true)))

struct InstanceMeta
    path::String
    name::String
    instance::HCInstance
    benchmark::Float64
    nbr_nurses::Int
    cap::Float64
    lb_demand::Int
end

function load_meta(path::String)
    data = JSON3.read(read(path, String))
    inst = load_instance(path)
    total_demand = sum(inst.demand)
    lb = max(1, ceil(Int, total_demand / inst.capacity_nurse))
    return InstanceMeta(
        path,
        basename(path),
        inst,
        Float64(data["benchmark"]),
        Int(data["nbr_nurses"]),
        Float64(data["capacity_nurse"]),
        lb,
    )
end

const METAS = [load_meta(p) for p in INSTANCE_FILES]
const META_BY_NAME = Dict(m.name => m for m in METAS)

@inline feasible_breakdown(b) = (b.cap_violation <= 1e-9 && b.return_overtime <= 1e-9 && b.late <= 1e-9)

function route_limits(cfg_nt, meta::InstanceMeta)
    if cfg_nt.routing_mode == :static
        max_n = cfg_nt.max_nurses
        min_n = cfg_nt.min_active_mode == :lb ? meta.lb_demand : 1
        min_n = min(min_n, max_n)
        return max_n, min_n
    end

    # dynamic routing mode
    max_n = meta.lb_demand + cfg_nt.route_slack
    max_n = max(max_n, meta.lb_demand)
    max_n = min(max_n, cfg_nt.max_cap)
    max_n = min(max_n, meta.nbr_nurses)

    min_n = cfg_nt.min_active_mode == :lb ? meta.lb_demand : 1
    min_n = min(min_n, max_n)

    return max_n, min_n
end

function build_cfg(cfg_nt, meta::InstanceMeta, pop_size, generations)
    max_n, min_n = route_limits(cfg_nt, meta)

    weights = FitnessWeights(
        w_travel=1.0,
        w_wait=0.0,
        w_capacity=cfg_nt.w_cap,
        w_return=cfg_nt.w_ret,
        w_late=cfg_nt.w_late,
        w_early=0.0,
    )
    schedule = PenaltySchedule(
        min_scale=cfg_nt.s_min,
        max_scale=cfg_nt.s_max,
        power=cfg_nt.s_power,
        mag_scale=cfg_nt.s_mag,
    )

    return GAConfig(
        p_c=cfg_nt.p_c,
        p_m=cfg_nt.p_m,
        p_ls=cfg_nt.p_ls,
        selector=TournamentSelector(cfg_nt.k),
        crossover=O1XCrossover(min_frac=0.07, max_frac=cfg_nt.o1xmax),
        mutator=SwapAnyMutator(),
        local_search=TwoOptLocalSearch(),
        survivor=ElitistSelector(num_elites=cfg_nt.elites),
        generator=SweepTWGenerator(meta.path; num_routes=max_n, min_active_routes=min_n, allow_empty_routes=true),
        pop_size=pop_size,
        max_generations=generations,
        min_active_routes=min_n,
        fitness_weights=weights,
        penalty_schedule=schedule,
        keep_history=false,
        verbose=false,
        log_every=0,
        solution_output_file=nothing,
        instance_json_file=meta.path,
    )
end

function run_instance(cfg_nt, meta::InstanceMeta; pop_size::Int, generations::Int, seed::Int)
    ga_cfg = build_cfg(cfg_nt, meta, pop_size, generations)
    res = GA(meta.path, ga_cfg; rng=StableRNG(seed))
    b = fitness_breakdown(
        res.best_individual,
        meta.instance;
        weights=ga_cfg.fitness_weights,
        schedule=ga_cfg.penalty_schedule,
        generation=generations,
        max_generations=generations,
    )
    gap = (b.travel / meta.benchmark - 1.0) * 100.0

    max_n, min_n = route_limits(cfg_nt, meta)

    return (
        instance=meta.name,
        travel=b.travel,
        gap=gap,
        feasible=feasible_breakdown(b),
        active=active_route_count(res.best_individual),
        late=b.late,
        ret=b.return_overtime,
        cap=b.cap_violation,
        max_n=max_n,
        min_n=min_n,
    )
end

function evaluate_config(cfg_nt, metas; pop_size::Int, generations::Int, seed::Int)
    rows = NamedTuple[]
    for meta in metas
        push!(rows, run_instance(cfg_nt, meta; pop_size=pop_size, generations=generations, seed=seed))
    end

    gaps = [r.gap for r in rows]
    feasible_count = count(r -> r.feasible, rows)
    # Strong feasibility priority, then gap.
    score = mean(gaps) + 1000.0 * (length(rows) - feasible_count)

    return (
        cfg=cfg_nt,
        score=score,
        mean_gap=mean(gaps),
        median_gap=median(gaps),
        worst_gap=maximum(gaps),
        feasible_count=feasible_count,
        n=length(rows),
        rows=rows,
    )
end

@inline function cfg_label(cfg_nt)
    route_part = cfg_nt.routing_mode == :static ? @sprintf("mode=static maxn=%d", cfg_nt.max_nurses) : @sprintf("mode=dynamic slack=%d cap=%d min=%s", cfg_nt.route_slack, cfg_nt.max_cap, String(cfg_nt.min_active_mode))
    return @sprintf(
        "pc=%.2f pm=%.2f pls=%.2f k=%d o1x=%.2f e=%d | %s | w=(%.1f,%.1f,%.1f) s=(%.1f,%.1f,%.1f,%.2f)",
        cfg_nt.p_c,
        cfg_nt.p_m,
        cfg_nt.p_ls,
        cfg_nt.k,
        cfg_nt.o1xmax,
        cfg_nt.elites,
        route_part,
        cfg_nt.w_cap,
        cfg_nt.w_ret,
        cfg_nt.w_late,
        cfg_nt.s_min,
        cfg_nt.s_max,
        cfg_nt.s_power,
        cfg_nt.s_mag,
    )
end

function print_rank(io, rows; title::String, top_k::Int=10)
    println(io, title)
    for (rank, r) in enumerate(rows[1:min(top_k, length(rows))])
        println(
            io,
            @sprintf(
                "#%d score=%.2f mean_gap=%.2f%% med=%.2f%% worst=%.2f%% feasible=%d/%d | %s",
                rank,
                r.score,
                r.mean_gap,
                r.median_gap,
                r.worst_gap,
                r.feasible_count,
                r.n,
                cfg_label(r.cfg),
            )
        )
    end
    println(io)
end

function run_retune2()
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(pwd(), "results", "retune")
    mkpath(out_dir)
    summary_file = joinpath(out_dir, "retune2_summary_$(stamp).txt")

    baseline_current = (
        p_c=0.85, p_m=0.10, p_ls=0.15, k=3, o1xmax=0.15, elites=10,
        routing_mode=:static, max_nurses=8, route_slack=0, max_cap=8, min_active_mode=:one,
        w_cap=8.0, w_ret=6.0, w_late=18.0,
        s_min=0.5, s_max=4.0, s_power=1.4, s_mag=0.02,
    )

    baseline_dynamic = (
        p_c=0.85, p_m=0.10, p_ls=0.15, k=3, o1xmax=0.15, elites=10,
        routing_mode=:dynamic, max_nurses=0, route_slack=2, max_cap=14, min_active_mode=:lb,
        w_cap=8.0, w_ret=6.0, w_late=18.0,
        s_min=0.5, s_max=4.0, s_power=1.4, s_mag=0.02,
    )

    pcs = [0.82, 0.88, 0.94]
    pms = [0.08, 0.12, 0.16]
    plss = [0.15, 0.22, 0.30]
    ks = [2, 3]
    o1xs = [0.15, 0.22, 0.30]
    elites = [6, 10]
    route_slacks = [1, 2, 3, 4]
    max_caps = [12, 14, 16]
    min_modes = [:lb, :one]

    w_caps = [8.0, 14.0, 20.0]
    w_rets = [6.0, 10.0, 14.0]
    w_lates = [18.0, 28.0, 40.0]

    schedules = [
        (0.5, 4.0, 1.4, 0.02),
        (1.0, 6.0, 1.2, 0.03),
        (1.0, 8.0, 1.0, 0.04),
    ]

    Random.seed!(3708)
    candidates = [baseline_dynamic]
    while length(candidates) < 18
        sch = rand(schedules)
        cfg = (
            p_c=rand(pcs),
            p_m=rand(pms),
            p_ls=rand(plss),
            k=rand(ks),
            o1xmax=rand(o1xs),
            elites=rand(elites),
            routing_mode=:dynamic,
            max_nurses=0,
            route_slack=rand(route_slacks),
            max_cap=rand(max_caps),
            min_active_mode=rand(min_modes),
            w_cap=rand(w_caps),
            w_ret=rand(w_rets),
            w_late=rand(w_lates),
            s_min=sch[1],
            s_max=sch[2],
            s_power=sch[3],
            s_mag=sch[4],
        )
        cfg in candidates || push!(candidates, cfg)
    end

    stage_rows = NamedTuple[]
    println("Stage A: tuning on all train instances (light budget)...")
    for (i, cfg) in enumerate(candidates)
        r = evaluate_config(cfg, METAS; pop_size=110, generations=300, seed=42)
        push!(stage_rows, r)
        println(@sprintf("  [%d/%d] score=%.2f mean_gap=%.2f%% feasible=%d/10 | %s", i, length(candidates), r.score, r.mean_gap, r.feasible_count, cfg_label(cfg)))
    end
    sort!(stage_rows; by=r -> r.score)

    top_cfgs = [r.cfg for r in stage_rows[1:4]]

    validate_rows = NamedTuple[]
    println("Stage B: deeper validation for top configs...")
    for (i, cfg) in enumerate(top_cfgs)
        r = evaluate_config(cfg, METAS; pop_size=200, generations=900, seed=42)
        push!(validate_rows, r)
        println(@sprintf("  [%d/%d] score=%.2f mean_gap=%.2f%% feasible=%d/10 | %s", i, length(top_cfgs), r.score, r.mean_gap, r.feasible_count, cfg_label(cfg)))
    end
    sort!(validate_rows; by=r -> r.score)
    best_cfg = validate_rows[1].cfg

    println("Final benchmark run (current baseline vs dynamic baseline vs tuned)...")
    final_current = evaluate_config(baseline_current, METAS; pop_size=250, generations=1200, seed=42)
    final_dynamic = evaluate_config(baseline_dynamic, METAS; pop_size=250, generations=1200, seed=42)
    final_tuned = evaluate_config(best_cfg, METAS; pop_size=250, generations=1200, seed=42)

    open(summary_file, "w") do io
        println(io, "Retune2 Summary")
        println(io, "Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)

        print_rank(io, stage_rows; title="Stage A Top", top_k=10)
        print_rank(io, validate_rows; title="Stage B Top", top_k=4)

        println(io, "Final comparison (all instances, pop=250, gen=1200):")
        println(io, "Current baseline:")
        println(io, "  ", cfg_label(baseline_current))
        println(io, @sprintf("  mean_gap=%.2f%% median_gap=%.2f%% worst_gap=%.2f%% feasible=%d/%d", final_current.mean_gap, final_current.median_gap, final_current.worst_gap, final_current.feasible_count, final_current.n))
        println(io)

        println(io, "Dynamic baseline:")
        println(io, "  ", cfg_label(baseline_dynamic))
        println(io, @sprintf("  mean_gap=%.2f%% median_gap=%.2f%% worst_gap=%.2f%% feasible=%d/%d", final_dynamic.mean_gap, final_dynamic.median_gap, final_dynamic.worst_gap, final_dynamic.feasible_count, final_dynamic.n))
        println(io)

        println(io, "Best tuned:")
        println(io, "  ", cfg_label(best_cfg))
        println(io, @sprintf("  mean_gap=%.2f%% median_gap=%.2f%% worst_gap=%.2f%% feasible=%d/%d", final_tuned.mean_gap, final_tuned.median_gap, final_tuned.worst_gap, final_tuned.feasible_count, final_tuned.n))
        println(io)

        println(io, "Per-instance comparison (current -> dynamic -> tuned):")
        for meta in METAS
            c = only(filter(r -> r.instance == meta.name, final_current.rows))
            d = only(filter(r -> r.instance == meta.name, final_dynamic.rows))
            t = only(filter(r -> r.instance == meta.name, final_tuned.rows))
            println(
                io,
                @sprintf(
                    "%s | gap %.2f%% -> %.2f%% -> %.2f%% | travel %.2f -> %.2f -> %.2f | feasible %s -> %s -> %s",
                    meta.name,
                    c.gap,
                    d.gap,
                    t.gap,
                    c.travel,
                    d.travel,
                    t.travel,
                    string(c.feasible),
                    string(d.feasible),
                    string(t.feasible),
                )
            )
        end
    end

    println("DONE")
    println("SUMMARY_FILE: ", summary_file)
    println("BEST_CONFIG: ", cfg_label(best_cfg))
    println(@sprintf("CURRENT_FINAL mean_gap=%.2f%% feasible=%d/%d", final_current.mean_gap, final_current.feasible_count, final_current.n))
    println(@sprintf("DYNAMIC_FINAL mean_gap=%.2f%% feasible=%d/%d", final_dynamic.mean_gap, final_dynamic.feasible_count, final_dynamic.n))
    println(@sprintf("TUNED_FINAL mean_gap=%.2f%% feasible=%d/%d", final_tuned.mean_gap, final_tuned.feasible_count, final_tuned.n))
end

run_retune2()
