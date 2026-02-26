using HomeCareGA
using StableRNGs
using JSON3
using Statistics
using Printf
using Random
using Dates

const TRAIN_DIR = joinpath(pwd(), "train")
const INSTANCE_FILES = sort(filter(f -> endswith(f, ".json"), readdir(TRAIN_DIR, join=true)))
const BENCHMARK = Dict{String, Float64}()
for fp in INSTANCE_FILES
    data = JSON3.read(read(fp, String))
    BENCHMARK[basename(fp)] = Float64(data["benchmark"])
end

@inline feasible_breakdown(b) = (b.cap_violation <= 1e-9 && b.return_overtime <= 1e-9 && b.late <= 1e-9)

function build_cfg(cfg_nt, instance_path, pop_size, generations)
    weights = FitnessWeights(
        w_travel=1.0,
        w_wait=0.0,
        w_capacity=8.0,
        w_return=6.0,
        w_late=18.0,
        w_early=0.0,
    )
    schedule = PenaltySchedule(
        min_scale=0.5,
        max_scale=4.0,
        power=1.4,
        mag_scale=0.02,
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
        generator=SweepTWGenerator(instance_path; num_routes=cfg_nt.max_nurses, min_active_routes=1, allow_empty_routes=true),
        pop_size=pop_size,
        max_generations=generations,
        min_active_routes=1,
        fitness_weights=weights,
        penalty_schedule=schedule,
        keep_history=false,
        verbose=false,
        log_every=0,
        solution_output_file=nothing,
        instance_json_file=instance_path,
    )
end

function run_instance(cfg_nt, instance_path; pop_size::Int, generations::Int, seed::Int)
    instance = load_instance(instance_path)
    ga_cfg = build_cfg(cfg_nt, instance_path, pop_size, generations)
    res = GA(instance_path, ga_cfg; rng=StableRNG(seed))
    b = fitness_breakdown(
        res.best_individual,
        instance;
        weights=ga_cfg.fitness_weights,
        schedule=ga_cfg.penalty_schedule,
        generation=generations,
        max_generations=generations,
    )
    bench = BENCHMARK[basename(instance_path)]
    gap = (b.travel / bench - 1.0) * 100.0

    return (
        instance=basename(instance_path),
        travel=b.travel,
        gap=gap,
        feasible=feasible_breakdown(b),
        active=active_route_count(res.best_individual),
        late=b.late,
        ret=b.return_overtime,
        cap=b.cap_violation,
    )
end

function evaluate_config(cfg_nt, instances; pop_size::Int, generations::Int, seed::Int)
    rows = NamedTuple[]
    for fp in instances
        push!(rows, run_instance(cfg_nt, fp; pop_size=pop_size, generations=generations, seed=seed))
    end

    gaps = [r.gap for r in rows]
    feasible_count = count(r -> r.feasible, rows)
    # Very high penalty for infeasible outcomes.
    score = mean(gaps) + 500.0 * (length(rows) - feasible_count)

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
    return @sprintf(
        "pc=%.2f pm=%.2f pls=%.2f k=%d o1x=%.2f elites=%d maxn=%d",
        cfg_nt.p_c,
        cfg_nt.p_m,
        cfg_nt.p_ls,
        cfg_nt.k,
        cfg_nt.o1xmax,
        cfg_nt.elites,
        cfg_nt.max_nurses,
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

function run_retune()
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(pwd(), "results", "retune")
    mkpath(out_dir)
    summary_file = joinpath(out_dir, "retune_summary_$(stamp).txt")

    baseline = (
        p_c=0.85,
        p_m=0.10,
        p_ls=0.15,
        k=3,
        o1xmax=0.15,
        elites=10,
        max_nurses=8,
    )

    pcs = [0.80, 0.85, 0.90, 0.95]
    pms = [0.06, 0.10, 0.14, 0.18]
    plss = [0.10, 0.15, 0.20, 0.25]
    ks = [2, 3, 4]
    o1xs = [0.15, 0.22, 0.30]
    elites = [6, 10, 14]
    maxns = [7, 8, 9, 10]

    Random.seed!(2026)
    candidates = [baseline]
    while length(candidates) < 25
        cfg = (
            p_c=rand(pcs),
            p_m=rand(pms),
            p_ls=rand(plss),
            k=rand(ks),
            o1xmax=rand(o1xs),
            elites=rand(elites),
            max_nurses=rand(maxns),
        )
        cfg in candidates || push!(candidates, cfg)
    end

    stage1_instances = filter(fp -> basename(fp) in Set(["train_0.json", "train_1.json", "train_4.json", "train_8.json"]), INSTANCE_FILES)

    stage1_rows = NamedTuple[]
    println("Stage 1: coarse search on 4 instances...")
    for (i, cfg) in enumerate(candidates)
        r = evaluate_config(cfg, stage1_instances; pop_size=90, generations=220, seed=42)
        push!(stage1_rows, r)
        println(@sprintf("  [%d/%d] score=%.2f mean_gap=%.2f%% feasible=%d/4 | %s", i, length(candidates), r.score, r.mean_gap, r.feasible_count, cfg_label(cfg)))
    end
    sort!(stage1_rows; by=r -> r.score)

    top_cfgs = [r.cfg for r in stage1_rows[1:5]]

    stage2_rows = NamedTuple[]
    println("Stage 2: validation on all 10 instances...")
    for (i, cfg) in enumerate(top_cfgs)
        r = evaluate_config(cfg, INSTANCE_FILES; pop_size=160, generations=600, seed=42)
        push!(stage2_rows, r)
        println(@sprintf("  [%d/%d] score=%.2f mean_gap=%.2f%% feasible=%d/10 | %s", i, length(top_cfgs), r.score, r.mean_gap, r.feasible_count, cfg_label(cfg)))
    end
    sort!(stage2_rows; by=r -> r.score)
    best_cfg = stage2_rows[1].cfg

    println("Final benchmark run (baseline vs best) on all instances...")
    baseline_final = evaluate_config(baseline, INSTANCE_FILES; pop_size=250, generations=1200, seed=42)
    best_final = evaluate_config(best_cfg, INSTANCE_FILES; pop_size=250, generations=1200, seed=42)

    open(summary_file, "w") do io
        println(io, "Retune Summary")
        println(io, "Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)

        print_rank(io, stage1_rows; title="Stage 1 Top (coarse)", top_k=10)
        print_rank(io, stage2_rows; title="Stage 2 Top (all instances)", top_k=5)

        println(io, "Baseline config:")
        println(io, "  ", cfg_label(baseline))
        println(io, @sprintf("  mean_gap=%.2f%% median_gap=%.2f%% worst_gap=%.2f%% feasible=%d/%d", baseline_final.mean_gap, baseline_final.median_gap, baseline_final.worst_gap, baseline_final.feasible_count, baseline_final.n))
        println(io)

        println(io, "Best tuned config:")
        println(io, "  ", cfg_label(best_cfg))
        println(io, @sprintf("  mean_gap=%.2f%% median_gap=%.2f%% worst_gap=%.2f%% feasible=%d/%d", best_final.mean_gap, best_final.median_gap, best_final.worst_gap, best_final.feasible_count, best_final.n))
        println(io)

        println(io, "Per-instance comparison (baseline -> tuned):")
        for fp in INSTANCE_FILES
            b_row = only(filter(r -> r.instance == basename(fp), baseline_final.rows))
            t_row = only(filter(r -> r.instance == basename(fp), best_final.rows))
            println(
                io,
                @sprintf(
                    "%s | gap %.2f%% -> %.2f%% | travel %.2f -> %.2f | feasible %s -> %s",
                    basename(fp),
                    b_row.gap,
                    t_row.gap,
                    b_row.travel,
                    t_row.travel,
                    string(b_row.feasible),
                    string(t_row.feasible),
                )
            )
        end
    end

    println("DONE")
    println("SUMMARY_FILE: ", summary_file)
    println("BEST_CONFIG: ", cfg_label(best_cfg))
    println(@sprintf("BASELINE_FINAL mean_gap=%.2f%% feasible=%d/%d", baseline_final.mean_gap, baseline_final.feasible_count, baseline_final.n))
    println(@sprintf("TUNED_FINAL mean_gap=%.2f%% feasible=%d/%d", best_final.mean_gap, best_final.feasible_count, best_final.n))
end

run_retune()
