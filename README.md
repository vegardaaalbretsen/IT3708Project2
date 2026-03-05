# Home Care GA (Julia)

Genetic algorithm solution for IT3708 Project 2 (home-care VRP with time windows).

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run one instance

```bash
julia --project=. run.jl --instance resources/train_0.json --time-limit 60 --population 80 --generations 1000
```

Use multiple CPU threads (recommended):

```bash
julia -t auto --project=. run.jl --instance resources/train_0.json --time-limit 60 --population 80 --generations 1000
```

## Run all training instances

```bash
julia --project=. run.jl --run-all --time-limit 60 --population 80 --generations 1000
```

## Outputs

Saved under `results/<instance>/`:

- `best_solution.txt` (formatted solution output)
- `best_solution.png` (plot with one color per route)
- `metrics.csv` (core metrics per generation: best/median/worst feasible travel + feasibility ratio)
- `fitness_spread.png` (high-resolution plot of best/median/worst travel per generation)

## Code Structure

- `src/HomeCareGA.jl`: Module entrypoint + includes/exports.
- `src/model.jl`: Core data types (`Instance`, `Patient`, `Candidate`, `GAConfig`, ...).
- `src/instance_io.jl`: Loading instance JSON files.
- `src/evaluation.jl`: Route/candidate evaluation and feasibility scoring.
- `src/repair.jl`: Repair/insertion utilities to keep solutions valid.
- `src/operators.jl`: Construction, crossover, mutation, and local improvement operators.
- `src/ga.jl`: GA loop (selection, elitism, population evolution).
- `src/reporting.jl`: Text report formatting.
- `src/plotting.jl`: Plot rendering for best solution.
- `src/config.jl`: Default GA configuration.
- `src/api.jl`: High-level `solve_instance` orchestration.
