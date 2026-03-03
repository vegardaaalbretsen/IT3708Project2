# IT3708 – Project 2  
## Genetic Algorithm for Home Care Routing

This repository contains an implementation of a **Genetic Algorithm (GA)** for solving a Home Care Routing Problem (HCRP), developed as part of the course **IT3708 – Bio-Inspired Artificial Intelligence**.

The goal is to assign patients to nurses and determine visit sequences while respecting operational constraints such as:

- Nurse capacity
- Working time limits
- Patient time windows
- Service durations
- Travel times

The solution uses a permutation-based chromosome representation with route delimiters and genetic operators adapted for routing problems.

---

# Problem Representation

## Chromosome Encoding

Each individual is represented as:

```julia
Vector{Int}
```

Patients are encoded as integers `1..N`.  
Routes are separated using `-1` as a delimiter.

Example:

```julia
[4, 2, 7, -1, 1, 6, -1, 3, 5]
```

This represents three nurse routes:

- Route 1: `[4, 2, 7]`
- Route 2: `[1, 6]`
- Route 3: `[3, 5]`

---

# Genetic Algorithm Components

## Parent Selection

Implemented strategies:

- **Tournament Selection**
- **Roulette Wheel Selection** (adapted for minimization)

Tournament size controls selection pressure.

---

## Crossover Operators

Implemented:

- **O1X (Order 1 Crossover)**

Key properties:

- Permutation-safe
- Preserves relative order
- Supports delimiter handling (`-1`)

Implementation strategy:

1. Remove delimiters  
2. Perform O1X on the pure permutation  
3. Reinsert delimiters according to original route structure  

Window size is controlled via:

```julia
struct O1XCrossover
    min_frac::Float64
    max_frac::Float64
end
```

---

## Mutation Operators

Implemented mutation types:

- Swap
- Insert
- Scramble
- Inversion

All mutation operators preserve permutation validity.

---

## Survivor Selection

Implemented:

- **Elitist Generational Replacement**

Keeps `num_elites` best individuals from the parent population and fills remaining slots with the best offspring.

---

# Fitness Function

The fitness function evaluates:

- Total travel time
- Capacity violations
- Return time violations
- Waiting time
- Treatment window violations

Constraints are handled as **soft penalties**:

```julia
fitness = travel_cost
        + α * capacity_violation
        + β * overtime_violation
        + γ * time_window_violation
```

Lower fitness values are better.

---

# Testing

The repository includes unit tests for:

- Mutation operators
- Crossover operators
- Selection operators
- Tournament selection pressure

Run tests:

```julia
pkg> activate .
pkg> test
```

---

# Run the GA (shared script)

Use `run_ga.jl` as the shared, editable experiment file.

How to use it:

1. Open `run_ga.jl`.
2. Edit the variables at the top (instance, population size, generations, probabilities, seed, etc.).
3. Run:

   ```bash
   julia --project=. run_ga.jl
   ```

   For better performance, multithreading is recommended:

   ```bash
   julia -t auto --project=. run_ga.jl
   ```

   This enables Julia to use multiple CPU threads for population fitness evaluation.

Notes:

- Set `OUTPUT_FILE = nothing` in `run_ga.jl` to disable writing a solution file.
- The script builds `GAConfig` and runs `GA(instance_file, config; rng=StableRNG(seed))`.
- Default instance is `train/train_0.json`.
- `MAX_NURSES` is an upper bound for route slots (`nothing` reads `nbr_nurses` from instance).
- `MUTATOR = :swap_any` lets the GA move `-1` separators, so it can decide how many nurses are actively used.

---

# Project Structure

Includes `run_ga.jl` at the repository root for shared GA runs.

```
IT3708Project2/
│
├── src/
│   ├── HomeCareGA.jl
│   ├── ga_config.jl
│   └── operators/
│       ├── crossover.jl
│       ├── mutation.jl
│       └── selection.jl
│
├── test/
│   ├── crossover.jl
│   ├── mutation.jl
│   ├── selection.jl
│   └── runtests.jl
│
└── Project.toml
```

---

# Performance Considerations

The implementation focuses on:

- Type stability
- Preallocated buffers for crossover
- Avoiding unnecessary allocations
- In-place operations where appropriate
- `@inbounds` in performance-critical loops

---

# Requirements

- Julia ≥ 1.11
- StableRNGs
- StatsBase

Install dependencies:

```julia
pkg> instantiate
```

---

# Author

Vegard Aa Albretsen, Erlend Vitsø, Olav Aspem
NTNU – IT3708 Bio-Inspired Artificial Intelligence
