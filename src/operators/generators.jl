abstract type Generator end

"""
Population generators:
    Random
    Heuristic?
"""
Base.@kwdef struct RandomGenerator <: Generator
    num_jobs::Int      # Number of patients/jobs to visit
    num_routes::Int    # Number of nurses/routes
end

"""
Generate a random population where each individual is a permutation of jobs (1 to num_jobs)
with separators (-1) dividing the different routes.

Chromosome length = num_jobs + (num_routes - 1) separators

Returns: Vector{Vector{Int}} - Population of chromosomes
"""
function generate(
    method::RandomGenerator;
    pop_size::Int,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Vector{Vector{Int}}
    
    population = Vector{Vector{Int}}(undef, pop_size)
    num_separators = method.num_routes - 1
    chrom_length = method.num_jobs + num_separators
    
    for i in 1:pop_size
        # Create a random permutation of jobs
        jobs = shuffle(rng, 1:method.num_jobs)
        
        # Create separators
        separators = fill(-1, num_separators)
        
        # Combine jobs and separators, then shuffle to get random placement
        chromosome = vcat(jobs, separators)
        shuffle!(rng, chromosome)
        
        population[i] = chromosome
    end
    
    return population
end