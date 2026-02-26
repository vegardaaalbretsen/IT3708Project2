using Test
using StableRNGs
using HomeCareGA

myrng = StableRNG(123)

function _route_lengths(individual::Vector{Int})
    lens = Int[]
    cur = 0
    for g in individual
        if g == -1
            push!(lens, cur)
            cur = 0
        else
            cur += 1
        end
    end
    push!(lens, cur)
    return lens
end

@inline _active_routes(individual::Vector{Int}) = count(>(0), _route_lengths(individual))

"""
Generator operators to be implemented, and therefore tested:
    Random
    Heuristic?
"""

@info("Testing generators")
@testset verbose=true "Generator Test" begin
    @testset "RandomGenerator" begin
        """
        Tests that the random generator:
            Returns a population of the correct size
            Each individual has the correct chromosome length
            Each individual contains all jobs as a permutation
            Each individual contains the correct number of separators (-1)
            Deterministic with same seed
        """
        num_jobs = 9
        num_routes = 4
        pop_size = 5
        chrom_length = num_jobs + (num_routes - 1)  # Expected length
        
        gen = RandomGenerator(num_jobs=num_jobs, num_routes=num_routes)
        population = generate(gen, pop_size=pop_size, rng=myrng)
        
        @testset "population size and chromosome length" begin
            @test length(population) == pop_size
            for individual in population
                @test length(individual) == chrom_length
            end
        end
        
        @testset "job permutations and separators" begin
            for individual in population
                # Extract jobs (all non-separator elements)
                jobs = filter(!=(-1), individual)
                
                # Should have exactly num_jobs elements
                @test length(jobs) == num_jobs
                
                # Should be a permutation of 1:num_jobs
                @test sort(jobs) == 1:num_jobs
                
                # Should have exactly num_separators
                num_separators = chrom_length - num_jobs
                @test count(==(-1), individual) == num_separators
            end
        end
        
        @testset "deterministic with same seed" begin
            rng1 = StableRNG(456)
            rng2 = StableRNG(456)
            
            pop1 = generate(gen, pop_size=3, rng=rng1)
            pop2 = generate(gen, pop_size=3, rng=rng2)
            
            @test pop1 == pop2
        end
        
        @testset "different seeds produce different populations" begin
            rng1 = StableRNG(111)
            rng2 = StableRNG(222)
            
            pop1 = generate(gen, pop_size=1, rng=rng1)
            pop2 = generate(gen, pop_size=1, rng=rng2)
            
            @test pop1 != pop2
        end
    end

    @testset "canonicalize_chromosome" begin
        raw = [3, -1, -1, 2, 1, -1]
        canon = canonicalize_chromosome(raw)

        @test canon == [2, 1, -1, 3, -1, -1]
        @test canonicalize_chromosome(canon) == canon
        @test count(==(-1), canon) == count(==(-1), raw)
        @test sort(filter(!=(-1), canon)) == sort(filter(!=(-1), raw))
    end

    @testset "SweepTWGenerator" begin
        instance_path = normpath(joinpath(@__DIR__, "..", "train", "train_1.json"))
        gen = SweepTWGenerator(instance_path; num_routes=25, shuffle_route_sizes=false)

        pop_size = 5
        chrom_length = gen.num_jobs + gen.num_routes - 1
        population = generate(gen; pop_size=pop_size, rng=StableRNG(987))

        @test length(population) == pop_size
        for individual in population
            @test length(individual) == chrom_length
            @test count(==(-1), individual) == gen.num_routes - 1
            @test sort(filter(!=(-1), individual)) == collect(1:gen.num_jobs)
        end

        rng1 = StableRNG(555)
        rng2 = StableRNG(555)
        pop1 = generate(gen; pop_size=3, rng=rng1)
        pop2 = generate(gen; pop_size=3, rng=rng2)
        @test pop1 == pop2
    end

    @testset "SweepTWGenerator can initialize fewer active routes than num_routes" begin
        instance_path = normpath(joinpath(@__DIR__, "..", "train", "train_1.json"))
        gen = SweepTWGenerator(
            instance_path;
            num_routes=10,
            shuffle_route_sizes=true,
            allow_empty_routes=true
        )
        population = generate(gen; pop_size=20, rng=StableRNG(2026))

        has_empty_route = any(any(==(0), _route_lengths(ind)) for ind in population)
        @test has_empty_route
    end

    @testset "min_active_routes is enforced by generators" begin
        rgen = RandomGenerator(num_jobs=12, num_routes=8, min_active_routes=6)
        rpop = generate(rgen; pop_size=30, rng=StableRNG(7))
        @test all(_active_routes(ind) >= 6 for ind in rpop)

        instance_path = normpath(joinpath(@__DIR__, "..", "train", "train_1.json"))
        sgen = SweepTWGenerator(
            instance_path;
            num_routes=12,
            min_active_routes=9,
            allow_empty_routes=true
        )
        spop = generate(sgen; pop_size=30, rng=StableRNG(8))
        @test all(_active_routes(ind) >= 9 for ind in spop)
    end

    @testset "enforce_min_active_routes repairs underfilled chromosomes" begin
        raw = [1, 2, 3, 4, -1, -1, -1]
        repaired = enforce_min_active_routes(raw, 3)

        @test _active_routes(repaired) == 3
        @test count(==(-1), repaired) == count(==(-1), raw)
        @test sort(filter(!=(-1), repaired)) == sort(filter(!=(-1), raw))
    end
end
