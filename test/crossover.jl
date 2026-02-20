using Test
using StableRNGs
using HomeCareGA

myrng = StableRNG(123)

"""
Crossover operators to be implemented, therefore tested:
    O1X
    PMX
    Edge recombination

    Another method also on p. 82 in lecture 6
"""

@info("Testing crossovers")
@testset verbose=true "Crossover Test" begin
    @testset "OX1" begin
    op = O1XCrossover(0.05, 0.30)
    p1 = Int[1,2,3,4,5,6,7,8]
    p2 = Int[4,3,2,1,8,7,6,5] 

        @testset "deterministic basics" begin
            a, b = Int(3), Int(6)
            c1, c2 = crossover(op, p1, p2; a, b)

            @test length(c1) == length(p1)
            @test sort(c1) == sort(p1)
            @test c1[a:b] == p1[a:b]

            @test length(c2) == length(p2)
            @test sort(c2) == sort(p2)
            @test c2[a:b] == p2[a:b]
        end

        @testset "bad cutpoints" begin
            c1a, c2a = crossover(op, p1, p2; a=6, b=3)
            c1b, c2b = crossover(op, p1, p2; a=3, b=6)
            @test c1a == c1b
            @test c2a == c2b
        end

        @testset "random invariants (seeded)" begin
            rng = StableRNG(123)
            for _ in 1:15
                c1, c2 = crossover(op, p1, p2, rng)
                @test sort(c1) == sort(p1)
                @test sort(c2) == sort(p2)
            end
        end
    end
end