using Test

using StableRNGs
include("../src/operators/crossover.jl") 

myrng = StableRNG(123)

"""
Crossover operators to be implemented, therefore tested:
    O1X
    PMX
    Edge recombination

    Another method also on p. 82 in lecture 6
"""

@info("Testing crossovers")
@testset verbose=true "Crossoverr Test" begin
    @testset "O1X" begin
        # Should:
        # 1 Choose arbitrary part from first parent
        # 2 Copy this part to the first child
        # 3 Copy the number s that are not in the first part, to the first child
        #     Starting right from cut point of the copied part
        #     usin the order of the second parent, and
        #     wrapping around the end of the list
        
        # 4 Analogous for the second child, but with the roles of the parents reversed
    end
end