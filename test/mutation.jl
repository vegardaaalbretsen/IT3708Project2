using Test

using StableRNGs
include("../src/operators/mutation.jl") 

myrng = StableRNG(123)

"""
Mutation operators to be implemented, and therefore tested:
    Swap
    Insert
    Scramble
    Inversion
"""

@info("Testing mutators")
@testset verbose=true "Mutator Test" begin
    @testset "Swap" begin
        """
        Tests that the swap mutator:
            Does not change the length of the individual
            Does not change the unique values in the individual
            Does not change number of nurse-tours (by counting number of split symbols, -1)
            Has placed the i-th allele in x at the j-th position in c and opposite
        """
        M = SwapMutator()
        x = [1,2,3,-1,4,5,-1,6,7]
        c = mutate(mutator=M,individual=x,rng=myrng)
        
        @test length(x) == length(c)
        @test sort(filter(!=(-1), c)) == sort(filter(!=(-1), x))
        @test findall(==(-1), c) == findall(==(-1), x)
        changed = [i for i in eachindex(x) if x[i] != c[i] && x[i] != -1]
        @test length(changed) == 2

        i,j = changed
        @test c[i] == x[j]
        @test x[i] == c[j]
    end

    @testset "Insert" begin
        
    end

    @testset "Scramble" begin
        
    end

    @testset "Inversion" begin
        
    end
end