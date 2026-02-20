"""
Mutation operators:
    Swap
    Insert
    Scramble
    Inversion
"""



abstract type Mutator end


struct SwapMutator <: Mutator end
"""
    mutate(mutator::SwapMutator, individual)

Randomly swap the position of two alleles in the `ind` candidate solution.
"""
function mutate(; mutator::SwapMutator, individual, rng)
    new_ind = copy(individual)
    n = length(new_ind)
    i,j = rand(rng, 1:n, 2)

    # If i or j hits -1, move one further
    while new_ind[i] == -1
        i = i == n ? 1 : i + 1
    end

    while new_ind[j] == -1
        j = j == n ? 1 : j + 1
    end
    
    # Ensure that i and j are not the same index
    while i == j
        j = rand(rng, 1:n)
        while new_ind[j] == -1
            j = j == n ? 1 : j + 1
        end
    end

    # Swap
    new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
    return new_ind
end