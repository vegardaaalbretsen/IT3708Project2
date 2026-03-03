"""
Mutation operators:
    Swap
    Insert
    Scramble
    Inversion
"""



abstract type Mutator end


struct SwapMutator <: Mutator end
struct SwapAnyMutator <: Mutator end
"""
    mutate(; mutator::Mutator, individual, rng)

Public mutation entrypoint. Dispatches by `mutator` type.
"""
function mutate(; mutator::Mutator, individual, rng)
    return _mutate(mutator, individual, rng)
end

"""
    _mutate(mutator::SwapMutator, individual, rng)

Randomly swap two non-separator alleles.
Keeps separator (`-1`) positions fixed.
"""
function _mutate(mutator::SwapMutator, individual, rng)
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

"""
    _mutate(mutator::SwapAnyMutator, individual, rng)

Randomly swap two positions, including route separators (-1).
This lets the GA change route boundaries and therefore the number
of active (non-empty) routes during evolution.
"""
function _mutate(mutator::SwapAnyMutator, individual, rng)
    new_ind = copy(individual)
    n = length(new_ind)
    i, j = rand(rng, 1:n, 2)
    while i == j
        j = rand(rng, 1:n)
    end
    new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
    return new_ind
end
