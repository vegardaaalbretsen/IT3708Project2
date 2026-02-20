# TODO

"""
Crossover operators:
    O1X
    PMX
    Edge recombination

    Another method also on p. 82 in lecture 6
"""


abstract type Crossover end

struct O1XCrossover <: Crossover end

function crossover(::O1XCrossover, parent1::Vector{Int}, parent2::Vector{Int})
    # O1X crossover operator
    n = length(parent1)
    child = fill(0, n)

    # Randomly select two crossover points
    point1, point2 = sort(rand(1:n, 2))

    # Copy the segment from parent1 to child
    child[point1:point2] .= parent1[point1:point2]

    # Fill the remaining positions with genes from parent2 in order
    pos = 1
    for gene in parent2
        if gene ∉ child
            while child[pos] != 0
                pos += 1
            end
            child[pos] = gene
        end
    end

    return child
end