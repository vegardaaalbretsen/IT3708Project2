"""
Crossover operators:
    O1X
    PMX
    Edge recombination

    Another method also on p. 82 in lecture 6
"""


abstract type Crossover end

struct O1XCrossover <: Crossover
    min_frac::Float64
    max_frac::Float64
end

function crossover(op::O1XCrossover, parent1::Vector{Int}, parent2::Vector{Int}, rng::AbstractRNG)
    # O1X crossover operator
    if has_delims(parent1) || has_delims(parent2)
        lens1 = route_lengths(parent1)  
        lens2 = route_lengths(parent2)  
        p1 = strip_delims(parent1)
        p2 = strip_delims(parent2)

        cperm1, cperm2 = crossover(op, p1, p2, rng)  # calls the permutation version
        return insert_delims(cperm1, lens1), insert_delims(cperm2, lens2)
    end

    N = length(parent1)
    a, b = sort(rand(rng, 1:N, 2))
    c1 = O1X_core(parent1=parent1, parent2=parent2, a=a, b=b)
    c2 = O1X_core(parent1=parent2, parent2=parent1, a=a, b=b)
    return c1, c2
end

# One deterministic for testing
function crossover(op::O1XCrossover, parent1::Vector{Int}, parent2::Vector{Int}; a::Int, b::Int)
    # O1X crossover operator
    N = length(parent1)
    @assert length(parent2) == N
    if a > b
        a, b = b, a
    end
    c1 = O1X_core(parent1=parent1, parent2=parent2, a=a, b=b)
    c2 = O1X_core(parent1=parent2, parent2=parent1, a=a, b=b)
    return c1, c2 
end


function O1X_core(; parent1::Vector{Int}, parent2::Vector{Int}, a::Int, b::Int)
    N = length(parent1)
    @assert length(parent2) == N
    @assert 1 ≤ a ≤ b ≤ N
    child = similar(parent1)
    fill!(child, 0)

    used = falses(N)

    # Copy the segment from parent1 to child
    @inbounds for i in a:b
        g=parent1[i]
        child[i] = g
        used[Int(g)] = true
    end

    # Fill the remaining positions with genes from parent2 in order
    writepos = (b % N) + 1
    @inbounds for k in 0:N-1
        idx = ((b + k - 1) % N) + 1
        g = parent2[idx]
        gi = Int(g)

        if !used[gi]
            # Advance writepos to next empty slot
            while child[writepos] != 0
                writepos = (writepos % N) + 1
            end
            child[writepos] = g
            used[gi] = true
        end
    end

    return child
end



# ---- delimiter helpers ----
@inline has_delims(x::Vector{Int}) = any(==(-1), x)

strip_delims(x::Vector{Int}) = [g for g in x if g != -1]

function route_lengths(x::Vector{Int})
    lens = Int[]
    cur = 0
    @inbounds for g in x
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

function insert_delims(perm::Vector{Int}, lens::Vector{Int})
    @assert sum(lens) == length(perm)
    out = Vector{Int}(undef, length(perm) + (length(lens)-1))
    p = 1
    o = 1
    for r in 1:length(lens)
        L = lens[r]
        @inbounds for _ in 1:L
            out[o] = perm[p]
            p += 1
            o += 1
        end
        if r != length(lens)
            out[o] = -1
            o += 1
        end
    end
    return out
end