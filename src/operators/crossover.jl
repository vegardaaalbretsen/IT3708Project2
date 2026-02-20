using Random: AbstractRNG, rand

abstract type Crossover end



Base.@kwdef struct O1XCrossover <: Crossover
    min_frac::Float64 = 0.05
    max_frac::Float64 = 0.30
end
"""
    O1XCrossover(min_frac, max_frac)

    Order-1 crossover operator (O1X). Preserves the relative order of elements from both parents.
    - min_frac: Minimum fraction of chromosome length for crossover segment
    - max_frac: Maximum fraction of chromosome length for crossover segment
"""
# -------------------------
# Public crossover
# -------------------------
function crossover(op::O1XCrossover,
                   parent1::Vector{Int},
                   parent2::Vector{Int},
                   rng::AbstractRNG)

    # If -1 encoding, strip -> crossover on permutation -> reinsert (pattern per parent)
    if has_delims(parent1) || has_delims(parent2)
        lens1 = route_lengths(parent1)
        lens2 = route_lengths(parent2)

        p1 = strip_delims(parent1)
        p2 = strip_delims(parent2)

        cperm1, cperm2 = crossover_perm(op, p1, p2, rng)

        return insert_delims(cperm1, lens1), insert_delims(cperm2, lens2)
    end

    # Otherwise pure permutation crossover
    return crossover_perm(op, parent1, parent2, rng)
end

# Deterministic (for tests): pure permutation only
function crossover(op::O1XCrossover,
                   parent1::Vector{Int},
                   parent2::Vector{Int};
                   a::Int, b::Int)
    N = length(parent1)
    @assert length(parent2) == N
    if a > b
        a, b = b, a
    end
    return crossover_perm(op, parent1, parent2, a, b)
end

# -------------------------
# Permutation crossover (internal)
# -------------------------
function crossover_perm(op::O1XCrossover,
                        parent1::Vector{Int},
                        parent2::Vector{Int},
                        rng::AbstractRNG)
    N = length(parent1)
    @assert length(parent2) == N

    a, b = choose_o1x_cuts(op, N, rng)

    child1 = Vector{Int}(undef, N)
    child2 = Vector{Int}(undef, N)
    used1  = Vector{Bool}(undef, N)
    used2  = Vector{Bool}(undef, N)

    o1x_core!(child1, used1, parent1, parent2, a, b)
    o1x_core!(child2, used2, parent2, parent1, a, b)

    return child1, child2
end

function crossover_perm(op::O1XCrossover,
                        parent1::Vector{Int},
                        parent2::Vector{Int},
                        a::Int, b::Int)
    N = length(parent1)
    @assert length(parent2) == N
    @assert 1 ≤ a ≤ b ≤ N

    child1 = Vector{Int}(undef, N)
    child2 = Vector{Int}(undef, N)
    used1  = Vector{Bool}(undef, N)
    used2  = Vector{Bool}(undef, N)

    o1x_core!(child1, used1, parent1, parent2, a, b)
    o1x_core!(child2, used2, parent2, parent1, a, b)

    return child1, child2
end

# -------------------------
# O1X core (allocation-free)
# Assumes genes are 1..N (after stripping -1)
# -------------------------
function o1x_core!(child::Vector{Int},
                   used::Vector{Bool},
                   parent1::Vector{Int},
                   parent2::Vector{Int},
                   a::Int, b::Int)
    N = length(parent1)
    @assert length(parent2) == N
    @assert length(child) == N
    @assert length(used) == N
    @assert 1 ≤ a ≤ b ≤ N

    # If your genes are always 1..N, keep this assert (cheap and prevents crashes)
    @assert minimum(parent1) ≥ 1 && maximum(parent1) ≤ N
    @assert minimum(parent2) ≥ 1 && maximum(parent2) ≤ N

    fill!(child, 0)
    fill!(used, false)

    @inbounds for i in a:b
        g = parent1[i]
        child[i] = g
        used[g] = true
    end

    writepos = (b % N) + 1

    @inbounds for k in 0:N-1
        idx = (b + k) % N + 1
        g = parent2[idx]
        if !used[g]
            while child[writepos] != 0
                writepos = (writepos % N) + 1
            end
            child[writepos] = g
            used[g] = true
        end
    end

    return child
end

# -------------------------
# Cut selection using min/max frac
# -------------------------
function choose_o1x_cuts(op::O1XCrossover, N::Int, rng::AbstractRNG)
    @assert 0 < op.min_frac ≤ op.max_frac ≤ 1

    wmin = max(2, floor(Int, op.min_frac * N))
    wmax = max(wmin, floor(Int, op.max_frac * N))
    wmax = min(wmax, N)

    w = rand(rng, wmin:wmax)
    a = rand(rng, 1:(N - w + 1))
    b = a + w - 1
    return a, b
end

# -------------------------
# -1 delimiter helpers
# -------------------------
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