@inline function _route_lt(a::Vector{Int}, b::Vector{Int})
    @inbounds for i in 1:min(length(a), length(b))
        if a[i] != b[i]
            return a[i] < b[i]
        end
    end
    return length(a) < length(b)
end

"""
Return a deterministic representation of a route-delimited chromosome.

Keeps the same number of route separators (`-1`) while:
- removing empty routes from the middle,
- sorting non-empty routes lexicographically,
- padding empty routes at the end.
"""
function canonicalize_chromosome(individual::Vector{Int})::Vector{Int}
    isempty(individual) && return Int[]

    num_routes = count(==(-1), individual) + 1
    routes = Vector{Vector{Int}}()
    route = Int[]

    @inbounds for gene in individual
        if gene == -1
            push!(routes, route)
            route = Int[]
        else
            push!(route, gene)
        end
    end
    push!(routes, route)

    non_empty = [r for r in routes if !isempty(r)]
    sort!(non_empty; lt=_route_lt)

    while length(non_empty) < num_routes
        push!(non_empty, Int[])
    end

    out = Int[]
    sizehint!(out, length(individual))
    @inbounds for r in 1:num_routes
        append!(out, non_empty[r])
        if r < num_routes
            push!(out, -1)
        end
    end

    return out
end
