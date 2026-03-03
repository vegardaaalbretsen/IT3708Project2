"""
Comparison function for sorting routes
    
Uses the ID of the first patient 
as an "anchor" to ensure deterministic route ordering. This prevents 
identical sets of routes from being treated as different chromosomes 
while preserving the internal sequence optimized by the GA.
"""
@inline function _route_lt(a::Vector{Int}, b::Vector{Int})
    return a[1] < b[1]
end

function _split_routes(individual::Vector{Int})::Vector{Vector{Int}}
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
    return routes
end

@inline function active_route_count(individual::Vector{Int})::Int
    routes = _split_routes(individual)
    return count(r -> !isempty(r), routes)
end

function _join_routes(routes::Vector{Vector{Int}})::Vector{Int}
    num_routes = length(routes)
    out = Int[]
    sizehint!(out, sum(length, routes) + max(0, num_routes - 1))
    @inbounds for r in 1:num_routes
        append!(out, routes[r])
        if r < num_routes
            push!(out, -1)
        end
    end
    return out
end

"""
Ensure at least `min_active_routes` non-empty routes while preserving:
- same patients,
- same number of separators (`-1`),
- deterministic canonical ordering.
"""
function enforce_min_active_routes(individual::Vector{Int}, min_active_routes::Int)::Vector{Int}
    min_active_routes >= 1 || error("min_active_routes must be >= 1.")
    isempty(individual) && return Int[]

    canon = canonicalize_chromosome(individual)
    num_routes = count(==(-1), canon) + 1
    num_jobs = count(!=(-1), canon)

    min_active_routes <= num_routes || error("min_active_routes cannot exceed number of route slots.")
    min_active_routes <= num_jobs || error("min_active_routes cannot exceed number of jobs.")

    routes = _split_routes(canon)
    non_empty = [r for r in routes if !isempty(r)]

    while length(non_empty) < min_active_routes
        # Split one customer from the currently longest route into a new singleton route.
        lens = length.(non_empty)
        idx = argmax(lens)
        length(non_empty[idx]) > 1 || error("Cannot enforce min_active_routes with current chromosome.")
        moved = pop!(non_empty[idx])
        push!(non_empty, [moved])
    end

    sort!(non_empty; lt=_route_lt)
    while length(non_empty) < num_routes
        push!(non_empty, Int[])
    end

    return _join_routes(non_empty)
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


"""
[Deprecated]
@inline function _route_lt(a::Vector{Int}, b::Vector{Int})
    @inbounds for i in 1:min(length(a), length(b))
        if a[i] != b[i]
            return a[i] < b[i]
        end
    end
    return length(a) < length(b)
end
"""
