
module TTNUtils
"""
Utilities for TTN project — fused-topology helpers and index mappings.

Conventions:
- `localdims` is a Vector{Int} giving the physical dimension for each original site (1-based indexing).
- `groups` is a Vector of vectors of integer site indices (1-based) describing which original sites are fused together.
- For mixed-radix packing we treat the first site in each group as the least-significant digit.

Exported functions:
- `fused_localdims(localdims, groups)` -> Vector of fused dims (product per group).
- `compress_indexset(orig, groups, localdims)` -> compress a full original index vector into fused indices per group.
- `expand_fused_indexset(fused, groups, localdims)` -> inverse mapping returning the original full index vector.
- `fuse_graph_pairs(g, groups)` -> fused graph where each group is a vertex.
- `make_fused_wrapper(f_orig, groups, localdims)` -> returns a function accepting fused indices and calling `f_orig` with expanded indices.
- `bits2decimal(v)` -> convert a vector of bits (1/2) to a decimal number between 0 and 1.
- `seed_pivots!(tci)` -> seed initial pivots for SimpleTCI based on binary representations.
- `sampled_errors(f, ttn, nsamples, bits)` -> compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits.
"""

export fused_preparations
using NamedGraphs: NamedGraph, add_edge!, vertices, NamedEdge, has_edge
include(joinpath(@__DIR__, "..", "src", "topologies.jl"))
using Graphs
using TreeTCI

"""
    fuse_graph_pairs(g::NamedGraph, groups::Vector{Vector{Int}})

Given an input graph `g` with vertices labeled 1..N and a partition `groups`
(vector of index vectors), return a new `NamedGraph` with one vertex per group
and an edge between group a and group b if any vertex in a is connected to any
vertex in b in `g`.
"""
function fuse_graph_pairs(g::NamedGraph, groups::Vector{Vector{Int}})
    m = length(groups)
    fg = NamedGraph(m)
    # For each pair of groups check if there's any edge between their members
    for a in 1:m
        for b in (a+1):m
            found = false
            for u in groups[a]
                for v in groups[b]
                    if has_edge(g, u, v) || has_edge(g, v, u)
                        add_edge!(fg, a, b)
                        found = true
                        break
                    end
                end
                if found
                    break
                end
            end
        end
    end
    return fg
end

"""
    fused_localdims(localdims::Vector{Int}, groups::Vector{Vector{Int}})

Compute the local dimensions for each fused group as the product of the
constituent local dims.
"""
function fused_localdims(localdims::Vector{Int}, groups::Vector{Vector{Int}})
    return [prod(localdims[idx] for idx in group) for group in groups]
end

"""
    compress_indexset(orig::Vector{Int}, groups::Vector{Vector{Int}}, localdims::Vector{Int})

Convert an original indexset (one integer per original site) to an indexset
for the fused groups. The mapping uses a mixed-radix ordering where the first
site in the group is the least significant digit.
"""
function compress_indexset(orig::Vector{Int}, groups::Vector{Vector{Int}}, localdims::Vector{Int})
    fused = Int[]
    for group in groups
        idx = 0
        radix = 1
        for site in group
            i = orig[site] - 1 # zero-based
            idx += i * radix
            radix *= localdims[site]
        end
        push!(fused, idx + 1)
    end
    return fused
end

"""
    expand_fused_indexset(fused::Vector{Int}, groups::Vector{Vector{Int}}, localdims::Vector{Int})

Inverse of `compress_indexset`: expands fused indices back to original per-site
indices.
"""
function expand_fused_indexset(fused::Vector{Int}, groups::Vector{Vector{Int}}, localdims::Vector{Int})
    N = sum(length(g) for g in groups)
    orig = Vector{Int}(undef, N)
    for (gi, group) in enumerate(groups)
        val = fused[gi] - 1
        for site in group
            d = localdims[site]
            orig[site] = (val % d) + 1
            val ÷= d
        end
    end
    return orig
end

"""
    make_fused_wrapper(f_orig, groups, localdims)

Return a function `f_fused` that accepts fused indexsets (one integer per
group) and calls `f_orig` with the expanded original indexset.
"""
function make_fused_wrapper(f_orig, groups::Vector{Vector{Int}}, localdims::Vector{Int})
    return function (fused_idx)
        orig = expand_fused_indexset(fused_idx, groups, localdims)
        return f_orig(orig)
    end
end

"""
    fused_preparations(
        f,
        groups::Vector{Vector{Int}},
        localdims::Vector{Int},
    )   
Prepare fused graph, fused localdims and fused function wrapper.
"""

function fused_preparations(
    f,
    groups::Vector{Vector{Int}},
    localdims::Vector{Int},
)
    """ Prepare fused graph, fused localdims and fused function wrapper."""
    baseg = BTTN(length(localdims) ÷ 2) # base graph with 2R vertices (x and y branches)
    fused_g = fuse_graph_pairs(baseg, groups)
    fused_dims = fused_localdims(localdims, groups)
    f_fused = make_fused_wrapper(f, groups, localdims)
    return fused_g, fused_dims, f_fused
end


function bits2decimal(v::AbstractVector{<:Integer})
    """Convert a vector of bits (1/2) to a decimal number between 0 and 1"""
    sum = 0.0
    for i in 1:length(v)
        sum += (v[i] - 1) * 2.0^(-i)
    end
    return sum
end

function seed_pivots!(tci)
    """ Seed initial pivots for the SimpleTCI tci based on binary representations. """

    for i in 1:10
        # Generate a random input of length equal to number of vertices
        x = rand(1:2, length(vertices(tci.g)))
        TreeTCI.addglobalpivots!(tci, [x])
    end
end

end # module
