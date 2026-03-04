using Random
using Plots
using Graphs
using TreeTCI
using NamedGraphs

function sampled_error(f, ttn, nsamples, bits, d)
    """Compute sampled errors between function f and ttn approximation over nsamples random inputs."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end
    error_l1 = 0.0
    for _ in 1:nsamples
        x = rand(1:2, d * bits)
        approx = TreeTCI.evaluate(eval_ttn, x)
        error_l1 += abs(f(x) - approx)
    end
    return error_l1 / nsamples
end

function bits2decimal(v::AbstractVector{<:Integer})
    """Convert a vector of bits (1/2) to a decimal number between 0 and 1"""
    sum = 0.0
    for i in 1:length(v)
        sum += (v[i] - 1) * 2.0^(-i)
    end
    return sum
end

function seed_pivots!(tci, npivots::Int64, f)
    """ Seed initial pivots for the SimpleTCI tci based on binary representations. """
    count = 0
    while count < npivots
        # Generate a random input of length equal to number of vertices
        x = rand(1:2, length(vertices(tci.g)))
        if abs(f(x)) > 0.5
            TreeTCI.addglobalpivots!(tci, [x])
            count += 1
        else
            continue
        end
    end
end

##############################################
# Classical topologies
##############################################

"""
    QTT_Block(R::Int)

Returns a sequential tensor train.

Generate a NamedGraph representing a tensor train topology for 2D functions,
with `R` bits precision per dimension (total 2R vertices).
All x coordinates first, then all y coordinates: x₁ — x₂ — ... — x_R — y₁ — y₂ — ... — y_R
"""
function QTT_Block(R::Int, d::Int=2)
    @assert R ≥ 1 "R must be at least 1"
    @assert d ≥ 1 "dimension d must be at least 1"
    N = d * R
    g = NamedGraph(N)
    for i in 1:(N-1)
        add_edge!(g, i, i + 1)
    end
    return g
end

"""
    QTT_Alt(R::Int)

Returns an interleaved tensor train.

Generate a NamedGraph representing a tensor train topology for 2D functions,
with `R` bits precision per dimension (total 2R vertices).
Vertices alternate between x and y coordinates: x₁ — y₁ — x₂ — y₂ — ... — x_R — y_R
"""
function QTT_Alt(R::Int, d::Int=2)
    @assert R ≥ 1 "R must be at least 1"
    @assert d ≥ 1 "dimension d must be at least 1"
    N = d * R
    g = NamedGraph(N)
    # connect dims across each position j: dim1_j - dim2_j - ... - dimd_j
    for j in 1:R
        for dim in 1:(d-1)
            add_edge!(g, (dim - 1) * R + j, dim * R + j)
        end
        # connect last dim at j to first dim at j+1 to create a chain along j
        if j < R
            add_edge!(g, (d - 1) * R + j, 1 + j)
        end
    end
    return g
end

"""
    CTTN(R::Int)

Returns a Comb TTN.

Generate a NamedGraph representing a quantics Comb TTN topology for 2D functions,
with `R` bits precision per dimension (total 2R vertices).

Structure:
  x-chain: 1 — 2 — 3 — ... — R
  y-chain: (R+1) — (R+2) — ... — (2R)
  Link: 1 — (R+1)
"""
function CTTN(R::Int, d::Int=2)
    @assert R ≥ 2 "R must be at least 2"
    @assert d ≥ 1 "dimension d must be at least 1"
    N = d * R
    g = NamedGraph(N)

    # build a linear chain for each dimension
    for dim in 0:(d-1)
        base = dim * R
        for i in 1:(R-1)
            add_edge!(g, base + i, base + i + 1)
        end
    end

    # link the root of the first chain to the roots of the other chains
    for dim in 1:(d-1)
        add_edge!(g, 1, dim * R + 1)
    end

    return g
end

"""
    BTTN(R::Int)

Returns a Binary TTN.

Generate a NamedGraph representing a quantics Binary TTN topology for 2D functions,
with `R` bits precision per dimension (total 2R vertices).

Structure:
  - x-representation: binary tree of depth ≈ log2(R)
  - y-representation: same structure shifted by R
  - Link between roots: 1 — (R+1)
"""
function BTTN(R::Int, d::Int=2)
    @assert R ≥ 2 "R must be at least 2"
    @assert d ≥ 1 "dimension d must be at least 1"
    N = d * R
    g = NamedGraph(N)

    # For each dimension build a binary tree on the block 1..R with offset
    for dim in 0:(d-1)
        base = dim * R
        for i in 1:R
            left = 2i
            right = 2i + 1
            if left ≤ R
                add_edge!(g, base + i, base + left)
            end
            if right ≤ R
                add_edge!(g, base + i, base + right)
            end
        end
    end

    # connect the roots of all dimensions to the first root (1)
    for dim in 1:(d-1)
        add_edge!(g, 1, dim * R + 1)
    end

    return g
end

""" 
    ITTN(R::Int, d::Int=2)

Returns an Interleaved TTN.

Generate a NamedGraph representing a quantics Interleaved TTN topology for 2D functions,
with `R` bits precision per dimension (total 2R vertices).
y1  y2  y3  ...  yR
 |   |   |        |
x1--x2--x3-- ... --xR
"""

function ITTN(R::Int, d::Int=2)
    @assert R ≥ 3 "R must be at least 3"
    @assert d == 2 "This ITTN implementation assumes d=2"

    N = 2R
    g = NamedGraph(N)

    # x-chain
    for i in 1:(R-1)
        add_edge!(g, i, i + 1)
    end

    # vertical links (each y_i attached to x_i)
    for i in 1:R
        add_edge!(g, i, R + i)
    end

    return g
end

