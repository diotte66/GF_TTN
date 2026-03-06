using Random
using Plots
using Graphs
using TreeTCI
using NamedGraphs

function sampled_error(f, ttn, nsamples::Int64, bits::Int64)
    """ Compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end
    error_l1 = 0.0
    for _ in 1:nsamples
        x = rand(1:2, bits)
        approx = TreeTCI.evaluate(eval_ttn, x)
        err = abs(f(x) - approx)
        error_l1 += err
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

# =======================================================
# Kadanoff-Baym Green's function on the L-shaped contour
# -------------------------------------------------------

function plot_KB_greens(; A=30.0, B=0.0, tmax=1.0, beta=1.0, N=400)
    """
    Plot G_k^0(z,z') on the Kadanoff-Baym L-shaped contour.
    Contour ordering: C1 (0->tmax) ≺ C2 (tmax->0) ≺ C3 (0->-ibeta)
    θ_C(z,z') = 1 if z ≻ z' (z comes LATER in contour ordering), 0 otherwise.

    G^0_k(z,z') = -i[θ_C(z,z') - f_T] * exp(-i*A*(t-t')) * exp(-B*|t-t'|)
    where the exponential uses the real-time difference for real-time components,
    and imaginary-time difference for Matsubara components.
    """

    f_T(ε) = 0
    fT0 = f_T(0.0)  # epsilon_k = 0 for simplicity

    # Contour index: C1=1, C2=2, C3=3
    # Within each segment, contour position increases with the segment parameter
    # C1: position increases with t (0 -> tmax)
    # C2: position increases as t decreases (tmax -> 0), so index N+1..2N corresponds to tmax..0
    # C3: position increases with tau (0 -> beta)

    # θ_C(z, z') = 1 if z ≻ z' i.e. z comes later in contour
    function theta_C(seg1, idx1, seg2, idx2)
        if seg1 > seg2
            return 1.0  # z on later segment => z ≻ z'
        elseif seg1 < seg2
            return 0.0
        else
            # same segment: compare position within segment
            # C1: larger index = later in contour
            # C2: larger index = later in contour (since C2 goes tmax->0, idx N+1 is tmax, 2N is t=0)
            # C3: larger index = later in contour
            return idx1 > idx2 ? 1.0 : 0.0
        end
    end

    # The full propagator: G = -i * [θ_C - f_T] * kernel
    # kernel depends on the block:
    #   real-real:  exp(-i*A*(t1-t2)) * exp(-B*|t1-t2|)
    #   real-imag:  exp(-i*A*t1) * exp(-A*tau2)          (integral along mixed contour)
    #   imag-real:  exp(+i*A*t2) * exp(-A*(beta-tau1))   (conjugate)
    #   imag-imag:  exp(-A*(tau1-tau2))

    function kernel(seg1, t1, tau1, seg2, t2, tau2)
        if seg1 <= 2 && seg2 <= 2
            # real-real block
            dt = t1 - t2
            return exp(-1im * A * dt) * exp(-B * abs(dt))
        elseif seg1 <= 2 && seg2 == 3
            # G^⌐: real time z1, imaginary time z2
            # integral from z2 (imaginary) to z1 (real) along contour
            return exp(-1im * A * t1) * exp(-B * t1) * exp(-A * tau2)
        elseif seg1 == 3 && seg2 <= 2
            # G^⌏: imaginary z1, real z2
            return exp(1im * A * t2) * exp(-B * t2) * exp(-A * (beta - tau1))
        else
            # Matsubara-Matsubara
            dtau = tau1 - tau2
            return exp(-A * dtau) * exp(-B * abs(dtau))
        end
    end

    # Build full matrix
    M = 3 * N
    Z = zeros(ComplexF64, M, M)

    ts = range(0, tmax, length=N)
    taus = range(0, beta, length=N)

    for i in 1:M
        for j in 1:M
            # Determine segment and local index
            seg_i = i <= N ? 1 : (i <= 2N ? 2 : 3)
            seg_j = j <= N ? 1 : (j <= 2N ? 2 : 3)

            loc_i = seg_i == 1 ? i : (seg_i == 2 ? i - N : i - 2N)
            loc_j = seg_j == 1 ? j : (seg_j == 2 ? j - N : j - 2N)

            # Physical time/tau values
            # C1: t increases 0->tmax with index
            # C2: t decreases tmax->0 with index (contour goes backward)
            t1 = seg_i == 1 ? ts[loc_i] : (seg_i == 2 ? ts[N+1-loc_i] : 0.0)
            t2 = seg_j == 1 ? ts[loc_j] : (seg_j == 2 ? ts[N+1-loc_j] : 0.0)
            tau1 = seg_i == 3 ? taus[loc_i] : 0.0
            tau2 = seg_j == 3 ? taus[loc_j] : 0.0

            θ = theta_C(seg_i, loc_i, seg_j, loc_j)
            K = kernel(seg_i, t1, tau1, seg_j, t2, tau2)

            Z[i, j] = -1im * (θ - fT0) * K
        end
    end

    # ── Plotting ──────────────────────────────────────────────────────────────
    mkpath("PDF/GF")

    tick_pos = [0, N, 2N, 3N]
    tick_labs = ["0", L"$t_{\max}$", "0", L"$-i\beta$"]
    clims = (-1.0, 1.0)

    function make_panel(mat, title_str)
        p = heatmap(1:M, 1:M, mat; clims=clims,
            xticks=(tick_pos, tick_labs),
            yticks=(tick_pos, tick_labs),
            xlabel=L"$z_2$", ylabel=L"$z_1$",
            title=title_str,
            aspect_ratio=:equal,
            size=(700, 700),
            guidefont=font(13), tickfont=font(10),
            colorbar_tickfont=font(9),
            xlims=(0, M), ylims=(0, M),   # ← c'est ça le fix
            left_margin=5mm, bottom_margin=5mm)
        vline!([N, 2N], color=:white, linewidth=2, label=false)
        hline!([N, 2N], color=:white, linewidth=2, label=false)
        return p
    end

    p_real = make_panel(real.(Z), "")
    p_imag = make_panel(imag.(Z), "")

    savefig(p_real, "PDF/GF/KB_greens_real.pdf")
    savefig(p_imag, "PDF/GF/KB_greens_imag.pdf")

    combined = plot(p_real, p_imag, layout=(2, 1), size=(800, 1400),
        left_margin=-8mm, bottom_margin=40px)
    savefig(combined, "PDF/GF/KB_greens_combined.pdf")
    display(combined)

    return Z
end
