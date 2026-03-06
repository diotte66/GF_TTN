# This file defines the Green's functions and the main experiment loop for testing TTN compression on momentum-independent Green's functions.

# =======================================================
# Libraries and utilities
# -------------------------------------------------------

using Random
using Plots
using LaTeXStrings
using Plots.PlotMeasures
using Statistics
include(joinpath(@__DIR__, "src", "utils_gf.jl"))
using TreeTCI
using Serialization
using CSV
using DataFrames

# =======================================================
# Green's functions
# -------------------------------------------------------

function gf(v, T, η)
    """Green's function in 2D with parameters T (frequency) and η (damping)."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    x = T * x
    y = T * y
    return 1im * exp(-1im * (x - y)) * exp(-η * abs(x - y))
end

function dcgf(v, T, η)
    """Green's function under a DC field in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    E = 10.0
    return 1im * exp(-1im * T * (sin(E * x) / E + (cos(E * x) - 1) / E)) *
           exp(1im * T * (sin(E * y) / E + (cos(E * y) - 1) / E)) *
           exp(-η * abs(x - y))
end

function acgf(v, T, η)
    """Green's function under an AC field in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    ω = 1.0
    integrand(t) = cos(-T * sin(ω * t) / ω) + sin(-T * sin(ω * t) / ω)
    Intx = Integrals.quadgk(integrand, 0, x)[1]
    Inty = Integrals.quadgk(integrand, 0, y)[1]
    return 1im * exp(-1im * Intx) * exp(1im * Inty) * exp(-η * abs(x - y))
end

# =======================================================
# Build and optimise the two TTNs
# -------------------------------------------------------

localdims = fill(2, 32)
g1 = ITTN(16, 2)
g2 = QTT_Alt(16, 2)

f = v -> gf(v, 10.0, 0.5)

ttn1 = TreeTCI.SimpleTCI{ComplexF64}(f, localdims, g1)
ttn2 = TreeTCI.SimpleTCI{ComplexF64}(f, localdims, g2)
seed_pivots!(ttn1, 5, f)
seed_pivots!(ttn2, 5, f)
TreeTCI.optimize!(ttn1, f;
    tolerance=1e-16, maxiter=5, maxbonddim=5,
    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
err1 = sampled_error(f, ttn1, 1000, 32)
println("L1 error for ITTN    : $err1")
TreeTCI.optimize!(ttn2, f;
    tolerance=1e-16, maxiter=5, maxbonddim=5,
    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
err2 = sampled_error(f, ttn2, 1000, 32)
println("L1 error for QTT_Alt : $err2")

# =======================================================
# Decompress both TTNs onto a 2D grid and compare
# -------------------------------------------------------

# Build the TreeTensorNetworks for evaluation
sitetensors1 = TreeTCI.fillsitetensors(ttn1, f)
net1         = TreeTCI.TreeTensorNetwork(ttn1.g, sitetensors1)
sitetensors2 = TreeTCI.fillsitetensors(ttn2, f)
net2         = TreeTCI.TreeTensorNetwork(ttn2.g, sitetensors2)

"""
    decode_bits(ix, iy, R) -> Vector{Int}

Convert grid indices (ix, iy) ∈ {1,…,2^R} to a bit vector of length 2R
using the interleaved or sequential encoding consistent with bits2decimal.
The first R entries encode x, the last R encode y (both in {1,2}).
"""
function decode_bits(ix::Int, iy::Int, R::Int)
    v = Vector{Int}(undef, 2R)
    for k in 1:R
        v[k]     = ((ix - 1) >> (R - k)) & 1 + 1   # x bits
        v[R + k] = ((iy - 1) >> (R - k)) & 1 + 1   # y bits
    end
    return v
end

R  = 16
N  = 2^R                    # full grid would be 2^16 × 2^16 — too large
# Use a coarser grid for visualisation: stride every 2^s points
s  = 8                      # stride exponent: sample every 2^8 = 256 points
Ng = 2^(R - s)              # number of grid points per axis (= 256)

xs = [(bits2decimal([(((i-1) >> (R-k)) & 1) + 1 for k in 1:R])) * 10.0 for i in 1:Ng:N]
ys = xs

# Evaluate exact function, ITTN approximation and QTT_Alt approximation on the grid
Z_exact = Matrix{ComplexF64}(undef, length(xs), length(ys))
Z_ttn1  = Matrix{ComplexF64}(undef, length(xs), length(ys))
Z_ttn2  = Matrix{ComplexF64}(undef, length(xs), length(ys))

println("Evaluating on $(length(xs))×$(length(ys)) grid …")
for (i, ix) in enumerate(1:Ng:N)
    for (j, iy) in enumerate(1:Ng:N)
        v            = decode_bits(ix, iy, R)
        Z_exact[i,j] = f(v)
        Z_ttn1[i,j]  = TreeTCI.evaluate(net1, v)
        Z_ttn2[i,j]  = TreeTCI.evaluate(net2, v)
    end
end

Z_err1 = abs.(Z_exact .- Z_ttn1)
Z_err2 = abs.(Z_exact .- Z_ttn2)

# =======================================================
# Plots
# -------------------------------------------------------

mkpath("PDF/GF_visual")

function make_comparison_plot(Z_exact, Z_ttn1, Z_ttn2, Z_err1, Z_err2, xs, ys; part=real, part_name="Re")
    clim_val = maximum(abs.(part.(Z_exact)))
    clim_err = max(maximum(Z_err1), maximum(Z_err2))

    kw_val = (xlabel=L"t", ylabel=L"t'", guidefontsize=10, tickfontsize=8,
              left_margin=8mm, bottom_margin=8mm, right_margin=6mm,
                 clims=(-clim_val, clim_val), framestyle=:box)
    kw_err = (xlabel=L"t", ylabel=L"t'", guidefontsize=10, tickfontsize=8,
              left_margin=8mm, bottom_margin=8mm, right_margin=6mm,
                clims=(0, clim_err), framestyle=:box)

    p1 = heatmap(xs, ys, part.(Z_exact)';
        title=L"G^<_0(t,t')", kw_val...)
    p2 = heatmap(xs, ys, Z_err1';
        title=L"G^<_0 - G^<_{\mathrm{ITTN}}", kw_err...)
    p3 = heatmap(xs, ys, Z_err2';
        title=L"G^<_0 - G^<_{\mathrm{QTT}}", kw_err...)

    fig = plot(p1, p2, p3;
        layout        = (1, 3),
        size          = (1050, 340),
        top_margin    = 6mm,
    )
    return fig
end

# Real part
fig_re = make_comparison_plot(Z_exact, Z_ttn1, Z_ttn2, Z_err1, Z_err2, xs, ys;
    part=real, part_name="Re")
savefig(fig_re, "PDF/GF_visual/comparison_real.pdf")
display(fig_re)
println("Saved → PDF/GF_visual/comparison_real.pdf")

# Imaginary part
fig_im = make_comparison_plot(Z_exact, Z_ttn1, Z_ttn2, Z_err1, Z_err2, xs, ys;
    part=imag, part_name="Im")
savefig(fig_im, "PDF/GF_visual/comparison_imag.pdf")
display(fig_im)
println("Saved → PDF/GF_visual/comparison_imag.pdf")