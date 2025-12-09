using TreeTCI
using Random
using Plots
using Graphs
using NamedGraphs

#------------------------------------------------------
# General utility functions for QTCI simulations
#------------------------------------------------------

function bits2decimal(v::AbstractVector{<:Integer})
    """Convert a vector of bits (1/2) to a decimal number between 0 and 1"""
    sum = 0.0
    for i in 1:length(v)
        sum += (v[i] - 1) * 2.0^(-i)
    end
    return sum
end

function sampled_error(f, ttn, nsamples::Int64, bits::Int64, d::Int64)
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

function seed_pivots!(tci, npivots::Int64)
    """ Seed initial pivots for the SimpleTCI tci based on binary representations. """
    for i in 1:npivots
        # Generate a random input of length equal to number of vertices
        x = rand(1:2, length(vertices(tci.g)))
        TreeTCI.addglobalpivots!(tci, [x])
    end
end

#-------------------------------------------------------------
# Specific utility functions for Green's function simulations
#-------------------------------------------------------------

function ϵ(kx::Float64, ky::Float64)
    """ Fictious dispersion relation """
    return -2.0 * (cos(2π * kx) + cos(2π * ky))
end

function δ(kx::Float64, ky::Float64)
    """ Fictious dispersion relation """
    return -2.0 * (sin(2π * kx) + sin(2π * ky))
end

function n(kx::Float64, ky::Float64, β::Float64)
    """ Fermi-Dirac distribution """
    return 1.0 / (exp(β * ϵ(kx, ky)) + 1.0)
end

function assign(v::AbstractVector{Int64}, nk::Int64, R::Int64)
    """ Assign variables from bit vector v using explicit `nk` and `R`.

    v is expected to be a vector with length 2*nk + 2*R (kx bits, ky bits, t1 bits, t2 bits).
    """
    kx = bits2decimal(v[1:nk])
    ky = bits2decimal(v[nk+1:2*nk])
    t1 = bits2decimal(v[2*nk+1:2*nk+1+R])
    t2 = bits2decimal(v[2*nk+2+R:end])
    return kx, ky, t1, t2
end

function plotgk(kx, ky)
    f(x, y) = -1im * exp(-1im * ϵ(kx, ky) * (x - y)) * exp(-2.0 * abs(x - y))

    xs = ys = range(0, 1, length=400)
    zs = [f(x, y) for x in xs, y in ys]
    # Plots/GR cannot compare Complex values when computing extrema; convert to real
    zs_imag = imag.(zs)
    zs_real = real.(zs)
    realpart = heatmap(xs, ys, zs_real, title="Real part of G($kx,$ky)", xlabel="t", ylabel="t'", size=(600, 500))
    imagpart = heatmap(xs, ys, zs_imag, title="Imaginary part of G($kx,$ky)", xlabel="t", ylabel="t'", size=(600, 500))
    display(plot(realpart, imagpart, layout=(1, 2), size=(1200, 500)))
end

function plotdcgk(kx, ky)
    E = 50.0
    f(x, y) = -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * x) / E) + δ(kx, ky) * (cos(E * x) - 1) / E - ϵ(kx, ky) * sin(E * y) / E - δ(kx, ky) * (cos(E * y) - 1) / E)) * exp(-abs(x - y)) * n(kx, ky, 10.0)
    xs = ys = range(0, 1, length=400)
    zs = [f(x, y) for x in xs, y in ys]
    # Plots/GR cannot compare Complex values when computing extrema; convert to real
    zs_imag = imag.(zs)
    zs_real = real.(zs)
    realpart = heatmap(xs, ys, zs_real, title="Real part of G($kx,$ky)", xlabel="t", ylabel="t'", size=(600, 500))
    imaginarypart = heatmap(xs, ys, zs_imag, title="Imaginary part of G($kx,$ky)", xlabel="t", ylabel="t'", size=(600, 500))
    display(plot(realpart, imaginarypart, layout=(1, 2), size=(1200, 500)))
end

#-------------------------------------------------------------
# Specific utility functions for different topologies
#-------------------------------------------------------------

function QTTSEQ(Rt::Int64, nk::Int64)
    """ QTT Sequential topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)
    total_bits = 2 * Rt + 2 * nk
    for i in 1:(total_bits-1)
        add_edge!(g, i, i + 1)
    end
    return g
end

function QTTALT(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(2*nk)
        add_edge!(g, i, i + 1)
    end

    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT2(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, 1, nk + 1)
    add_edge!(g, 2 * nk, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT3(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:nk
        add_edge!(g, i, nk + i)
        if i < nk
            add_edge!(g, nk + i, i + 1)
        end
    end
    add_edge!(g, 2 * nk, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT4(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:nk
        add_edge!(g, i, nk + i)
        if i < nk
            add_edge!(g, nk + i, i + 1)
        end
    end
    add_edge!(g, 1, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function CTTNALT(Rt::Int64, nk::Int64)
    """ CTTN Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    add_edge!(g, 1, nk + 1)
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, nk + 1, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function CTTNALT2(Rt::Int64, nk::Int64)
    """ CTTN Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    add_edge!(g, 1, 2 * nk + 1)
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, nk + 1, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function CTTNSEQ(Rt::Int64, nk::Int64)
    """ CTTN Sequential topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    add_edge!(g, 1, nk + 1)
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, nk + 1, 2 * nk + 1)
    for i in 1:(Rt-1)
        add_edge!(g, 2 * nk + i, 2 * nk + i + 1)
    end
    add_edge!(g, 2 * nk + 1, 2 * nk + Rt + 1)
    for i in 1:(Rt-1)
        add_edge!(g, 2 * nk + Rt + i, 2 * nk + Rt + i + 1)
    end
    return g
end

function BTTN(Rt::Int64, nk::Int64)
    """ BTTN topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    g = NamedGraph()
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    lengths = (nk, nk, Rt, Rt)

    start = 1
    for L in lengths
        for i in 1:L
            left = 2 * i
            right = 2 * i + 1
            if left <= L
                add_edge!(g, start + i - 1, start + left - 1)
            end
            if right <= L
                add_edge!(g, start + i - 1, start + right - 1)
            end
        end

        if start != 1
            add_edge!(g, 1, start)
        end

        start += L
    end
    return g
end

