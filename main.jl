using Random
using Plots
include(joinpath(@__DIR__, "src", "utils_gk.jl"))
using TreeTCI

function gkdc(v, nk::Int64, R::Int64, β::Float64, E::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * t1) / E) + δ(kx, ky) * (cos(E * t1) - 1) / E - ϵ(kx, ky) * sin(E * t2) / E - δ(kx, ky) * (cos(E * t2) - 1) / E)) * exp(-abs(t1 - t2)) * n(kx, ky, β)
end

function gk(v, nk, R)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * ϵ(kx, ky) * (t1 - t2)) * exp(-abs(t1 - t2))
end

function main()
    """
        main function for the compression of momentum dependent Green's function G_{kx,ky}(t,t')
    """
    Rt = 16  # number of time bits
    Nk = 256  # size of the lattice in one dimension (number of k-points bits)
    nk = Int(log2(Nk)) # number of bits to encode the Nk kx,ky points 
    d = 4  # total dimensions: kx, ky, t, t'
    localdims = fill(2, 2 * Rt + 2 * nk)

    β = 10.0  # inverse temperature
    E = 10.0  # electric field amplitude

    topo = Dict(
        #"QTT_Seq" => QTTSEQ(Rt, nk),
        #"QTT_Alt" => QTTALT(Rt, nk),
        #"QTT_Alt2" => QTTALT2(Rt, nk),
        #"QTT_Alt3" => QTTALT3(Rt, nk),
        #"QTT_Alt4" => QTTALT4(Rt, nk),
        "CTTN_Alt" => CTTNALT(Rt, nk),
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        #"CTTN_Seq" => CTTNSEQ(Rt, nk),
        #"BTTN" => BTTN(Rt, nk)
    )

    ntopos = length(topo)

    for (j, (toponame, topology)) in enumerate(topo)
        println("------------ Topology: $toponame ------------")
        kwargs = (
            maxiter=5,
            maxbonddim=60,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gk(v, nk, Rt), localdims, topology)
        seed_pivots!(ttn, 10)
        ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gk(v, nk, Rt); kwargs...)
        println(" Bonds: ", bonds)

    end
end