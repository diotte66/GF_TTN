using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "utils_gk.jl"))
using TreeTCI

function gkdc(v, nk::Int64, R::Int64, β::Float64, E::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * t1) / E) + δ(kx, ky) * (cos(E * t1) - 1) / E - ϵ(kx, ky) * sin(E * t2) / E - δ(kx, ky) * (cos(E * t2) - 1) / E)) * exp(-abs(t1 - t2)) * n(kx, ky, β)
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
        "QTT_Seq" => QTTSEQ(Rt, nk),
        "QTT_Alt" => QTTALT(Rt, nk),
        "QTT_Alt2" => QTTALT2(Rt, nk),
        "QTT_Alt3" => QTTALT3(Rt, nk),
        "QTT_Alt4" => QTTALT4(Rt, nk),
        "CTTN_Alt" => CTTNALT(Rt, nk),
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        "CTTN_Seq" => CTTNSEQ(Rt, nk),
        #"BTTN" => BTTN(Rt, nk)
    )

    ntopos = length(topo)
    nsteps = 20
    step = 3
    maxit = 5
    nsamples = 1000

    error_l1 = zeros(ntopos, nsteps)
    error_pivot = zeros(ntopos, nsteps)
    mem = zeros(ntopos, nsteps)
    rankendlist = zeros(ntopos, nsteps)
    ranklist = zeros(ntopos, nsteps)

    for i in 1:nsteps
        maxbd = step * i
        println("Max bond dimension: $maxbd")
        tstart = @elapsed begin

            for (j, (toponame, topology)) in enumerate(topo)
                println("Topology: $toponame")
                kwargs = (
                    maxiter=maxit,
                    maxbonddim=maxbd,
                    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
                )
                ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, nk, Rt, β, E), localdims, topology)
                seed_pivots!(ttn, 10)
                ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, nk, Rt, β, E); kwargs...)
                mem[j, i] = Base.summarysize(ttn)
                errl1 = sampled_error(v -> gkdc(v, nk, Rt, β, E), ttn, nsamples, 2 * Rt + 2 * nk, d)
                error_l1[j, i] = errl1
                rankendlist[j, i] = ranks[end]
                ranklist[j, i] = maxbd
                println(" Sampled L1 error: ", errl1)
            end
        end
        println(" Time for maxbonddim $maxbd : $tstart seconds")
    end

    # ensure enough margin so titles/labels/legends are not clipped
    p1 = plot(title="G_k<(t,t') compression (nk = $Nk)", xlabel="Max bond dimension",
        ylabel="Sampled L1 error", yscale=:log10, legend=:topright)
    p2 = plot(title="G_k<(t,t') compression (nk = $Nk)", xlabel="Memory (kB)",
        ylabel="Sampled L1 error", yscale=:log10, legend=:topright)

    for (j, (toponame, topology)) in enumerate(topo)
        plot!(p1, ranklist[j, :], error_l1[j, :], label=toponame, marker=:o)
        plot!(p2, mem[j, :] / 1000, error_l1[j, :], label=toponame, marker=:o)
    end

    # large canvas; avoid explicit margin objects that may mix Measure types
    plt = plot(p1, p2, layout=(1, 2), size=(900, 550))

    savefig(plt, "SVG/GKDC/gkdc_l1_nk=$Nk.svg")
    display(plt)

end