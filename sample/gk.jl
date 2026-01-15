using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "utils_gk.jl"))
using TreeTCI

function gk(v, nk, R, T)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * ϵ(kx, ky) * T * (t1 - t2)) * exp(-T * (ϵ(kx, ky))^2 * abs(t1 - t2))
end

function main()
    """
        main function for the compression of momentum dependent Green's function G_{kx,ky}(t,t')
    """
    Rt = 30  # number of time bits
    Nk = 256  # size of the lattice in one dimension (number of k-points bits)
    nk = Int(log2(Nk)) # number of bits to encode the Nk kx,ky points 
    d = 4  # total dimensions: kx, ky, t, t'
    localdims = fill(2, 2 * Rt + 2 * nk)

    topo = Dict(
        #"QTT_Seq" => QTTSEQ(Rt, nk),
        #"QTT_Alt" => QTTALT(Rt, nk),
        #"QTT_Alt2" => QTTALT2(Rt, nk),
        #"QTT_Alt3" => QTTALT3(Rt, nk),
        "QTT_Alt4" => QTTALT4(Rt, nk),
        #"CTTN_Alt" => CTTNALT(Rt, nk),
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        #"CTTN_Seq" => CTTNSEQ(Rt, nk),
        #"BTTN" => BTTN(Rt, nk)
    )

    ntopos = length(topo)

    maxit = 5
    nsamples = 1000


    times = [40.0, 200.0, 500.0]
    maxbond = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    error_l1 = zeros(ntopos, length(times), length(maxbond))
    mem = zeros(ntopos, length(times), length(maxbond))

    for (k, tmax) in enumerate(times)
        println("============ Time T = $tmax ============")
        for (idx_bond, maxbd) in enumerate(maxbond)
            println("Max bond dimension: $maxbd")
            tstart = @elapsed begin
                for (j, (toponame, topology)) in enumerate(topo)
                    println("Topology: $toponame")
                    kwargs = (
                        maxiter=maxit,
                        maxbonddim=maxbd,
                        sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
                    )
                    ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gk(v, nk, Rt, tmax), localdims, topology)
                    seed_pivots!(ttn, 25)
                    ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gk(v, nk, Rt, tmax); kwargs...)
                    mem[j, k, idx_bond] = Base.summarysize(ttn)
                    errl1 = sampled_error(v -> gk(v, nk, Rt, tmax), ttn, nsamples, 2 * Rt + 2 * nk, d)
                    error_l1[j, k, idx_bond] = errl1
                    println(" Sampled L1 error: ", errl1)
                end
            end
            println(" Time for maxbonddim $maxbd : $tstart seconds")
        end
    end

    subplots = []
    for (i, T) in enumerate(times)
        subplot = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="L1 Error", yscale=:log10)
        for (j, (toponame, topology)) in enumerate(topo)
            plot!(subplot, maxbond, error_l1[j, i, :], label="$toponame, T=$T", marker=:o)
        end
        push!(subplots, subplot)
    end
    plt = plot(subplots..., layout=(2, 2), size=(1200, 900))
    savefig(plt, "gk_compression_error.svg")
    display(plt)

    subplots2 = []
    for (i, T) in enumerate(times)
        subplot2 = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="Memory usage (Bytes)")
        for (j, (toponame, topology)) in enumerate(topo)
            plot!(subplot2, maxbond, mem[j, i, :], label="$toponame, T=$T", marker=:o)
        end
        push!(subplots2, subplot2)
    end
    plt2 = plot(subplots2..., layout=(2, 2), size=(1200, 900))
    savefig(plt2, "gk_compression_memory.svg")
    display(plt2)


end