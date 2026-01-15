using Random
using Plots
include(joinpath(@__DIR__, "src", "utils_gk.jl"))
using TreeTCI
using ITensors

function gkdc(v, nk::Int64, R::Int64, β::Float64, E::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * t1) / E) + δ(kx, ky) * (cos(E * t1) - 1) / E - ϵ(kx, ky) * sin(E * t2) / E - δ(kx, ky) * (cos(E * t2) - 1) / E)) * exp(-(ϵ(kx, ky)^2) * abs(t1 - t2)) * n(kx, ky, β)
end

function gk(v, nk::Int64, R::Int64, T::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    ϵ, t1, t2 = assign2(v, nk, R)
    return -1im * exp(-1im * ϵ * T * (t1 - t2)) * exp(-(ϵ^2) * T * abs(t1 - t2))
end

function simpleTCItoTreeTensorNetwork(ttn::TreeTCI.SimpleTCI, cutdim::Int, nk::Int, Rt::Int, T::Float64)
    """ Convert a SimpleTCI to an ITensor Network by filling site tensors and truncating bonds.

    Accepts `nk`, `Rt`, and `T` explicitly so the site-filling closure can call `gk` with the
    correct parameters instead of relying on outer-scope variables.
    """
    sitetensors = TreeTCI.fillsitetensors(ttn, v -> gk(v, nk, Rt, T))
    ttn_concrete = TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    ψ = TreeTCI.convert_ITensorNetwork(ttn_concrete, cutdim)
    return ψ
end

function topo1(nk::Int64, Rt::Int64)
    N = nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:Rt
        add_edge!(g, nk + i, nk + Rt + i)
        if i < Rt
            add_edge!(g, nk + Rt + i, nk + Rt + i + 1)
        end
    end
    step = div(Rt, nk)
    if step >= 1
        for i in 0:nk-1
            add_edge!(g, i + 1, nk + 1 + step * i)
        end
    end
    return g
end

function topo2(nk::Int64, Rt::Int64)
    N = nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:Rt
        add_edge!(g, nk + i, nk + Rt + i)
        if i < Rt
            add_edge!(g, nk + Rt + i, nk + Rt + i + 1)
        end
    end

    # Lower k-space chain: 1 to div(nk,2)
    if div(nk, 2) >= 1
        for i in 1:(div(nk, 2)-1)
            add_edge!(g, i, i + 1)
        end
    end

    # Upper k-space chain: div(nk,2)+1 to nk
    if nk > div(nk, 2)
        for i in (div(nk, 2)+1):(nk-1)
            add_edge!(g, i, i + 1)
        end
    end

    # Connect k-space chains to time nodes
    add_edge!(g, nk, nk + 1)           # upper chain to first time bit
    add_edge!(g, 1, nk + 2 * Rt)       # lower chain to last time bit

    return g
end

function topo3(nk::Int64, Rt::Int64)
    N = nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:Rt
        add_edge!(g, nk + i, nk + Rt + i)
        if i < Rt
            add_edge!(g, nk + Rt + i, nk + Rt + i + 1)
        end
    end

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end

    add_edge!(g, nk, nk + 1)
    return g
end

function main()
    """
        main function for the compression of momentum dependent Green's function G_{kx,ky}(t,t')
    """
    Rt = 30  # number of time bits
    nk = 15  # number of bits to encode the epsilon points

    localdims = fill(2, nk + 2 * Rt)

    β = 10.0  # inverse temperature 
    E = 10.0  # electric field amplitude

    topo = Dict(
        "Topo1" => topo1(nk, Rt),
        "Topo2" => topo2(nk, Rt),
        "Topo3" => topo3(nk, Rt)
    )

    ntopos = length(topo)
    times = [1.0, 40.0, 200.0, 500.0]
    maxbond = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    error_l1 = zeros(ntopos, length(times), length(maxbond))
    mem = zeros(ntopos, length(times), length(maxbond))

    for (i, T) in enumerate(times)
        println("============ Time T = $T ============")
        for (j, (toponame, topology)) in enumerate(topo)
            println("------------ Topology: $toponame ------------")
            println("For step $i we have tmax = $T")
            for maxbd in maxbond
                println("Max bond dimension: $maxbd")
                kwargs = (
                    maxiter=5,
                    maxbonddim=maxbd,
                    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
                )
                ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gk(v, nk, Rt, T), localdims, topology)
                seed_pivots!(ttn, 10)
                ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gk(v, nk, Rt, T); kwargs...)
                errl1 = sampled_error(v -> gk(v, nk, Rt, T), ttn, 1000, nk + 2 * Rt, 2)
                mem[j, i, findfirst(==(maxbd), maxbond)] = Base.summarysize(ttn)
                println(" Sampled L1 error: ", errl1)
                error_l1[j, i, findfirst(==(maxbd), maxbond)] = errl1
            end
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