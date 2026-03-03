# SETUP WITH MOMENTUM DEPENDENCE COMPRESSION ALONG THE BZ

using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "utils_gk.jl"))
using TreeTCI

"""
    gk(v, nk, R, T, β, η) -> ComplexF64

Equilibrium Green's function G_{kx,ky}(t,t').
Encodes kx, ky, t, t' from bits `v` using `nk` and `R`.
β is the inverse temperature and η is the damping factor ~(U/t)^2.
"""

function ϵ1(kx, ky)
    return -4 * cos(kx) - cos(ky)
end

function ϵ2(kx, ky)
    return -2 * (cos(kx) + cos(ky)) - 4 * cos(kx) * cos(ky)
end

function gk(v, nk, R, T, β::Float64, η::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * n(kx, ky, β) * exp(-1im * ϵ1(kx, ky) * T * (t1 - t2)) * exp(-T * (ϵ1(kx, ky))^2 * η * abs(t1 - t2))
end

"""
    gkdc(v, nk, R, T, β, E) -> ComplexF64
Equilibrium Green's function with energy-dependent compression along the diagonal in the BZ.
Encodes kx, ky, t, t' from bits `v` using `nk` and `R`.
β is the inverse temperature, E controls the electric field amplitude, and η is the damping factor ~(U/t)^2.
"""

function gkdc(v, nk::Int64, R::Int64, T::Float64, β::Float64, η::Float64, E::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * T * t1) / E) + δ(kx, ky) * (cos(E * T * t1) - 1) / E - ϵ(kx, ky) * sin(E * T * t2) / E - δ(kx, ky) * (cos(E * T * t2) - 1) / E)) * exp(-T * (ϵ(kx, ky)^2) * 0.01 * abs(t1 - t2)) * n(kx, ky, β)
end

function integrand(t)
    A = 20.0
    ω = 1.0
    return cos(-A * sin(ω * t) / ω) + sin(-A * sin(ω * t) / ω)
end

function acgf(v)
    """Green's function under an AC field in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    B = 3.0
    Intx = Integrals.quadgk(integrand, 0, x)[1]
    Inty = Integrals.quadgk(integrand, 0, y)[1]
    return 1im * exp(-1im * Intx) * exp(1im * Inty) * exp(-B * abs(x - y))
end

function gkac(v, nk, R, T, β, η, E)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') under an AC field """
    kx, ky, t1, t2 = assign(v, nk, R)
    A = 20.0
    ω = 1.0
    return cos(-A * sin(ω * T * t1) / ω) + sin(-A * sin(ω * T * t1) / ω) *
                                           cos(-A * sin(ω * T * t2) / ω) + sin(-A * sin(ω * T * t2) / ω) *
                                                                           exp(-T * (ϵ(kx, ky)^2) * 0.01 * abs(t1 - t2)) * n(kx, ky, β)
end


"""
    build_topologies(nk, Rt) -> Dict{String, NamedGraph}

Convenience selector for topologies to test.
"""
function build_topologies(nk::Int, Rt::Int)
    return Dict(
        #"QTT_Alt4" => QTTALT4(Rt, nk),
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        "QTT_Alt2" => QTTALT2(Rt, nk),
        "QTT_Alt" => QTTALT(Rt, nk),
        #"BTTN" => BTTN(Rt, nk),
        #"BTTN2" => BTTN2(Rt, nk),
        #"BTTN3" => BTTN3(Rt, nk)
    )
end
"""
    default_config() -> NamedTuple

Provide canonical parameters and derived values.
"""
function default_config()
    Rt = 30 # number of time bits
    Nk = 32 # number of k-points in each direction
    nk = Int(log2(Nk)) # number of bits to encode the Nk kx,ky points
    β = 10.0 # inverse temperature
    η = 0.01 # damping factor ~(U/t)^2
    E = 1.0 # electric field amplitude for gkdc
    # list of `nk` values (bits) to sweep when varying momentum resolution
    nk_list = [nk, nk + 1, nk + 2, nk + 3]
    maxit = 5 # maximum number of iterations
    nsamples = 1000 # number of samples for error estimation
    times = [50.0, 100.0, 150.0, 200.0] # max times to evaluate
    maxbond = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] # max bond dimensions
    localdims = fill(2, 2 * Rt + 2 * nk) # local dimensions for all vertices
    topo = build_topologies(nk, Rt)
    return (Rt=Rt, Nk=Nk, nk=nk, nk_list=nk_list, β=β, η=η, E=E, maxit=maxit, nsamples=nsamples,
        times=times, maxbond=maxbond, localdims=localdims, topo=topo)
end

"""
    run_experiment(cfg) -> (error_l1, mem, topo_names)

Run compression across times and bond dimensions for all selected topologies.
"""
function run_experiment(cfg)
    ntopos = length(cfg.topo)
    error_l1 = zeros(ntopos, length(cfg.times), length(cfg.maxbond))
    mem = zeros(ntopos, length(cfg.times), length(cfg.maxbond))
    topo_names = collect(keys(cfg.topo))

    for (k, tmax) in enumerate(cfg.times)
        println("============ Time T = $tmax ============")
        for (idx_bond, maxbd) in enumerate(cfg.maxbond)
            println("Max bond dimension: $maxbd")
            for (j, name) in enumerate(topo_names)
                topology = cfg.topo[name]
                println("Topology: $name")
                kwargs = (
                    maxiter=cfg.maxit,
                    maxbonddim=maxbd,
                    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
                )
                ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η, cfg.E), cfg.localdims, topology)
                # utils_gk.jl defines seed_pivots!(tci, npivots)
                seed_pivots!(ttn, 25, v -> gkdc(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η, cfg.E))
                ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η, cfg.E); kwargs...)
                mem[j, k, idx_bond] = Base.summarysize(ttn)
                errl1 = sampled_error(v -> gkdc(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η, cfg.E), ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
                error_l1[j, k, idx_bond] = errl1
                println(" Sampled L1 error: ", errl1)
            end
        end
    end

    return error_l1, mem, topo_names
end

"""
    run_experiment_over_nk(cfg, nk_list) -> (error_l1, mem, topo_names, nk_list)

Run compression across different `nk` values (vary momentum resolution) for a fixed time (T=100).
Returns arrays shaped (ntopos, length(nk_list), length(cfg.maxbond)).

"""

function run_experiment_over_nk(cfg)
    nk_list = cfg.nk_list
    first_topo = build_topologies(Int(nk_list[1]), cfg.Rt)
    topo_names = collect(keys(first_topo))
    ntopos = length(topo_names)
    error_l1 = zeros(ntopos, length(nk_list), length(cfg.maxbond))
    mem = zeros(ntopos, length(nk_list), length(cfg.maxbond))

    tmax = 100.0
    for (ik, nkv) in enumerate(nk_list)
        println("============ nk = $nkv ============")
        topo = build_topologies(Int(nkv), cfg.Rt)
        for (idx_bond, maxbd) in enumerate(cfg.maxbond)
            println("Max bond dimension: $maxbd")
            for (j, name) in enumerate(topo_names)
                println("Topology: $name")
                topology = topo[name]
                localdims = fill(2, 2 * cfg.Rt + 2 * Int(nkv))
                kwargs = (
                    maxiter=cfg.maxit,
                    maxbonddim=maxbd,
                    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
                )
                ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, Int(nkv), cfg.Rt, tmax, cfg.β, cfg.η, cfg.E), localdims, topology)
                seed_pivots!(ttn, 25, v -> gkdc(v, Int(nkv), cfg.Rt, tmax, cfg.β, cfg.η, cfg.E))
                ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, Int(nkv), cfg.Rt, tmax, cfg.β, cfg.η, cfg.E); kwargs...)
                mem[j, ik, idx_bond] = Base.summarysize(ttn)
                errl1 = sampled_error(v -> gkdc(v, Int(nkv), cfg.Rt, tmax, cfg.β, cfg.η, cfg.E), ttn, cfg.nsamples, 2 * cfg.Rt + 2 * Int(nkv))
                error_l1[j, ik, idx_bond] = errl1
                println(" Sampled L1 error: ", errl1)
            end
        end
    end
    return error_l1, mem, topo_names, cfg.nk_list
end

"""
    plot_error_curves(error_l1, times, maxbond, topo_names; outfile)

Create and save error curves per topology and time.
"""
function plot_error_curves(error_l1, times, maxbond, topo_names; outfile="gkdc_compression_error.svg")
    subplots = []
    for (i, T) in enumerate(times)
        subplot = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="L1 Error", yscale=:log10)
        for (j, name) in enumerate(topo_names)
            plot!(subplot, maxbond, error_l1[j, i, :], label="$name, T=$T", marker=:o)
        end
        push!(subplots, subplot)
    end
    n = length(times)
    ncols = 2
    nrows = ceil(Int, n / ncols)
    plt = plot(subplots..., layout=(nrows, ncols), size=(1200, 900))
    savefig(plt, outfile)
    display(plt)
end

"""
    plot_memory_curves(mem, times, maxbond, topo_names; outfile)

Create and save memory usage curves per topology and time.
"""
function plot_memory_curves(mem, times, maxbond, topo_names; outfile="gkdc_compression_memory.svg")
    subplots = []
    for (i, T) in enumerate(times)
        subplot2 = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="Memory usage (Bytes)")
        for (j, name) in enumerate(topo_names)
            plot!(subplot2, maxbond, mem[j, i, :], label="$name, T=$T", marker=:o)
        end
        push!(subplots, subplot2)
    end
    n = length(times)
    ncols = 2
    nrows = ceil(Int, n / ncols)
    plt2 = plot(subplots..., layout=(nrows, ncols), size=(1200, 900))
    savefig(plt2, outfile)
    display(plt2)
end

function plot_error_curves_nk(error_l1, nk_list, maxbond, topo_names, outfile="gk_compression_error_nk.svg")
    subplots = []
    for (i, nk) in enumerate(nk_list)
        subplot = plot(title="nk = $nk", xlabel="Max Bond Dimension", ylabel="L1 Error", yscale=:log10)
        for (j, name) in enumerate(topo_names)
            plot!(subplot, maxbond, error_l1[j, i, :], label="$name, nk=$nk", marker=:o)
        end
        push!(subplots, subplot)
    end
    n = length(nk_list)
    ncols = 2
    nrows = ceil(Int, n / ncols)
    plt = plot(subplots..., layout=(nrows, ncols), size=(1200, 900))
    savefig(plt, outfile)
    display(plt)

end

function plot_memory_curves_nk(mem, nk_list, maxbond, topo_names, outfile="gk_compression_memory_nk.svg")
    subplots = []
    for (i, nk) in enumerate(nk_list)
        subplot2 = plot(title="nk = $nk", xlabel="Max Bond Dimension", ylabel="Memory usage (Bytes)")
        for (j, name) in enumerate(topo_names)
            plot!(subplot2, maxbond, mem[j, i, :], label="$name, nk=$nk", marker=:o)
        end
        push!(subplots, subplot2)
    end
    n = length(nk_list)
    ncols = 2
    nrows = ceil(Int, n / ncols)
    plt2 = plot(subplots..., layout=(nrows, ncols), size=(1200, 900))
    savefig(plt2, outfile)
    display(plt2)
end

function bondplottrain()
    times = [100.0]
    listlist = Vector{Int}[]
    cfg = default_config()
    for i in 1:length(times)
        kwargs = (
            maxiter=cfg.maxit,
            tolerance=1e-6,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        time = times[i]
        ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), cfg.localdims, cfg.topo["QTT_Alt"])
        # utils_gk.jl defines seed_pivots!(tci, npivots)
        seed_pivots!(ttn, 25, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E))
        ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E); kwargs...)
        errl1 = sampled_error(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
        println(" Sampled L1 error: ", errl1)
        mem = Base.summarysize(ttn)
        println("Memory usage (Bytes): ", mem)
        list = trainbondlist(bonds, cfg.nk, cfg.Rt)
        push!(listlist, list)
    end
    multipletrainbondplot(listlist, cfg.nk, cfg.Rt, times)
end

function bondplottrain4()
    times = [20.0, 40.0, 80.0, 160.0, 200.0, 400.0, 800.0]
    listlist = Vector{Int}[]
    cfg = default_config()
    for i in 1:length(times)
        kwargs = (
            maxiter=cfg.maxit,
            tolerance=1e-6,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        time = times[i]
        ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), cfg.localdims, cfg.topo["QTT_Alt4"])
        # utils_gk.jl defines seed_pivots!(tci, npivots)
        seed_pivots!(ttn, 25, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E))
        ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E); kwargs...)
        errl1 = sampled_error(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
        println(" Sampled L1 error: ", errl1)
        list = trainbondlist4(bonds, cfg.nk, cfg.Rt)
        push!(listlist, list)
    end
    multipletrainbondplot4(listlist, cfg.nk, cfg.Rt, times)
end

function bondplotfork()
    times = [100.0]
    listlistkx = Vector{Int}[]
    listlistky = Vector{Int}[]
    cfg = default_config()
    for i in 1:length(times)
        kwargs = (
            maxiter=cfg.maxit,
            tolerance=1e-6,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        time = times[i]
        ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), cfg.localdims, cfg.topo["CTTN_Alt2"])
        seed_pivots!(ttn, 25, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E))
        ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E); kwargs...)
        errl1 = sampled_error(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
        mem = Base.summarysize(ttn)
        println("Memory usage (Bytes): ", mem)
        println(" Sampled L1 error: ", errl1)
        listkx, listky = forkbondlist(bonds, cfg.nk, cfg.Rt)
        push!(listlistkx, listkx)
        push!(listlistky, listky)
    end
    println(listlistkx)
    multipleforkbondplot(listlistkx, listlistky, cfg.nk, cfg.Rt, times)
end

function bondplotfork2()
    times = [50.0]
    cfg = default_config()
    for i in 1:length(times)
        kwargs = (
            maxiter=cfg.maxit,
            tolerance=1e-6,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        time = times[i]
        ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), cfg.localdims, cfg.topo["CTTN_Alt2"])
        seed_pivots!(ttn, 25, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E))
        ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E); kwargs...)
        errl1 = sampled_error(v -> gkdc(v, cfg.nk, cfg.Rt, time, cfg.β, cfg.η, cfg.E), ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
        mem = Base.summarysize(ttn)
        println("Memory usage (Bytes): ", mem)
        println(" Sampled L1 error: ", errl1)
        println("Ranks: ", ranks)
        println("Bonds: ", bonds)
    end
end

"""
    preview_gk(kx, ky)

Quick visualization of G(t,t') at a fixed k using utils.
"""
function preview_gk(kx::Float64, ky::Float64, β::Float64, η::Float64, tmax::Float64)
    plotgk(kx, ky, β, η, tmax)
end

"""
    main()

Entry point: configure, run, and plot results.
"""
function main()
    cfg = default_config()
    error_l1, mem, topo_names = run_experiment(cfg)
    plot_error_curves(error_l1, cfg.times, cfg.maxbond, topo_names)
    plot_memory_curves(mem, cfg.times, cfg.maxbond, topo_names)
end

function main_nk()
    cfg = default_config()
    error_l1, mem, topo_names, nk_list = run_experiment_over_nk(cfg)
    plot_error_curves_nk(error_l1, nk_list, cfg.maxbond, topo_names)
    plot_memory_curves_nk(mem, nk_list, cfg.maxbond, topo_names)
end