# SETUP WITH ENERGY DEPENDENCE COMPRESSION ALONG THE DIAGONAL IN THE BZ

# ========== Imports and Includes ==========
using Random
using Plots
include(joinpath(@__DIR__, "../src", "utils_gk.jl"))
using TreeTCI
using ITensors

# ========== Green's Function ==============
function gk(v, nk, Rt, T, β, η)
    ϵ, t1, t2 = assign2(v, nk, Rt)
    return -1im * n(ϵ, β) * exp(-1im * ϵ * T * (t1 - t2)) * exp(-T * ϵ^2 * η * abs(t1 - t2))
end

# ========== Topology Builders =============
function build_topologies(nk, Rt)
    return Dict(
        "Topo1" => topo1(nk, Rt),
        "Topo2" => topo2(nk, Rt),
        "Topo3" => topo3(nk, Rt),
    )
end

function topo1(nk, Rt)
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

function topo2(nk, Rt)
    N = nk + 2 * Rt
    g = NamedGraph(N)
    for i in 1:Rt
        add_edge!(g, nk + i, nk + Rt + i)
        if i < Rt
            add_edge!(g, nk + Rt + i, nk + Rt + i + 1)
        end
    end
    if div(nk, 2) >= 1
        for i in 1:(div(nk, 2)-1)
            add_edge!(g, i, i + 1)
        end
    end
    if nk > div(nk, 2)
        for i in (div(nk, 2)+1):(nk-1)
            add_edge!(g, i, i + 1)
        end
    end
    add_edge!(g, nk, nk + 1)
    add_edge!(g, 1, nk + 2 * Rt)
    return g
end

function topo3(nk, Rt)
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

# ========== Configuration ================
function default_config()
    Rt = 30 # time resolution
    nk = 15 # number of energy points
    β = 10.0 # inverse temperature
    η = 0.01 # damping factor ~U^2
    maxit = 5 # maximum number of iterations
    nsamples = 1000 # number of samples for error estimation
    times = [100.0, 200.0, 400.0, 800.0] # max times to evaluate
    maxbond = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] # max bond dimensions
    localdims = fill(2, nk + 2 * Rt) # local dimensions for all vertices
    topo = build_topologies(nk, Rt)
    return (Rt=Rt, nk=nk, β=β, η=η, maxit=maxit, nsamples=nsamples, times=times, maxbond=maxbond, localdims=localdims, topo=topo)
end

# ========== Experiment Runner ============
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
                ttn = TreeTCI.SimpleTCI{ComplexF64}(v -> gk(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η), cfg.localdims, topology)
                seed_pivots!(ttn, 10, v -> gk(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η))
                ranks, errors, bonds = TreeTCI.optimize!(ttn, v -> gk(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η); kwargs...)
                mem[j, k, idx_bond] = Base.summarysize(ttn)
                errl1 = sampled_error(v -> gk(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η), ttn, cfg.nsamples, cfg.nk + 2 * cfg.Rt)
                error_l1[j, k, idx_bond] = errl1
                println(" Sampled L1 error: ", errl1)
            end
        end
    end
    return error_l1, mem, topo_names
end

# ========== Plotting =====================
function plot_error_curves(error_l1, times, maxbond, topo_names; outfile="gk_compression_error.svg")
    subplots = []
    for (i, T) in enumerate(times)
        subplot = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="L1 Error", yscale=:log10)
        for (j, name) in enumerate(topo_names)
            plot!(subplot, maxbond, error_l1[j, i, :], label="$name, T=$T", marker=:o)
        end
        push!(subplots, subplot)
    end
    nplots = length(subplots)
    if nplots == 0
        return
    end
    ncols = min(2, nplots)
    nrows = ceil(Int, nplots / ncols)
    plt = plot(subplots..., layout=(nrows, ncols), size=(1200, 900))
    savefig(plt, outfile)
    display(plt)
end

function plot_memory_curves(mem, times, maxbond, topo_names; outfile="gk_compression_memory.svg")
    subplots = []
    for (i, T) in enumerate(times)
        subplot2 = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="Memory usage (Bytes)")
        for (j, name) in enumerate(topo_names)
            plot!(subplot2, maxbond, mem[j, i, :], label="$name, T=$T", marker=:o)
        end
        push!(subplots, subplot2)
    end
    nplots2 = length(subplots)
    if nplots2 == 0
        return
    end
    ncols2 = min(2, nplots2)
    nrows2 = ceil(Int, nplots2 / ncols2)
    plt2 = plot(subplots..., layout=(nrows2, ncols2), size=(1200, 900))
    savefig(plt2, outfile)
    display(plt2)
end

# ========== Main Entry ===================
function main()
    cfg = default_config()
    error_l1, mem, topo_names = run_experiment(cfg)
    plot_error_curves(error_l1, cfg.times, cfg.maxbond, topo_names)
    plot_memory_curves(mem, cfg.times, cfg.maxbond, topo_names)
end