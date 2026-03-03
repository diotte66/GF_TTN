using Random
using Plots
using LaTeXStrings
using Plots.PlotMeasures
include(joinpath(@__DIR__, "..", "src", "utils_gk.jl"))
using TreeTCI
using Serialization
using DelimitedFiles

function gkdc(v, nk::Int64, R::Int64, T::Float64, β::Float64, η::Float64, E::Float64)
    """Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t')"""
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * exp(-1im * (
               ϵ(kx, ky) * sin(E * T * t1) / E +
               δ(kx, ky) * (cos(E * T * t1) - 1) / E -
               ϵ(kx, ky) * sin(E * T * t2) / E -
               δ(kx, ky) * (cos(E * T * t2) - 1) / E)) *
           exp(-T * (ϵ(kx, ky)^2) * η * abs(t1 - t2)) *
           n(kx, ky, β)
end

function plotdcgf()
    A = 5.0
    B = 5.0
    E = 10.0
    kx, ky = 0.0, 0.0
    T = 100.0
    f(x, y) = -1im * exp(-1im * (
                  ϵ(kx, ky) * sin(E * T * x) / E +
                  δ(kx, ky) * (cos(E * T * x) - 1) / E -
                  ϵ(kx, ky) * sin(E * T * y) / E -
                  δ(kx, ky) * (cos(E * T * y) - 1) / E)) *
              exp(-T * (ϵ(kx, ky)^2) * η * abs(x - y)) *
              n(kx, ky, β)
    xs = ys = range(0, 1, length=400)
    zs = [f(x, y) for x in xs, y in ys]
    p1 = heatmap(xs, ys, real.(zs), title="",
        xlabel=L"$z_1$", ylabel=L"$z_2$",
        guidefont=font(12), tickfont=font(10),
        left_margin=10mm, bottom_margin=10mm)
    p2 = heatmap(xs, ys, imag.(zs), title="",
        xlabel=L"$z_1$", ylabel=L"$z_2$",
        guidefont=font(12), tickfont=font(10),
        left_margin=10mm, bottom_margin=10mm)
    mkpath("PDF/DCGF")
    P = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    savefig(P, "PDF/DCGF/dcgf_function.pdf")
    display(P)
end

function default_config()
    Rt = 30
    Nk = 32
    nk = Int(log2(Nk))
    β = 10.0
    η = 0.01
    E = 1.0
    nk_list = [nk, nk + 1, nk + 2, nk + 3]
    maxit = 5
    nsamples = 1000
    times = [100.0]#, 100.0, 150.0, 200.0]
    maxbond = [50, 100, 150, 200, 250, 300, 350, 400]
    localdims = fill(2, 2 * Rt + 2 * nk)
    topo = Dict(
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        "QTT_Alt2" => QTTALT2(Rt, nk),
        "QTT_Alt" => QTTALT(Rt, nk),
        #"BTTN2" => BTTN2(Rt, nk),
        #"BTTN3" => BTTN3(Rt, nk),
    )
    return (Rt=Rt, Nk=Nk, nk=nk, nk_list=nk_list, β=β, η=η, E=E,
        maxit=maxit, nsamples=nsamples, times=times, maxbond=maxbond,
        localdims=localdims, topo=topo)
end

function run_experiment(cfg)
    """Run compression across times and bond dimensions for all topologies."""
    ntopos = length(cfg.topo)
    topo_names = collect(keys(cfg.topo))
    error_l1 = zeros(ntopos, length(cfg.times), length(cfg.maxbond))
    mem = zeros(ntopos, length(cfg.times), length(cfg.maxbond))

    for (k, tmax) in enumerate(cfg.times)
        println("============ Time T = $tmax ============")
        f = v -> gkdc(v, cfg.nk, cfg.Rt, tmax, cfg.β, cfg.η, cfg.E)
        for (idx_bond, maxbd) in enumerate(cfg.maxbond)
            println("Max bond dimension: $maxbd")
            for (j, name) in enumerate(topo_names)
                println("Topology: $name")
                ttn = TreeTCI.SimpleTCI{ComplexF64}(f, cfg.localdims, cfg.topo[name])
                seed_pivots!(ttn, 25, f)
                ranks, errors, bonds = TreeTCI.optimize!(ttn, f;
                    maxiter=cfg.maxit, maxbonddim=maxbd,
                    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
                mem[j, k, idx_bond] = Base.summarysize(ttn)
                error_l1[j, k, idx_bond] = sampled_error(f, ttn, cfg.nsamples,
                    2 * cfg.Rt + 2 * cfg.nk)
                println("  Sampled L1 error: ", error_l1[j, k, idx_bond])
            end
        end
    end

    return error_l1, mem, topo_names
end

function run_experiment_over_nk(cfg)
    """Run compression across different nk values for a fixed time (T=100)."""
    topo_names = collect(keys(build_topologies(cfg.nk_list[1], cfg.Rt)))
    ntopos = length(topo_names)
    error_l1 = zeros(ntopos, length(cfg.nk_list), length(cfg.maxbond))
    mem = zeros(ntopos, length(cfg.nk_list), length(cfg.maxbond))
    tmax = 100.0

    for (ik, nkv) in enumerate(cfg.nk_list)
        println("============ nk = $nkv ============")
        topo = build_topologies(Int(nkv), cfg.Rt)
        localdims = fill(2, 2 * cfg.Rt + 2 * Int(nkv))
        for (idx_bond, maxbd) in enumerate(cfg.maxbond)
            println("Max bond dimension: $maxbd")
            for (j, name) in enumerate(topo_names)
                println("Topology: $name")
                f = v -> gkdc(v, Int(nkv), cfg.Rt, tmax, cfg.β, cfg.η, cfg.E)
                ttn = TreeTCI.SimpleTCI{ComplexF64}(f, localdims, topo[name])
                seed_pivots!(ttn, 25, f)
                ranks, errors, bonds = TreeTCI.optimize!(ttn, f;
                    maxiter=cfg.maxit, maxbonddim=maxbd,
                    sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
                mem[j, ik, idx_bond] = Base.summarysize(ttn)
                error_l1[j, ik, idx_bond] = sampled_error(f, ttn, cfg.nsamples,
                    2 * cfg.Rt + 2 * Int(nkv))
                println("  Sampled L1 error: ", error_l1[j, ik, idx_bond])
            end
        end
    end

    return error_l1, mem, topo_names, cfg.nk_list
end

function build_topologies(nk::Int, Rt::Int)
    return Dict(
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        "QTT_Alt2" => QTTALT2(Rt, nk),
        "QTT_Alt" => QTTALT(Rt, nk),
        "BTTN2" => BTTN2(Rt, nk),
        "BTTN3" => BTTN3(Rt, nk),
    )
end

function save_results(prefix, cfg, error_l1, mem, topo_names)
    outdir = joinpath("results", "GKDC")
    mkpath(outdir)

    # Serialized full dump
    serfile = joinpath(outdir, string(prefix, "_gkdc_results.jls"))
    open(serfile, "w") do io
        serialize(io, (cfg, error_l1, mem, topo_names))
    end

    # CSV: error_l1 (topo, time, maxbond, value)
    errfile = joinpath(outdir, string(prefix, "_error_l1.csv"))
    open(errfile, "w") do io
        write(io, "topo,time,maxbond,value\n")
        for j in 1:length(topo_names)
            for (k, tmax) in enumerate(cfg.times)
                for (idx_bond, mb) in enumerate(cfg.maxbond)
                    write(io, "$(topo_names[j]),$(tmax),$(mb),$(error_l1[j, k, idx_bond])\n")
                end
            end
        end
    end

    # CSV: memory usage (topo, time, maxbond, bytes)
    memfile = joinpath(outdir, string(prefix, "_memory.csv"))
    open(memfile, "w") do io
        write(io, "topo,time,maxbond,bytes\n")
        for j in 1:length(topo_names)
            for (k, tmax) in enumerate(cfg.times)
                for (idx_bond, mb) in enumerate(cfg.maxbond)
                    write(io, "$(topo_names[j]),$(tmax),$(mb),$(mem[j, k, idx_bond])\n")
                end
            end
        end
    end

    return serfile, errfile, memfile
end

function save_results_nk(prefix, cfg, error_l1, mem, topo_names)
    outdir = joinpath("results", "GK")
    mkpath(outdir)

    serfile = joinpath(outdir, string(prefix, "_gk_nk_results.jls"))
    open(serfile, "w") do io
        serialize(io, (cfg, error_l1, mem, topo_names))
    end

    errfile = joinpath(outdir, string(prefix, "_nk_error_l1.csv"))
    open(errfile, "w") do io
        write(io, "topo,nk,maxbond,value\n")
        for j in 1:length(topo_names)
            for (ik, nkv) in enumerate(cfg.nk_list)
                for (idx_bond, mb) in enumerate(cfg.maxbond)
                    write(io, "$(topo_names[j]),$(nkv),$(mb),$(error_l1[j, ik, idx_bond])\n")
                end
            end
        end
    end

    memfile = joinpath(outdir, string(prefix, "_nk_memory.csv"))
    open(memfile, "w") do io
        write(io, "topo,nk,maxbond,bytes\n")
        for j in 1:length(topo_names)
            for (ik, nkv) in enumerate(cfg.nk_list)
                for (idx_bond, mb) in enumerate(cfg.maxbond)
                    write(io, "$(topo_names[j]),$(nkv),$(mb),$(mem[j, ik, idx_bond])\n")
                end
            end
        end
    end

    return serfile, errfile, memfile
end

function main()
    cfg = default_config()
    error_l1, mem, topo_names = run_experiment(cfg)
    save_results("gkdc", cfg, error_l1, mem, topo_names)
end

function main_nk()
    cfg = default_config()
    error_l1, mem, topo_names, nk_list = run_experiment_over_nk(cfg)
    save_results_nk("gkdc", cfg, error_l1, mem, topo_names)
end