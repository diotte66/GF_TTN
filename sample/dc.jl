using Random
using Plots
using LaTeXStrings
using Plots.PlotMeasures
include(joinpath(@__DIR__, "..", "src", "topologies.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
using TreeTCI
using Serialization
using DelimitedFiles

function sampled_error(f, ttn, nsamples, bits, d)
    """Compute sampled errors between function f and ttn approximation over nsamples random inputs."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end
    error_l1 = 0.0
    for _ in 1:nsamples
        x = rand(1:2, d * bits)
        approx = TreeTCI.evaluate(eval_ttn, x)
        error_l1 += abs(f(x) - approx)
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

function dcgf(v)
    """Green's function under a DC field in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    A = 5.0
    B = 5.0
    E = 10.0
    return 1im * exp(-1im * A * (sin(E * x) / E + (cos(E * x) - 1) / E)) *
           exp(1im * A * (sin(E * y) / E + (cos(E * y) - 1) / E)) *
           exp(-B * abs(x - y))
end

function default_config()
    R = 16
    d = 2
    localdims = fill(2, d * R)
    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d),
        "TTTN" => TTTN(R, d),
    )
    maxit = 5
    nsamples = 1000
    maxbond = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]
    return (R=R, d=d, localdims=localdims, topo=topo, maxit=maxit, nsamples=nsamples, maxbond=maxbond)
end

function plotdcgf()
    A = 5.0
    B = 5.0
    E = 10.0
    f(x, y) = 1im * exp(-1im * A * (sin(E * x) / E + (cos(E * x) - 1) / E)) *
              exp(1im * A * (sin(E * y) / E + (cos(E * y) - 1) / E)) *
              exp(-B * abs(x - y))
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

function run_experiment(cfg)
    topo = cfg.topo
    ntopos = length(topo)
    nbonds = length(cfg.maxbond)
    topo_names = collect(keys(topo))

    error_l1 = zeros(ntopos, nbonds)
    mem = zeros(ntopos, nbonds)
    error_pivot = zeros(ntopos, nbonds)
    rankendlist = zeros(ntopos, nbonds)
    ranklist = zeros(ntopos, nbonds)

    for (idx_bond, maxbd) in enumerate(cfg.maxbond)
        println("Max bond dimension: $maxbd")
        for (j, (toponame, topology)) in enumerate(topo)
            println("Topology: $toponame")
            ttn = TreeTCI.SimpleTCI{ComplexF64}(dcgf, cfg.localdims, topology)
            ranks, errors = TreeTCI.optimize!(ttn, dcgf;
                tolerance=1e-16, maxiter=cfg.maxit, maxbonddim=maxbd,
                sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
            errl1 = sampled_error(dcgf, ttn, cfg.nsamples, cfg.R, cfg.d)
            mem[j, idx_bond] = Base.summarysize(ttn)
            error_l1[j, idx_bond] = errl1
            error_pivot[j, idx_bond] = errors[end]
            rankendlist[j, idx_bond] = ranks[end]
            ranklist[j, idx_bond] = maxbd
            println("  Sampled L1 error: ", errl1)
        end
    end

    return error_l1, mem, error_pivot, rankendlist, ranklist, topo_names
end

function save_results(prefix, cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names)
    outdir = joinpath("results", "DCGF")
    mkpath(outdir)

    serfile = joinpath(outdir, string(prefix, "_dcgf_results.jls"))
    open(serfile, "w") do io
        serialize(io, (cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names))
    end

    errfile = joinpath(outdir, string(prefix, "_error_l1.csv"))
    open(errfile, "w") do io
        write(io, "topo,maxbond,value\n")
        for j in 1:length(topo_names)
            for (idx_bond, mb) in enumerate(cfg.maxbond)
                write(io, "$(topo_names[j]),$(mb),$(error_l1[j, idx_bond])\n")
            end
        end
    end

    memfile = joinpath(outdir, string(prefix, "_memory.csv"))
    open(memfile, "w") do io
        write(io, "topo,maxbond,bytes\n")
        for j in 1:length(topo_names)
            for (idx_bond, mb) in enumerate(cfg.maxbond)
                write(io, "$(topo_names[j]),$(mb),$(mem[j, idx_bond])\n")
            end
        end
    end

    return serfile, errfile, memfile
end

function main()
    cfg = default_config()
    error_l1, mem, error_pivot, rankendlist, ranklist, topo_names = run_experiment(cfg)
    save_results("dcgf", cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names)
end