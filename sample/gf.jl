using Random
using Plots
using LaTeXStrings
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

function gf(v)
    """Green's function in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    if x > y
        return 0
    else
        return 1im * exp(-1im * 30.0 * (x - y)) # added exponential decay for better numerical stability
    end
end

function default_config()
    R = 16  # number of bits per dimension
    d = 2   # spatial dimension
    localdims = fill(2, d * R)
    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d),
    )
    maxit = 5
    nsamples = 1000
    maxbond = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    return (R=R, d=d, localdims=localdims, topo=topo, maxit=maxit, nsamples=nsamples, maxbond=maxbond)
end

function plotgf()
    f(x, y) = y > x ? 0 : 1im * exp(-1im * 30.0 * (x - y)) # added exponential decay for better numerical stability
    xs = ys = range(0, 1, length=1000)
    zs = [f(x, y) for x in xs, y in ys]
    realpart = heatmap(xs, ys, real.(zs), xlabel=L"$t$", ylabel=L"$t'$",
        size=(600, 500), guidefont=font(12), tickfont=font(10))
    imagpart = heatmap(xs, ys, imag.(zs), xlabel=L"$t$", ylabel=L"$t'$",
        size=(600, 500), guidefont=font(12), tickfont=font(10))
    mkpath("PDF/GF")
    savefig(plot(realpart, imagpart, layout=(1, 2), size=(1200, 500)), "PDF/GF/gf_function.pdf")
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
            ttn = TreeTCI.SimpleTCI{ComplexF64}(gf, cfg.localdims, topology)
            ranks, errors = TreeTCI.optimize!(ttn, gf;
                tolerance=1e-16, maxiter=cfg.maxit, maxbonddim=maxbd,
                sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
            errl1 = sampled_error(gf, ttn, cfg.nsamples, cfg.R, cfg.d)
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
    outdir = joinpath("results", "GF")
    mkpath(outdir)

    # Serialized full dump
    serfile = joinpath(outdir, string(prefix, "_gf_results.jls"))
    open(serfile, "w") do io
        serialize(io, (cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names))
    end

    # CSV: error_l1 (topo, maxbond, value)
    errfile = joinpath(outdir, string(prefix, "_error_l1.csv"))
    open(errfile, "w") do io
        write(io, "topo,maxbond,value\n")
        for j in 1:length(topo_names)
            for (idx_bond, mb) in enumerate(cfg.maxbond)
                write(io, "$(topo_names[j]),$(mb),$(error_l1[j, idx_bond])\n")
            end
        end
    end

    # CSV: memory usage (topo, maxbond, bytes)
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
    save_results("gf", cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names)
end