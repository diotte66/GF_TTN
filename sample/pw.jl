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

function make_pw(d::Int, n_waves::Int, R::Int; seed=42)
    """
    Build a plane wave function for arbitrary spatial dimension d.
      - d:       spatial dimension
      - n_waves: number of wavevectors
      - R:       number of bits per dimension
    Returns a closure f(v) that encodes the plane wave sum.
    """
    rng = MersenneTwister(seed)
    k = randn(rng, d, n_waves)   # (d × n_waves) wavevector matrix

    function pw(v)
        # Split bit vector into d equal blocks, each of length R
        r = [bits2decimal(v[(i-1)*R+1:i*R]) for i in 1:d]
        s = 0.0
        for i in 1:n_waves
            s += cos(i * r' * k[:, i])
        end
        return s
    end

    return pw, k
end

function default_config(; d=3, n_waves=30, R=16, seed=42)
    localdims = fill(2, d * R)
    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d),
    )
    maxit = 5
    nsamples = 1000
    maxbond = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
        33, 36, 39, 42, 45, 48, 51, 54, 57, 60]

    pw, k = make_pw(d, n_waves, R; seed=seed)

    return (R=R, d=d, n_waves=n_waves, seed=seed, localdims=localdims,
        topo=topo, maxit=maxit, nsamples=nsamples, maxbond=maxbond,
        pw=pw, k=k)
end

function plotpw(cfg)
    """Plot a 2D slice of the plane wave function (first two dimensions)."""
    @assert cfg.d >= 2 "Need at least d=2 to plot a 2D slice"
    xs = ys = range(0, 1, length=400)

    # Fix all dimensions beyond the first two to 0.5
    function f2d(x, y)
        r = fill(0.5, cfg.d)
        r[1] = x
        r[2] = y
        s = 0.0
        for i in 1:cfg.n_waves
            s += cos(i * dot(r, cfg.k[:, i]))
        end
        return s
    end

    zs = [f2d(x, y) for x in xs, y in ys]
    p = heatmap(xs, ys, zs,
        xlabel=L"$x_1$", ylabel=L"$x_2$",
        colorbar_title=L"Amplitude $f$",
        size=(700, 600), guidefont=font(12), tickfont=font(10),
        colorbar_tickfont=font(9),
        left_margin=10mm, bottom_margin=10mm)
    mkpath("PDF/PW")
    savefig(p, "PDF/PW/plane_wave_d$(cfg.d)_amplitude.pdf")
    display(p)
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
            ttn = TreeTCI.SimpleTCI{Float64}(cfg.pw, cfg.localdims, topology)
            ranks, errors = TreeTCI.optimize!(ttn, cfg.pw;
                maxiter=cfg.maxit, maxbonddim=maxbd,
                sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
            errl1 = sampled_error(cfg.pw, ttn, cfg.nsamples, cfg.R, cfg.d)
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
    outdir = joinpath("results", "PW")
    mkpath(outdir)

    serfile = joinpath(outdir, string(prefix, "_pw_results.jls"))
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

function main(; d=3, n_waves=30, R=16, seed=42)
    cfg = default_config(d=d, n_waves=n_waves, R=R, seed=seed)
    println("Running plane wave experiment: d=$d, n_waves=$n_waves, R=$R")
    error_l1, mem, error_pivot, rankendlist, ranklist, topo_names = run_experiment(cfg)
    save_results("pw_d$(d)", cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names)
end