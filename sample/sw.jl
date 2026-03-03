using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "topologies.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
using .TTNUtils: bits2decimal
using TreeTCI
using Serialization
using DelimitedFiles

function sampled_error(f, ttn, nsamples, bits, d)
    """ Compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end
    error_l1 = 0.0
    for _ in 1:nsamples
        # Generate a random 3R sequence of 1s and 2s
        x = rand(1:2, d * bits)
        # Evaluate the concrete TreeTensorNetwork (it provides evaluate/call)
        approx = TreeTCI.evaluate(eval_ttn, x)
        err = abs(f(x) - approx)
        error_l1 += err
    end
    return error_l1 / nsamples
end

function sw(v)
    #generate a function representing a 2D spherical wave centered at (0.5, 0.5) and plot it using Plots within the unit square [0, 1] x [0, 1]
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    return sin(20 * sqrt((x - 0.5)^2 + (y - 0.5)^2)) / (1 + 10 * sqrt((x - 0.5)^2 + (y - 0.5)^2))

end

function plotsw()
    f(x, y) = sin(20 * sqrt((x - 0.5)^2 + (y - 0.5)^2)) / (1 + 10 * sqrt((x - 0.5)^2 + (y - 0.5)^2))
    xs = ys = range(0, 1, length=1000)
    zs = [f(x, y) for x in xs, y in ys]
    heatmap(xs, ys, zs, xlabel=L"$x$", ylabel=L"$y$", colorbar_title=L"Amplitude $f$", size=(900, 700), guidefont=font(12), tickfont=font(10), colorbar_tickfont=font(9))
    savefig("PDF/SW/spherical_wave_function.pdf")
end

function default_config()
    R = 16 # number of bits per dimension
    d = 2  # spatial dimension
    localdims = fill(2, d * R)
    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d),
    )
    maxit = 5
    nsamples = 1000
    maxbond = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]
    return (R=R, d=d, localdims=localdims, topo=topo, maxit=maxit, nsamples=nsamples, maxbond=maxbond)
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
            ttn = TreeTCI.SimpleTCI{Float64}(sw, cfg.localdims, topology)
            ranks, errors = TreeTCI.optimize!(ttn, sw; maxiter=cfg.maxit, maxbonddim=maxbd, sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
            errl1 = sampled_error(sw, ttn, cfg.nsamples, cfg.R, cfg.d)
            mem[j, idx_bond] = Base.summarysize(ttn)
            error_l1[j, idx_bond] = errl1
            error_pivot[j, idx_bond] = errors[end]
            rankendlist[j, idx_bond] = ranks[end]
            ranklist[j, idx_bond] = maxbd
            println(" Sampled L1 error: ", errl1)
        end
    end

    return error_l1, mem, error_pivot, rankendlist, ranklist, topo_names
end

function save_results(prefix, cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names)
    outdir = joinpath("results", "SW")
    mkpath(outdir)
    serfile = joinpath(outdir, string(prefix, "_sw_results.jls"))
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
    save_results("sw", cfg, error_l1, mem, error_pivot, rankendlist, ranklist, topo_names)
end
