using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "topologies.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
using .TTNUtils: bits2decimal, fused_preparations, compress_indexset
using TreeTCI

"""Sample L1 error of `ttn` approximation to function `f`.

If `ttn` is a `TreeTCI.SimpleTCI`, the function is converted to a concrete
`TreeTensorNetwork` before evaluation.

Arguments
- `f`: function to approximate
- `ttn`: TreeTCI model or concrete TreeTensorNetwork
- `nsamples`: number of random samples to average
- `bits`, `d`: used to build random inputs of length `d * bits`
"""
function sampled_error(f, ttn, nsamples, bits, d)
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

function dcgf(v)
    """ Fictious Green's function undergoing a DC field in 2D. """
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    A = 10.0 # amplitude of the dc field
    B = 10.0 # damping factor
    return 1im * exp(-1im * A * (sin(10x) + cos(10x))) * exp(1im * A * (sin(10y) + cos(10y))) * exp(-B * abs(x - y))
end

function init(topo::Dict{String,NamedGraph{Int64}}, f; R::Int, d::Int, localdims::Vector{Int})
    """Run structural optimization cycles over the provided topologies.

    For each topology we perform a sequence of `ttnopt` cycles with increasing
    `max_degree` (1, 2, 3). Each resulting graph is stored in `newtopo` with
    suffixes `_struct`, `_struct2`, and `_maxdeg3` respectively.
    """
    newtopo = Dict()
    kwargs = (
        maxiter=5,
        sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        tolerance=1e-10,
    )

    println("---- Starting structural optimization runs ----")
    for (name, graph) in pairs(topo)
        println("Topology $name:")

        # initial cross-interpolation and diagnostics
        ttn, ranks, errors, bonds = TreeTCI.crossinterpolate(ComplexF64, f, localdims, graph; kwargs...)
        println("  Sampled L1 error: ", sampled_error(f, ttn, 1000, R, d))
        println("  Max bond dim: ", ranks[end])

        center_vertex = 1

        for i in 1:3
            maxdeg = i
            try
                g_tmp, original_entanglements, entanglements = TreeTCI.ttnopt(ttn; ortho_vertex=center_vertex, max_degree=maxdeg, T0=0.0)
                newtopo[string(name, "_maxdeg", string(i))] = g_tmp
            catch e
                @warn "ttnopt failed for topology $name with maxdeg=$maxdeg: $e"
                continue
            end
        end
    end
    return newtopo
end

function main()
    # Parameters for TCI
    R = 16 # number of bits per dimension
    d = 2 # spatial dimension
    localdims = fill(2, d * R) # local dimensions for d dimensions with R bits each

    topo = Dict(
        #"BTTN" => BTTN(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        #"QTT_Seq" => QTT_Block(R, d),
        #"CTTN" => CTTN(R, d),
        #"TTTN" => TTTN(R, d)
    )

    newtopo = init(Dict("QTT_Int" => QTT_Alt(R, d)), dcgf; R=R, d=d, localdims=localdims)

    fulltopo = merge(topo, newtopo)

    ntopos = length(fulltopo)
    nsteps = 6
    step = 10
    maxit = 5
    nsamples = 1000

    # Output storage
    error_l1 = zeros(ntopos, nsteps)
    error_pivot = zeros(ntopos, nsteps)
    rankendlist = zeros(ntopos, nsteps)
    ranklist = zeros(ntopos, nsteps)
    mem = zeros(ntopos, nsteps)

    kwargs = (
        maxiter=maxit,
        sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        tolerance=1e-10,
        maxbonddim=60,
    )

    for (i, (name, graph)) in enumerate(fulltopo)
        for k in 1:nsteps
            kwargs = merge(kwargs, (maxbonddim=step * k,))
            println("Topology: $name")
            ttn, ranks, errors, bonds = TreeTCI.crossinterpolate(ComplexF64, dcgf, localdims, graph; kwargs...)
            mem[i, k] = Base.summarysize(ttn)
            errl1 = sampled_error(dcgf, ttn, nsamples, R, d)
            error_l1[i, k] = errl1
            rankendlist[i, k] = ranks[end]
            ranklist[i, k] = step * k
            println(" Sampled L1 error: ", errl1)
        end
    end

    p1 = plot(title="Sampled L1 Error vs Max Bond Dimension", xlabel="Max Bond Dimension", ylabel="Sampled L1 Error", yscale=:log10)
    p2 = plot(title="Sampled L1 Error vs Memory", xlabel="Memory (kB)", ylabel="Sampled L1 Error", yscale=:log10)
    topo_names = collect(keys(fulltopo))
    for i in 1:ntopos
        plot!(p1, step:step:step*nsteps, error_l1[i, :], label=topo_names[i], marker=:o)
        plot!(p2, mem[i, :] / 1000, error_l1[i, :], label=topo_names[i], marker=:o)
    end
    plt = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    savefig(plt, "dcgf_sampled_l1_error.svg")
    display(plt)


end