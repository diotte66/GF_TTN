# This file defines the Green's functions and the main experiment loop for testing TTN compression on momentum-dependent Green's functions.

# =======================================================
# Libraries and utilities
# -------------------------------------------------------

using Random
using Plots
using CSV
using DataFrames
include(joinpath(@__DIR__, "..", "src", "utils_gk.jl"))
using TreeTCI

# =======================================================
# Green's function (Equilibrium, with/without fields)
# -------------------------------------------------------

"""
    gk(v, nk, R, T, β, η) -> ComplexF64

Equilibrium Green's function G_{kx,ky}(t,t').
Encodes kx, ky, t, t' from bits `v` using `nk` and `R`.
β is the inverse temperature and η is the damping factor ~(U/t)^2.
"""

function gk(v, nk, R, T, β::Float64, η::Float64)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky, t1, t2 = assign(v, nk, R)
    return -1im * n(kx, ky, β) * exp(-1im * ϵ(kx, ky) * T * (t1 - t2)) * exp(-T * (ϵ(kx, ky))^2 * η * abs(t1 - t2))
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
    return -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * T * t1) / E) + δ(kx, ky) * (cos(E * T * t1) - 1) / E - ϵ(kx, ky) * sin(E * T * t2) / E - δ(kx, ky) * (cos(E * T * t2) - 1) / E)) * exp(-T * (ϵ(kx, ky)^2) * η * abs(t1 - t2)) * n(kx, ky, β)
end

"""
    gkac(v, nk, R, T, β, E) -> ComplexF64
Momentum-dependent equilibrium Green's function under an AC field.
Encodes kx, ky, t, t' from bits `v` using `nk` and `R`.
β is the inverse temperature, E controls the electric field amplitude, and η is the damping factor ~(U/t)^2.
"""

function gkac(v, nk, R, T, β, η, E)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') under an AC field """
    kx, ky, t1, t2 = assign(v, nk, R)
    ω = 1.0
    intregand(t) = cos(-E * sin(ω * t) / ω) + sin(-E * sin(ω * t) / ω)
    Intx = Integrals.quadgk(intregand, 0, T * t1)[1]
    Inty = Integrals.quadgk(intregand, 0, T * t2)[1]
    return 1im * exp(-1im * Intx) * exp(1im * Inty) * exp(-T * (ϵ(kx, ky)^2) * η * abs(t1 - t2)) * n(kx, ky, β)
end

# =======================================================
# Setup preparation
# -------------------------------------------------------

"""
    build_topologies(nk, Rt) -> Dict{String, NamedGraph}

Convenience selector for topologies to test.
"""
function build_topologies(nk::Int, Rt::Int)
    return Dict(
        "CTTN_Alt2" => CTTNALT2(Rt, nk),
        "QTT_Alt2" => QTTALT2(Rt, nk),
        "ITTN" => ITTN(Rt, nk),
    )
end

"""
    make_gf(name, cfg) -> Function

Return a single-argument closure `v -> GF(v, ...)` for the Green's function
identified by `name` ∈ {"gk", "gkac", "gkdc"}, capturing all parameters from `cfg`.

The closure has the same signature expected everywhere in the code: `f(v)`.
"""
function make_gf(name::String, cfg)
    if name == "gk"
        return (v, T, η) -> gk(v, cfg.nk, cfg.Rt, T, cfg.β, η)
    elseif name == "gkac"
        return (v, T, η) -> gkac(v, cfg.nk, cfg.Rt, T, cfg.β, η, cfg.E)
    elseif name == "gkdc"
        return (v, T, η) -> gkdc(v, cfg.nk, cfg.Rt, T, cfg.β, η, cfg.E)
    else
        error("Unknown Green's function: '$name'. Choose among \"gk\", \"gkac\", \"gkdc\".")
    end
end

"""
    default_config() -> NamedTuple

Provide canonical parameters and derived values.
`gf_name`  selects which Green's function to use: "gk" | "gkac" | "gkdc".
`gf`       is the corresponding closure `(v, T, η) -> ComplexF64`.
"""
function default_config(; gf_name::String="gkdc")
    Rt = 30                               # number of time bits
    Nk = 32.                              # number of k-points in each direction
    nk = Int(log2(Nk))                    # number of bits to encode the Nk kx,ky points
    β = 10.0                              # inverse temperature
    E = 1.0                               # electric field amplitude for gkdc / gkac
    maxit = 5                             # maximum number of iterations
    nsamples = 1000                       # number of samples for error estimation
    times = [50.0, 100.0, 150.0, 200.0]
    etas = [0.001, 0.01, 0.1, 0.0, 1.0]   # ← η sweep
    maxbond = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    localdims = fill(2, 2 * Rt + 2 * nk)
    topo = build_topologies(nk, Rt)

    _partial = (nk=nk, Rt=Rt, β=β, E=E)
    gf = make_gf(gf_name, _partial)       # closure: (v, T, η) -> ComplexF64

    return (Rt=Rt, Nk=Nk, nk=nk, β=β, E=E, maxit=maxit, nsamples=nsamples,
        times=times, etas=etas, maxbond=maxbond, localdims=localdims,
        topo=topo, gf_name=gf_name, gf=gf)
end

# =======================================================
# Running the experiment
# -------------------------------------------------------

"""
    run_experiment(cfg) -> DataFrame

Run compression across **all topologies × all times × all η values × all bond dimensions**.
Returns a tidy DataFrame with one row per (topology, T, η, maxbond) combination,
containing the sampled L1 error and memory usage.

The DataFrame is also written to `<gf_name>.csv` in the current directory.
"""
function run_experiment(cfg)
    rows = NamedTuple{
        (:gf_name, :topology, :T, :eta, :maxbond, :error_l1, :mem_bytes),
        Tuple{String,String,Float64,Float64,Int,Float64,Int}
    }[]

    topo_names = collect(keys(cfg.topo))

    for (k, tmax) in enumerate(cfg.times)
        println("\n============ Time T = $tmax ============")
        for (ie, η) in enumerate(cfg.etas)
            println("  ---- η = $η ----")
            for (idx_bond, maxbd) in enumerate(cfg.maxbond)
                println("  Max bond dimension: $maxbd")
                for name in topo_names
                    topology = cfg.topo[name]
                    println("    Topology: $name")

                    f = v -> cfg.gf(v, tmax, η)

                    kwargs = (
                        maxiter=cfg.maxit,
                        maxbonddim=maxbd,
                        sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
                    )

                    ttn = TreeTCI.SimpleTCI{ComplexF64}(f, cfg.localdims, topology)
                    seed_pivots!(ttn, 25, f)
                    TreeTCI.optimize!(ttn, f; kwargs...)

                    mem_bytes = Base.summarysize(ttn)
                    errl1 = sampled_error(f, ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)

                    println("      Sampled L1 error: $errl1  |  Memory: $mem_bytes bytes")

                    push!(rows, (
                        gf_name=cfg.gf_name,
                        topology=name,
                        T=tmax,
                        eta=η,
                        maxbond=maxbd,
                        error_l1=errl1,
                        mem_bytes=mem_bytes,
                    ))
                end
            end
        end
    end

    df = DataFrame(rows)

    # ── Save to CSV ────────────────────────────────────────────────────────────
    csv_path = joinpath(@__DIR__, "results", "$(cfg.gf_name).csv")
    CSV.write(csv_path, df)
    println("\nResults saved to: $csv_path")
    # ──────────────────────────────────────────────────────────────────────────

    return df
end

# =======================================================
# Additional analysis and plotting functions
# -------------------------------------------------------

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
        η = cfg.etas[2]    # use default η = 0.01
        f = v -> cfg.gf(v, time, η)
        ttn = TreeTCI.SimpleTCI{ComplexF64}(f, cfg.localdims, cfg.topo["QTT_Alt2"])
        seed_pivots!(ttn, 25, f)
        ranks, errors, bonds = TreeTCI.optimize!(ttn, f; kwargs...)
        errl1 = sampled_error(f, ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
        println(" Sampled L1 error: ", errl1)
        mem = Base.summarysize(ttn)
        println("Memory usage (Bytes): ", mem)
        list = trainbondlist(bonds, cfg.nk, cfg.Rt)
        push!(listlist, list)
    end
    multipletrainbondplot(listlist, cfg.nk, cfg.Rt, times)
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
        η = cfg.etas[2]    # use default η = 0.01
        f = v -> cfg.gf(v, time, η)
        ttn = TreeTCI.SimpleTCI{ComplexF64}(f, cfg.localdims, cfg.topo["CTTN_Alt2"])
        seed_pivots!(ttn, 25, f)
        ranks, errors, bonds = TreeTCI.optimize!(ttn, f; kwargs...)
        errl1 = sampled_error(f, ttn, cfg.nsamples, 2 * cfg.Rt + 2 * cfg.nk)
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

# =======================================================
# Main function and plotting
# -------------------------------------------------------

"""
    main(; gf_name)

Entry point: configure, run, and plot results.
Uses `gf_name` keyword to select the Green's function (default: "gk").

The experiment sweeps over **all topologies**, **all times**, **all η values**, and
**all bond dimensions**. Results are saved to `<gf_name>.csv`.

Examples:
    main()                  # uses gk
    main(gf_name="gkdc")    # uses gkdc
    main(gf_name="gkac")    # uses gkac
"""
function main(; gf_name::String="gk")
    cfg = default_config(; gf_name)
    println("Using Green's function: $(cfg.gf_name)")
    println("Topologies : $(collect(keys(cfg.topo)))")
    println("Times      : $(cfg.times)")
    println("η values   : $(cfg.etas)")
    println("Bond dims  : $(cfg.maxbond)")
    df = run_experiment(cfg)
    return df
end