# This file defines the Green's functions and the main experiment loop for testing TTN compression on momentum-independent Green's functions.

# =======================================================
# Libraries and utilities
# -------------------------------------------------------

using Random
using Plots
using LaTeXStrings
using Plots.PlotMeasures
using Statistics
include(joinpath(@__DIR__, "..", "src", "utils_gf.jl"))
using TreeTCI
using Serialization
using CSV
using DataFrames

# =======================================================
# Green's functions
# -------------------------------------------------------

function gf(v, T, η)
    """Green's function in 2D with parameters T (frequency) and η (damping)."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    x = T * x
    y = T * y
    if x > y
        return 1im * exp(-1im * (x - y)) * exp(-η * (x - y))
    else
        return 0
    end
end

function dcgf(v, T, η)
    """Green's function under a DC field in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    E = 10.0
    return 1im * exp(-1im * T * (sin(E * x) / E + (cos(E * x) - 1) / E)) *
           exp(1im * T * (sin(E * y) / E + (cos(E * y) - 1) / E)) *
           exp(-η * abs(x - y))
end

function acgf(v, T, η)
    """Green's function under an AC field in 2D."""
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    ω = 1.0
    integrand(t) = cos(-T * sin(ω * t) / ω) + sin(-T * sin(ω * t) / ω)
    Intx = Integrals.quadgk(integrand, 0, x)[1]
    Inty = Integrals.quadgk(integrand, 0, y)[1]
    return 1im * exp(-1im * Intx) * exp(1im * Inty) * exp(-η * abs(x - y))
end

# =======================================================
# Green's function selector
# -------------------------------------------------------

function make_gf(gf_name::String)
    if gf_name == "gf"
        return (v, T, η) -> gf(v, T, η)
    elseif gf_name == "dcgf"
        return (v, T, η) -> dcgf(v, T, η)
    elseif gf_name == "acgf"
        return (v, T, η) -> acgf(v, T, η)
    else
        error("Unknown Green's function: '$gf_name'. Choose among \"gf\", \"dcgf\", \"acgf\".")
    end
end

# =======================================================
# Configuration
# -------------------------------------------------------

"""
    default_config(; gf_name) -> NamedTuple

Configuration for the full systematic sweep.
  times    : oscillation frequency values  (formerly A_list)
  etas     : damping values                (formerly B_list)
  gf_name  : which Green's function to compress — "gf" | "dcgf" | "acgf"
"""
function default_config(; gf_name::String="gf")
    R = 20
    d = 2
    localdims = fill(2, d * R)
    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d),
        "ITTN" => ITTN(R, d),
    )
    maxit = 5
    nsamples = 1000
    maxbond = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    times = [1.0, 5.0, 15.0, 30.0, 50.0, 100.0, 200.0, 400.0]
    etas = [0.5, 2.0, 5.0, 10.0]

    f = make_gf(gf_name)

    return (R=R, d=d, localdims=localdims, topo=topo, maxit=maxit,
        nsamples=nsamples, maxbond=maxbond, times=times, etas=etas,
        gf_name=gf_name, gf=f)
end

# =======================================================
# Experiment
# -------------------------------------------------------

"""
    run_experiment(cfg) -> DataFrame

Full sweep: topology × T × η × maxbond.
Results are saved to `<gf_name>.csv` in the script directory.
"""
function run_experiment(cfg)
    rows = NamedTuple{
        (:gf_name, :topology, :T, :eta, :maxbond, :error_l1, :mem_bytes),
        Tuple{String,String,Float64,Float64,Int,Float64,Int}
    }[]

    for T in cfg.times
        for η in cfg.etas
            println("\n===== T = $T, η = $η =====")
            f = v -> cfg.gf(v, T, η)
            for maxbd in cfg.maxbond
                println("  Bond = $maxbd")
                for (toponame, topology) in cfg.topo
                    println("    Topology: $toponame")
                    ttn = TreeTCI.SimpleTCI{ComplexF64}(f, cfg.localdims, topology)
                    seed_pivots!(ttn, 5, f)
                    TreeTCI.optimize!(ttn, f;
                        tolerance=1e-16, maxiter=cfg.maxit, maxbonddim=maxbd,
                        sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
                    errl1 = sampled_error(f, ttn, cfg.nsamples, cfg.R, cfg.d)
                    mem_bytes = Base.summarysize(ttn)
                    println("      $toponame → L1 = $errl1  |  mem = $mem_bytes bytes")
                    push!(rows, (
                        gf_name=cfg.gf_name,
                        topology=toponame,
                        T=T,
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
    csv_path = joinpath(@__DIR__, "$(cfg.gf_name).csv")
    CSV.write(csv_path, df)
    println("\nResults saved to: $csv_path")
    return df
end

# =======================================================
# Main
# -------------------------------------------------------

"""
    main(; gf_name)

Entry point. Select the Green's function to compress:
  "gf"   – standard equilibrium GF  (default)
  "dcgf" – GF under DC field
  "acgf" – GF under AC field

Runs the full sweep (topology × T × η × bond dim) and saves results to `<gf_name>.csv`.

Examples:
    main()                 # uses gf
    main(gf_name="dcgf")   # uses dcgf
    main(gf_name="acgf")   # uses acgf
"""
function main(; gf_name::String="gf")
    cfg = default_config(; gf_name)
    println("Green's function : $(cfg.gf_name)")
    println("Topologies       : $(collect(keys(cfg.topo)))")
    println("Times            : $(cfg.times)")
    println("η values         : $(cfg.etas)")
    println("Bond dims        : $(cfg.maxbond)")
    df = run_experiment(cfg)
    plot_systematic(cfg, df)
    plot_systematic_memory(cfg, df)
    return df
end