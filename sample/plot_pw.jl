using CSV
using DataFrames
using Plots
using LaTeXStrings
using Plots.PlotMeasures
using Statistics
using Printf

# =======================================================
# Load data
# -------------------------------------------------------

function load_all(datadir::String)
    data = Dict{Int, NamedTuple}()
    for d in 1:3
        err_path = joinpath(datadir, "pw_d$(d)_error_l1.csv")
        mem_path = joinpath(datadir, "pw_d$(d)_memory.csv")
        err_df   = CSV.read(err_path, DataFrame)
        mem_df   = CSV.read(mem_path, DataFrame)
        data[d]  = (err=err_df, mem=mem_df)
    end
    return data
end

# =======================================================
# Style helpers
# -------------------------------------------------------

const MARKERS   = [:circle, :square, :diamond, :utriangle, :dtriangle, :hexagon]
const LSTYLES   = [:solid, :dash, :dot, :dashdot, :dashdotdot, :solid]

function topo_style(topo_names)
    Dict(name => (marker=MARKERS[mod1(i,end)], ls=LSTYLES[mod1(i,end)])
         for (i, name) in enumerate(topo_names))
end

# =======================================================
# Plot 1 — L₁ error vs bond dimension, one panel per d
# All topologies overlaid, black lines
# -------------------------------------------------------

function plot_error_per_dim(data::Dict; outdir="PDF/PW")
    mkpath(outdir)

    # Collect all topology names across all dims for a consistent style map
    all_topos = unique(vcat([unique(data[d].err.topo) for d in 1:3]...))
    style     = topo_style(all_topos)

    fig = plot(
        layout        = (1, 3),
        size          = (950, 320),
        left_margin   = 10mm,
        bottom_margin = 8mm,
        top_margin    = 4mm,
        right_margin  = 2mm,
    )

    for d in 1:3
        df = sort(data[d].err, [:topo, :maxbond])
        topos = unique(df.topo)
        for topo in topos
            s = filter(r -> r.topo == topo, df)
            plot!(fig[d], s.maxbond, s.value;
                label         = d == 1 ? topo : "",
                color         = :black,
                marker        = style[topo].marker,
                linestyle     = style[topo].ls,
                markersize    = 4,
                linewidth     = 1.6,
                yscale        = :log10,
                xlabel        = L"\chi_{\max}",
                ylabel        = d == 1 ? L"L_1\ \mathrm{error}" : "",
                title         = L"d = %$d",
                titlefontsize = 11,
                guidefontsize = 10,
                tickfontsize  = 9,
                legend        = d == 1 ? :topright : false,
                grid          = true,
                gridalpha     = 0.2,
                framestyle    = :box,
            )
        end
    end

    path = joinpath(outdir, "pw_error_vs_bond.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Plot 2 — Pareto: memory vs L₁ error, one panel per d
# -------------------------------------------------------

function plot_pareto_per_dim(data::Dict; outdir="PDF/PW")
    mkpath(outdir)

    all_topos = unique(vcat([unique(data[d].err.topo) for d in 1:3]...))
    style     = topo_style(all_topos)

    fig = plot(
        layout        = (1, 3),
        size          = (950, 340),
        left_margin   = 16mm,
        bottom_margin = 14mm,
        top_margin    = 4mm,
        right_margin  = 6mm,
    )

    for d in 1:3
        err_df = sort(data[d].err, [:topo, :maxbond])
        mem_df = sort(data[d].mem, [:topo, :maxbond])
        topos  = unique(err_df.topo)

        for topo in topos
            e = sort(filter(r -> r.topo == topo, err_df), :maxbond)
            m = sort(filter(r -> r.topo == topo, mem_df), :maxbond)
            err_col = names(e)[end]
            mem_col = names(m)[end]
            mem_vals = m[!, mem_col] ./ 1e6
            # 4 evenly spaced ticks between min and max memory
            xmin, xmax = extrema(mem_vals)
            xtick_vals = range(xmin, xmax, length=4) |> collect
            xtick_vals = round.(xtick_vals, digits=2)

            plot!(fig[d], mem_vals, e[!, err_col];
                label         = d == 1 ? topo : "",
                color         = :black,
                marker        = style[topo].marker,
                linestyle     = style[topo].ls,
                markersize    = 4,
                linewidth     = 1.6,
                yscale        = :log10,
                xlabel        = "Memory (MB)",
                xticks        = xtick_vals,
                ylabel        = d == 1 ? L"L_1\ \mathrm{error}" : "",
                title         = L"d = %$d",
                titlefontsize = 11,
                guidefontsize = 10,
                tickfontsize  = 9,
                legend        = d == 1 ? :topright : false,
                grid          = true,
                gridalpha     = 0.2,
                framestyle    = :box,
            )
        end
    end

    path = joinpath(outdir, "pw_pareto.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Plot 5 — Memory vs bond dimension, one panel per d
# All topologies overlaid, black lines
# -------------------------------------------------------

function plot_memory_per_dim(data::Dict; outdir="PDF/PW")
    mkpath(outdir)

    all_topos = unique(vcat([unique(data[d].mem.topo) for d in 1:3]...))
    style     = topo_style(all_topos)

    fig = plot(
        layout        = (1, 3),
        size          = (950, 320),
        left_margin   = 10mm,
        bottom_margin = 8mm,
        top_margin    = 4mm,
        right_margin  = 2mm,
    )

    for d in 1:3
        df      = sort(data[d].mem, [:topo, :maxbond])
        mem_col = names(df)[end]
        topos   = unique(df.topo)
        for topo in topos
            s = filter(r -> r.topo == topo, df)
            plot!(fig[d], s.maxbond, s[!, mem_col] ./ 1e6;
                label         = d == 1 ? topo : "",
                color         = :black,
                marker        = style[topo].marker,
                linestyle     = style[topo].ls,
                markersize    = 4,
                linewidth     = 1.6,
                xlabel        = L"\chi_{\max}",
                ylabel        = d == 1 ? "Memory (MB)" : "",
                title         = L"d = %$d",
                titlefontsize = 11,
                guidefontsize = 10,
                tickfontsize  = 9,
                legend        = d == 1 ? :topright : false,
                grid          = true,
                gridalpha     = 0.2,
                framestyle    = :box,
            )
        end
    end

    path = joinpath(outdir, "pw_memory_vs_bond.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Plot 3 — Error ratio topo / QTT_Seq across dimensions
# One panel per topology (excluding reference), x-axis = d
# Shows at each d the ratio at the maximum shared bond dim
# -------------------------------------------------------

function plot_ratio_vs_dim(data::Dict; ref="QTT_Seq", outdir="PDF/PW")
    mkpath(outdir)

    all_topos    = unique(vcat([unique(data[d].err.topo) for d in 1:3]...))
    other_topos  = filter(t -> t != ref, all_topos)
    style        = topo_style(all_topos)
    bonds        = sort(unique(data[1].err.maxbond))

    fig = plot(
        size          = (700, 420),
        left_margin   = 14mm,
        bottom_margin = 12mm,
        top_margin    = 6mm,
        right_margin  = 6mm,
        xlabel        = "Dimension d",
        ylabel        = L"\log_{10}\!\left(\varepsilon_{\mathrm{topo}} / \varepsilon_{\mathrm{%$ref}}\right)",
        title         = "Error ratio vs dimension  (bond = $(bonds[end]))",
        titlefontsize = 11,
        guidefontsize = 10,
        tickfontsize  = 9,
        xticks        = ([1, 2, 3], ["d=1", "d=2", "d=3"]),
        legend        = :topright,
        grid          = true,
        gridalpha     = 0.2,
        framestyle    = :box,
    )

    maxbd = bonds[end]

    for topo in other_topos
        ratios = Float64[]
        for d in 1:3
            df   = data[d].err
            vt   = filter(r -> r.topo == topo && r.maxbond == maxbd, df)
            vr   = filter(r -> r.topo == ref  && r.maxbond == maxbd, df)
            push!(ratios,
                (nrow(vt) == 0 || nrow(vr) == 0 || vr.value[1] == 0) ? NaN :
                    log10(vt.value[1] / vr.value[1]))
        end
        plot!(fig, [1, 2, 3], ratios;
            label      = topo,
            color      = :black,
            marker     = style[topo].marker,
            linestyle  = style[topo].ls,
            markersize = 5,
            linewidth  = 1.6,
        )
    end

    hline!(fig, [0.0]; color=:gray, linestyle=:dash, linewidth=1, label=ref * " (ref)")

    path = joinpath(outdir, "pw_ratio_vs_dim.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Plot 4 — Error vs bond for each topology, one panel per topo
# All three dimensions overlaid (different line styles)
# -------------------------------------------------------

function plot_error_per_topo(data::Dict; outdir="PDF/PW")
    mkpath(outdir)

    all_topos = unique(vcat([unique(data[d].err.topo) for d in 1:3]...))
    ntopos    = length(all_topos)
    dim_ls    = [:solid, :dash, :dot]
    dim_mk    = [:circle, :square, :diamond]

    fig = plot(
        layout        = (1, ntopos),
        size          = (260 * ntopos, 320),
        left_margin   = 10mm,
        bottom_margin = 8mm,
        top_margin    = 4mm,
        right_margin  = 2mm,
    )

    for (j, topo) in enumerate(all_topos)
        for d in 1:3
            df = sort(filter(r -> r.topo == topo, data[d].err), :maxbond)
            isempty(df) && continue
            plot!(fig[j], df.maxbond, df.value;
                label         = j == 1 ? "d = $d" : "",
                color         = :black,
                marker        = dim_mk[d],
                linestyle     = dim_ls[d],
                markersize    = 4,
                linewidth     = 1.6,
                yscale        = :log10,
                xlabel        = L"\chi_{\max}",
                ylabel        = j == 1 ? L"L_1\ \mathrm{error}" : "",
                title         = topo,
                titlefontsize = 10,
                guidefontsize = 10,
                tickfontsize  = 9,
                legend        = j == 1 ? :topright : false,
                grid          = true,
                gridalpha     = 0.2,
                framestyle    = :box,
            )
        end
    end

    path = joinpath(outdir, "pw_error_per_topo.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Main
# -------------------------------------------------------

"""
    main(datadir; outdir)

Load all six CSVs from `datadir` and produce four comparative plots.

    main("results/PW")
    main("results/PW"; outdir="PDF/PW")
"""
function main(datadir::String; outdir::String="PDF/PW")
    println("Loading data from $datadir …")
    data = load_all(datadir)
    for d in 1:3
        topos = unique(data[d].err.topo)
        println("  d=$d : $(nrow(data[d].err)) rows, topologies: $topos")
    end

    plot_error_per_dim(data;    outdir)   # Plot 1
    plot_pareto_per_dim(data;   outdir)   # Plot 2
    plot_memory_per_dim(data;   outdir)   # Plot 5
    plot_ratio_vs_dim(data;     outdir)   # Plot 3
    plot_error_per_topo(data;   outdir)   # Plot 4

    println("\nDone. All figures saved to $outdir/")
end