using CSV
using DataFrames
using Plots
using LaTeXStrings
using Plots.PlotMeasures
using Statistics

# =======================================================
# Load data
# -------------------------------------------------------

function load_data(csv_path::String)
    df = CSV.read(csv_path, DataFrame)
    return df
end

# =======================================================
# Plot 1 — L₁ error vs bond dimension
# One panel per (T, η), all topologies overlaid
# -------------------------------------------------------

function plot_error_vs_bond(df::DataFrame; outdir="PDF")
    mkpath(outdir)

    times      = sort(unique(df.T))
    etas       = sort(unique(df.eta))
    topo_names = unique(df.topology)
    nT         = length(times)
    nη         = length(etas)

    markers = [:circle, :square, :diamond, :utriangle, :dtriangle]
    lstyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

    fig = plot(
        layout     = (nT, nη),
        size       = (320 * nη, 260 * nT),
        left_margin  = 12mm,
        bottom_margin = 10mm,
        top_margin   = 4mm,
        right_margin = 4mm,
    )

    for (iT, T) in enumerate(times)
        for (ie, η) in enumerate(etas)
            idx = (iT - 1) * nη + ie
            sub = filter(r -> r.T == T && r.eta == η, df)

            for (j, name) in enumerate(topo_names)
                s = sort(filter(r -> r.topology == name, sub), :maxbond)
                isempty(s) && continue
                plot!(fig[idx], s.maxbond, s.error_l1;
                    label      = (iT == 1 && ie == 1) ? name : "",
                    color      = :black,
                    marker     = markers[mod1(j, end)],
                    linestyle  = lstyles[mod1(j, end)],
                    markersize = 4,
                    linewidth  = 1.8,
                    yscale     = :log10,
                    xlabel     = iT == nT  ? "Bond dimension"  : "",
                    ylabel     = ie == 1   ? L"L_1\ \mathrm{error}" : "",
                    title      = L"T=%$(T),\ \eta=%$(η)",
                    titlefontsize  = 9,
                    guidefontsize  = 9,
                    tickfontsize   = 8,
                    legend         = (idx == 1) ? :topright : false,
                    grid       = true,
                    gridalpha  = 0.25,
                )
            end
        end
    end

    path = joinpath(outdir, "$(first(df.gf_name))_error_vs_bond.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Plot 2 — Pareto: memory vs L₁ error
# Averaged over all (T, η); one curve per topology
# -------------------------------------------------------

function plot_pareto(df::DataFrame; outdir="PDF")
    mkpath(outdir)

    topo_names = unique(df.topology)
    markers    = [:circle, :square, :diamond, :utriangle, :dtriangle]
    lstyles    = [:solid, :dash, :dot, :dashdot, :dashdotdot]

    fig = plot(
        xlabel        = "Memory (MB)",
        ylabel        = L"L_1\ \mathrm{error}\ \mathrm{(mean\ over\ }T,\eta\mathrm{)}",
        xscale        = :log10,
        yscale        = :log10,
        legend        = :topright,
        size          = (680, 480),
        guidefontsize = 11,
        tickfontsize  = 9,
        legendfontsize = 9,
        left_margin   = 12mm,
        bottom_margin = 10mm,
        grid          = true,
        gridalpha     = 0.25,
        framestyle    = :box,
    )

    for (j, name) in enumerate(topo_names)
        s = filter(r -> r.topology == name, df)
        agg = combine(groupby(s, :maxbond),
            :mem_bytes => (x -> mean(x) / 1e6) => :mem_mb,
            :error_l1  => mean                  => :error_l1,
        )
        sort!(agg, :maxbond)

        plot!(fig, agg.mem_mb, agg.error_l1;
            label      = name,
            color      = :black,
            marker     = markers[mod1(j, end)],
            linestyle  = lstyles[mod1(j, end)],
            markersize = 5,
            linewidth  = 1.8,
        )
    end

    path = joinpath(outdir, "$(first(df.gf_name))_pareto.pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Plot 3 — Heatmap  log₁₀[ err(CTTN_Alt2) / err(QTT_Alt2) ]
# One heatmap per bond dimension (or pick a fixed one)
# -------------------------------------------------------

"""
    plot_heatmap_ratio(df; ref, target, bond, outdir)

Heatmap of log₁₀(err(`target`) / err(`ref`)) on the (T × η) grid,
for a single `bond` dimension value.
Negative (blue/green) → `target` is better than `ref`.
Positive (red)        → `ref` is better.
"""
function plot_heatmap_ratio(
        df::DataFrame;
        ref    = "QTT",
        target = "CTTN",
        bond   = 40,        # if nothing, use the maximum bond dimension
        outdir = "PDF",
    )
    mkpath(outdir)

    bond = isnothing(bond) ? maximum(df.maxbond) : bond
    times = sort(unique(df.T))
    etas  = sort(unique(df.eta))

    ratio_mat = [begin
        vt = filter(r -> r.topology == target && r.T == T && r.eta == η && r.maxbond == bond, df)
        vr = filter(r -> r.topology == ref    && r.T == T && r.eta == η && r.maxbond == bond, df)
        (nrow(vt) == 0 || nrow(vr) == 0 || vr.error_l1[1] == 0) ? NaN :
            log10(vt.error_l1[1] / vr.error_l1[1])
    end for T in times, η in etas]   # (nT × nη) matrix

    clim_abs = max(maximum(abs.(filter(!isnan, vec(ratio_mat)))), 0.01)
    clim_abs = min(clim_abs, 1.5)

    fig = heatmap(
        string.(etas),
        string.(times),
        ratio_mat;
        xlabel        = L"\eta\ \mathrm{(damping)}",
        ylabel        = L"T\ \mathrm{(time)}",
        title         = L"\log_{10}\!\left(\frac{\varepsilon_{%$(target)}}{\varepsilon_{%$(ref)}}\right)\ @\ \chi_{\max}=%$(bond)",
        clims         = (-clim_abs, clim_abs),
        size          = (560, 400),
        guidefontsize = 11,
        tickfontsize  = 9,
        left_margin   = 12mm,
        bottom_margin = 10mm,
        right_margin  = 18mm,
        framestyle    = :box,
        annotate_kw   = (fontsize=9,),
    )

    path = joinpath(outdir, "$(first(df.gf_name))_heatmap_$(target)_vs_$(ref)_bond$(bond).pdf")
    savefig(fig, path)
    println("Saved → $path")
    display(fig)
    return fig
end

# =======================================================
# Optional: heatmaps for all bond dimensions
# -------------------------------------------------------

function plot_all_heatmaps(df::DataFrame; ref="QTT", target="CTTN", outdir="PDF")
    for bond in sort(unique(df.maxbond))
        plot_heatmap_ratio(df; ref, target, bond, outdir)
    end
end

# =======================================================
# Main
# -------------------------------------------------------

"""
    main(csv_path; outdir)

Load results from `csv_path` and produce all three plots.

    main("gk.csv")
    main("gkdc.csv"; outdir="PDF/gkdc")
"""
function main(csv_path::String; outdir::String="PDF/GK")
    println("Loading $csv_path …")
    df = load_data(csv_path)
    println("  $(nrow(df)) rows — topologies: $(unique(df.topology))")

    plot_error_vs_bond(df;   outdir)
    plot_pareto(df;          outdir)
    plot_heatmap_ratio(df;   outdir=outdir)   # default: CTTN_Alt2 vs QTT_Alt2 at max bond

    println("\nDone. All figures saved to $outdir/")
end