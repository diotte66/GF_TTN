using Plots
using Plots.PlotMeasures
using Statistics
using LaTeXStrings

# ── Choose your dataset here ────────────────────
dataname = "GK"   # e.g. "PW", "SW", "GF", "GK" 
# ────────────────────────────────────────────────

const RESULTS_DIR = joinpath("results", dataname)
const OUTPUT_DIR = joinpath("PDF", dataname)
const PREFIX = lowercase(dataname)

"""Simple visualiser for results saved by `sample/pw.jl`.
Expects files in `results/<dataname>/`:
  <prefix>_error_l1.csv  (topo,maxbond,value)
  <prefix>_memory.csv    (topo,maxbond,bytes)
"""

function _read_csv_rows(path)
    lines = readlines(path)
    if isempty(lines)
        return String[]
    end
    hdr = first(lines)
    rows = lines[2:end]
    return [split(r, ',') for r in rows]
end

function load_results(prefix; dir=RESULTS_DIR)
    errfile = joinpath(dir, string(prefix, "_error_l1.csv"))
    memfile = joinpath(dir, string(prefix, "_memory.csv"))
    @assert isfile(errfile) "Error CSV not found: $errfile"
    @assert isfile(memfile) "Memory CSV not found: $memfile"

    errrows = _read_csv_rows(errfile)
    memrows = _read_csv_rows(memfile)

    topo_names = unique([r[1] for r in errrows])
    sort!(topo_names)
    maxbonds = unique([parse(Int, r[2]) for r in errrows])
    sort!(maxbonds)

    nt = length(topo_names)
    nb = length(maxbonds)
    error_l1 = fill(NaN, nt, nb)
    memmat = fill(NaN, nt, nb)

    topomap = Dict(name => i for (i, name) in enumerate(topo_names))
    bondmap = Dict(b => i for (i, b) in enumerate(maxbonds))

    for r in errrows
        error_l1[topomap[r[1]], bondmap[parse(Int, r[2])]] = parse(Float64, r[3])
    end
    for r in memrows
        memmat[topomap[r[1]], bondmap[parse(Int, r[2])]] = parse(Float64, r[3])
    end

    return topo_names, maxbonds, error_l1, memmat
end

function _make_common_plot_opts(; xlabel=L"$D_{\max}$")
    return (size=(900, 480), guidefont=font(12), tickfont=font(10),
        legendfont=font(10), xlabel=xlabel)
end

function plot_style_black(topo_names, maxbonds, mat; ylabel, outfile, log_scale=true)
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5]
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    opts = _make_common_plot_opts()
    p = plot(; opts..., ylabel=ylabel, legend=:topright, grid=true,
        framestyle=:box, yscale=log_scale ? :log10 : :identity)
    for j in 1:length(topo_names)
        plot!(p, maxbonds, mat[j, :], label=topo_names[j], color=:black,
            linestyle=linestyles[mod1(j, length(linestyles))],
            marker=markers[mod1(j, length(markers))], markersize=6,
            markerstrokecolor=:black, markerstrokewidth=0.7, linewidth=1.4)
    end
    mkpath(OUTPUT_DIR)
    plot!(p, left_margin=25px, bottom_margin=25px)
    savefig(p, outfile)
    return p
end

function plot_style_color(topo_names, maxbonds, mat; ylabel, outfile, log_scale=true)
    palette = [:black, :blue, :orange, :green4, :purple, :brown]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5]
    opts = _make_common_plot_opts()
    p = plot(; opts..., ylabel=ylabel, legend=:topright, grid=true,
        framestyle=:box, yscale=log_scale ? :log10 : :identity)
    for j in 1:length(topo_names)
        plot!(p, maxbonds, mat[j, :], label=topo_names[j],
            color=palette[mod1(j, length(palette))],
            marker=markers[mod1(j, length(markers))], markersize=6,
            markerstrokecolor=:black, markerstrokewidth=0.6, linewidth=1.6)
    end
    mkpath(OUTPUT_DIR)
    plot!(p, left_margin=120px, bottom_margin=80px)
    savefig(p, outfile)
    return p
end

function plot_both(prefix=PREFIX)
    topo_names, maxbonds, error_l1, memmat = load_results(prefix)

    p_err_black = plot_style_black(topo_names, maxbonds, error_l1;
        ylabel=L"Error $\varepsilon$",
        outfile=joinpath(OUTPUT_DIR, string(prefix, "_error_black.pdf")))
    plot!(p_err_black, yscale=:log10)

    p_err_color = plot_style_color(topo_names, maxbonds, error_l1;
        ylabel=L"Error $\varepsilon$",
        outfile=joinpath(OUTPUT_DIR, string(prefix, "_error_color.pdf")))
    plot!(p_err_color, yscale=:log10)

    p_mem_black = plot_style_black(topo_names, maxbonds, memmat;
        ylabel="Memory (bytes)",
        outfile=joinpath(OUTPUT_DIR, string(prefix, "_memory_black.pdf")), log_scale=false)
    p_mem_color = plot_style_color(topo_names, maxbonds, memmat;
        ylabel="Memory (bytes)",
        outfile=joinpath(OUTPUT_DIR, string(prefix, "_memory_color.pdf")), log_scale=false)

    combined_black = plot(p_err_black, p_mem_black, layout=(2, 1), size=(900, 900))
    plot!(combined_black, left_margin=120px, bottom_margin=80px, xlabel=L"$D_{\max}$")
    savefig(combined_black, joinpath(OUTPUT_DIR, string(prefix, "_error_memory_black.pdf")))

    combined_color = plot(p_err_color, p_mem_color, layout=(2, 1), size=(900, 900))
    plot!(combined_color, left_margin=120px, bottom_margin=80px, xlabel=L"$D_{\max}$")
    savefig(combined_color, joinpath(OUTPUT_DIR, string(prefix, "_error_memory_color.pdf")))

    return (p_err_black, p_err_color, p_mem_black, p_mem_color)
end

function plot_KB_greens(; A=30.0, B=0.0, tmax=1.0, beta=1.0, N=400)
    """
    Plot G_k^0(z,z') on the Kadanoff-Baym L-shaped contour.
    Contour ordering: C1 (0->tmax) ≺ C2 (tmax->0) ≺ C3 (0->-ibeta)
    θ_C(z,z') = 1 if z ≻ z' (z comes LATER in contour ordering), 0 otherwise.

    G^0_k(z,z') = -i[θ_C(z,z') - f_T] * exp(-i*A*(t-t')) * exp(-B*|t-t'|)
    where the exponential uses the real-time difference for real-time components,
    and imaginary-time difference for Matsubara components.
    """

    f_T(ε) = 0
    fT0 = f_T(0.0)  # epsilon_k = 0 for simplicity

    # Contour index: C1=1, C2=2, C3=3
    # Within each segment, contour position increases with the segment parameter
    # C1: position increases with t (0 -> tmax)
    # C2: position increases as t decreases (tmax -> 0), so index N+1..2N corresponds to tmax..0
    # C3: position increases with tau (0 -> beta)

    # θ_C(z, z') = 1 if z ≻ z' i.e. z comes later in contour
    function theta_C(seg1, idx1, seg2, idx2)
        if seg1 > seg2
            return 1.0  # z on later segment => z ≻ z'
        elseif seg1 < seg2
            return 0.0
        else
            # same segment: compare position within segment
            # C1: larger index = later in contour
            # C2: larger index = later in contour (since C2 goes tmax->0, idx N+1 is tmax, 2N is t=0)
            # C3: larger index = later in contour
            return idx1 > idx2 ? 1.0 : 0.0
        end
    end

    # The full propagator: G = -i * [θ_C - f_T] * kernel
    # kernel depends on the block:
    #   real-real:  exp(-i*A*(t1-t2)) * exp(-B*|t1-t2|)
    #   real-imag:  exp(-i*A*t1) * exp(-A*tau2)          (integral along mixed contour)
    #   imag-real:  exp(+i*A*t2) * exp(-A*(beta-tau1))   (conjugate)
    #   imag-imag:  exp(-A*(tau1-tau2))

    function kernel(seg1, t1, tau1, seg2, t2, tau2)
        if seg1 <= 2 && seg2 <= 2
            # real-real block
            dt = t1 - t2
            return exp(-1im * A * dt) * exp(-B * abs(dt))
        elseif seg1 <= 2 && seg2 == 3
            # G^⌐: real time z1, imaginary time z2
            # integral from z2 (imaginary) to z1 (real) along contour
            return exp(-1im * A * t1) * exp(-B * t1) * exp(-A * tau2)
        elseif seg1 == 3 && seg2 <= 2
            # G^⌏: imaginary z1, real z2
            return exp(1im * A * t2) * exp(-B * t2) * exp(-A * (beta - tau1))
        else
            # Matsubara-Matsubara
            dtau = tau1 - tau2
            return exp(-A * dtau) * exp(-B * abs(dtau))
        end
    end

    # Build full matrix
    M = 3 * N
    Z = zeros(ComplexF64, M, M)

    ts = range(0, tmax, length=N)
    taus = range(0, beta, length=N)

    for i in 1:M
        for j in 1:M
            # Determine segment and local index
            seg_i = i <= N ? 1 : (i <= 2N ? 2 : 3)
            seg_j = j <= N ? 1 : (j <= 2N ? 2 : 3)

            loc_i = seg_i == 1 ? i : (seg_i == 2 ? i - N : i - 2N)
            loc_j = seg_j == 1 ? j : (seg_j == 2 ? j - N : j - 2N)

            # Physical time/tau values
            # C1: t increases 0->tmax with index
            # C2: t decreases tmax->0 with index (contour goes backward)
            t1 = seg_i == 1 ? ts[loc_i] : (seg_i == 2 ? ts[N+1-loc_i] : 0.0)
            t2 = seg_j == 1 ? ts[loc_j] : (seg_j == 2 ? ts[N+1-loc_j] : 0.0)
            tau1 = seg_i == 3 ? taus[loc_i] : 0.0
            tau2 = seg_j == 3 ? taus[loc_j] : 0.0

            θ = theta_C(seg_i, loc_i, seg_j, loc_j)
            K = kernel(seg_i, t1, tau1, seg_j, t2, tau2)

            Z[i, j] = -1im * (θ - fT0) * K
        end
    end

    # ── Plotting ──────────────────────────────────────────────────────────────
    mkpath("PDF/GF")

    tick_pos = [0, N, 2N, 3N]
    tick_labs = ["0", L"$t_{\max}$", "0", L"$-i\beta$"]
    clims = (-1.0, 1.0)

    function make_panel(mat, title_str)
        p = heatmap(1:M, 1:M, mat; clims=clims,
            xticks=(tick_pos, tick_labs),
            yticks=(tick_pos, tick_labs),
            xlabel=L"$z_2$", ylabel=L"$z_1$",
            title=title_str,
            aspect_ratio=:equal,
            size=(700, 700),
            guidefont=font(13), tickfont=font(10),
            colorbar_tickfont=font(9),
            xlims=(0, M), ylims=(0, M),   # ← c'est ça le fix
            left_margin=5mm, bottom_margin=5mm)
        vline!([N, 2N], color=:white, linewidth=2, label=false)
        hline!([N, 2N], color=:white, linewidth=2, label=false)
        return p
    end

    p_real = make_panel(real.(Z), "")
    p_imag = make_panel(imag.(Z), "")

    savefig(p_real, "PDF/GF/KB_greens_real.pdf")
    savefig(p_imag, "PDF/GF/KB_greens_imag.pdf")

    combined = plot(p_real, p_imag, layout=(2, 1), size=(800, 1400),
        left_margin=-8mm, bottom_margin=40px)
    savefig(combined, "PDF/GF/KB_greens_combined.pdf")
    display(combined)

    return Z
end

function plot_systematic(cfg, error_l1, topo_names)
    mkpath("PDF/GF_systematic")
    palette = [:black, :blue, :orange, :green4]
    markers = [:circle, :square, :diamond, :utriangle]
    nA = length(cfg.A_list)
    nB = length(cfg.B_list)

    # ── Plot 1: grille A × B ──────────────────────────────────────────────────
    fig = plot(layout=(nA, nB), size=(300 * nB, 280 * nA),
        left_margin=10mm, bottom_margin=8mm)
    for (iA, A) in enumerate(cfg.A_list)
        for (iB, B) in enumerate(cfg.B_list)
            idx = (iA - 1) * nB + iB
            for (j, name) in enumerate(topo_names)
                plot!(fig[idx], cfg.maxbond, error_l1[j, iA, iB, :],
                    label=(iA == 1 && iB == 1) ? name : "",
                    color=palette[mod1(j, end)],
                    marker=markers[mod1(j, end)],
                    markersize=4, linewidth=1.4, yscale=:log10,
                    xlabel=iA == nA ? "Bond dim" : "",
                    ylabel=iB == 1 ? "L1 error" : "",
                    title="A=$(A), B=$(B)",
                    titlefontsize=8, guidefontsize=8, tickfontsize=7,
                    legend=idx == 1 ? :topright : false)
            end
        end
    end
    savefig(fig, "PDF/GF_systematic/grid_A_B.pdf")
    display(fig)

    # ── Plot 2: ratio vs QTT_Int ───────────────────────────────────────────────
    ref_idx = findfirst(==("QTT_Int"), topo_names)
    fig2 = plot(layout=(nA, nB), size=(300 * nB, 280 * nA),
        left_margin=10mm, bottom_margin=8mm)
    for (iA, A) in enumerate(cfg.A_list)
        for (iB, B) in enumerate(cfg.B_list)
            idx = (iA - 1) * nB + iB
            for (j, name) in enumerate(topo_names)
                j == ref_idx && continue
                ratio = error_l1[j, iA, iB, :] ./ error_l1[ref_idx, iA, iB, :]
                plot!(fig2[idx], cfg.maxbond, ratio,
                    label=(iA == 1 && iB == 1) ? "$(name)/QTT_Int" : "",
                    color=palette[mod1(j, end)],
                    marker=markers[mod1(j, end)],
                    markersize=4, linewidth=1.4,
                    xlabel=iA == nA ? "Bond dim" : "",
                    ylabel=iB == 1 ? "error ratio" : "",
                    title="A=$(A), B=$(B)",
                    titlefontsize=8, guidefontsize=8, tickfontsize=7,
                    legend=idx == 1 ? :topright : false)
            end
            hline!(fig2[idx], [1.0], color=:red, linestyle=:dash,
                linewidth=1, label=false)
        end
    end
    savefig(fig2, "PDF/GF_systematic/ratio_TTN_QTT.pdf")
    display(fig2)

    # ── Plot 3: heatmap du gain au bond max ───────────────────────────────────
    ref_idx = findfirst(==("QTT_Int"), topo_names)
    for (j, name) in enumerate(topo_names)
        j == ref_idx && continue
        gain = [log10(error_l1[j, iA, iB, end] / error_l1[ref_idx, iA, iB, end])
                for iA in 1:nA, iB in 1:nB]
        p = heatmap(string.(cfg.B_list), string.(cfg.A_list), gain,
            xlabel="B (damping)",
            ylabel="A (frequency)",
            title="log₁₀($(name) / QTT_Int) at bond=$(cfg.maxbond[end])",
            color=:RdBu, clims=(-1, 1),
            guidefont=font(11), tickfont=font(9),
            left_margin=10mm, bottom_margin=8mm)
        savefig(p, "PDF/GF_systematic/heatmap_gain_$(name).pdf")
        display(p)
    end
end

function plot_systematic_memory(cfg, error_l1, mem, topo_names)
    mkpath("PDF/GF_systematic")
    palette = [:black, :blue, :orange, :green4]
    markers = [:circle, :square, :diamond, :utriangle]
    nA = length(cfg.A_list)
    nB = length(cfg.B_list)

    # ── Plot 1: memory grid A × B ─────────────────────────────────────────────
    fig = plot(layout=(nA, nB), size=(300 * nB, 280 * nA),
        left_margin=10mm, bottom_margin=8mm)
    for (iA, A) in enumerate(cfg.A_list)
        for (iB, B) in enumerate(cfg.B_list)
            idx = (iA - 1) * nB + iB
            for (j, name) in enumerate(topo_names)
                plot!(fig[idx], cfg.maxbond, mem[j, iA, iB, :] ./ 1e6,
                    label=(iA == 1 && iB == 1) ? name : "",
                    color=palette[mod1(j, end)],
                    marker=markers[mod1(j, end)],
                    markersize=4, linewidth=1.4,
                    xlabel=iA == nA ? "Bond dim" : "",
                    ylabel=iB == 1 ? "Memory (MB)" : "",
                    title="A=$(A), B=$(B)",
                    titlefontsize=8, guidefontsize=8, tickfontsize=7,
                    legend=idx == 1 ? :topleft : false)
            end
        end
    end
    savefig(fig, "PDF/GF_systematic/grid_memory_A_B.pdf")
    display(fig)

    # ── Plot 2: memory ratio TTN/QTT vs bond dim ──────────────────────────────
    ref_idx = findfirst(==("QTT_Int"), topo_names)
    fig2 = plot(layout=(nA, nB), size=(300 * nB, 280 * nA),
        left_margin=10mm, bottom_margin=8mm)
    for (iA, A) in enumerate(cfg.A_list)
        for (iB, B) in enumerate(cfg.B_list)
            idx = (iA - 1) * nB + iB
            for (j, name) in enumerate(topo_names)
                j == ref_idx && continue
                ratio = mem[j, iA, iB, :] ./ mem[ref_idx, iA, iB, :]
                plot!(fig2[idx], cfg.maxbond, ratio,
                    label=(iA == 1 && iB == 1) ? "$(name)/QTT_Int" : "",
                    color=palette[mod1(j, end)],
                    marker=markers[mod1(j, end)],
                    markersize=4, linewidth=1.4,
                    xlabel=iA == nA ? "Bond dim" : "",
                    ylabel=iB == 1 ? "memory ratio" : "",
                    title="A=$(A), B=$(B)",
                    titlefontsize=8, guidefontsize=8, tickfontsize=7,
                    legend=idx == 1 ? :topright : false)
            end
            hline!(fig2[idx], [1.0], color=:red, linestyle=:dash,
                linewidth=1, label=false)
        end
    end
    savefig(fig2, "PDF/GF_systematic/ratio_memory_TTN_QTT.pdf")
    display(fig2)

    # ── Plot 3: memory vs error scatter (Pareto plot) ─────────────────────────
    # Each point = one (topology, A, B, bond) combination
    # Points below and to the left are better (less memory, less error)
    fig3 = plot(xlabel="Memory (MB)", ylabel="L1 error",
        xscale=:log10, yscale=:log10,
        legend=:topright, size=(800, 600),
        guidefont=font(12), tickfont=font(10),
        left_margin=10mm, bottom_margin=10mm,
        framestyle=:box, grid=true)
    for (j, name) in enumerate(topo_names)
        # Flatten over all (A, B) pairs, keep bond dimension as the sweep axis
        mem_flat = vec(mean(mem[j, :, :, :], dims=(1, 2)))  # average over A,B
        error_flat = vec(mean(error_l1[j, :, :, :], dims=(1, 2)))
        scatter!(fig3, mem_flat ./ 1e6, error_flat,
            label=name,
            color=palette[mod1(j, end)],
            marker=markers[mod1(j, end)],
            markersize=6,
            markerstrokewidth=0.5)
        # Connect points by bond dimension order
        plot!(fig3, mem_flat ./ 1e6, error_flat,
            color=palette[mod1(j, end)],
            linewidth=1.0,
            label=false)
    end
    savefig(fig3, "PDF/GF_systematic/pareto_memory_error.pdf")
    display(fig3)
end

function main(argv)
    prefix = length(argv) >= 1 ? argv[1] : PREFIX
    println("Loading results with prefix=", prefix, " from ", RESULTS_DIR)
    plot_both(prefix)
    println("Plots saved to ", OUTPUT_DIR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end