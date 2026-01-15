using TreeTCI
using Random
using Plots
using Graphs
using NamedGraphs

#------------------------------------------------------
# General utility functions for QTCI simulations
#------------------------------------------------------

function bits2decimal(v::AbstractVector{<:Integer})
    """Convert a vector of bits (1/2) to a decimal number between 0 and 1"""
    sum = 0.0
    for i in 1:length(v)
        sum += (v[i] - 1) * 2.0^(-i)
    end
    return sum
end

function sampled_error(f, ttn, nsamples::Int64, bits::Int64, d::Int64)
    """ Compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end
    error_l1 = 0.0
    for _ in 1:nsamples
        x = rand(1:2, bits)
        approx = TreeTCI.evaluate(eval_ttn, x)
        err = abs(f(x) - approx)
        error_l1 += err
    end
    return error_l1 / nsamples
end

function seed_pivots!(tci, npivots::Int64)
    """ Seed initial pivots for the SimpleTCI tci based on binary representations. """
    for i in 1:npivots
        # Generate a random input of length equal to number of vertices
        x = rand(1:2, length(vertices(tci.g)))
        TreeTCI.addglobalpivots!(tci, [x])
    end
end

#-------------------------------------------------------------
# Specific utility functions for Green's function simulations
#-------------------------------------------------------------

function ϵ(kx::Float64, ky::Float64)
    """ Fictious dispersion relation """
    return -2.0 * (cos(2π * kx) + cos(2π * ky))
end

function δ(kx::Float64, ky::Float64)
    """ Fictious dispersion relation """
    return -2.0 * (sin(2π * kx) + sin(2π * ky))
end

function n(kx::Float64, ky::Float64, β::Float64)
    """ Fermi-Dirac distribution """
    return 1.0 / (exp(β * ϵ(kx, ky)) + 1.0)
end

function assign(v::AbstractVector{Int64}, nk::Int64, R::Int64)
    """ Assign variables from bit vector v using explicit `nk` and `R`.

    v is expected to be a vector with length 2*nk + 2*R (kx bits, ky bits, t1 bits, t2 bits).
    """
    kx = bits2decimal(v[1:nk])
    ky = bits2decimal(v[nk+1:2*nk])
    t1 = bits2decimal(v[2*nk+1:2*nk+1+R])
    t2 = bits2decimal(v[2*nk+2+R:end])
    return kx, ky, t1, t2
end

function assign2(v::AbstractVector{Int64}, nk::Int64, Rt::Int64)
    """ Assign variables from bit vector v using explicit `nk` and `Rt`.

    v is expected to be a vector with length nk + 2*Rt (epsilon bits, t1 bits, t2 bits).
    """
    ϵ = bits2decimal(v[1:nk])
    ϵ = -4 + (8 * ϵ)  # map to [-4, 4]
    t1 = bits2decimal(v[nk+1:nk+Rt])
    t2 = bits2decimal(v[nk+Rt+1:end])
    return ϵ, t1, t2
end

function plotgk(kx, ky)
    T = 40.0
    f(x, y) = -1im * exp(-1im * ϵ(kx, ky) * T * (x - y)) * exp(-(ϵ(kx, ky)^2) * abs(x - y))

    xs = ys = range(0, 1, length=400)
    zs = [f(x, y) for x in xs, y in ys]
    # Plots/GR cannot compare Complex values when computing extrema; convert to real
    zs_imag = imag.(zs)
    zs_real = real.(zs)
    println("Max real part: ", maximum(zs_real))
    println("Max imag part: ", maximum(zs_imag))
    realpart = heatmap(xs, ys, zs_real, title="Real part of G(2π,2π)", xlabel="t", ylabel="t'", size=(600, 500))
    imagpart = heatmap(xs, ys, zs_imag, title="Imaginary part of G(2π,2π)", xlabel="t", ylabel="t'", size=(600, 500))
    display(plot(realpart, imagpart, layout=(1, 2), size=(1200, 500)))
end

function plotdcgk(kx, ky)
    E = 50.0
    f(x, y) = -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * x) / E) + δ(kx, ky) * (cos(E * x) - 1) / E - ϵ(kx, ky) * sin(E * y) / E - δ(kx, ky) * (cos(E * y) - 1) / E)) * exp(-(ϵ(kx, ky)^2) * abs(x - y)) * n(kx, ky, 10.0)
    xs = ys = range(0, 1, length=400)
    zs = [f(x, y) for x in xs, y in ys]
    # Plots/GR cannot compare Complex values when computing extrema; convert to real
    zs_imag = imag.(zs)
    zs_real = real.(zs)
    realpart = heatmap(xs, ys, zs_real, title="Real part of G($kx,$ky)", xlabel="t", ylabel="t'", size=(600, 500))
    imaginarypart = heatmap(xs, ys, zs_imag, title="Imaginary part of G($kx,$ky)", xlabel="t", ylabel="t'", size=(600, 500))
    display(plot(realpart, imaginarypart, layout=(1, 2), size=(1200, 500)))
end

#-------------------------------------------------------------
# Specific utility functions for different topologies
#-------------------------------------------------------------

function QTTSEQ(Rt::Int64, nk::Int64)
    """ QTT Sequential topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)
    total_bits = 2 * Rt + 2 * nk
    for i in 1:(total_bits-1)
        add_edge!(g, i, i + 1)
    end
    return g
end

function QTTALT(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(2*nk)
        add_edge!(g, i, i + 1)
    end

    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT2(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, 1, nk + 1)
    add_edge!(g, 2 * nk, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT3(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:nk
        add_edge!(g, i, nk + i)
        if i < nk
            add_edge!(g, nk + i, i + 1)
        end
    end
    add_edge!(g, 2 * nk, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT4(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:nk
        add_edge!(g, i, nk + i)
        if i < nk
            add_edge!(g, nk + i, i + 1)
        end
    end
    add_edge!(g, 1, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function QTTALT5(Rt::Int64, nk::Int64)
    """ QTT Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    # the first 2nk bits are fused as i and i + nk in a chain of the form : (1, nk+1) -> {(2,nk+2),(3,nk+3)}, (3, nk+3) -> {(4,nk+4),(5,nk+5)} ...

    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function CTTNALT(Rt::Int64, nk::Int64)
    """ CTTN Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    add_edge!(g, 1, nk + 1)
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, nk + 1, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function CTTNALT2(Rt::Int64, nk::Int64)
    """ CTTN Alternating topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    add_edge!(g, 1, 2 * nk + 1)
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, nk + 1, 2 * nk + 1)
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + i + Rt)
        if i < Rt
            add_edge!(g, 2 * nk + i + Rt, 2 * nk + i + 1)
        end
    end
    return g
end

function CTTNSEQ(Rt::Int64, nk::Int64)
    """ CTTN Sequential topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    add_edge!(g, 1, nk + 1)
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end
    add_edge!(g, nk + 1, 2 * nk + 1)
    for i in 1:(Rt-1)
        add_edge!(g, 2 * nk + i, 2 * nk + i + 1)
    end
    add_edge!(g, 2 * nk + 1, 2 * nk + Rt + 1)
    for i in 1:(Rt-1)
        add_edge!(g, 2 * nk + Rt + i, 2 * nk + Rt + i + 1)
    end
    return g
end

function BTTN(Rt::Int64, nk::Int64)
    """ BTTN topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    g = NamedGraph()
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    lengths = (nk, nk, Rt, Rt)

    start = 1
    for L in lengths
        for i in 1:L
            left = 2 * i
            right = 2 * i + 1
            if left <= L
                add_edge!(g, start + i - 1, start + left - 1)
            end
            if right <= L
                add_edge!(g, start + i - 1, start + right - 1)
            end
        end

        if start != 1
            add_edge!(g, 1, start)
        end

        start += L
    end
    return g
end

function bondslistkx(bonds, nk, Rt)
    """Return an array of bond dimensions in the requested order.

    Order produced:
      - nk -> nk-1, nk-1 -> nk-2, ..., 2 -> 1
      - 1 -> 2*nk + 1
      - interleaved chain starting at `m = 2*nk + 1` and `r = m + Rt`:
          sequence S = [m, r, m+1, r+1, m+2, r+2, ..., m+(2*nk-1), r+(2*nk-1)]
        return the bdims for consecutive pairs in S (S[i] -> S[i+1])

    The function accepts `bonds` as any iterable of objects that expose
    `src`, `dst`, and `bdim` (NamedTuple, Dict-like, struct, or Tuple/Array)
    and will match edges regardless of ordering (src,dst) vs (dst,src).
    Missing edges are returned as `0` with a warning.
    """

    # Build target edge pair list
    pairs = Tuple{Int,Int}[]

    # first nk-1 bonds: 2*nk -> 2*nk-1 -> ... -> nk+2 -> nk+1
    for i in (nk):-1:(2)
        push!(pairs, (i, i - 1))
    end

    # connect 1 -> middle start
    m = 2 * nk + 1
    push!(pairs, (1, m))

    # build interleaved sequence S = [m, r, m+1, r+1, ...]
    r = m + Rt
    S = Int[]
    for t in 0:(Rt-1)
        push!(pairs, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs, (m + Rt + t, m + t + 1))
        end
    end

    # Helper to extract fields robustly
    function _getfield(x, sym::Symbol, idx::Int)
        try
            if hasproperty(x, sym)
                return getproperty(x, sym)
            end
        catch
        end
        # NamedTuple and Dict access
        try
            if isa(x, NamedTuple) && haskey(x, sym)
                return x[sym]
            end
        catch
        end
        try
            if isa(x, AbstractDict) && haskey(x, sym)
                return x[sym]
            end
        catch
        end
        # Tuple/Array positional fallback
        try
            if isa(x, Tuple) || isa(x, AbstractVector)
                return x[idx]
            end
        catch
        end
        return nothing
    end

    # Build lookup map of undirected edges -> bdim (take first occurrence)
    bondmap = Dict{Tuple{Int,Int},Int}()
    for b in bonds
        src = _getfield(b, :src, 1)
        dst = _getfield(b, :dst, 2)
        bdim = _getfield(b, :bdim, 3)

        if src === nothing || dst === nothing || bdim === nothing
            @warn "Skipping bond with unrecognized format" b = b
            continue
        end

        # coerce to Int where reasonable
        src_i = Int(round(NaN == bdim ? parse(Int, string(src)) : tryparse(Int, string(src)) !== nothing ? tryparse(Int, string(src)) : Int(src)))
        dst_i = Int(round(NaN == bdim ? parse(Int, string(dst)) : tryparse(Int, string(dst)) !== nothing ? tryparse(Int, string(dst)) : Int(dst)))
        # bdim coercion
        bdim_i = 0
        if isa(bdim, Number)
            bdim_i = Int(round(bdim))
        else
            bdim_i = tryparse(Int, string(bdim)) === nothing ? 0 : tryparse(Int, string(bdim))
        end

        key = src_i <= dst_i ? (src_i, dst_i) : (dst_i, src_i)
        if !haskey(bondmap, key)
            bondmap[key] = bdim_i
        end
    end

    # collect bdims in the requested order
    result = Int[]
    for (a, b) in pairs
        k = a <= b ? (a, b) : (b, a)
        val = get(bondmap, k, 0)
        if val == 0
            @warn "Bond not found for edge" edge = k
        end
        push!(result, val)
    end

    return result
end

function bondslistky(bonds, nk, Rt)
    """Return an array of bond dimensions in the requested order.

    Order produced:
      - nk -> nk-1, nk-1 -> nk-2, ..., 2 -> 1
      - 1 -> 2*nk + 1
      - interleaved chain starting at `m = 2*nk + 1` and `r = m + Rt`:
          sequence S = [m, r, m+1, r+1, m+2, r+2, ..., m+(2*nk-1), r+(2*nk-1)]
        return the bdims for consecutive pairs in S (S[i] -> S[i+1])

    The function accepts `bonds` as any iterable of objects that expose
    `src`, `dst`, and `bdim` (NamedTuple, Dict-like, struct, or Tuple/Array)
    and will match edges regardless of ordering (src,dst) vs (dst,src).
    Missing edges are returned as `0` with a warning.
    """

    # Build target edge pair list
    pairs = Tuple{Int,Int}[]

    # first nk-1 bonds: 2*nk -> 2*nk-1 -> ... -> nk+2 -> nk+1
    for i in (2*nk):-1:(nk+2)
        push!(pairs, (i, i - 1))
    end

    # connect nk+1 -> middle start
    m = 2 * nk + 1
    push!(pairs, (nk + 1, m))

    # build interleaved sequence S = [m, r, m+1, r+1, ...]
    r = m + Rt
    S = Int[]
    for t in 0:(Rt-1)
        push!(pairs, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs, (m + Rt + t, m + t + 1))
        end
    end

    # Helper to extract fields robustly
    function _getfield(x, sym::Symbol, idx::Int)
        try
            if hasproperty(x, sym)
                return getproperty(x, sym)
            end
        catch
        end
        # NamedTuple and Dict access
        try
            if isa(x, NamedTuple) && haskey(x, sym)
                return x[sym]
            end
        catch
        end
        try
            if isa(x, AbstractDict) && haskey(x, sym)
                return x[sym]
            end
        catch
        end
        # Tuple/Array positional fallback
        try
            if isa(x, Tuple) || isa(x, AbstractVector)
                return x[idx]
            end
        catch
        end
        return nothing
    end

    # Build lookup map of undirected edges -> bdim (take first occurrence)
    bondmap = Dict{Tuple{Int,Int},Int}()
    for b in bonds
        src = _getfield(b, :src, 1)
        dst = _getfield(b, :dst, 2)
        bdim = _getfield(b, :bdim, 3)

        if src === nothing || dst === nothing || bdim === nothing
            @warn "Skipping bond with unrecognized format" b = b
            continue
        end

        # coerce to Int where reasonable
        src_i = Int(round(NaN == bdim ? parse(Int, string(src)) : tryparse(Int, string(src)) !== nothing ? tryparse(Int, string(src)) : Int(src)))
        dst_i = Int(round(NaN == bdim ? parse(Int, string(dst)) : tryparse(Int, string(dst)) !== nothing ? tryparse(Int, string(dst)) : Int(dst)))
        # bdim coercion
        bdim_i = 0
        if isa(bdim, Number)
            bdim_i = Int(round(bdim))
        else
            bdim_i = tryparse(Int, string(bdim)) === nothing ? 0 : tryparse(Int, string(bdim))
        end

        key = src_i <= dst_i ? (src_i, dst_i) : (dst_i, src_i)
        if !haskey(bondmap, key)
            bondmap[key] = bdim_i
        end
    end

    # collect bdims in the requested order
    result = Int[]
    for (a, b) in pairs
        k = a <= b ? (a, b) : (b, a)
        val = get(bondmap, k, 0)
        if val == 0
            @warn "Bond not found for edge" edge = k
        end
        push!(result, val)
    end

    return result
end

function bondslistQTTALT(Rt::Int64, nk::Int64, bonds)
    """ Return bond dimensions in QTTALT order. """
    # Build target edge pair list
    pairs = Tuple{Int,Int}[]

    # first nk-1 bonds: 2*nk -> 2*nk-1 -> ... -> nk+2 -> nk+1
    for i in (2*nk):-1:(nk+1)
        push!(pairs, (i, i - nk))
        if i - nk > 1
            push!(pairs, (i - nk, i - 1))
        end
    end

    # connect nk+1 -> middle start
    m = 2 * nk + 1
    push!(pairs, (1, m))

    # build interleaved sequence S = [m, r, m+1, r+1, ...]
    r = m + Rt
    S = Int[]
    for t in 0:(Rt-1)
        push!(pairs, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs, (m + Rt + t, m + t + 1))
        end
    end

    # Helper to extract fields robustly
    function _getfield(x, sym::Symbol, idx::Int)
        try
            if hasproperty(x, sym)
                return getproperty(x, sym)
            end
        catch
        end
        # NamedTuple and Dict access
        try
            if isa(x, NamedTuple) && haskey(x, sym)
                return x[sym]
            end
        catch
        end
        try
            if isa(x, AbstractDict) && haskey(x, sym)
                return x[sym]
            end
        catch
        end
        # Tuple/Array positional fallback
        try
            if isa(x, Tuple) || isa(x, AbstractVector)
                return x[idx]
            end
        catch
        end
        return nothing
    end

    # Build lookup map of undirected edges -> bdim (take first occurrence)
    bondmap = Dict{Tuple{Int,Int},Int}()
    for b in bonds
        src = _getfield(b, :src, 1)
        dst = _getfield(b, :dst, 2)
        bdim = _getfield(b, :bdim, 3)

        if src === nothing || dst === nothing || bdim === nothing
            @warn "Skipping bond with unrecognized format" b = b
            continue
        end

        # coerce to Int where reasonable
        src_i = Int(round(NaN == bdim ? parse(Int, string(src)) : tryparse(Int, string(src)) !== nothing ? tryparse(Int, string(src)) : Int(src)))
        dst_i = Int(round(NaN == bdim ? parse(Int, string(dst)) : tryparse(Int, string(dst)) !== nothing ? tryparse(Int, string(dst)) : Int(dst)))
        # bdim coercion
        bdim_i = 0
        if isa(bdim, Number)
            bdim_i = Int(round(bdim))
        else
            bdim_i = tryparse(Int, string(bdim)) === nothing ? 0 : tryparse(Int, string(bdim))
        end

        key = src_i <= dst_i ? (src_i, dst_i) : (dst_i, src_i)
        if !haskey(bondmap, key)
            bondmap[key] = bdim_i
        end
    end

    # collect bdims in the requested order
    result = Int[]
    for (a, b) in pairs
        k = a <= b ? (a, b) : (b, a)
        val = get(bondmap, k, 0)
        if val == 0
            @warn "Bond not found for edge" edge = k
        end
        push!(result, val)
    end

    return result
end

function bondlistplotQTTALT(bonddimlist, Rt, nk, times)
    # collect global max for plotting/annotation
    vals = Int[]
    for ii in 1:size(bonddimlist, 1), jj in 1:size(bonddimlist, 2)
        v = bonddimlist[ii, jj]
        if !isempty(v)
            append!(vals, v)
        end
    end
    global_max = isempty(vals) ? 1 : maximum(vals)

    # prepare palette (one color per time index)
    colors = [:blue, :red, :green, :orange, :purple, :brown, :magenta, :cyan]

    # compute xmax (max bond index across all vectors)
    xmax = 1
    for ii in 1:size(bonddimlist, 1), jj in 1:size(bonddimlist, 2)
        v = bonddimlist[ii, jj]
        if !isempty(v)
            xmax = max(xmax, length(v))
        end
    end

    p1 = plot(title="Nk scaling for tmax=$(times[1])", xlabel="Bond number",
        ylabel="Bond dimension", xlim=(1, xmax), ylim=(0, global_max * 1.2), framestyle=:box,
        ticks=:auto)
    for (i, nk) in enumerate(nks)
        col = colors[mod1(i, length(colors))]
        vec = bonddimlist[1, i]
        maxnks = maximum(nks)
        if !isempty(vec)
            plot!(p1, (maxnks+1-nk):(maxnks+2*Rt-1), vec, label="Nk=$(2^nk)", color=col)
        end
    end

    # vertical divider at nk with annotation
    xdiv = maximum(nks)

    # shaded backgrounds: left region and right region (pale colors)
    ymin = 0
    ymax = global_max * 1.2
    # left rectangle: x in [1, xdiv)
    left_x = [1, xdiv, xdiv, 1]
    left_y = [ymin, ymin, ymax, ymax]
    plot!(p1, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)
    # right
    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [ymin, ymin, ymax, ymax]
    plot!(p1, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)
    vline!(p1, [xdiv], color=:black, linestyle=:dash, linewidth=0.5)
    savefig(p1, "gk_bonddim_QTTALT_t=$(times[1]).svg")
    display(p1)
end

function bondlistplotfixedtime(bonddimlist1, bonddimlist2, Rt, nks, T)
    # collect global max for plotting/annotation
    vals = Int[]
    for ii in 1:size(bonddimlist1, 1), jj in 1:size(bonddimlist1, 2)
        v1 = bonddimlist1[ii, jj]
        if !isempty(v1)
            append!(vals, v1)
        end
        v2 = bonddimlist2[ii, jj]
        if !isempty(v2)
            append!(vals, v2)
        end
    end
    global_max = isempty(vals) ? 1 : maximum(vals)

    # prepare palette (one color per time index)
    colors = [:blue, :red, :green, :orange, :purple, :brown, :magenta, :cyan]

    # compute xmax (max bond index across all vectors)
    xmax = 1
    for ii in 1:size(bonddimlist1, 1), jj in 1:size(bonddimlist1, 2)
        v1 = bonddimlist1[ii, jj]
        v2 = bonddimlist2[ii, jj]
        if !isempty(v1)
            xmax = max(xmax, length(v1))
        end
        if !isempty(v2)
            xmax = max(xmax, length(v2))
        end
    end

    p1 = plot(title="Nk scaling for tmax=$T", xlabel="Bond number",
        ylabel="Bond dimension", xlim=(1, xmax), ylim=(0, global_max * 1.2), framestyle=:box,
        ticks=:auto)
    for (i, nk) in enumerate(nks)
        col = colors[mod1(i, length(colors))]
        vec1 = bonddimlist1[1, i]
        vec2 = bonddimlist2[1, i]
        maxnks = maximum(nks)
        if !isempty(vec1)
            plot!(p1, (maxnks+1-nk):(maxnks+2*Rt-1), vec1, label="kx Nk=$(2^nk)", color=col)
        end
        if !isempty(vec2)
            plot!(p1, (maxnks+1-nk):(maxnks+2*Rt-1), vec2, label="ky Nk=$(2^nk)", color=col, seriestype=:scatter)
        end
    end

    # vertical divider at nk with annotation
    xdiv = maximum(nks)

    # shaded backgrounds: left region and right region (pale colors)
    ymin = 0
    ymax = global_max * 1.2
    # left rectangle: x in [1, xdiv)
    left_x = [1, xdiv, xdiv, 1]
    left_y = [ymin, ymin, ymax, ymax]
    plot!(p1, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)
    # right rectangle: x in (xdiv, xmax]
    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [ymin, ymin, ymax, ymax]
    plot!(p1, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)

    vline!(p1, [xdiv], color=:black, linestyle=:dash, linewidth=0.5)
    savefig(p1, "gk_bonddim_T=$T.svg")
    display(p1)
end

function bondlistplotfixednk(bonddimlist1::Vector{Vector{Int}}, bonddimlist2::Vector{Vector{Int}}, Rt, nk, times, Nk)
    # This function expects 1D vectors `bonddimlist1` and `bonddimlist2` where
    # each element is a Vector{Int} of bond dimensions for a given time.

    # collect global max for plotting/annotation
    vals = Int[]
    for i in 1:length(bonddimlist1)
        v1 = bonddimlist1[i]
        if !isempty(v1)
            append!(vals, v1)
        end
        v2 = bonddimlist2[i]
        if !isempty(v2)
            append!(vals, v2)
        end
    end
    global_max = isempty(vals) ? 1 : maximum(vals)

    # prepare palette (one color per time index)
    colors = [:blue, :red, :green, :orange, :purple, :brown, :magenta, :cyan]

    # compute xmax (max bond index across all vectors)
    xmax = 1
    for i in 1:length(bonddimlist1)
        v1 = bonddimlist1[i]
        v2 = bonddimlist2[i]
        if !isempty(v1)
            xmax = max(xmax, length(v1))
        end
        if !isempty(v2)
            xmax = max(xmax, length(v2))
        end
    end

    p1 = plot(title="tmax scaling for Nk=$Nk", xlabel="Bond number",
        ylabel="Bond dimension", xlim=(1, xmax), ylim=(0, global_max * 1.2), framestyle=:box,
        ticks=:auto)

    # Plot a small selection of times (only if indices exist)
    sel = [2, 5, 8]
    for idx in sel
        if idx <= length(times)
            col = colors[mod1(idx, length(colors))]
            vec1 = bonddimlist1[idx]
            vec2 = bonddimlist2[idx]
            if !isempty(vec1)
                plot!(p1, vec1, label="kx tmax=$(times[idx])", color=col)
            end
            if !isempty(vec2)
                plot!(p1, vec2, label="ky tmax=$(times[idx])", color=col, seriestype=:scatter)
            end
        end
    end

    # vertical divider at nk
    xdiv = nk

    # shaded backgrounds: left region and right region (pale colors)
    ymin = 0
    ymax = global_max * 1.2
    # clamp xdiv to plotting range for the shading polygons
    xdivc = clamp(xdiv, 1, xmax)
    left_x = [1, xdivc, xdivc, 1]
    left_y = [ymin, ymin, ymax, ymax]
    plot!(p1, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)
    right_x = [xdivc, xmax, xmax, xdivc]
    right_y = [ymin, ymin, ymax, ymax]
    plot!(p1, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)

    # vertical dotted line at exact 2*nk
    vline!(p1, [xdiv], color=:black, linestyle=:dot, linewidth=0.8)

    savefig(p1, "gk_bonddim_N=$Nk.svg")
    display(p1)
    return p1
end