using Random
using Plots
using Graphs
using TreeTCI
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

function sampled_error(f, ttn, nsamples::Int64, bits::Int64)
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

function seed_pivots!(tci, npivots::Int64, f)
    """ Seed initial pivots for the SimpleTCI tci based on binary representations. """
    count = 0
    while count < npivots
        # Generate a random input of length equal to number of vertices
        x = rand(1:2, length(vertices(tci.g)))
        if abs(f(x)) > 0.5
            TreeTCI.addglobalpivots!(tci, [x])
            count += 1
        else
            continue
        end
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

function n(ϵ::Float64, β::Float64)
    """ Fermi-Dirac distribution """
    return 1.0 / (exp(β * ϵ) + 1.0)
end

function assign(v::AbstractVector{Int64}, nk::Int64, R::Int64)
    """ Assign variables from bit vector v using explicit `nk` and `R`.

    v is expected to be a vector with length 2*nk + 2*R (kx bits, ky bits, t1 bits, t2 bits).
    """
    kx = bits2decimal(v[1:nk])
    ky = bits2decimal(v[nk+1:2*nk])
    t1 = bits2decimal(v[2*nk+1:2*nk+R])
    t2 = bits2decimal(v[2*nk+R+1:2*nk+2*R])
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

function plotgk(kx, ky, β, η, tmax)
    f(x, y) = -1im * n(kx, ky, β) * exp(-1im * ϵ(kx, ky) * tmax * (x - y)) * exp(-(ϵ(kx, ky)^2) * η * tmax * abs(x - y))

    xs = ys = range(0, 1, length=400)
    zs = [f(x, y) for x in xs, y in ys]
    # Plots/GR cannot compare Complex values when computing extrema; convert to real
    zs_imag = imag.(zs)
    zs_real = real.(zs)
    println("Max real part: ", maximum(zs_real))
    println("Max imag part: ", maximum(zs_imag))
    realpart = heatmap(xs, ys, zs_real, title="Real part of G(0,0)", xlabel="t", ylabel="t'", size=(600, 500))
    imagpart = heatmap(xs, ys, zs_imag, title="Imaginary part of G(0,0)", xlabel="t", ylabel="t'", size=(600, 500))
    savefig(plot(realpart, imagpart, layout=(1, 2), size=(1200, 500)), "gk_$(kx)_$(ky)_tmax$(tmax).svg")
    display(plot(realpart, imagpart, layout=(1, 2), size=(1200, 500)))

end

function plotgk2(kx, ky, β, η, tmax)
    f(x, y) = -1im * n(kx, ky, β) * exp(-1im * ϵ(kx, ky) * tmax * (x - y)) * exp(-(ϵ(kx, ky)^2) * η * tmax * abs(x - y))

    ys = 0.5
    xs = range(0, 1, length=400)
    zs = [f(x, ys) for x in xs]
    # Plots/GR cannot compare Complex values when computing extrema; convert to real    
    zs_imag = imag.(zs)
    zs_real = real.(zs)
    plt1 = plot(xs, zs_real, label="Real part")
    plot!(plt1, xs, zs_imag, label="Imaginary part")

end

function plotdcgk(kx, ky)
    E = 1.0
    T = 50.0
    f(x, y) = -1im * exp(-1im * ((ϵ(kx, ky) * sin(E * T * x) / E) + δ(kx, ky) * (cos(E * T * x) - 1) / E - ϵ(kx, ky) * sin(E * T * y) / E - δ(kx, ky) * (cos(E * T * y) - 1) / E)) * exp(-(ϵ(kx, ky)^2) * 0.01 * T * abs(x - y)) * n(kx, ky, 10.0)
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
    add_edge!(g, 1, 2 * nk)
    add_edge!(g, nk + 1, 2 * nk + 1)
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

    for i in nk:-1:2
        add_edge!(g, i, i - 1)
    end
    add_edge!(g, 1, 2 * nk)
    for i in (2*nk):-1:(nk+2)
        add_edge!(g, i, i - 1)
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

function CTTNALT3(Rt::Int64, nk::Int64)
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

function BTTN2(Rt::Int64, nk::Int64)
    """ BTTN topology for momentum-dependent Green's function G_{kx,ky}(t,t') """
    g = NamedGraph()
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    lengths = (Rt, Rt)

    # Tree for time bits 
    start = 2 * nk + 1
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

        start += L
    end
    # Connect both time trees
    add_edge!(g, 2 * nk + 1, 2 * nk + Rt + 1)
    # Kx bits 
    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end
    # Connect kx bits to time t tree 
    add_edge!(g, 1, 2 * nk + 1)
    # Ky bits
    for i in (nk+1):(2*nk-1)
        add_edge!(g, i, i + 1)
    end
    # Connect ky bits to time t' tree
    add_edge!(g, nk + 1, 2 * nk + Rt + 1)
    return g
end

function BTTN3(Rt::Int64, nk::Int64)
    """BTTN topology for momentum-dependent Green's function G_{kx,ky}(t,t')"""
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    # Helper: build a binary tree over nodes in a given range [from, from+L-1]
    function build_binary_tree!(g, from, L)
        for i in 1:L
            left = 2 * i
            right = 2 * i + 1
            if left <= L
                add_edge!(g, from + i - 1, from + left - 1)
            end
            if right <= L
                add_edge!(g, from + i - 1, from + right - 1)
            end
        end
    end

    # Time bits t1: nodes 2*nk+1 ... 2*nk+Rt
    build_binary_tree!(g, 2 * nk + 1, Rt)

    # Time bits t2: nodes 2*nk+Rt+1 ... 2*nk+2*Rt
    build_binary_tree!(g, 2 * nk + Rt + 1, Rt)

    # kx bits: nodes 1 ... nk  (chain)
    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end

    # ky bits: nodes nk+1 ... 2*nk  (chain)
    for i in (nk+1):(2*nk-1)
        add_edge!(g, i, i + 1)
    end

    # Connect kx chain end to ky chain start
    add_edge!(g, nk, nk + 1)

    # Connect kx/ky junction to root of t1 tree
    add_edge!(g, 1, 2 * nk + 1)

    # Connect t1 root to t2 root
    add_edge!(g, 2 * nk + 1, 2 * nk + Rt + 1)

    return g
end


""" 
    ITTN(R::Int, d::Int=2)

Returns an Interleaved TTN.

Generate a NamedGraph representing a quantics Interleaved TTN topology for 2D functions,
with `R` bits precision per dimension (total 2R vertices).
yR   ...   y2   y1
 |         |    |
xR - ... - x2 - x1 - kx1 - ... - kxnk - ky1 - ... - kynk
"""
function ITTN(nk::Int64, Rt::Int64)
    """
    Interleaved Tree Tensor Network topology for G_{kx,ky}(t,t').

    Layout (indices 1-based):
      kx-chain  : 1 … nk
      ky-chain  : nk+1 … 2nk
      t1-chain  : 2nk+1 … 2nk+Rt          (forward time)
      t2-chain  : 2nk+Rt+1 … 2nk+2Rt      (backward time)

    Structure:
      • kx-chain and ky-chain are each linear; joined at their ends: nk — (nk+1)
      • t1 and t2 chains are interleaved: each t1[i] — t2[i] (vertical rung)
        and t1[i] — t1[i+1] along the chain
      • Bridge: 2nk (last ky node) — (2nk+1) (first t1 node)

    Maximum degree is 3 everywhere (verified below).
    """
    N = 2 * nk + 2 * Rt
    g = NamedGraph(N)

    # ── kx chain: 1 – 2 – … – nk ──────────────────────────────────────────
    for i in 1:(nk-1)
        add_edge!(g, i, i + 1)
    end

    # ── ky chain: (nk+1) – (nk+2) – … – 2nk ───────────────────────────────
    for i in 1:(nk-1)
        add_edge!(g, nk + i, nk + i + 1)
    end

    # ── join kx and ky at their meeting ends: nk — (nk+1) ──────────────────
    add_edge!(g, nk, nk + 1)

    # ── bridge from k-block to t-block: 2nk — (2nk+1) ─────────────────────
    # Node 2nk has degree 1 (end of ky-chain), so adding this keeps degree ≤ 2.
    add_edge!(g, 1, 2 * nk + 1)

    # ── t1 chain: (2nk+1) – (2nk+2) – … – (2nk+Rt) ────────────────────────
    for i in 1:(Rt-1)
        add_edge!(g, 2 * nk + i, 2 * nk + i + 1)
    end

    # ── vertical rungs t1[i] — t2[i] ────────────────────────────────────────
    # t2 nodes: 2nk+Rt+1 … 2nk+2Rt   (leaves, degree 1 each)
    # t1 interior nodes then have degree 3: left-chain, right-chain, rung  ✓
    # t1[1] = 2nk+1 has: bridge-edge, right-chain, rung  → degree 3         ✓
    # t1[Rt] = 2nk+Rt has: left-chain, rung             → degree 2          ✓
    for i in 1:Rt
        add_edge!(g, 2 * nk + i, 2 * nk + Rt + i)
    end

    return g
end

#-------------------------------------------------------------
# Plotting and analysis utilities for bond dimensions
#-------------------------------------------------------------

function forkbondlist(bonds, nk::Int, Rt::Int)
    """
    Return two ordered bond dimension lists: one for kx, one for ky.

    kx order:
      - kx chain: nk -> nk-1 -> ... -> 2 -> 1
      - bridge: 1 -> 2*nk + 1
      - time: alternating [2*nk+1, 2*nk+1+Rt, 2*nk+2, 2*nk+2+Rt, ...]

    ky order:
      - ky chain: 2*nk -> ... -> nk+2 -> nk+1
      - bridge: nk + 1 -> 2*nk + 1
      - time: alternating [2*nk+1, 2*nk+1+Rt, 2*nk+2, 2*nk+2+Rt, ...]

    Returns: (bonddims_kx, bonddims_ky)
    """

    # Helper: attempt to read a field as Int
    function _as_int(x)
        if x isa Integer
            return Int(x)
        end
        px = tryparse(Int, string(x))
        return px === nothing ? nothing : Int(px)
    end

    # Extract a field by name or position
    function _getfield_any(x, sym::Symbol, idx::Int)
        # property access
        try
            if hasproperty(x, sym)
                v = getproperty(x, sym)
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        # NamedTuple / Dict
        try
            if isa(x, NamedTuple) && haskey(x, sym)
                v = x[sym]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        try
            if isa(x, AbstractDict) && haskey(x, sym)
                v = x[sym]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        # positional (Tuple/Vector)
        try
            if (isa(x, Tuple) || isa(x, AbstractVector)) && length(x) >= idx
                v = x[idx]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        return nothing
    end

    # Build map of undirected edge -> bdim (first occurrence wins)
    bondmap = Dict{Tuple{Int,Int},Int}()
    for b in bonds
        src = _getfield_any(b, :src, 1)
        dst = _getfield_any(b, :dst, 2)
        bdim_val = _getfield_any(b, :bdim, 3)

        if src === nothing || dst === nothing || bdim_val === nothing
            @warn "Skipping bond with unrecognized format" b = b
            continue
        end

        key = src <= dst ? (src, dst) : (dst, src)
        if !haskey(bondmap, key)
            bondmap[key] = bdim_val
        end
    end

    # Build kx order
    pairs_kx = Tuple{Int,Int}[]
    for i in nk:-1:2
        push!(pairs_kx, (i, i - 1))
    end
    m = 2 * nk + 1
    push!(pairs_kx, (1, m))
    for t in 0:(Rt-1)
        push!(pairs_kx, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs_kx, (m + Rt + t, m + t + 1))
        end
    end

    # Build ky order
    pairs_ky = Tuple{Int,Int}[]
    for i in (2*nk):-1:(nk+2)
        push!(pairs_ky, (i, i - 1))
    end
    push!(pairs_ky, (nk + 1, m))
    for t in 0:(Rt-1)
        push!(pairs_ky, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs_ky, (m + Rt + t, m + t + 1))
        end
    end

    # Collect in requested order
    result_kx = Int[]
    for (a, b) in pairs_kx
        k = a <= b ? (a, b) : (b, a)
        val = get(bondmap, k, 0)
        if val == 0
            @warn "Bond not found for edge" edge = k
        end
        push!(result_kx, val)
    end

    result_ky = Int[]
    for (a, b) in pairs_ky
        k = a <= b ? (a, b) : (b, a)
        val = get(bondmap, k, 0)
        if val == 0
            @warn "Bond not found for edge" edge = k
        end
        push!(result_ky, val)
    end

    return result_kx, result_ky
end

function forkbondplot(bonddims_kx::AbstractVector{<:Integer}, bonddims_ky::AbstractVector{<:Integer}, nk::Int, Rt::Int; outfile="gk_bonddim_fork.svg", title="Bond dimensions for equilibrium Green's function (CTTN)")
    # Plot both kx (line) and ky (scatter) on same graph, dashed line at nk, orange/lightgreen bands
    if isempty(bonddims_kx) || isempty(bonddims_ky)
        @warn "bonddims_kx or bonddims_ky is empty; nothing to plot"
        return nothing
    end

    xmax = max(length(bonddims_kx), length(bonddims_ky))
    ymax = max(maximum(bonddims_kx), maximum(bonddims_ky))
    xdiv = nk

    p = plot(1:length(bonddims_kx), bonddims_kx;
        title=title,
        xlabel="Bond number",
        ylabel="Bond dimension",
        xlim=(1, xmax),
        ylim=(0, ymax * 1.1),
        label="kx",
        framestyle=:box,
        color=:black)

    plot!(p, 1:length(bonddims_ky), bonddims_ky;
        label="ky",
        color=:black,
        marker=:o,
        markersize=3,
        linestyle=:solid)

    # shaded backgrounds
    left_x = [1, xdiv, xdiv, 1]
    left_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)

    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)

    # dashed divider at nk
    vline!(p, [xdiv], color=:black, linestyle=:dash, linewidth=0.8)

    savefig(p, outfile)
    display(p)
    return p
end

function multipleforkbondplot(listlistkx::AbstractVector{<:AbstractVector{<:Integer}}, listlistky::AbstractVector{<:AbstractVector{<:Integer}}, nk::Int, Rt::Int, times::AbstractVector{<:Real}; outfile="gk_bonddim_fork_multi.svg", title="Bond dimensions over times (CTTN)")
    # Validate inputs
    if isempty(listlistkx) || isempty(listlistky)
        @warn "listlistkx or listlistky is empty; nothing to plot"
        return nothing
    end
    if length(listlistkx) != length(listlistky)
        @warn "listlistkx and listlistky must have the same number of time series"
        return nothing
    end

    nseries = length(listlistkx)
    # Determine global x/y ranges
    xmax = nk + 2 * Rt  # max possible bond count based on topology
    ymax = maximum([maximum(v) for v in vcat(listlistkx..., listlistky...)])
    xdiv = nk

    # Prepare color palette (sample viridis gradient at evenly spaced positions)
    if nseries == 1
        palette = [get(cgrad(:viridis), 0.5)]
    else
        palette = [get(cgrad(:viridis), (i - 1) / (nseries - 1)) for i in 1:nseries]
    end

    p = plot(title=title,
        xlabel="Bond number",
        ylabel="Bond dimension",
        xlim=(1, xmax),
        ylim=(0, ymax * 1.1),
        framestyle=:box)

    # Use the provided `times` vector for labels and ensure its length matches the series
    if length(times) != nseries
        @warn "Length of `times` does not match number of series; falling back to indices for labels"
    end
    for i in 1:nseries
        kx = listlistkx[i]
        ky = listlistky[i]
        color = palette[i]
        lblkx = (length(times) == nseries) ? "kx T=$(times[i])" : "kx t=$(i)"
        lblky = (length(times) == nseries) ? "ky T=$(times[i])" : "ky t=$(i)"
        plot!(p, 1:length(kx), kx; label=lblkx, color=color, linewidth=2)
        plot!(p, 1:length(ky), ky; label=lblky, color=color, marker=:o, markersize=3, linestyle=:dash)
    end

    # shaded backgrounds
    left_x = [1, xdiv, xdiv, 1]
    left_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)

    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)

    # dashed divider at nk
    vline!(p, [xdiv], color=:black, linestyle=:dash, linewidth=0.8)

    savefig(p, outfile)
    display(p)
    return p
end

function trainbondlist(bonds, nk::Int, Rt::Int)
    """
    Build bond-dimension vector following the ordering induced by the `QTTALT`
    topology.

    Ordering used:
      1) k-section: consecutive bonds (1,2), (2,3), ..., (2*nk, 2*nk+1)
      2) time bridge/section: alternating (m+t, m+Rt+t), (m+Rt+t, m+t+1), ...

    Accepts the same variety of `bonds` representations as the other helpers.
    """

    # Build ordered edge list according to QTTALT connectivity
    pairs = Tuple{Int,Int}[]

    # 1) k-section (ascending consecutive bonds 1->2->...->2*nk->m)
    for i in 1:(2*nk)
        push!(pairs, (i, i + 1))
    end

    # 2) time section (ascending, alternating t / t')
    m = 2 * nk + 1
    for t in 0:(Rt-1)
        push!(pairs, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs, (m + Rt + t, m + t + 1))
        end
    end

    # Helper: attempt to read a field as Int
    function _as_int(x)
        if x isa Integer
            return Int(x)
        end
        px = tryparse(Int, string(x))
        return px === nothing ? nothing : Int(px)
    end

    # Extract a field by name or position
    function _getfield_any(x, sym::Symbol, idx::Int)
        # property access
        try
            if hasproperty(x, sym)
                v = getproperty(x, sym)
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        # NamedTuple / Dict
        try
            if isa(x, NamedTuple) && haskey(x, sym)
                v = x[sym]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        try
            if isa(x, AbstractDict) && haskey(x, sym)
                v = x[sym]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        # positional (Tuple/Vector)
        try
            if (isa(x, Tuple) || isa(x, AbstractVector)) && length(x) >= idx
                v = x[idx]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        return nothing
    end

    # Build map of undirected edge -> bdim (first occurrence wins)
    bondmap = Dict{Tuple{Int,Int},Int}()
    for b in bonds
        src = _getfield_any(b, :src, 1)
        dst = _getfield_any(b, :dst, 2)
        bdim_val = _getfield_any(b, :bdim, 3)

        if src === nothing || dst === nothing || bdim_val === nothing
            @warn "Skipping bond with unrecognized format" b = b
            continue
        end

        key = src <= dst ? (src, dst) : (dst, src)
        if !haskey(bondmap, key)
            bondmap[key] = bdim_val
        end
    end

    # Collect in requested order
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

function trainbondplot(bonddims::AbstractVector{<:Integer}, nk::Int, Rt::Int; outfile="gk_bonddim_QTTALT.svg", title="Bond dimensions for equilibrium Green's function (QTT-ALT)")
    # Minimal plot: bond number vs bond dimension, keep dashed line at 2*nk and orange/lightgreen bands.
    if isempty(bonddims)
        @warn "bonddims is empty; nothing to plot"
        return nothing
    end

    xmax = length(bonddims)
    ymax = maximum(bonddims)
    xdiv = 2 * nk

    p = plot(1:xmax, bonddims;
        title=title,
        xlabel="Bond number",
        ylabel="Bond dimension",
        xlim=(1, xmax),
        ylim=(0, ymax * 1.1),
        legend=false,
        framestyle=:box,
        color=:black)

    # shaded backgrounds
    left_x = [1, xdiv, xdiv, 1]
    left_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)

    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)

    # dashed divider at 2*nk
    vline!(p, [xdiv], color=:black, linestyle=:dash, linewidth=0.8)

    savefig(p, outfile)
    display(p)
    return p
end

function multipletrainbondplot(listbonddims::AbstractVector{<:AbstractVector{<:Integer}}, nk::Int, Rt::Int, times::AbstractVector{<:Real}; outfile="gk_bonddim_QTTALT_multi.svg", title="Bond dimensions over times (QTT-ALT)")
    # Validate inputs
    if isempty(listbonddims)
        @warn "listbonddims is empty; nothing to plot"
        return nothing
    end

    nseries = length(listbonddims)
    # Determine global x/y ranges
    xmax = 2 * nk + 2 * Rt  # max possible bond count based on topology
    ymax = maximum([maximum(v) for v in listbonddims])
    xdiv = 2 * nk

    # Prepare color palette (sample viridis gradient at evenly spaced positions)
    if nseries == 1
        palette = [get(cgrad(:viridis), 0.5)]
    else
        palette = [get(cgrad(:viridis), (i - 1) / (nseries - 1)) for i in 1:nseries]
    end

    p = plot(title=title,
        xlabel="Bond number",
        ylabel="Bond dimension",
        xlim=(1, xmax),
        ylim=(0, ymax * 1.1),
        framestyle=:box)

    # Use the provided `times` vector for labels and ensure its length matches the series
    if length(times) != nseries
        @warn "Length of `times` does not match number of series; falling back to indices for labels"
    end
    for i in 1:nseries
        bonddims = listbonddims[i]
        color = palette[i]
        lbl = (length(times) == nseries) ? "T=$(times[i])" : "t=$(i)"
        plot!(p, 1:length(bonddims), bonddims; label=lbl, color=color, linewidth=2, marker=:o)
    end

    # shaded backgrounds
    left_x = [1, xdiv, xdiv, 1]
    left_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)
    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent,
        label=false)
    # dashed divider at 2*nk
    vline!(p, [xdiv], color=:black, linestyle=:dash, linewidth=0.8)
    savefig(p, outfile)
    display(p)
    return p
end

function trainbondlist4(bonds, nk::Int, Rt::Int)
    """
    Return bond dimensions ordered for the QTT-ALT topology.

    Ordering rules:
      1. k-section (2*nk bonds): traverse kx/ky alternated from high to low index
         for i = 2*nk : -1 : (nk+1)
             (i, i-nk)        # vertical ky_i -> kx_i
             if i-nk > 1: (i-nk, i-1)  # diagonal to next
      2. Bridge from k to time: (1, 2*nk + 1)
      3. Time section (2*Rt bonds), alternating t / t', increasing index:
         let m = 2*nk + 1, r = m + Rt
         sequence S = [m, r, m+1, r+1, ..., m+Rt-1, r+Rt-1]
         add consecutive pairs from S.

    Accepts any iterable of bonds exposing src, dst, bdim (NamedTuple, struct, Dict,
    tuple/vector with positional fields). Missing edges are returned as 0 with a warning.
    """

    # Build ordered edge list
    pairs = Tuple{Int,Int}[]

    # 1) k-section (descending)
    for i in (2*nk):-1:(nk+1)
        push!(pairs, (i, i - nk))
        if i - nk > 1
            push!(pairs, (i - nk, i - 1))
        end
    end

    # 2) bridge to time bits
    m = 2 * nk + 1
    push!(pairs, (1, m))

    # 3) time section (ascending, alternating t / t')
    for t in 0:(Rt-1)
        push!(pairs, (m + t, m + Rt + t))
        if t < Rt - 1
            push!(pairs, (m + Rt + t, m + t + 1))
        end
    end

    # Helper: attempt to read a field as Int
    function _as_int(x)
        if x isa Integer
            return Int(x)
        end
        px = tryparse(Int, string(x))
        return px === nothing ? nothing : Int(px)
    end

    # Extract a field by name or position
    function _getfield_any(x, sym::Symbol, idx::Int)
        # property access
        try
            if hasproperty(x, sym)
                v = getproperty(x, sym)
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        # NamedTuple / Dict
        try
            if isa(x, NamedTuple) && haskey(x, sym)
                v = x[sym]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        try
            if isa(x, AbstractDict) && haskey(x, sym)
                v = x[sym]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        # positional (Tuple/Vector)
        try
            if (isa(x, Tuple) || isa(x, AbstractVector)) && length(x) >= idx
                v = x[idx]
                vi = _as_int(v)
                vi !== nothing && return vi
            end
        catch
        end
        return nothing
    end

    # Build map of undirected edge -> bdim (first occurrence wins)
    bondmap = Dict{Tuple{Int,Int},Int}()
    for b in bonds
        src = _getfield_any(b, :src, 1)
        dst = _getfield_any(b, :dst, 2)
        bdim_val = _getfield_any(b, :bdim, 3)

        if src === nothing || dst === nothing || bdim_val === nothing
            @warn "Skipping bond with unrecognized format" b = b
            continue
        end

        key = src <= dst ? (src, dst) : (dst, src)
        if !haskey(bondmap, key)
            bondmap[key] = bdim_val
        end
    end

    # Collect in requested order
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

function trainbondplot4(bonddims::AbstractVector{<:Integer}, nk::Int, Rt::Int; outfile="gk_bonddim_QTTALT.svg", title="Bond dimensions for equilibrium Green's function (QTT-ALT)")
    # Minimal plot: bond number vs bond dimension, keep dashed line at 2*nk and orange/lightgreen bands.
    if isempty(bonddims)
        @warn "bonddims is empty; nothing to plot"
        return nothing
    end

    xmax = length(bonddims)
    ymax = maximum(bonddims)
    xdiv = 2 * nk

    p = plot(1:xmax, bonddims;
        title=title,
        xlabel="Bond number",
        ylabel="Bond dimension",
        xlim=(1, xmax),
        ylim=(0, ymax * 1.1),
        legend=false,
        framestyle=:box,
        color=:black)

    # shaded backgrounds
    left_x = [1, xdiv, xdiv, 1]
    left_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)

    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)

    # dashed divider at 2*nk
    vline!(p, [xdiv], color=:black, linestyle=:dash, linewidth=0.8)

    savefig(p, outfile)
    display(p)
    return p
end

function multipletrainbondplot4(listbonddims::AbstractVector{<:AbstractVector{<:Integer}}, nk::Int, Rt::Int, times::AbstractVector{<:Real}; outfile="gk_bonddim_QTTALT4_multi.svg", title="Bond dimensions over times (QTT-ALT4)")
    # Validate inputs
    if isempty(listbonddims)
        @warn "listbonddims is empty; nothing to plot"
        return nothing
    end

    nseries = length(listbonddims)
    # Determine global x/y ranges
    xmax = 2 * nk + 2 * Rt  # max possible bond count based on topology
    ymax = maximum([maximum(v) for v in listbonddims])
    xdiv = 2 * nk

    # Prepare color palette (sample viridis gradient at evenly spaced positions)
    if nseries == 1
        palette = [get(cgrad(:viridis), 0.5)]
    else
        palette = [get(cgrad(:viridis), (i - 1) / (nseries - 1)) for i in 1:nseries]
    end

    p = plot(title=title,
        xlabel="Bond number",
        ylabel="Bond dimension",
        xlim=(1, xmax),
        ylim=(0, ymax * 1.1),
        framestyle=:box)

    # Use the provided `times` vector for labels and ensure its length matches the series
    if length(times) != nseries
        @warn "Length of `times` does not match number of series; falling back to indices for labels"
    end
    for i in 1:nseries
        bdims = listbonddims[i]
        color = palette[i]
        lbl = (length(times) == nseries) ? "T=$(times[i])" : "t=$(i)"
        plot!(p, 1:length(bdims), bdims; label=lbl, color=color, linewidth=2, marker=:o)
    end

    # shaded backgrounds
    left_x = [1, xdiv, xdiv, 1]
    left_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, left_x, left_y, seriestype=:shape, fillcolor=:orange, fillalpha=0.08, linecolor=:transparent, label=false)

    right_x = [xdiv, xmax, xmax, xdiv]
    right_y = [0, 0, ymax * 1.1, ymax * 1.1]
    plot!(p, right_x, right_y, seriestype=:shape, fillcolor=:lightgreen, fillalpha=0.08, linecolor=:transparent, label=false)
    savefig(p, outfile)
    display(p)
    return p
end