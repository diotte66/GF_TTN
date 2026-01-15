import TensorCrossInterpolation as TCI
using Plots
include(joinpath(@__DIR__, "src", "utils_gk.jl"))

function gk(v, nk, R, T)
    """ Momentum-dependent equilibrium Green's function G_{kx,ky}(t,t') """
    kx, ky = bits2decimal(reverse(v[1:2:2*nk])), bits2decimal(reverse(v[2:2:2*nk]))
    t1, t2 = bits2decimal(v[2*nk+1:2:2*nk+2*R]), bits2decimal(v[2*nk+2:2:2*nk+2*R])
    return -1im * exp(-1im * ϵ(kx, ky) * T * (t1 - t2)) * exp(-T * abs(t1 - t2))
end

function sampled_errors(f, ttn, nsamples::Int, bits::Int)
    """ Compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits."""
    error_inf = 0.0
    error_l1 = 0.0
    for _ in 1:nsamples
        # Generate a random 3R sequence of 1s and 2s
        x = rand(1:2, bits)
        # Evaluate the concrete TreeTensorNetwork (it provides evaluate/call)
        err = abs(f(x) - ttn(x))
        error_inf = max(error_inf, err)
        error_l1 += err
    end
    return error_inf, error_l1 / nsamples
end

function main()
    nk = 8
    Rt = 30
    localdims = fill(2, 2 * Rt + 2 * nk)
    times = [1.0, 40.0, 200.0, 500.0]
    maxbond = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    error_l1 = zeros(length(times), length(maxbond))
    mem = zeros(length(times), length(maxbond))

    for (i, T) in enumerate(times)
        for (j, maxbd) in enumerate(maxbond)
            println("Time T = $T, Max bond dimension: $maxbd")
            kwargs = (
                maxiter=5,
                maxbonddim=maxbd
            )
            ttn = TCI.TensorCI2{ComplexF64}(v -> gk(v, nk, Rt, T), localdims)
            ranks, errors = TCI.optimize!(ttn, v -> gk(v, nk, Rt, T); kwargs...)
            mem[i, j] = Base.summarysize(ttn)
            errinf, errl1 = sampled_errors(v -> gk(v, nk, Rt, T), ttn, 1000, 2 * Rt + 2 * nk)
            error_l1[i, j] = errl1
            println("L1 Error: $errl1, Memory usage: $(mem[i,j]) bytes")
        end
    end

    subplots1 = []
    for (i, T) in enumerate(times)
        subplot1 = plot(title="Time T = $T", xlabel="Max Bond Dimension", ylabel="L1 Error", yscale=:log10)
        plot!(subplot1, maxbond, error_l1[i, :], label="", marker=:o)
        push!(subplots1, subplot1)
    end
    plt1 = plot(subplots1..., layout=(2, 2), size=(1200, 900))
    savefig(plt1, "gk_simple_compression_error.svg")
    display(plt1)
end