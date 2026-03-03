using Random
using Plots
using Graphs
using TreeTCI
using NamedGraphs

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

