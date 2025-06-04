using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using DataFrames, DataFrameMacros

include(srcdir("wasserstein_distance.jl"))
include(srcdir("milad_circuit.jl"))

df = collect_results(datadir("MiladCircuitDistances"))

for df_nqubits in @groupby(df, :Nqubits, :i1, :i2, :optimizer)
    i1 = df_nqubits[1, :i1]
    i2 = df_nqubits[1, :i2]
    Nqubits = df_nqubits[1, :Nqubits]
    Npoints = df_nqubits[1, :Npoints]
    optimizer = df_nqubits[1, :optimizer]
    theta2s = df_nqubits[1, :theta2range]

    sorted_df = sort(df_nqubits, :theta1)
    theta1s = sorted_df[:, :theta1]

    heatmap_size = (length(theta1s), length(theta2s))
    data_infidelity = fill(NaN, heatmap_size)
    data_W1 = fill(NaN, heatmap_size)

    for (i, row) in enumerate(eachrow(sorted_df))
        data_infidelity[i,:] .= row[:infidelity_vec]
        data_W1[i,:] .= row[:W1_vec]
    end

    d = @dict(Nqubits, Npoints, optimizer, i1, i2)
    results = merge(d, @dict(theta1s, theta2s, data_infidelity, data_W1))
    results = Dict(string(k) => v for (k, v) in results)
    wsave(datadir("MiladCircuitHeatmaps", savename(d, "jld2")), results)
end

