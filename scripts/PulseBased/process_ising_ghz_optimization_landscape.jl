using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using DataFrames, DataFrameMacros

include(srcdir("wasserstein_distance.jl"))
include(srcdir("ising_model_hamiltonian.jl"))

df = collect_results(datadir("PulseGhzDistances"))

for df_nqubits in @groupby(df, :Nqubits, :i1, :i2, :optimizer, :controlFuncType, :controlPermType, :J, :T)
    i1 = df_nqubits[1, :i1]
    i2 = df_nqubits[1, :i2]
    Nqubits = df_nqubits[1, :Nqubits]
    Npoints = df_nqubits[1, :Npoints]
    optimizer = df_nqubits[1, :optimizer]
    theta2s = df_nqubits[1, :theta2range]
    controlFuncType = df_nqubits[1, :controlFuncType]
    controlPermType = df_nqubits[1, :controlPermType]
    J = df_nqubits[1, :J]
    T = df_nqubits[1, :T]

    sorted_df = sort(df_nqubits, :theta1)
    theta1s = sorted_df[:, :theta1]

    heatmap_size = (length(theta1s), length(theta2s))
    data_infidelity = fill(NaN, heatmap_size)

    for (i, row) in enumerate(eachrow(sorted_df))
        data_infidelity[i,:] .= row[:infidelity_vec]
    end
    if (optimizer == "none")
        data_W1 = missing
    else
        data_W1 = fill(NaN, heatmap_size)
        for (i, row) in enumerate(eachrow(sorted_df))
            data_W1[i,:] .= row[:W1_vec]
        end
    end

    d = @dict(Nqubits, Npoints, optimizer, i1, i2, controlFuncType, controlPermType, J, T)
    results = merge(d, @dict(theta1s, theta2s, data_infidelity, data_W1))
    results = Dict(string(k) => v for (k, v) in results)
    wsave(datadir("PulseGhzHeatmaps", savename(d, "jld2")), results)
end

