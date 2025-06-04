using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using DataFrames, DataFrameMacros, CairoMakie

function main()
    df = collect_results(datadir("MiladCircuitHeatmaps"))

    mkpath(plotsdir("MiladCircuitHeatmaps")) # Where we will store plots

    for df_nqubits in @groupby(df, :Nqubits, :i1, :i2, :optimizer)
        @assert size(df_nqubits, 1) == 1
        Nqubits = df_nqubits[1,:Nqubits]
        i1 = df_nqubits[1,:i1]
        i2 = df_nqubits[1,:i2]
        optimizer = df_nqubits[1,:optimizer]
        Npoints = df_nqubits[1,:Npoints]
        x = df_nqubits[1,:theta1s]
        y = df_nqubits[1,:theta2s]

        data_inf = df_nqubits[1,:data_infidelity]
        dict_inf = @dict(Nqubits, i1, i2, Npoints, cost="infidelity")

        fig_inf = Figure()
        ax_inf = Axis(
            fig_inf[1,1], xlabel="theta $i1", ylabel="theta $i2",
            title=savename(dict_inf, connector=", ")
        )
        hm_inf = heatmap!(ax_inf, x, y, data_inf)
        Colorbar(fig_inf[1,end+1], hm_inf)
        save(plotsdir("MiladCircuitHeatmaps", savename(dict_inf, "png")), fig_inf)

        if !(optimizer == "none") # Only make W1 plot if W1 distances were computed
            data_W1 = df_nqubits[1,:data_W1]
            dict_W1 = @dict(Nqubits, i1, i2, optimizer, Npoints, cost="W1")

            fig_W1 = Figure()
            ax_W1 = Axis(
                fig_W1[1,1], xlabel="theta $i1", ylabel="theta $i2",
                title=savename(dict_W1, connector=", ")
            )
            hm_W1 = heatmap!(ax_W1, x, y, data_W1)
            Colorbar(fig_W1[1,end+1], hm_W1)
            save(plotsdir("MiladCircuitHeatmaps", savename(dict_W1, "png")), fig_W1)
        end
    end
end

main()
