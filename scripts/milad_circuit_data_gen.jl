using DrWatson
@quickactivate "JuliaPulseExperiments"
using Random, Yao, ProgressMeter
include(srcdir("milad_circuit.jl"))

function makesim(d::Dict)
    @unpack Nqubits, i1, i2, theta1, theta2 = d
    angles = rand(MersenneTwister(0), Nqubits) .* 2pi
    angles[1] = pi
    angles[i1] = theta1
    angles[i2] = theta2

    final_state = run_milad_circuit(angles) |> statevec
    fulld = Dict{String, Any}(copy(d))
    fulld["final_state"] = final_state
    return fulld
end

function main(partition_size=101)
    allparams = Dict{String, Any}(
        "Nqubits" => collect(3:8),
        "i1" => [2],
        "i2" => [3],
        "theta1" => collect(LinRange(0,2pi,partition_size)),
        "theta2" => collect(LinRange(0,2pi,partition_size)),
    )

    dicts = dict_list(allparams)
    @showprogress for (i, dicts_partition) in enumerate(Iterators.partition(dicts, partition_size))
        partition_sim_names = savename.(dicts_partition)
        partition_sim_results = [makesim(d) for d in dicts_partition]
        name_result_pairs = zip(partition_sim_names, partition_sim_results)
        wsave(
            datadir("MiladCircuitStates", "results_partition_$i.jld2"),
            Dict(name_result_pairs)
        )
    end
end

main()
