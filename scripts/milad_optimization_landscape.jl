using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using QuantumToolbox, Yao, Random, ProgressMeter

includet(srcdir("wasserstein_distance.jl"))
includet(srcdir("milad_circuit.jl"))

"""
Iterate over theta2 for fixed theta1. Get Infidelity and W1 distances as vectors.
"""
function makesim(d::Dict)
    @unpack Nqubits, i1, i2, theta1, Npoints = d
    angles = rand(MersenneTwister(0), Nqubits) .* 2pi
    angles[1] = pi
    angles[i1] = theta1

    theta2range = LinRange(0, 2pi, Npoints)
    ghz_dm = ghz_operator(Nqubits)

    infidelity_vec = Vector{Float64}(undef, Npoints)
    W1_vec = Vector{Float64}(undef, Npoints)

    for (k, theta2) in enumerate(theta2range)
        angles[i2] = theta2
        final_state = run_milad_circuit(angles) |> statevec

        infidelity_vec[k] = ghz_infidelity(final_state)

        dims = ntuple(_ -> 2, Nqubits)
        final_dm = Qobj(final_state, dims=dims) |> ket2dm
        W1_vec[k] = W1_primal(final_dm, ghz_dm)

        #GC.gc() # Being safe about running out of memory
    end

    fulld = Dict{String, Any}(copy(d))
    fulld["angles"] = angles
    fulld["theta2range"] = theta2range
    fulld["infidelity_vec"] = infidelity_vec
    fulld["W1_vec"] = W1_vec

    return fulld
end

"""
Only really optimal when Nqubits % 4 == 0
"""
function optimal_angles(Nqubits)
    angles_optimal = fill(1pi, Nqubits)
    angles_optimal[1] /= 2
    return angles_optimal
end

function ghz_operator(Nqubits)
    return QuantumToolbox.ghz_state(Nqubits) |> ket2dm
end

function run_milad_circuit_operator(angles)
    dims = ntuple(_ -> 2, length(angles))
    Nqubits = length(angles)
    circuit_output_vec = run_milad_circuit(angles) |> statevec
    return Qobj(circuit_output_vec, dims=dims)  |> ket2dm
end


function get_chunk(v, task_idx, ntasks)
    # task_idx is 0-indexed, to be consistent with slurm
    @assert task_idx < ntasks
    # Divide `v` into `ntasks` parts
    chunksize, rem = divrem(length(v), ntasks)
    start_idx = task_idx * chunksize + min(task_idx, rem) + 1 # The min accounts for offsets due to extra elements in earlier partitions
    end_idx = start_idx + chunksize - 1 + (task_idx < rem ? 1 : 0)
    return view(v, start_idx:end_idx)
end

function main(obj_type=:infidelity)
    Npoints = 11
    max_Nqubits = DrWatson.readenv("MAX_NQUBITS", 4)
    allparams = Dict{String, Any}(
        "Nqubits" => collect(3:max_Nqubits),
        "i1" => [2],
        "i2" => [3],
        "theta1" => collect(LinRange(0,2pi,Npoints)),
        "Npoints" => [Npoints],
    )

    dicts = dict_list(allparams)

    # For this to work, all job arrays should start at 0 and use stepsize 1
    slurm_task_id = DrWatson.readenv("SLURM_ARRAY_TASK_ID", 0)
    slurm_ntasks = DrWatson.readenv("SLURM_ARRAY_TASK_COUNT", 1)

    @showprogress for d in get_chunk(dicts, slurm_task_id, slurm_ntasks)
        f = makesim(d)
        #wsave(datadir("MiladCircuitDistances", savename(d, "jld2")), f)
        produce_or_load(makesim, d, datadir("MiladCircuitDistances"), loadfile=false)
        #wsave(datadir("MiladCircuitDistances", savename(d, "jld2")), f)
    end
end

main()
