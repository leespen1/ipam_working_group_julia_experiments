using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using QuantumToolbox, Yao, Random, ProgressMeter, Dates, MosekTools, SCS

include(srcdir("wasserstein_distance.jl"))
include(srcdir("milad_circuit.jl"))

"""
Iterate over theta2 for fixed theta1. Get Infidelity and W1 distances as vectors.
"""
function makesim(d::Dict)
    @unpack Nqubits, i1, i2, theta1, Npoints, angleInitialization, optimizer = d

    infidelity_vec = Vector{Float64}(undef, Npoints)
    W1_vec = Vector{Float64}(undef, Npoints)
    terminationStatus_vec = Vector{Int}(undef, Npoints)

    theta2range = LinRange(0, 2pi, Npoints)
    ghz_dm = ghz_operator(Nqubits, rel_phase=(-im)^Nqubits)

    angles = Vector{Float64}(undef, Nqubits)
    if angleInitialization == "random"
        angles .= rand(MersenneTwister(0), Nqubits) .* 2pi
    elseif angleInitialization == "optimal"
        angles[2:end] .= pi
        angles[1] = pi/2
    elseif angleInitialization == "pirandom"
        angles .= rand(MersenneTwister(0), Nqubits) .* 2pi
        angles[1] = pi
    elseif angleInitialization == "pidiv2random"
        angles .= rand(MersenneTwister(0), Nqubits) .* 2pi
        angles[1] = pi/2
    end
        

    if optimizer == "mosek"
        optimizer_obj = MosekTools.Optimizer
    elseif optimizer == "scs"
        optimizer_obj = SCS.Optimizer
    elseif optimizer == "none"
        optimizer_obj = nothing
    else
        error("Invalid optimizer string $(d["optimizer"])")
    end

    # Optional silent optimization set by environment
    silent = tryparse(Bool, get(ENV, "SILENT", "true"))

    dims = ntuple(_ -> 2, Nqubits)

    angles[i1] = theta1
    for (k, theta2) in enumerate(theta2range)
        angles[i2] = theta2
        final_state = run_milad_circuit(angles) |> statevec

        infidelity_vec[k] = ghz_infidelity(final_state, rel_phase = (-im)^Nqubits)

        if !isnothing(optimizer_obj)
            final_dm = Qobj(final_state, dims=dims) |> ket2dm
            opt_model = W1_primal(final_dm, ghz_dm, optimizer_obj, silent=silent)

            W1_vec[k] = JuMP.objective_value(opt_model)
            terminationStatus_vec[k] = Int(JuMP.termination_status(opt_model))
        end

        GC.gc() # Being safe about running out of memory
    end

    fulld = Dict{String, Any}(copy(d))
    fulld["angles"] = angles
    fulld["theta2range"] = theta2range
    fulld["infidelity_vec"] = infidelity_vec
    if !isnothing(optimizer_obj)
        fulld["W1_vec"] = W1_vec
        fulld["terminationStatus_vec"] = terminationStatus_vec
    end

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

"""
Construct the GHZ-like state

|000…⟩ + rel_phase*|111…⟩

as a density matrix.
"""
function ghz_operator(Nqubits; rel_phase=1)
    ghz_vec = zeros(2^Nqubits)
    ghz_vec[1] = 1/sqrt(2)
    ghz_vec[end] = 1/sqrt(2)
    dims = ntuple(_ -> 2, Nqubits)
    return Qobj(ghz_vec, dims=dims) |> ket2dm
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

function main()
    Npoints = DrWatson.readenv("NPOINTS", 11)
    max_Nqubits = DrWatson.readenv("MAX_NQUBITS", 4)
    min_Nqubits = DrWatson.readenv("MIN_NQUBITS", 3)
    i1 = DrWatson.readenv("I1", 2)
    i2 = DrWatson.readenv("I2", 3)
    grad = DrWatson.readenv("GRAD", false)
    angleInitialization = get(ENV, "ANGLE_INITIALIZATION", "pirandom") |> lowercase
    optimizer = get(ENV, "OPTIMIZER", "mosek") |> lowercase

    # For this to work, all job arrays should start at 0 and use stepsize 1
    slurm_task_id = DrWatson.readenv("SLURM_ARRAY_TASK_ID", 0)
    slurm_ntasks = DrWatson.readenv("SLURM_ARRAY_TASK_COUNT", 1)



    println("Running test using following environment variables:")
    @show Npoints, max_Nqubits, slurm_task_id, slurm_ntasks

    allparams = Dict{String, Any}(
        "Npoints" => Npoints,
        "Nqubits" => collect(min_Nqubits:max_Nqubits),
        "i1" => i1,
        "i2" => i2,
        "grad" => grad,
        "angleInitialization" => angleInitialization,
        "optimizer" => optimizer,
        "theta1" => collect(LinRange(0,2pi,Npoints)),
    )

    dicts = dict_list(allparams)

    my_chunk = get_chunk(dicts, slurm_task_id, slurm_ntasks)
    ntasks_in_chunk = length(my_chunk)

    if haskey(ENV, "SLURM_JOB_ID") # Change log output if in SLURM environment
        for (i,d) in enumerate(my_chunk)
            println("[$(Dates.now())] Running simulation $i/$ntasks_in_chunk, Nqubits=$(d["Nqubits"]), Npoints=$(d["Npoints"]), theta1=$(d["theta1"]) ...")
            produce_or_load(makesim, d, datadir("MiladCircuitDistances"), loadfile=false)
            #wsave(datadir("MiladCircuitDistances", savename(d, "jld2")), makesim(d))
        end
    else
        @showprogress for d in my_chunk
            produce_or_load(makesim, d, datadir("MiladCircuitDistances"), loadfile=false)
            #wsave(datadir("MiladCircuitDistances", savename(d, "jld2")), makesim(d))
        end
    end
end

main()
