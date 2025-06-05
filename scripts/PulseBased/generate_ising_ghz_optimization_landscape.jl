using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using QuantumToolbox, Random, ProgressMeter, Dates, MosekTools, SCS, StaticArrays

include(srcdir("wasserstein_distance.jl"))
include(srcdir("ising_model_hamiltonian.jl"))


function ghz_infidelity(state::Qobj)
    1 - 0.5*abs2(first(state.data) + last(state.data))
end


function ghz_operator(Nqubits)
    return QuantumToolbox.ghz_state(Nqubits) |> ket2dm
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


"""
Iterate over theta2 for fixed theta1. Get Infidelity and W1 distances as vectors.
"""
function makesim(d::Dict)
    @unpack Nqubits, i1, i2, theta1, Npoints, maxAmplitude, controlFuncType, controlPermType, J, T, seed = d

    infidelity_vec = Vector{Float64}(undef, Npoints)
    W1_vec = Vector{Float64}(undef, Npoints)

    optimizer_str = d["optimizer"]
    if lowercase(optimizer_str) == "mosek"
        optimizer = MosekTools.Optimizer
    elseif lowercase(optimizer_str) == "scs"
        optimizer = SCS.Optimizer
    elseif lowercase(optimizer_str) == "none"
        optimizer = nothing
    else
        error("Invalid optimizer string $optimizer_str")
    end

    if controlFuncType == "sin"
        control = sincontrol
        input_length = Val(1)
    elseif controlFuncType == "sincos"
        control = sincoscontrol
        input_length = Val(2)
    else
        error("Invalid control function type $controlFuncType")
    end

    if controlPermType == "invariant"
        Ht = generic_ising_spinchain_perm_invariant(
            Val(Nqubits), control, input_length, J
        )
        Nparams = 2*getVal(input_length)
    elseif controlPermType == "normal"
        Ht = generic_ising_spinchain_perm_invariant(
            Val(Nqubits), control, input_length, J
        )
        Nparams = 2*Nqubits*getVal(input_length)
    else
        error("Invalid control function type $controlPermType")
    end

    controlVector = (0.5 .- rand(MersenneTwister(seed), Nparams)) .* (2*maxAmplitude)
    controlVector[i1] = theta1

    # Optional silent optimization set by environment (not in dict because I don't want it in savename)
    silent = tryparse(Bool, get(ENV, "SILENT", "true"))

    theta2range = LinRange(-maxAmplitude, maxAmplitude, Npoints)
    ghz_dm = ghz_operator(Nqubits)
    ground_state = basis(2^Nqubits, 0, dims=ntuple(_ -> 2, Nqubits))
    tlist = SVector(0.0,T)

    for (k, theta2) in enumerate(theta2range)
        controlVector[i2] = theta2

        sol = sesolve(Ht, ground_state, tlist, params=controlVector,
                      progress_bar=Val(false))
        final_state = last(sol.states)
        infidelity_vec[k] = ghz_infidelity(final_state)

        if !isnothing(optimizer)
            dims = ntuple(_ -> 2, Nqubits)
            final_dm = ket2dm(final_state)
            W1_vec[k] = W1_primal(final_dm, ghz_dm, optimizer, silent=silent)
        end

        GC.gc() # Being safe about running out of memory
    end

    fulld = Dict{String, Any}(copy(d))
    fulld["controlVector"] = controlVector
    fulld["theta2range"] = theta2range
    fulld["infidelity_vec"] = infidelity_vec
    if !isnothing(optimizer)
        fulld["W1_vec"] = W1_vec
    end

    return fulld
end


function main()
    maxAmplitude = DrWatson.readenv("MAX_AMPLITUDE", 1.0)
    controlFuncType = get(ENV, "CONTROL_FUNC_TYPE", "sin") |> lowercase
    controlPermType = get(ENV, "CONTROL_PERM_TYPE", "invariant") |> lowercase
    Npoints = DrWatson.readenv("NPOINTS", 11)
    max_Nqubits = DrWatson.readenv("MAX_NQUBITS", 4)
    J = DrWatson.readenv("J", 0.1)
    T = DrWatson.readenv("T", 100.0)
    seed = DrWatson.readenv("SEED", 0)
    optimizer_str = get(ENV, "OPTIMIZER", "mosek") |> lowercase

    # For this to work, all job arrays should start at 0 and use stepsize 1
    slurm_task_id = DrWatson.readenv("SLURM_ARRAY_TASK_ID", 0)
    slurm_ntasks = DrWatson.readenv("SLURM_ARRAY_TASK_COUNT", 1)


    println("Running test using following environment variables:")
    @show Npoints, max_Nqubits, slurm_task_id, slurm_ntasks

    allparams = Dict{String, Any}(
        "Nqubits" => collect(1:max_Nqubits),
        "i1" => [1],
        "i2" => [2],
        "maxAmplitude" => maxAmplitude,
        "theta1" => collect(LinRange(-maxAmplitude, maxAmplitude, Npoints)),
        "Npoints" => [Npoints],
        "optimizer" => optimizer_str,
        "controlFuncType" => controlFuncType,
        "controlPermType" => controlPermType,
        "J" => J,
        "T" => T,
        "seed" => seed,
    )

    dicts = dict_list(allparams)

    my_chunk = get_chunk(dicts, slurm_task_id, slurm_ntasks)
    ntasks_in_chunk = length(my_chunk)

    if haskey(ENV, "SLURM_JOB_ID") # Change log output if in SLURM environment
        for (i,d) in enumerate(my_chunk)
            println("[$(Dates.now())] Running simulation $i/$ntasks_in_chunk, Nqubits=$(d["Nqubits"]), Npoints=$(d["Npoints"]), theta1=$(d["theta1"]) ...")
            produce_or_load(makesim, d, datadir("PulseGhzDistances"), loadfile=false)
            #wsave(datadir("PulseGhzDistances", savename(d, "jld2")), makesim(d))
        end
    else
        @showprogress for d in my_chunk
            produce_or_load(makesim, d, datadir("PulseGhzDistances"), loadfile=false)
            #wsave(datadir("PulseGhzDistances", savename(d, "jld2")), makesim(d))
        end
    end
end

main()
