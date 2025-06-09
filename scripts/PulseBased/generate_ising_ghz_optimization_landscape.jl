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
    @unpack Nqubits, i1, i2, theta1, Npoints, maxAmplitude, controlFuncType,
            controlPermType, J, T, seed, optimizer, grad = d

    infidelity_vec = Vector{Float64}(undef, Npoints)
    W1_vec = Vector{Float64}(undef, Npoints)
    terminationStatus_vec = Vector{Int}(undef, Npoints)
    final_states_mat = Matrix{ComplexF64}(undef, 2^Nqubits, Npoints)

    if optimizer == "mosek"
        optimizer_obj = MosekTools.Optimizer
    elseif optimizer == "scs"
        optimizer_obj = SCS.Optimizer
    elseif optimizer == "none"
        optimizer_obj = nothing
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
        Ht = generic_ising_spinchain_independent(
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

    grad_mat_inf = Matrix{Float64}(undef, Nparams, Npoints)
    grad_mat_W1 = Matrix{Float64}(undef, Nparams, Npoints)
    theta2range = LinRange(-maxAmplitude, maxAmplitude, Npoints)
    ghz_dm = ghz_operator(Nqubits)
    ground_state = basis(2^Nqubits, 0, dims=ntuple(_ -> 2, Nqubits))
    tlist = SVector(0.0,T)
    dims = ntuple(_ -> 2, Nqubits)

    for (k, theta2) in enumerate(theta2range)
        controlVector[i2] = theta2

        sol = sesolve(Ht, ground_state, tlist, params=controlVector,
                      progress_bar=Val(false))
        final_state = last(sol.states)
        infidelity_vec[k] = ghz_infidelity(final_state)
        final_states_mat[:,k] .= final_state.data

        if !isnothing(optimizer_obj)
            final_dm = ket2dm(final_state)
            opt_model = W1_dual(final_dm, ghz_dm, optimizer_obj, silent=silent)

            W1_vec[k] = JuMP.objective_value(opt_model)
            terminationStatus_vec[k] = Int(JuMP.termination_status(opt_model))
        end

        if grad
            h = 1e-5
            controlvec_fd = copy(controlVector)
            for findiff_i in 1:Nqubits
                controlvec_fd .= controlVector
                controlvec_fd[findiff_i] += h
                sol_fd = sesolve(Ht, ground_state, tlist, params=controlVector,
                              progress_bar=Val(false))
                final_state_fd = last(sol_fd.states)
                infidelity_fd = ghz_infidelity(final_state_fd)
                grad_mat_inf[findiff_i,k] = (infidelity_fd - infidelity_vec[k])/h

                if !isnothing(optimizer_obj)
                    final_dm_fd = ket2dm(final_state_fd)
                    opt_model_fd = W1_dual(final_dm_fd, ghz_dm, optimizer_obj, silent=silent)
                    W1_fd = JuMP.objective_value(opt_model_fd)
                    grad_mat_W1[findiff_i,k] = (W1_fd - W1_vec[k])/h
                end
            end
        end

        GC.gc() # Being safe about running out of memory
    end

    fulld = Dict{String, Any}(copy(d))
    fulld["controlVector"] = controlVector
    fulld["theta2range"] = theta2range
    fulld["infidelity_vec"] = infidelity_vec
    fulld["finalSates_mat"] = final_states_mat
    if !isnothing(optimizer_obj)
        fulld["W1_vec"] = W1_vec
        fulld["terminationStatus_vec"] = terminationStatus_vec
    end

    if grad
        fulld["grad_mat_inf"] = grad_mat_inf
        if !isnothing(optimizer_obj)
            fulld["grad_mat_W1"] = grad_mat_W1
        end
    end

    return fulld
end


function main()
    Npoints = DrWatson.readenv("NPOINTS", 11)
    max_Nqubits = DrWatson.readenv("MAX_NQUBITS", 4)
    min_Nqubits = DrWatson.readenv("MIN_NQUBITS", 1)
    i1 = DrWatson.readenv("I1", 1)
    i2 = DrWatson.readenv("I2", 2)
    grad = DrWatson.readenv("GRAD", false)
    maxAmplitude = DrWatson.readenv("MAX_AMPLITUDE", 1.0)
    J = DrWatson.readenv("J", 0.1)
    T = DrWatson.readenv("T", 100.0)
    seed = DrWatson.readenv("SEED", 0)
    controlFuncType = get(ENV, "CONTROL_FUNC_TYPE", "sin") |> lowercase
    controlPermType = get(ENV, "CONTROL_PERM_TYPE", "invariant") |> lowercase
    optimizer = get(ENV, "OPTIMIZER", "scs") |> lowercase

    # For this to work, all job arrays should start at 0 and use stepsize 1
    slurm_task_id = DrWatson.readenv("SLURM_ARRAY_TASK_ID", 0)
    slurm_ntasks = DrWatson.readenv("SLURM_ARRAY_TASK_COUNT", 1)

    println("Running test using following environment variables:")
    @show Npoints, max_Nqubits, slurm_task_id, slurm_ntasks

    allparams = Dict{String, Any}(
        "Npoints" => [Npoints],
        "Nqubits" => collect(1:max_Nqubits),
        "i1" => i1,
        "i2" => i2,
        "grad" => grad,
        "maxAmplitude" => maxAmplitude,
        "J" => J,
        "T" => T,
        "seed" => seed,
        "controlFuncType" => controlFuncType,
        "controlPermType" => controlPermType,
        "optimizer" => optimizer,
        "theta1" => collect(LinRange(-maxAmplitude, maxAmplitude, Npoints)),
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
