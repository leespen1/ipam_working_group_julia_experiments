using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using QuantumToolbox, StaticArrays, Zygote, SciMLSensitivity, Dates, Optim, Random
include(srcdir("ising_model_hamiltonian.jl"))


function ghz_infidelity(state::Qobj)
    1 - 0.5*abs2(first(state.data) + last(state.data))
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

function makesim(d::Dict)
    @unpack Nqubits, Namplitudes, maxAmplitude, J, T, seed, optimizer,
            timeLimit, iterations = d

    if optimizer == "neldermead"
        optimizer_obj = NelderMead()
    elseif optimizer == "gradientdescent"
        optimizer_obj = GradientDescent()
    elseif optimizer == "simulatedannealing"
        optimizer_obj = SimulatedAnnealing()
    elseif optimizer == "lbfgs"
        optimizer_obj = LBFGS()
    else
        error("Invalid optimizer $optimizer")
    end

    tlist = SVector(0,T)
    control = makegrapecontrol(T, Namplitudes)
    Ht = generic_ising_spinchain_independent(Nqubits, control, Namplitudes)
    ground_state = basis(2^Nqubits, 0, dims=ntuple(_ -> 2, Nqubits))
    Nparams = 2*Nqubits*Namplitudes
    controlVectorInitial = (0.5 .- rand(MersenneTwister(seed), Nparams)) .* (2*maxAmplitude)

    f(x) = begin 
        sol = sesolve(
            Ht, ground_state, tlist, params=x, progress_bar=Val(false),
            sensealg=BacksolveAdjoint(autojacvec = EnzymeVJP())
        )
        sol.states |> last |> ghz_infidelity
    end
    g(x) = Zygote.gradient(f, x)[1]

    opt_ret = optimize(
        f, g, controlVectorInitial, optimizer_obj,
        Optim.Options(time_limit=3600*timeLimit, iterations=iterations),
        inplace=false,
    )

    controlVectorOptimal = opt_ret.minimizer
    infidelity = opt_ret.minimum

    fulld = Dict{String, Any}(copy(d))
    fulld["controlVectorInitial"] = controlVectorInitial
    fulld["controlVectorOptimal"] = controlVectorOptimal
    fulld["infidelity"] = infidelity

    return fulld
end

function main()
    max_Nqubits = DrWatson.readenv("MAX_NQUBITS", 2)
    min_Nqubits = DrWatson.readenv("MIN_NQUBITS", 1)
    maxAmplitude = DrWatson.readenv("MAX_AMPLITUDE", 1.0)
    Namplitudes = DrWatson.readenv("N_AMPLITUDES", 25)
    J = DrWatson.readenv("J", 0.1)
    T = DrWatson.readenv("T", 100.0)
    max_seed = DrWatson.readenv("MAX_SEED", 0)
    min_seed = DrWatson.readenv("MIN_SEED", 0)
    timeLimit = DrWatson.readenv("TIMELIMIT", 8.0) # time limit (hours)
    iterations = DrWatson.readenv("ITERATIONS", 10_000)
    optimizer = get(ENV, "OPTIMIZER", "all") |> lowercase
    if optimizer == "all"
        optimizer = ["simulatedannealing", "neldermead", "gradientdescent", "lbfgs"]
    end

    # For this to work, all job arrays should start at 0 and use stepsize 1
    slurm_task_id = DrWatson.readenv("SLURM_ARRAY_TASK_ID", 0)
    slurm_ntasks = DrWatson.readenv("SLURM_ARRAY_TASK_COUNT", 1)

    allparams = Dict{String, Any}(
        "Nqubits" => collect(min_Nqubits:max_Nqubits),
        "maxAmplitude" => maxAmplitude,
        "Namplitudes" => Namplitudes,
        "J" => J,
        "T" => T,
        "seed" => collect(min_seed:max_seed),
        "optimizer" => optimizer,
        "timeLimit" => timeLimit,
        "iterations" => iterations,
    )

    dicts = dict_list(allparams)

    my_chunk = get_chunk(dicts, slurm_task_id, slurm_ntasks)
    ntasks_in_chunk = length(my_chunk)

    for (i,d) in enumerate(my_chunk)
        println("[$(Dates.now())] Running simulation $i/$ntasks_in_chunk, ", savename(d))
        produce_or_load(makesim, d, datadir("GrapeOptimization"), loadfile=false)
        #wsave(datadir("PulseGhzDistances", savename(d, "jld2")), makesim(d))
    end
end

main()

#=
Nqubits = 2
Namplitudes = 10
T = 100.0
tlist = SVector(0,T)
control = makegrapecontrol(T, Namplitudes)
Ht = generic_ising_spinchain_independent(Nqubits, control, Namplitudes)
ground_state = basis(2^Nqubits, 0, dims=ntuple(_ -> 2, Nqubits))
Nparams = 2*Nqubits*Namplitudes
x0 = rand(Nparams)

# Need inplace=false for automatic differentiation. Although it seems Enzyme
# might be compatible with in-place, based on the QuantumToolbox paper example.
f(x) = sesolve(Ht, ground_state, tlist, params=x, progress_bar=Val(false),
               #inplace=Val(false), 
               sensealg=BacksolveAdjoint(autojacvec = EnzymeVJP())
       ).states |> last |> ghz_infidelity
g(x) = Zygote.gradient(f, x)[1]
grad = Zygote.gradient(f, x0)[1]

#opt_ret = optimize(f, g, x0, LBFGS(), inplace=false)
opt_ret = optimize(f, g, x0, GradientDescent(), inplace=false)
#opt_ret = optimize(f, x0, NelderMead())
#opt_ret = optimize(f, x0, LBFGS(), autodiff=:forward) # Only allows forward and findiff

## Box constrained example
#lower = [1.25, -2.1]
#upper = [Inf, Inf]
#initial_x = [2.0, 2.0]
#inner_optimizer = GradientDescent()
#results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
=#
