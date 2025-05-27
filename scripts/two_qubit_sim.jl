#!/usr/bin/env julia
#SBATCH --job-name=two_qubit_sim     # Job name
#SBATCH --mail-type=NONE             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=8                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=20         # Maximum number of tasks on each node
#SBATCH --cpus-per-task=1            # Number of processors for each task (want several because the BLAS is multithreaded, even though my Julia code is not)
#SBATCH --mem-per-cpu=2G             # Memory (i.e. RAM) per NODE
#SBATCH --constraint=intel18         # Run on the
#SBATCH --time=12:00:00               # Wall time limit (days-hrs:min:sec)
#SBATCH --output=Log/two_qubit_sim_%A.log     # Path to the standard output and error files relative to the working directory
using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments") # Necessary when using sbatch with this file directory, since sbatch feeds this file into julia using stdin, hence __FILE__ is /.

using Distributed, SlurmClusterManager
try 
    addprocs(SlurmManager(), exeflags="--project=$(DrWatson.projectdir())")
catch
    println("SLURM environment not detected. Processes not added from SlurmManager")
end


@everywhere using DrWatson, StaticArrays, QuantumToolbox
@everywhere include(srcdir("utilities.jl"))

@everywhere begin
    function hamiltonian(control_type::Union{Symbol, Val})
        H1 = tensor(eye(2), sigmax())
        H2 = tensor(sigmax(), eye(2))
        H3 = tensor(projection(2,1,1), projection(2,1,1)) 
        if getVal(control_type) == :sines
            c1(p,t) = p[1]*sin(p[2]*t)
            c2(p,t) = p[3]*sin(p[4]*t)
            c3(p,t) = p[5]*sin(p[6]*t)
        else
            throw(ArgumentError("Invalid control type $(getVal(control_type))")) 
        end

        H_tuple = ((H1, c1), (H2, c2), (H3, c3))
        H_t = QobjEvo(H_tuple)
        return H_t
    end

    """
    Function to recursively unwrap Val() values
    """
    function unwrap_vals(tags)
        return Dict(k => getVal(v) for (k, v) in tags)
    end

    function my_sim(control_vector, init_basis_index)
        @assert length(control_vector) == 6
        H_t = hamiltonian(Val(:sines))
        tlist = LinRange(0,101,101)
        initial_state = basis(4, init_basis_index, dims=(2,2))
        sol = sesolve(H_t, initial_state, tlist, params=control_vector, progress_bar=Val(false))
        return get_data(sol.states[end])
    end

    function makesim(d::Dict)
        @unpack w1, w2, w3, a1, a2, a3, initialState = d
        #control_vector = SVector(w1, w2, w3, a1, a2, a3)
        control_vector = [w1, w2, w3, a1, a2, a3]
        final_state = my_sim(control_vector, initialState)
        fulld = copy(d)
        fulld["final_state"] = final_state
        return fulld
    end
end #@everywhere

function main()
    allparams = Dict(
        "w1" => collect(0:1:1), # Frequency of control 1
        "w2" => collect(0:1:1), # Frequency of control 2
        "w3" => collect(0:1:1), # Frequency of control 3
        "a1" => collect(0:1:0), # Amplitude of control 1
        "a2" => collect(0:1:0), # Amplitude of control 2
        "a3" => collect(0:1:0), # Amplitude of control 3
        "controlType" => Val(:sines),
        "initialState" => collect(0:3)
    )
    
    dicts = dict_list(allparams)

    @sync @distributed for d_chunk in chunked_partition(dicts, nworkers())
        for d in d_chunk
            d_unwrapped  = unwrap_vals(d)

            #output_name = datadir("two_qubit_simulations", savename("twoQubit", d_unwrapped))
            #produce_or_load(makesim, d, filename=output_name, tag=true, loadfile=false)

            # Alternative call, will rerun existing tests (but now overwrite)
            f = makesim(d)
            output_name = datadir("two_qubit_simulations", savename("twoQubit", d_unwrapped, "jld2"))
            @tagsave(output_name, f, safe=true)
        end
    end
end

main()
