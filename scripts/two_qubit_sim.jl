using DrWatson
@quickactivate "JuliaPulseExperiments"

using Distributed, SlurmClusterManager
try 
    addprocs(SlurmManager())
catch
    println("SLURM environment not detected. Processes not added from SlurmManager")
end

@everywhere begin 
    using DrWatson
    @quickactivate "JuliaPulseExperiments"
    using StaticArrays, QuantumToolbox

    include(srcdir("utilities.jl"))

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
        return hcat(get_data.(sol.states)...)
    end

    function makesim(d::Dict)
        @unpack w1, w2, w3, a1, a2, a3, initialState = d
        #control_vector = SVector(w1, w2, w3, a1, a2, a3)
        control_vector = [w1, w2, w3, a1, a2, a3]
        state_history = my_sim(control_vector, initialState)
        fulld = copy(d)
        fulld["state_history"] = state_history
        return fulld
    end
end # @everywhere


function main()
    allparams = Dict(
        "w1" => collect(0:0.1:3), # Frequency of control 1
        "w2" => [0], # Frequency of control 2
        "w3" => [0], # Frequency of control 3
        "a1" => [0,1], # Amplitude of control 1
        "a2" => [0], # Amplitude of control 2
        "a3" => [0], # Amplitude of control 3
        "controlType" => Val(:sine),
        "initialState" => collect(0:3)
    )
    
    dicts = dict_list(allparams)

    @sync @distributed for d in dicts
        # First way, always runs test, doesn't overwrite
        #f = makesim(d)
        #d_unwrapped  = unwrap_vals(d)
        #output_name = datadir("two_qubit_simulations", savename("twoQubit", d_unwrapped, "jld2"))
        #@tagsave(output_name, f, safe=true)
        
        # Second way, checks if sim has already been run
        d_unwrapped  = unwrap_vals(d)
        output_name = datadir("two_qubit_simulations", savename("twoQubit", d_unwrapped))
        produce_or_load(makesim, d, filename=output_name, tag=true, loadfile=false)
    end
end

main()
