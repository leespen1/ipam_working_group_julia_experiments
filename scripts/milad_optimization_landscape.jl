using DrWatson
quickactivate(pwd(), "JuliaPulseExperiments")

using QuantumToolbox, GLMakie, Random

includet(srcdir("wasserstein_distance.jl"))
includet(srcdir("milad_circuit.jl"))

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

function collect_data(Nqubits::Integer, i1::Integer, i2::Integer;
        i1_lims=(0,2pi),  i2_lims=(0,2pi), N_points=101, obj_type=:infidelity)
    # Q | Better to set non-changing angles as optimal or nonoptimal?
    # A | It's better to do non-optimal. If almost all angles are optimal, we don't see barren plateau
    #angles = optimal_angles(Nqubits)
    angles = rand(MersenneTwister(0), Nqubits) .* 2pi # Random in range [0,2pi]
    #angles .= pi
    angles[1] = pi # Maybe just θ₁ optimal is good

    ghz_dm = ghz_operator(Nqubits)

    i1_linrange = LinRange(i1_lims..., N_points) 
    i2_linrange = LinRange(i2_lims..., N_points) 
    obj_matrix = Matrix{Float64}(undef, N_points, N_points)
    for (j1, theta_i1) in enumerate(i1_linrange)
        println("Doing row #", j1)
        for (j2, theta_i2) in enumerate(i2_linrange)
            angles[i1] = theta_i1
            angles[i2] = theta_i2

            if obj_type == :infidelity
                obj_matrix[j1,j2] = infidelity_obj(angles)
            elseif obj_type == :w1_primal
                final_dm = run_milad_circuit_operator(angles)
                obj_matrix[j1,j2] = W1_primal(final_dm, ghz_dm)
            elseif obj_type == :w1_dual
                final_dm = run_milad_circuit_operator(angles)
                obj_matrix[j1,j2] = W1_dual(final_dm, ghz_dm)
            else
                error("Invalid objective type '$obj_type'")
            end
        end
    end
    return obj_matrix 
end


function main(obj_type=:infidelity)
    N_points = 26
    N_cols = 2
    hms = []
    fig = Figure()

    for (i, Nqubits) in enumerate(3:6)
        data = collect_data(Nqubits, 2, 3, N_points=N_points,
                                      obj_type=obj_type)
        ax_tmp, hm_tmp = GLMakie.heatmap(
            fig[1,end+2], (0,2pi), (0,2pi), data;
            colorscale=log10,
            colorrange=(0.5, 1),
            axis=(; title="$Nqubits Qubits"),

        )
        @show i, minimum(data)
        push!(hms, hm_tmp)
    end
    for i in 1:length(hms)
        Colorbar(fig[1,2i], hms[i])
    end
    return fig
end

#fig = main()

# TODO Want to show barren landscape. Should I set θ₁ optimally? What is optimal?
