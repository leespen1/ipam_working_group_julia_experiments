using Yao, BitBasis, LinearAlgebra, Zygote, Optim, Plots

function pretty_print_state(reg)
    amplitudes = state(reg)
    for (i, amp) in enumerate(amplitudes)
        if abs(amp) > 1e-10  # filter out negligible amplitudes
            bits = bitstring(i - 1)[end - nqubits + 1:end]  # index starts from 0
            println("$bits => $amp")
        end
    end
end

"""
    get_bitstrings(reg::AbstractRegister; show_amplitudes=false, atol=1e-10)

Return bitstrings with non-zero amplitudes in the quantum register `reg`.
If `show_amplitudes` is true, return a vector of `(bitstring, amplitude)` pairs.
"""
function get_bitstrings(reg::AbstractRegister; show_amplitudes=false, atol=1e-10)
    statevec = state(reg)  # Extract the state vector
    n = nqubits(reg)              # Number of qubits
    result = String[]

    for (i, amp) in enumerate(statevec)
        if abs(amp) > atol
            bits = bitstring(i - 1)[end-n+1:end]  # Convert index to bitstring
            show_amplitudes ? push!(result, "$(abs2(amp)) => $bits") : push!(result, bits)
        end
    end
    return result
end


function milad_circuit(angles)
	n_qubits = length(angles)
    # Maybe I should be using dispatch here?
	circuit = chain(
		n_qubits,
		put(1 => Rx(angles[1])),
		[control(i, i+1 => Rx(angles[i+1])) for i in 1:n_qubits-1]...,
	)
	return circuit
end

function run_milad_circuit(angles)
	n_qubits = length(angles)
    circuit = milad_circuit(angles)
    return zero_state(n_qubits) |> circuit
end


function infidelity(psi1, psi2)
    return 1-abs2(dot(psi1,psi2))
end

"""
Measure infidelity between state psi and the GHZ-like state

|000…⟩ + rel_phase*|111…⟩
"""
function ghz_infidelity(psi; rel_phase=1)
    #@assert size(psi,2) == 1
    return 1 - 0.5*abs2(first(psi) + last(psi)*conj(rel_phase))
end


function run_milad_circuit_rand(N_qubits, N_samples)
    N_amplitudes = 2^N_qubits
    output_states = Matrix{ComplexF64}(undef, N_amplitudes, N_samples)
    angles = Vector{Float64}(undef, N_qubits)
    for i in 1:N_samples
        rand!(angles)
        output_register = run_milad_circuit(angles)
        output_states[:,i] .= state(output_register)
    end
    return output_states
end

function pca_test(N_qubits, N_samples)
    #for n in N_qubits
    #    run_milad_circuit_rand
    #end
end

function infidelity_obj(angles)
    return angles |> run_milad_circuit |> state |> ghz_infidelity
end

function infidelity_obj_hardcoded(angles)
    @assert length(angles) % 4 == 0
    return 1 - 0.5*(cos(angles[1]/2)  + prod(x -> sin(x/2), angles[2:end]))^2
end

function objgrad(angles)
    return first(Zygote.gradient(obj, angles))
end

"""
Gradient
"""
function objgrad!(G, angles)
    G .= grad(angles)
end

function objgrad_fd(angles, h=1e-5)
    grad = zeros(length(angles))
    angles_L = zeros(length) 
    angles_R = zeros(length) 
    for i in eachindex(angles)
        angles_L .= angles
        angles_R .= angles
        angles_L[i] -= h
        angles_R[i] += h
        grad[i] = (obj(angles_R) - obj(angles_L))/(2*h)
    end
    return grad  
end
    

function ideal_angles(n_qubits)
    @assert n_qubits % 4 == 0
    angles = ones(n_qubits) .* pi
    angles[1] /= 2
    return angles
end

#opt = optimize(obj, init_angles)
#opt = optimize(obj, grad!, init_angles)
