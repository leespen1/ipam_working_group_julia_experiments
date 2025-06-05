using QuantumToolbox
include("utilities.jl")

function promote_subsys(A, i, Nqubits)
    @assert 1 <= getVal(i) <= getVal(Nqubits)
    subsys_ops = ntuple(j -> (getVal(j) == getVal(i)) ? A : qeye(2), Nqubits)
    return reduce(tensor, subsys_ops)
end

"""
Get the control matrices used in the Ising Hamiltonian.
"""
function X_hamiltonians(Nqubits)
    return ntuple(i -> promote_subsys(sigmax(), i, Nqubits), Nqubits)
end

function Z_hamiltonians(Nqubits)
    return ntuple(i -> promote_subsys(sigmaz(), i, Nqubits), Nqubits)
end

function ZZ_hamiltonians_spinchain(Nqubits)
    Zs = Z_hamiltonians(Nqubits)
    ZZs = ntuple(i -> Zs[i]*Zs[i+1], getVal(Nqubits)-1)
    return ZZs 
end

"""
Takes a function f whose input is array-like (but must support tuples!), and
ouput function which uses the 'control_index'th slice of the original
array-like input.

input_length should be a Val type for efficiney

Length of p given to the output function must not be smaller than
input_length*control_index.
"""
function offset_f(f::Function, control_index::Integer, input_length::Val{N}) where N
    @assert control_index >= 1
    offset = (control_index-1)*getVal(input_length)
    return (p,t) -> f(ntuple(i -> p[offset+i], input_length), t)
end

function sincoscontrol(p, t)
    sinval, cosval = sincos(t)
    return p[1]*sinval + p[2]*cosval
end

function sincontrol(p, t)
    return p[1]*sin(t)
end

"""
Ising Spinchain with coupling strength J.

Controls are linear combinations of sin and cos. All controls are independent.
Ordering is: X controls for each qubit, Z controls for each qubits, ZZ controls

J should be roughly one-tenth the strength of the Xs and Zs.
"""
function generic_ising_spinchain_independent(Nqubits::Union{Integer,Val},
        control::Function, input_length::Val, J::Real=0.1)

    Xs = X_hamiltonians(Nqubits)
    Zs = Z_hamiltonians(Nqubits)
    ZZs = J .* ZZ_hamiltonians_spinchain(Nqubits)

    X_controls = ntuple(i -> offset_f(control, i, input_length), Nqubits)
    Z_controls = ntuple(i -> offset_f(control, i+Nqubits, input_length), Nqubits)

    dynamic_Xs = [QobjEvo(H, f) for (H, f) in zip(Xs, X_controls)]
    dynamic_Zs = [QobjEvo(H, f) for (H, f) in zip(Zs, Z_controls)]

    controlled_hamiltonian = sum(dynamic_Xs) + sum(dynamic_Zs)
    if !isempty(ZZs)
        controlled_hamiltonian += sum(ZZs)
    end

    return controlled_hamiltonian
end

"""
Ising Spinchain with coupling strength J.

Controls are linear combinations of sin and cos. All controls are independent.
Ordering is: X controls for each qubit, Z controls for each qubits, ZZ controls

J should be roughly one-tenth the strength of the Xs and Zs.
"""
function generic_ising_spinchain_perm_invariant(Nqubits::Union{Integer,Val}, control::Function,
        input_length::Val, J::Real=0.1)

    Xs = X_hamiltonians(Nqubits)
    Zs = Z_hamiltonians(Nqubits)
    ZZs = J .* ZZ_hamiltonians_spinchain(Nqubits)

    X_controls = ntuple(i -> offset_f(control, 1, input_length), Nqubits)
    Z_controls = ntuple(i -> offset_f(control, 2, input_length), Nqubits)

    dynamic_Xs = [QobjEvo(H, f) for (H, f) in zip(Xs, X_controls)]
    dynamic_Zs = [QobjEvo(H, f) for (H, f) in zip(Zs, Z_controls)]

    controlled_hamiltonian = sum(dynamic_Xs) + sum(dynamic_Zs)
    if !isempty(ZZs)
        controlled_hamiltonian += sum(ZZs)
    end

    return controlled_hamiltonian
end
