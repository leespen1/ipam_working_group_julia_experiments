using QuantumControl
using GRAPE
using QuantumControl.Controls: get_controls, substitute
using QuantumPropagators: Cheby

# Shaped Amplitude "allows to have a control amplitude Ω(t) = S(t)ϵ(t) where
# S(t) is a fixed shape and ϵ(t) is the pulse directly tuned by the
# optimization ... Note that passing tlist to ShapedAmplitude discretizes both
# the control and the shape function to the midpoints of the tlist array.
#
# So I think if I just do S(t) = 1, and pass tlist, I can do GRAPE style PWC
# controls.
#
#   Product of a fixed shape and a control.
#
# ampl = ShapedAmplitude(control; shape=shape)
#
# produces an amplitude a(t) = S(t) ϵ(t), where S(t) corresponds to shape and ϵ(t)
# corresponds to control. Both control and shape should be either a vector of values
# defined on the midpoints of a time grid or a callable control(t), respectively
# shape(t). In the latter case, ampl will also be callable.
#
# Given their vector of values suggestion, I imagine if I just skip shaped
# amplitudes that I will do a GRAPE-style optimization

X = ComplexF64[0 1; 1  0]
Z = ComplexF64[1 0; 0 -1]
T = 50.0

tlist = collect(LinRange(0, T, 11))

initial_state = ComplexF64[1 ; 0]
target_state = ComplexF64[0 ; 1]
# Drift Z, controlled X
H = hamiltonian(Z, (X, ones(length(tlist))))

infidelity(ψ, trajectories) = sum(1 - abs2(dot(ψi, trajectoryi)) for (ψi, trajectoryi) in zip(ψ, trajectories))

trajectory = Trajectory(
    initial_state = initial_state,
    generator = H,
    target_state = target_state,
)

ret_prop = propagate_trajectory(
    trajectory,
    tlist,
    method=Cheby
)


problem = ControlProblem(
    [trajectory],
    tlist,
    J_T = infidelity, # Think this one is a required keyword arg. Might be default to just use target state
    #method=Cheby
)

ret = optimize(problem, method=GRAPE, J_T=infidelity)


#=
  In general,

  H = hamiltonian(terms...; check=true)

  constructs a Hamiltonian based on the given terms. Each term must be an operator or a
  tuple (op, ampl) of an operator and a control amplitude. Single operators are
  considered "drift" terms.

  In most cases, each control amplitude will simply be a control function or vector of
  pulse values. In general, ampl can be an arbitrary object that depends on one or more
  controls, which must be obtainable via get_controls(ampl). See
  QuantumPropagators.Interfaces.check_amplitude for the required interface.

# Will pass this into GRAPE
ControlProblem(
   trajectories,
   tlist;
   kwargs...
)

# Initial state, dynamics, and target state
Trajectory(
    initial_state,
    generator;
    target_state=nothing,
    weight=1.0,
    kwargs...
)

result = optimize(
    problem;
    method,  # mandatory keyword argument
    check=true,
    callback=nothing,
    print_iters=true,
    kwargs...
)

using GRAPE
result = optimize(problem; method=GRAPE, kwargs...)
=#
