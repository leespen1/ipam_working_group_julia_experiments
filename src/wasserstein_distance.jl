using LinearAlgebra, QuantumToolbox, MosekTools, JuMP, SparseArrays
using SCS # First order alternative to MosekTools, doesn't require a license. May be better for large systems.
#using Convex

"""
Compute the W1 distance (Quantum Earth Mover's Distance) between density
matrices sigma and rho using the primal formulation and solving as a PSD
program.
"""
function W1_primal(sigma::QuantumObject, rho::QuantumObject; silent=true)
    @assert size(sigma) == size(rho)
    N = size(sigma, 1)
    Nqubits = length(sigma.dims)
    dims = sigma.dims

    sigma_minus_rho = (sigma - rho).data

    model = Model(MosekTools.Optimizer)
    if silent
        set_silent(model)
    end
    var_type = Hermitian{GenericAffExpr{ComplexF64, VariableRef}, Matrix{GenericAffExpr{ComplexF64, VariableRef}}}
    sigmas = Vector{var_type}(undef, Nqubits)
    rhos = Vector{var_type}(undef, Nqubits)

    for i in 1:Nqubits
        # Set up sigma/rho Hermitian SemiDefinite matrix variables
        sigma_tmp = @variable(model, [1:N, 1:N] in HermitianPSDCone())
        rho_tmp = @variable(model, [1:N, 1:N] in HermitianPSDCone())
        sigmas[i] = sigma_tmp
        rhos[i] = rho_tmp

        # Assign partial trace constraints
        pt_sigma_tmp = partialtrace(sigma_tmp, i, dims)
        pt_rho_tmp = partialtrace(rho_tmp, i, dims)
        @constraint(model, pt_sigma_tmp == pt_rho_tmp)
    end

    sigmasum_minus_rhosum = sum(sigmas) - sum(rhos)
    @constraint(model, sigma_minus_rho == sigmasum_minus_rhosum)

    # tr(sigmas[i]) should be real, since each sigmas[i] is Hermitian, but still
    # need `real` here so JuMP recognizes that the objective function is real.
    @objective(model, Min, sum(real(tr(sigmas[i])) for i in 1:Nqubits))
    optimize!(model)
    return JuMP.objective_value(model)
end


"""
Compute the W1 distance (Quantum Earth Mover's Distance) between density
matrices sigma and rho using the dual formulation and solving as a PSD
program.
"""
function W1_dual(sigma::QuantumObject, rho::QuantumObject; silent=true)
    # To make this efficient inside a loop, I will want to reuse a single model
    @assert size(sigma) == size(rho)
    N = size(sigma, 1)
    Nqubits = length(sigma.dims)
    dims_vec = Vector(sigma.dims)
    N_div2 = div(N,2)
    sigma_minus_rho = (sigma-rho).data
    
    #model = Model(MosekTools.Optimizer)
    model = Model(SCS.Optimizer)
    if silent
        set_silent(model)
    end

    @variable(model, H[1:N,1:N], Hermitian) # Should it be PSD? I don't think so.

    for i in 1:Nqubits
        Htmp = @variable(model, [1:N_div2, 1:N_div2], Hermitian)
        K = customKron(Htmp, i)
        H_minus_K = H .- K

        @assert ishermitian(H_minus_K)
        @constraint(model, Hermitian(H_minus_K) <= 0.5I, HermitianPSDCone())
        @constraint(model, Hermitian(H_minus_K) >= -0.5I, HermitianPSDCone())
    end

    ## Objective as written in Matlab
    #@objective(model, Max, real(tr(H * sigma_minus_rho)))
    # Our equivalent objective
    # dot conjugates first arg, but that's fine since H is Hermitian
    @objective(model, Max, real(dot(H, sigma_minus_rho))) 
    optimize!(model)
    return JuMP.objective_value(model)
end


"""
Turn an operator acting on n qubits into an operator acting on n+1 qubits, with
the new qubit inserted at position i.
"""
function customKron(Hi, i)
    N = size(Hi, 1);
    n = floor(Int, log2(N));
    if i > (n+1) || i < 1
      throw("Invalid subsystem")
    end
    tensor_prod = kron(Hi, I(2))
    dims = ntuple(_ -> 2, 2*(n+1))
    # Reshape Hi into a 2⊗ … ⊗ 2 array (not sure what it means, physically)
    reshaped1 = reshape(tensor_prod, dims)
    idx = vcat(2:n-i+2, 1, n-i+3:n+1, n+3:2n-i+3, n+2, 2n-i+4:2n+2)
    # "Move the added subsystem to the i-th qubit place"
    reshaped2 = permutedims(reshaped1, idx)
    return reshape(reshaped2, 2^(n+1), 2^(n+1))
end

#=
function W1_primal_convex_version(sigma::QuantumObject, rho::QuantumObject)
    @assert size(sigma) == size(rho)
    N = size(sigma, 1)
    Nqubits = length(sigma.dims)
    dims_vec = Vector(sigma.dims)
    # Define optimization variables: semidefinite Hermitian matrices
    sigmas = [HermitianSemidefinite(N) for _ in 1:Nqubits]
    rhos = [HermitianSemidefinite(N) for _ in 1:Nqubits]

    # Objective: minimize sum of traces of sigmas
    #objective = sum(tr(sigmas[i]) for i in 1:Nqubits)
    # Mathematically, should always be real. But need `real` to enforce that for Convex.jl
    objective = sum(real(tr(sigmas[i])) for i in 1:Nqubits)

    # Constraints list
    constraints = []

    #Constant(sigma - rho)
    sigma_minus_rho = (sigma - rho).data
    # First constraint: sigma - rho == sum(sigmas) - sum(rhos)
    sigmasum = sum(sigmas)
    rhosum = sum(rhos)
    # Split this into real and imag part, which might be necessary
    push!(constraints, sigma_minus_rho == sigmasum - rhosum)

    # Second constraint: partial traces must match
    for i in 1:Nqubits
        # Take partial trace over qubit i (i.e., trace over that subsystem)
        pt_sigma = partialtrace(sigmas[i], i, dims_vec)
        pt_rho   = partialtrace(rhos[i], i, dims_vec)
        push!(constraints, pt_sigma == pt_rho)
    end

    # Set up and solve the problem
    problem = minimize(objective, constraints)
    solve!(problem, MosekTools.Optimizer; silent=true)

    return problem.optval
end
=#

# Below: code for computing partial traces, copied from Convex.jl

# Copyright (c) 2014: Madeleine Udell and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file or at https://opensource.org/license/bsd-2-clause

spidentity(T, d) = SparseArrays.sparse(one(T) * LinearAlgebra.I, d, d)

# We compute the partial trace of x by summing over
# (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I) for all j's
# in the system we want to trace out.
# This function returns the jth term in the sum, namely
# (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I).
function _term(x, j::Int, sys, dims)
    a = spidentity(Float64, 1)
    b = spidentity(Float64, 1)
    for (i_sys, dim) in enumerate(dims)
        if i_sys == sys
            # create a vector that is only 1 at its jth component
            v = spzeros(Float64, dim, 1)
            v[j] = 1
            a = kron(a, v')
            b = kron(b, v)
        else
            a = kron(a, spidentity(Float64, dim))
            b = kron(b, spidentity(Float64, dim))
        end
    end
    return a * x * b
end

"""
    partialtrace(x, sys::Int, dims)

Returns the partial trace of `x` over the `sys`th system, where `dims` is an
iterable of integers encoding the dimensions of each subsystem.
"""
function partialtrace(x, sys::Int, dims)
    if size(x, 1) != size(x, 2)
        throw(ArgumentError("Only square matrices are supported"))
    end
    if !(1 <= sys <= length(dims))
        msg = "Invalid system index, should between 1 and $(length(dims)), got $sys"
        throw(ArgumentError(msg))
    end
    if size(x, 1) != prod(dims)
        msg = "Dimension of system doesn't correspond to dimension of subsystems"
        throw(ArgumentError(msg))
    end
    return sum(j -> _term(x, j, sys, dims), 1:dims[sys])
end
