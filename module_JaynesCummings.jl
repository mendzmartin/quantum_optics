#!/usr/bin/julia

"""
    RUN COMMANDS
    Via REPL => julia
                include("module_JaynesCummings.jl")
    Via Bash => chmod +x module_JaynesCummings.jl
                ./JaynesCummings_model.jl
"""

function create_2x2_hamiltonian(H11::Float64,H12::Float64)
    # Definimos el hamiltoniano a diagonalizar
    H = Matrix{Float64}(undef, 2, 2) # creamos matriz
    H[:,:] = zeros(2,2)              # inicializamos matriz
    H[1,1] = α; H[1,2] = β;
    H[2,1] = H[1,2] ; H[2,2] = H[1,1];
    return H
end

function eigenvalues_problem(A_matrix,A_eigenvals,A_eigenvectors)
    A_eigenvals = copy(eigvals(A_matrix));
    A_eigenvectors = copy(eigvecs(A_matrix));
    return A_eigenvals,A_eigenvectors
end

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculamos cuál es la descomposición del estado inicial en la base de autoestados del hamiltoniano B:={|ϕj⟩}
# |ψ0⟩=∑aj|ϕj⟩ := estado inicial (vector columna)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function lineal_superposition(H_eigenvectors,ϕ)
    # vector columna de coeficientes de descomposición coeff_vector := [a1,a2,a3]
    coeff_vector = zero(Array{ComplexF64}(undef,length(H_eigenvectors[1,:])))
    coeff_vector = inv(H_eigenvectors)*ϕ
    return coeff_vector
end

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Realizamos la evolución temporal el hamiltoniano
# |ψ(t)⟩=∑exp[-i⋅ϵj⋅(t-t0)]|ψ0⟩
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function evolution(
    ψ0              :: Array{Float64},   # vector de estado inicial
    H_eigenvals     :: Array{Float64},   # vector de autovalores del hamiltoniano del sistema
    H_eigenvectors  :: Matrix{Float64},  # matriz de autovectores del hamiltoniano del sistema
    time_vector     :: Array{Float64})   # vector de tiempos para realizar evolución temporal

    # vector columna de coeficientes de descomposición coeff_vector := [a1,a2,a3]
    coeff_vect = lineal_superposition(H_eigenvectors,ψ0)

    dim_time = length(time_vector)
    dim_eigvecs = length(H_eigenvectors[1,:])
    dim_eigvals = length(H_eigenvals)

    # vector de estado evolucionado
    ψt = zeros(ComplexF64, dim_eigvecs, dim_time);

    for i in 1:dim_time
        for j in 1:dim_eigvecs
            for k in 1:dim_eigvals
                ψt[k,i]=exp(-im*H_eigenvals[k]*(time_vector[i]-time_vector[1]))*coeff_vect[k]*H_eigenvectors[j,k]
            end
        end
    end
    return ψt
end

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# funcion evolución de un estado coherente
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function evolution_coherent_state(  ϕ0::Array{Float64},
                                    αcoeff::Float64,
                                    H_eigenvals::Array{Float64},
                                    H_eigenvectors::Matrix{Float64},
                                    time_vector::Array{Float64})

    # vector columna de coeficientes de descomposición coeff_vector := [a1,a2,a3]
    coeff_vect = lineal_superposition(H_eigenvectors,ϕ0)
    coeff_vect[:]=αcoeff*coeff_vect[:]

    dim_time = length(time_vector)
    dim_eigvecs = length(H_eigenvectors[1,:])
    dim_eigvals = length(H_eigenvals)

    ψt = zeros(ComplexF64, dim_eigvecs, dim_time);

    for i in 1:dim_time
        for j in 1:dim_eigvecs
            for k in 1:dim_eigvals
                ψt[k,i]=exp(-im*H_eigenvals[k]*(time_vector[i]-time_vector[1]))*coeff_vect[k]*H_eigenvectors[j,k]
            end
        end
    end
    return ψt
end

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# función para medir el periodo y la frecuencia de oscilación
#   de una dada función oscilatoria.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function period_oscillator(time_vector,osc_function,ϵ)
    T=time_vector[1]
    counter=1;index=1;
    for i in 2:length(time_vector)
        if (abs(osc_function[i]-osc_function[1]) < ϵ)
            T=T+abs(time_vector[index]-time_vector[i]);counter=counter+1
            index=i
        end
    end
    ω=(2*π)/(T/Float64(counter))
    # retornamos tupla de frecuencia y período de oscilación
    return ω,T
end

function population_probability(time_vector,H1_eigenvectors,ψ_t,ϕ,params)
    # calculamos valores útiles
    dim_time = length(time_vector)
    dim_eigvecs = length(H_eigenvectors[1,:])

    # definimos vector de probabilidades (población del estado excitado)
    pϕ=Array{ComplexF64}(undef, dim_time); # vector complejo
    # vector de coeficientes aj tq |e⟩=∑aj|ϕj⟩, con |ϕj⟩ := base de autoestados de H
    coeff_vect_ϕ = lineal_superposition(H_eigenvectors,ϕ);
    coeff_vect_ϕ[:]=params*coeff_vect_ϕ[:];

    for i in 1:dim_time
        ψ_t[:,i]=ψ_t[:,i]/norm(ψ_t[:,i]);     # normalizamos la vector de estado

        pϕ[i]=adjoint(coeff_vect_ϕ)*ψ_t[:,i]; # computamos ⟨e|ψt⟩=[∑(aj)*⟨ϕj|][∑bk|ϕk⟩]
        pϕ[i]=abs(pϕ[i])*abs(pϕ[i])           # computamos |⟨e|ψt⟩|²
    end
    return real(pϕ)
end

