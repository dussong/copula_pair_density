using SparseArrays
using Combinatorics
using StaticArrays
using TensorOperations
using LinearAlgebra
#----------------------------------------------------------------------------
# Hamiltonian structures:
# ham1d for 1D situation; ham2d for 2D situation
#----------------------------------------------------------------------------
export Hamiltonian, ham1d, ham2d

abstract type Hamiltonian end

"""Hamiltonian
L : Interval length
N : Nb of discretization points of the interval [-L,L]
AΔ : Laplace matrix (1 body) for P1/P2 finite elements
AV : External potential matrix (1 body) with P1/P2 finite elements basis
C : Overlap matrix (1 body) for P1/P2 finite elements
Bee : Two-body matrices for P1/P2 finite elements
alpha_lap : Parameter in the kinetic potential
vext : External potential function
vee : Coulomb potential function
element : finite element basis
"""
struct ham1d <: Hamiltonian
    L::Float64
    N::Int64
    AΔ::SparseMatrixCSC{Float64,Int64}
    AV::SparseMatrixCSC{Float64,Int64}
    C::SparseMatrixCSC{Float64,Int64}
    Bee
    alpha_lap::Float64
    vext::Function
    vee::Function
    element::String
end

function ham1dGen(L::Float64, N::Int, alpha_lap::Float64, vext::Function, vee::Function, nx::Int64, ny::Int64, element::String)

    if element == "P1"
        AΔ, AV, C, Bee = matrix_1_2body_1d(L, N, vext, vee, nx, ny)
    elseif element == "P2"
        AΔ, AV, C, Bee = matrix_1_2body_2ndfem(L, N, vext, vee, nx, ny)
    end

    return ham1d(L, N, AΔ, AV, C, Bee, alpha_lap, vext, vee, element)
end
ham1d(L::Float64, N::Int; alpha_lap=1.0, vext::Function=(x -> x^2), vee::Function=(x -> 1.0 ./ abs.(x)), nx=3, ny=4, element="P1") = ham1dGen(L, N, alpha_lap, vext, vee, nx, ny, element)

struct ham2d <: Hamiltonian
    L::Vector{Float64}
    N::Vector{Int64}
    AΔ::SparseMatrixCSC{Float64,Int64}
    AV::SparseMatrixCSC{Float64,Int64}
    C::SparseMatrixCSC{Float64,Int64}
    Bee::SparseMatrixCSC{Float64,Int64}
    alpha_lap::Float64
    vext::Function
    vee::Function
    element::String
end

function ham2dGen(L::Vector{Float64}, N::Vector{Int64}, alpha_lap::Float64, vext::Function, vee::Function, nx1::Int64, ny1::Int64, nx2::Int64, ny2::Int64, element::String)
    AΔ, AV, C, Bee = matrix_1_2body_2d(L[1], L[2], N[1], N[2], vext, vee, nx1, ny1, nx2, ny2)
    return ham2d(L, N, AΔ, AV, C, Bee, alpha_lap, vext, vee, element::String)
end
ham2d(L::Vector{Float64}, N::Vector{Int64}; alpha_lap=1.0, vext::Function=((x, y) -> x^2 + y^2), vee::Function=((x, y) -> 1.0 / sqrt(1e-3 + x^2 + y^2)), nx1=4, ny1=4, nx2=4, ny2=4, element="P1") = ham2dGen(L, N, alpha_lap, vext, vee, nx1, ny1, nx2, ny2, element)
ham2d(L::Float64, N::Int64; alpha_lap=1.0, vext::Function=((x, y) -> x^2 + y^2), vee::Function=((x, y) -> 1.0 / sqrt(1e-3 + x^2 + y^2)), nx1=4, ny1=4, nx2=4, ny2=4, element="P1") = ham2d([L, L], [N, N]; alpha_lap=alpha_lap, vext=vext, vee=vee, nx1=nx1, ny1=ny1, nx2=nx2, ny2=ny2, element=element)

#-------------------------------------------------------------------------------
#   Assemble N-body Hamiltonian matrices
#-------------------------------------------------------------------------------
export hamiltonian

# PARAMETERS
# ne : number of electrons
# alpha_lap : scaling parameter ---- not used until adding three parts together
# L : size of domain, i.e., Ω = [-L,L]³
# vext : external potential
# vee : electron-electron interaction
# N : discretization parameter, for FE, it is number of grid points
# nx, ny: number of Gauss quadrature points in x and y directions
#
# RETURN
# H : ne-body Hamiltonian -1/2⋅∑ᵢ Δ_{xᵢ} + ∑ᵢ V(xᵢ) + ∑_{i<j} v_ee(xᵢ-xⱼ)
# M : overlap
#-------------------------------------------------------------------------------



function hamiltonian(ne::Int, combBasis::Array{Array{T,1},1},
    AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
    C::SparseMatrixCSC{Float64,Int64},
    B::Array{Float64,4}; alpha_lap=1.0) where {T<:Signed}

    N = C.n
    # initialize the sparse array
    indrow2body = Int64[]
    indcol2body = Int64[]
    valH = Float64[]
    valM = Float64[]

    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # collect the pairs for Coulomb interactions
    coulomb_which2 = collect(combinations(v, 2))

    index = tensor2vec(ne, N, combBasis)

    i = zeros(Int, ne)
    s = zeros(Int, ne)
    j = zeros(Int, ne)
    t = zeros(Int, ne)
    jp = zeros(Int, ne)
    tj = zeros(Int, ne)
    jptr = zeros(Int64, ne)
    pk = zeros(Int, ne)
    jn = (2N) .^ collect(0:ne-1)

    # loop for the matrix elements
    for count = 1:length(combBasis)
        si = combBasis[count]
        for l in 1:ne
            i[l] = si[l] > N ? si[l] - N : si[l]
            s[l] = si[l] > N ? 1 : 0
        end
        @. jptr = 0
        while jptr[ne] < C.colptr[i[ne]+1] - C.colptr[i[ne]]
            Cv = 1.0
            Av = 0.0
            for l in 1:ne
                j[l] = C.rowval[C.colptr[i[l]]+jptr[l]]
                Cv *= C[i[l], j[l]]
                Av += (0.5 * alpha_lap * AΔ[i[l], j[l]] + AV[i[l], j[l]]) / C[i[l], j[l]]
            end
            Av *= Cv

            Bv = 0.0
            for l = 1:length(coulomb_which2)
                ca = coulomb_which2[l][1]
                cb = coulomb_which2[l][2]
                Bv += Cv * B[i[ca], i[cb], j[ca], j[cb]] /
                      (C[i[ca], j[ca]] * C[i[cb], j[cb]])
            end

            for k = 1:length(p)
                @views pk = p[k]
                for l in 1:ne
                    t[l] = s[pk[l]]
                    jp[l] = j[pk[l]]
                    tj[l] = jp[l] + N * t[l] - 1
                end
                tj1 = dot(tj, jn) + 1
                ζ = index[tj1]
                if ζ > 0
                    push!(indrow2body, count)
                    push!(indcol2body, ζ)
                    push!(valH, ε[k] * (Av + Bv))
                    push!(valM, ε[k] * Cv)
                end # end issorted(tj)
            end # end loop through permutation

            # adjust jptr
            jptr[1] += 1
            for ℓ = 1:ne-1
                if jptr[ℓ] == C.colptr[i[ℓ]+1] - C.colptr[i[ℓ]]
                    jptr[ℓ] = 0
                    jptr[ℓ+1] += 1
                end
            end
        end # end while loop for jptr
    end # end loop for combBasis

    H = sparse(indrow2body, indcol2body, valH)
    M = sparse(indrow2body, indcol2body, valM)
    return (H + H') / 2, M
end

function hamiltonian(ne::Int, combBasis::Array{Array{T,1},1},
    AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
    C::SparseMatrixCSC{Float64,Int64},
    B::SparseMatrixCSC{Float64,Int64}; alpha_lap=1.0) where {T<:Signed}

    N = C.n
    # initialize the sparse array
    indrow2body = Int64[]
    indcol2body = Int64[]
    valH = Float64[]
    valM = Float64[]

    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # collect the pairs for Coulomb interactions
    coulomb_which2 = collect(combinations(v, 2))

    index = tensor2vec(ne, N, combBasis)

    i = zeros(Int, ne)
    s = zeros(Int, ne)
    j = zeros(Int, ne)
    t = zeros(Int, ne)
    jp = zeros(Int, ne)
    tj = zeros(Int, ne)
    jptr = zeros(Int64, ne)
    pk = zeros(Int, ne)
    jn = (2N) .^ collect(0:ne-1)

    # loop for the matrix elements
    for count = 1:length(combBasis)
        si = combBasis[count]
        for l in 1:ne
            i[l] = si[l] > N ? si[l] - N : si[l]
            s[l] = si[l] > N ? 1 : 0
        end
        @. jptr = 0
        while jptr[ne] < C.colptr[i[ne]+1] - C.colptr[i[ne]]
            Cv = 1.0
            Av = 0.0
            for l in 1:ne
                j[l] = C.rowval[C.colptr[i[l]]+jptr[l]]
                Cv *= C[i[l], j[l]]
                Av += (0.5 * alpha_lap * AΔ[i[l], j[l]] + AV[i[l], j[l]]) / C[i[l], j[l]]
            end
            Av *= Cv

            Bv = 0.0
            for l = 1:length(coulomb_which2)
                ca = coulomb_which2[l][1]
                cb = coulomb_which2[l][2]
                Bv += Cv * B[i[ca]+(i[cb]-1)*N, j[ca]+(j[cb]-1)*N] /
                      (C[i[ca], j[ca]] * C[i[cb], j[cb]])
            end
            for k = 1:length(p)
                @views pk = p[k]
                for l in 1:ne
                    t[l] = s[pk[l]]
                    jp[l] = j[pk[l]]
                    tj[l] = jp[l] + N * t[l] - 1
                end
                tj1 = dot(tj, jn) + 1
                ζ = index[tj1]
                if ζ > 0
                    push!(indrow2body, count)
                    push!(indcol2body, ζ)
                    push!(valH, ε[k] * (Av + Bv))
                    push!(valM, ε[k] * Cv)
                end # end issorted(tj)
            end # end loop through permutation

            # adjust jptr
            jptr[1] += 1
            for ℓ = 1:ne-1
                if jptr[ℓ] == C.colptr[i[ℓ]+1] - C.colptr[i[ℓ]]
                    jptr[ℓ] = 0
                    jptr[ℓ+1] += 1
                end
            end
        end # end while loop for jptr
    end # end loop for combBasis

    H = sparse(indrow2body, indcol2body, valH)
    M = sparse(indrow2body, indcol2body, valM)
    return (H + H') / 2, M
end

hamiltonian(ne, AΔ, AV, C, B; alpha_lap=1.0) = hamiltonian(ne, collect(combinations(1:2*C.n, ne)), AΔ, AV, C, B; alpha_lap=alpha_lap)

hamiltonian(ne::Int64, combBasis::Array{Array{Int64,1},1}, ham::Hamiltonian) = hamiltonian(ne, combBasis, ham.AΔ, ham.AV, ham.C, ham.Bee; alpha_lap=ham.alpha_lap)

hamiltonian(ne::Int64, ham::Hamiltonian) = hamiltonian(ne, ham.AΔ, ham.AV, ham.C, ham.Bee; alpha_lap=ham.alpha_lap)