
#----------------------------------------------------------------------------
# Hamiltonian structures:
# ham1d for 1D situation; ham2d for 2D situation
#----------------------------------------------------------------------------
export HamiltonianBoson, hamBs1d, hamBs2d

abstract type HamiltonianBoson end

"""Hamiltonian
L : Interval length
N : Nb of discretization points of the interval [-L,L]
AΔ : Laplace matrix (1 body) for P1 finite elements
AV : External potential matrix (1 body) with P1 finite elements basis
C : Overlap matrix (1 body) for P1 finite elements
B : Two-body matrices for P1 finite elements
alpha_lap : Parameter in the kinetic potential
"""
struct hamBs1d <: HamiltonianBoson
    L::Float64
    N::Int64
    AΔ::SparseMatrixCSC{Float64,Int64}
    AV::SparseMatrixCSC{Float64,Int64}
    C::SparseMatrixCSC{Float64,Int64}
    Bee::Array{Float64,4}
    alpha_lap::Float64
end

function hamBs1dGen(L::Float64, N::Int, alpha_lap::Float64, vext::Function, vee::Function, nx::Int64, ny::Int64)
    AΔ, AV, C, Bee = matrix_1_2body_1d(L, N, vext, vee, nx, ny)
    return hamBs1d(L, N, AΔ, AV, C, Bee, alpha_lap)
end
hamBs1d(L::Float64, N::Int; alpha_lap=1.0, vext::Function=(x -> x^2), vee::Function=(x -> 1.0 ./ abs.(x[1] - x[2])), nx=3, ny=4) = hamBs1dGen(L, N, alpha_lap, vext, vee, nx, ny)

struct hamBs2d <: HamiltonianBoson
    L::Float64
    N::Int64
    AΔ::SparseMatrixCSC{Float64,Int64}
    AV::SparseMatrixCSC{Float64,Int64}
    C::SparseMatrixCSC{Float64,Int64}
    Bee::SparseMatrixCSC{Float64,Int64}
    alpha_lap::Float64
end

function hamBs2dGen(L::Float64, N::Int64, alpha_lap::Float64, vext::Function, vee::Function, nx1::Int64, ny1::Int64, nx2::Int64, ny2::Int64)
    AΔ, AV, C, Bee = matrix_1_2body_2d(L, N, vext, vee, nx1, ny1, nx2, ny2)
    return hamBs2d(L, N, AΔ, AV, C, Bee, alpha_lap)
end
hamBs2d(L::Float64, N::Int64; alpha_lap=1.0, vext::Function=(x -> (x[1] .^ 2 .+ x[2] .^ 2)), vee::Function=(x -> 1.0 ./ abs.(x[1] - x[2])), nx1=4, ny1=4, nx2=3, ny2=3) = hamBs2dGen(L, N, alpha_lap, vext, vee, nx1, ny1, nx2, ny2)

#-------------------------------------------------------------------------------
#   Assemble N-body Hamiltonian matrices
#-------------------------------------------------------------------------------
export tensor2vec_bs, hamiltonian_bs
function tensor2vec_bs(ne::Int, N::Int, combBasis::Array{Array{T,1},1}) where {T<:Signed}
    ind = ones(Int, length(combBasis))
    vals = collect(1:length(combBasis))
    for count = 1:length(combBasis)
        for i = 1:ne
            ind[count] += (combBasis[count][i] - 1) * (N)^(i - 1)
        end
    end
    return sparsevec(ind, vals, N^ne)
end

# function ham_alpha_lap(HΔ::SparseMatrixCSC{Float64,Int64},
#                 M::SparseMatrixCSC{Float64,Int64}, alpha_lap::Float64)
#   H = 0.5 * alpha_lap * HΔ + M;
#   return H;
# end

function hamiltonian_bs(ne::Int, combBasis::Array{Array{T,1},1},
    AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
    C::SparseMatrixCSC{Float64,Int64},
    B::Array{Float64,4}; alpha_lap=1.0) where {T<:Signed}

    N = C.n
    # initialize the sparse array
    indrow2body = Int64[]
    indcol2body = Int64[]
    valH = Float64[]
    valM = Float64[]

    # collect the pairs for Coulomb interactions
    v = 1:ne
    p = collect(permutations(v))[:]
    coulomb_which2 = collect(combinations(v, 2))

    index = tensor2vec_bs(ne, N, combBasis)

    j = zeros(Int, ne)
    jptr = zeros(Int64, ne)
    pk = zeros(Int, ne)

    # loop for the 1-body matrix elements
    for count = 1:length(combBasis)
        i = combBasis[count]
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
                j1 = 1
                @views pk = p[k]
                for l = 1:ne
                    j1 += (j[pk[l]] - 1) * (N)^(l - 1)
                end
                ζ = index[j1]
                if ζ > 0
                    push!(indrow2body, count)
                    push!(indcol2body, ζ)
                    push!(valH, Av + Bv)
                    push!(valM, Cv)
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

function hamiltonian_bs(ne::Int, combBasis::Array{Array{T,1},1},
    AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
    C::SparseMatrixCSC{Float64,Int64},
    B::SparseMatrixCSC{Float64,Int64}; alpha_lap=1.0) where {T<:Signed}

    N = C.n
    # initialize the sparse array
    indrow2body = Int64[]
    indcol2body = Int64[]
    valH = Float64[]
    valM = Float64[]

    # collect the pairs for Coulomb interactions
    v = 1:ne
    p = collect(permutations(v))[:]
    coulomb_which2 = collect(combinations(v, 2))

    index = tensor2vec_bs(ne, N, combBasis)

    j = zeros(Int, ne)
    jp = zeros(Int, ne)
    jptr = zeros(Int64, ne)
    pk = zeros(Int, ne)
    jn = N .^ collect(0:ne-1)

    # loop for the 1-body matrix elements
    for count = 1:length(combBasis)
        i = combBasis[count]
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
                for l = 1:ne
                    jp[l] = j[pk[l]] - 1
                end
                j1 = dot(jp, jn) + 1
                ζ = index[j1]
                if ζ > 0
                    push!(indrow2body, count)
                    push!(indcol2body, ζ)
                    push!(valH, Av + Bv)
                    push!(valM, Cv)
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

hamiltonian_bs(ne, AΔ, AV, C, B; alpha_lap=1.0) = hamiltonian_bs(ne, collect(combinations(1:2*C.n, ne)), AΔ, AV, C, B; alpha_lap=alpha_lap)

hamiltonian_bs(ne::Int64, combBasis::Array{Array{Int64,1},1}, ham::HamiltonianBoson) = hamiltonian_bs(ne, combBasis, ham.AΔ, ham.AV, ham.C, ham.Bee; alpha_lap=ham.alpha_lap)

hamiltonian_bs(ne::Int64, ham::HamiltonianBoson) = hamiltonian_bs(ne, ham.AΔ, ham.AV, ham.C, ham.Bee; alpha_lap=ham.alpha_lap)