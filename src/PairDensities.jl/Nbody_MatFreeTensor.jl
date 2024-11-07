
# PARAMETERS
# ne: number of electrons/particles
# A : 1-body operator, e.g., -Δ, v_ext
# B : 2-body operator, e.g., v_ee
# C : overlap
# RETURNS
# H Ψ : many-body Hamiltonian for the 1-body and 2-body operator on a given vector Ψ
#-------------------------------------------------------------------------------
export ham_1B_free_tensor, ham_2B_free_tensor


function ham_1B_free_tensor(ne::Int64, Ψ::Array{Float64,1},
    A::SparseMatrixCSC{Float64,Int64},
    C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
    Φtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
    # indecies for the basis
    basis1body = 1:2*N
    combBasis = collect(combinations(basis1body, ne))
    # Φ = H⋅Ψ
    @assert length(Ψ) == length(combBasis)
    Φ = zeros(size(Ψ))
    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    #ik = zeros(Int,ne)
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            #ik = [ij[p[k][l]] for l = 1 : ne]
            #Ψtensor[CartesianIndex(Tuple(ik))] = Ψ[j] * ε[k]
            #=
            for l = 1 : ne
                ik[l] = ij[p[k][l]]
            end
            Ψtensor[CartesianIndex(Tuple(ik))] = Ψ[j] * ε[k]
            =#

            ik = ij[p[k][1]]
            for l = 2:ne
                ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
            end
            Ψtensor[ik] = Ψ[j] * ε[k]

        end
    end
    # loop through different spin configurations
    sptr = zeros(Int64, ne)
    for s = 1:2^ne
        sp = sptr * N
        for j = 1:ne # act A on the j-th particle
            φtensor = getindex(Ψtensor, ntuple(x -> sp[x]+1:sp[x]+N, ne)...)
            # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
            for i = 1:ne
                m = vcat(i, setdiff(1:ne, i))
                M = permutedims(φtensor, m)
                M = reshape(M, N, N^(ne - 1))
                if i == j
                    M = A * M
                else
                    M = C * M
                end
                φtensor = reshape(M, ntuple(x -> N, ne))
                φtensor = permutedims(φtensor, sortperm(m))

            end
            # assemble the value to Φtensor
            Φtensor[ntuple(x -> sp[x]+1:sp[x]+N, ne)...] += φtensor
        end
        # adjust sptr
        sptr[1] += 1
        for ℓ = 1:ne-1
            if sptr[ℓ] == 2
                sptr[ℓ] = 0
                sptr[ℓ+1] += 1
            end
        end
    end

    for i = 1:length(combBasis)

        il = combBasis[i]
        l = il[1]
        for j = 2:ne
            l += (il[j] - 1) * (2N)^(j - 1)
        end
        Φ[i] = Φtensor[l]

        #Φ[i] = Φtensor[CartesianIndex(Tuple(combBasis[i]))]
    end
    return Φ
end

function ham_2B_free_tensor(ne::Int64, Ψ::Array{Float64,1},
    B::Array{Float64,4},
    C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
    Φtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
    # indecies for the basis
    basis1body = 1:2*N
    combBasis = collect(combinations(basis1body, ne))
    # Φ = H⋅Ψ
    @assert length(Ψ) == length(combBasis)
    Φ = zeros(size(Ψ))
    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    coulomb_which2 = collect(combinations(v, 2))
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            #ik = [ij[p[k][l]] for l = 1 : ne]
            #Ψtensor[CartesianIndex(Tuple(ik))] = Ψ[j] * ε[k]

            ik = ij[p[k][1]]
            for l = 2:ne
                ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
            end
            Ψtensor[ik] = Ψ[j] * ε[k]

        end
    end
    # loop through different spin configurations
    sptr = zeros(Int64, ne)
    for s = 1:2^ne
        sp = sptr * N
        for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
            k = coulomb_which2[j][1]
            l = coulomb_which2[j][2]
            φtensor = getindex(Ψtensor, ntuple(x -> sp[x]+1:sp[x]+N, ne)...)
            # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
            for i = 1:ne
                if i != k && i != l
                    m = vcat(i, setdiff(1:ne, i))
                    M = permutedims(φtensor, m)
                    M = reshape(M, N, N^(ne - 1))
                    M = C * M
                    φtensor = reshape(M, ntuple(x -> N, ne))
                    φtensor = permutedims(φtensor, sortperm(m))
                elseif i == k
                    m = vcat(k, l, setdiff(1:ne, k, l))
                    M = permutedims(φtensor, m)
                    M = reshape(M, N, N, N^(ne - 2))
                    W = zeros(N, N, N^(ne - 2))
                    @tensor W[i, j, k] = B[i, j, a, b] * M[a, b, k]
                    φtensor = reshape(W, ntuple(x -> N, ne))
                    φtensor = permutedims(φtensor, sortperm(m))
                end
            end
            # assemble the value to Φtensor
            Φtensor[ntuple(x -> sp[x]+1:sp[x]+N, ne)...] += φtensor
        end
        # adjust sptr
        sptr[1] += 1
        for ℓ = 1:ne-1
            if sptr[ℓ] == 2
                sptr[ℓ] = 0
                sptr[ℓ+1] += 1
            end
        end
    end

    for i = 1:length(combBasis)
        il = combBasis[i]
        l = il[1]
        for j = 2:ne
            l += (il[j] - 1) * (2N)^(j - 1)
        end
        Φ[i] = Φtensor[l]
        #Φ[i] = Φtensor[CartesianIndex(Tuple(combBasis[i]))]
    end
    return Φ
end


function ham_2B_free_tensor(ne::Int64, Ψ::Array{Float64,1},
    B::SparseMatrixCSC{Float64,Int64},
    C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
    Φtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
    # indecies for the basis
    basis1body = 1:2*N
    combBasis = collect(combinations(basis1body, ne))
    # Φ = H⋅Ψ
    @assert length(Ψ) == length(combBasis)
    Φ = zeros(size(Ψ))
    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    coulomb_which2 = collect(combinations(v, 2))
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)

            ik = ij[p[k][1]]
            for l = 2:ne
                ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
            end
            Ψtensor[ik] = Ψ[j] * ε[k]

        end
    end
    # loop through different spin configurations
    sptr = zeros(Int64, ne)
    for s = 1:2^ne
        sp = sptr * N
        for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
            k = coulomb_which2[j][1]
            l = coulomb_which2[j][2]
            φtensor = getindex(Ψtensor, ntuple(x -> sp[x]+1:sp[x]+N, ne)...)
            # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
            for i = 1:ne
                if i != k && i != l
                    m = vcat(i, setdiff(1:ne, i))
                    M = permutedims(φtensor, m)
                    M = reshape(M, N, N^(ne - 1))
                    M = C * M
                    φtensor = reshape(M, ntuple(x -> N, ne))
                    φtensor = permutedims(φtensor, sortperm(m))
                elseif i == k
                    m = vcat(k, l, setdiff(1:ne, k, l))
                    M = permutedims(φtensor, m)
                    M = reshape(M, N^2, N^(ne - 2))
                    M = B * M
                    φtensor = reshape(M, ntuple(x -> N, ne))
                    φtensor = permutedims(φtensor, sortperm(m))
                end
            end
            # assemble the value to Φtensor
            Φtensor[ntuple(x -> sp[x]+1:sp[x]+N, ne)...] += φtensor
        end
        # adjust sptr
        sptr[1] += 1
        for ℓ = 1:ne-1
            if sptr[ℓ] == 2
                sptr[ℓ] = 0
                sptr[ℓ+1] += 1
            end
        end
    end

    for i = 1:length(combBasis)

        il = combBasis[i]
        l = il[1]
        for j = 2:ne
            l += (il[j] - 1) * (2N)^(j - 1)
        end
        Φ[i] = Φtensor[l]

    end
    return Φ
end
