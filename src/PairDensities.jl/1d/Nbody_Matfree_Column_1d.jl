#  1D situation

export ham_column_nonz, ham_column_nonz_rd, ham_column
#-------------------------------------------------------------------------------
# PARAMETERS
# A : 1-body operator, e.g., -Δ, v_ext
# B : 2-body operator, e.g., v_ee
# C : overlap
# combBasis : the index set of selected determinants
# Ψ : the coefficient of selected determinants
# RETURNS
# ΦH : H * Ψ 
# ΦM : M * Ψ 
#-------------------------------------------------------------------------------

#find the nonzero coordinates of H[:,k]
function ham_column_nonz(ne::Int64, ham::ham1d, kl::Array{Int64,1})
    N = ham.C.n

    #kl = num2seq_ns(2N,ne,k)
    m = kl .% N
    inftr = -ones(Int, ne)
    suptr = ones(Int, ne)
    for i = 1:ne
        if m[i] == 1
            inftr[i] = 0
        elseif m[i] == 0
            suptr[i] = 0
        end
    end

    jk = zeros(Int, ne)
    jktr = copy(inftr)
    l = Int64[]
    while jktr[ne] <= suptr[ne]
        jk = kl + jktr
        unique!(jk)
        if length(jk) == ne
            sort!(jk)
            j = seq2num_ns(2N, ne, jk)
            push!(l, j)
        end
        #adjust jktr
        jktr[1] += 1
        for ℓ = 1:ne-1
            if jktr[ℓ] == suptr[ℓ] + 1
                jktr[ℓ] = inftr[ℓ]
                jktr[ℓ+1] += 1
            end
        end
    end

    l = unique!(l)
    #l = sort(l)
    return l
end
ham_column_nonz(ne::Int64, ham::Hamiltonian, k::Int64) = ham_column_nonz(ne, ham, num2seq_ns(2ham.C.n, ne, k))

#Randomly find a part of the nonzero coordinates of H[:,k]
function ham_column_nonz_rd(ne::Int64, ham::ham1d, kl::Array{Int64,1})
    N = ham.C.n
    m = zeros(Int, ne)
    s = zeros(Int, ne)

    for i = 1:ne
        m[i] = kl[i] > N ? kl[i] - N : kl[i]
        s[i] = kl[i] > N ? N : 0
    end

    X = zeros(Int, 3, ne)
    for i = 1:ne
        for j = 1:3
            if 0 < m[i] + j - 2 < N + 1
                X[j, i] = m[i] + j - 2 + s[i]
            end
        end
    end

    jktr = rand(1:3, 2)
    jk = zeros(Int, ne)
    kktr = ones(Int64, ne)
    l = Int64[]
    for ti = 1:2^ne
        for i = 1:ne
            jk[i] = X[jktr[kktr[i]], i]
        end
        
        if !(0 in jk) && allunique(jk)
            sort!(jk)
            push!(l, seq2num_ns(2N, ne, jk))
        end

        #adjust jktr
        kktr[1] += 1
        for ℓ = 1:ne-1
            if kktr[ℓ] == 3
                kktr[ℓ] = 1
                kktr[ℓ+1] += 1
            end
        end
    end

    l = unique!(l)
    return l
end
ham_column_nonz_rd(ne::Int64, ham::Hamiltonian, k::Int64) = ham_column_nonz_rd(ne, ham, num2seq_ns(2ham.C.n, ne, k))

# map tensor to vector
# ind : the corresponding vector index of tensor index
# val_H : value of (H * Ψ)_{ind}
# val_M : value of (M * Ψ)_{ind}
function t2v_full_1d(ne::Int64, N::Int64, Φk::Vector{Int64}, Φktr::Vector{Int64}, p::Vector{Int64},
    X::Array{Int64,2}, W, Wm, Φm::Vector{Int64})
    ind = Int64[]
    val_H = Float64[]
    val_M = Float64[]
    for ti = 1:3^ne
        for i = 1:ne
            Φk[i] = X[Φktr[i], p[i]]
        end
        if !(0 in Φk) && issorted(Φk) && Φk[1] != Φk[2] && Φk[ne-1] != Φk[ne]
            @views Φk1 = Φk[1:ne-1]
            @views Φk2 = Φk[2:ne]
            @. Φm = Φk1 - Φk2
            if !(0 in Φm)
                push!(ind, seq2num_ns(2N, ne, Φk))
                push!(val_H, W[ti])
                push!(val_M, Wm[ti])
            end
        end
        Φktr[1] += 1
        for ℓ = 1:ne-1
            if Φktr[ℓ] == 4
                Φktr[ℓ] = 1
                Φktr[ℓ+1] += 1
            end
        end
    end
    return ind, val_H, val_M
end

function ham_column(ne::Int, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::ham1d)
    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    B = ham.Bee
    alpha = ham.alpha_lap

    N = C.n
    @assert N > 2
    @assert length(Ψ) == length(combBasis)
    Φktensor = zeros(Float64, ntuple(x -> 3, ne))
    Φktensor1 = zeros(Float64, ntuple(x -> 3, ne))
    Φktensorm = zeros(Float64, ntuple(x -> 3, ne))
    ϕktensor = zeros(Float64, ntuple(x -> 3, ne))
    ϕktensorm = zeros(Float64, ntuple(x -> 3, ne))
    W = zeros(Float64, ntuple(x -> 3, ne))
    Wm = zeros(Float64, ntuple(x -> 3, ne))
    ind = Int64[]
    vals_H = Float64[]
    vals_M = Float64[]
    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    coulomb_which2 = collect(combinations(v, 2))

    aΔ = [AΔ[1, 2], AΔ[2, 2], AΔ[3, 2]]
    c = [C[1, 2] C[2, 2] C[3, 2]]
    Cn = 1.0
    for i = 1:ne-2
        Cn = Cn * c
        Cn = Cn[:]
    end
    Cna = (Cn*c)[:]
    Cn = Cn'
    Cna = Cna'

    c = c'
    am = c ./ ne
    for j = 1:ne# act A on the j-th particle
        M = 1.0
        for i = ne:-1:1
            if i == j
                M = am * M'
            else
                M = c * M'
            end
            M = M[:]
        end
        broadcast!(+, ϕktensorm, ϕktensorm, reshape(M, ntuple(x -> 3, ne)))
    end

    Φk = zeros(Int64, ne)
    Φktr = ones(Int64, ne)
    kl = zeros(Int64, ne)
    pj = zeros(Int64, ne)
    av = zeros(3)
    b = zeros(9)
    X = zeros(Int, 3, ne)
    M1 = zeros(3, 3^(ne - 1))
    M2 = zeros(9, 3^(ne - 2))
    Φm = zeros(Int64, ne - 1)
    for l = 1:length(combBasis)
        Kl = combBasis[l]
        broadcast!(*, Φktensorm, ϕktensorm, Ψ[l])

        @. X = 0
        for i = 1:ne
            kl[i] = Kl[i] > N ? Kl[i] - N : Kl[i]
            st = Kl[i] > N ? N : 0
            for j = 1:3
                if 0 < kl[i] + j - 2 < N + 1
                    X[j, i] = kl[i] + j - 2 + st
                end
            end
        end

        @. ϕktensor = 0.0
        for j = 1:ne# act act A on the j-th particle
            @. av = 0.0
            for i = 1:3
                if 0 < kl[j] - 2 + i < N + 1
                    av[i] = AV[kl[j], kl[j]-2+i]
                end
            end
            # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
            @. av = 0.5 * alpha * aΔ + av
            mul!(M1, av, Cna)
            permutedims!(W, reshape(M1, ntuple(x -> 3, ne)), sortperm!(pj, append!([j], setdiff!(collect(1:ne), j))))
            broadcast!(+, ϕktensor, ϕktensor, W)
            #ϕktensor += permutedims(M,sortperm!(pj,append!([j],setdiff!(collect(1:ne),j))))
        end
        broadcast!(*, Φktensor1, ϕktensor, Ψ[l])

        @. ϕktensor = 0.0
        for j = 1:length(coulomb_which2)# act B on the p-th,q-th particle
            s = coulomb_which2[j][1]
            t = coulomb_which2[j][2]
            for i = 1:3
                for j = 1:3
                    if 0 < kl[s] - 2 + i <= N && 0 < kl[t] - 2 + j <= N
                        b[i+3*(j-1)] = B[kl[s], kl[t], kl[s]-2+i, kl[t]-2+j]
                    end
                end
            end
            # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
            mul!(M2, b, Cn)
            permutedims!(W, reshape(M2, ntuple(x -> 3, ne)), sortperm!(pj, append!([s, t], setdiff!(collect(1:ne), s, t))))
            broadcast!(+, ϕktensor, ϕktensor, W)
        end
        broadcast!(*, Φktensor, ϕktensor, Ψ[l])
        broadcast!(+, Φktensor, Φktensor, Φktensor1)

        #loop for the permutations
        for k = 1:length(p)
            permutedims!(W, Φktensor, p[k])
            broadcast!(*, W, W, ε[k])
            permutedims!(Wm, Φktensorm, p[k])
            broadcast!(*, Wm, Wm, ε[k])

            @. Φktr = 1
            indk, val_H, val_M = t2v_full_1d(ne, N, Φk, Φktr, p[k], X, W, Wm, Φm)
            append!(ind, indk)
            append!(vals_H, val_H)
            append!(vals_M, val_M)
        end
    end

    ΦH = sparsevec(ind, vals_H, (2 * N)^ne)
    ΦM = sparsevec(ind, vals_M, (2 * N)^ne)
    return ΦH, ΦM
end
