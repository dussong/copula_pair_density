
export ham_row_bs

function v2t_bs_1d(ne::Int64, kl::Vector{Int64}, N::Int64, Ψ::SparseVector{Float64,Int64}, Φktensor)
    X = zeros(Int,3,ne)
    Ψk = zeros(Int,ne)
    Ψktr = ones(Int,ne)
    for i = 1 : ne
        for j = 1 : 3
            if 0 < kl[i]+j-2 < N+1
                X[j,i] = kl[i]+j-2
            end
        end
    end
    while Ψktr[ne] <= 3
        for i = 1 : ne
            Ψk[i] = X[Ψktr[i],i]
        end
        if !(0 in Ψk) && allunique(Ψk)
            sort!(Ψk)
            j = seq2num_ns(N,ne,Ψk)
            l = Ψktr[1]
            for i = 2:ne
                l += (Ψktr[i]-1)*(3)^(i-1)
            end
            Φktensor[l] = Ψ[j]
        end

        #adjust Ψktr
        Ψktr[1] += 1
        for ℓ = 1 : ne-1
            if Ψktr[ℓ] == 4
               Ψktr[ℓ] = 1
               Ψktr[ℓ+1] += 1
           end
        end
    end

    return Φktensor;
end


function ham_row_bs(k::Vector{Int64}, ne::Int, Ψ::SparseVector{Float64,Int64}, ham::hamBs1d)

    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    B = ham.Bee
    alpha = ham.alpha_lap

    N = C.n
    n = length(k)
    ΦH = zeros(n)
    ΦM = zeros(n)
    Φktensor = zeros(Float64, ntuple(x -> 3, ne))

    Kl = map(x -> num2seq_ns(N, ne, x), k)
    v = 1:ne
    coulomb_which2 = collect(combinations(v, 2))

    c = [C[2, 1] C[2, 2] C[2, 3]]
    aΔ = [AΔ[2, 1] AΔ[2, 2] AΔ[2, 3]]
    am = c ./ ne

    kl = zeros(Int64, ne)
    av = zeros(3)
    b = zeros(9)
    for t = 1:n
        # reshape the vector Ψ to the (antisymmetric) tensor
        kl = Kl[t]
        @. Φktensor = 0.0
        Φktensor = v2t_bs_1d(ne, kl, N, Ψ, Φktensor)

        for j = 1:ne# act A on the j-th particle
            @. av = 0.0
            for i = 1:3
                if 0 < kl[j] - 2 + i <= N
                    av[i] = AV[kl[j], kl[j]-2+i]
                end
            end
            M = Φktensor
            Mc = Φktensor
            for i = 1:ne
                M = reshape(M, 3, 3^(ne - i))
                Mc = reshape(Mc, 3, 3^(ne - i))

                if i == j
                    M = (0.5 * alpha .* aΔ .+ av') * M
                    Mc = am * Mc
                else
                    M = c * M
                    Mc = c * Mc
                end
            end
            ΦH[t] += M[1]
            ΦM[t] += Mc[1]
        end

        for j = 1:length(coulomb_which2)# act B on the p-th,q-th particle
            p = coulomb_which2[j][1]
            q = coulomb_which2[j][2]
            M = permutedims(Φktensor, append!([p, q], setdiff!(collect(1:ne), p, q)))
            M = reshape(M, 3^2, 3^(ne - 2))
            @. b = 0.0
            for i = 1:3
                for j = 1:3
                    if 0 < kl[p] - 2 + i <= N && 0 < kl[q] - 2 + j <= N
                        b[i+(j-1)*3] = B[kl[p], kl[q], kl[p]-2+i, kl[q]-2+j]
                    end
                end
            end
            M = b' * M
            M = reshape(M, ntuple(x -> 3, ne - 2))
            for i = 1:ne-2
                M = reshape(M, 3, 3^(ne - i - 2))
                M = c * M
            end
            ΦH[t] += M[1]
        end
    end
    return ΦH, ΦM
end
