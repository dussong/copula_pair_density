

export ham_column_nonz_bs, ham_column_bs

#find the nonzero coordinates of H[:,k]
function ham_column_nonz_bs(ne::Int64, ham::hamBs2d, kl::Array{Int64,1})
    dof = ham.C.n
    N = Int(sqrt(dof))

    XY = zeros(Int, 9, ne)

    for i = 1:ne
        lx = kl[i] % N == 0 ? N : kl[i] % N
        ly = Int((kl[i] - lx) / N) + 1
        @views XY[:, i] = con_ij(lx, ly, N, 0)
    end

    l = Int[]
    Φk = zeros(Int64, ne)
    Φktr = rand(1:9, 5)
    kktr = ones(Int64, ne)
    for ti = 1:5^ne
        for i = 1:ne
            Φk[i] = XY[Φktr[kktr[i],i], i]
        end
        if !(0 in Φk)
            sort!(Φk)
            if Φk[1] != Φk[2] && Φk[ne-1] != Φk[ne]
                if ne <= 3
                    push!(l, seq2num_ns(dof, ne, Φk))
                else
                    @views Φm = Φk[1:ne-1] - Φk[2:ne]
                    if !(0 in Φm)
                        push!(l, seq2num_ns(dof, ne, Φk))
                    end
                end
            end
        end

        kktr[1] += 1
        for ℓ = 1:ne-1
            if kktr[ℓ] == 6
                kktr[ℓ] = 1
                kktr[ℓ+1] += 1
            end
        end
    end

    l = unique!(l)

    return l
end


function t2v_full_bs_2d(ne::Int64, dof::Int64, Φk::Vector{Int64}, Φktr::Vector{Int64}, p::Vector{Int64}, XY::Array{Int64,2}, W, Wm, Φm::Vector{Int64})
    ind = Int64[]
    val_H = Float64[]
    val_M = Float64[]
    for ti = 1:9^ne
        for i = 1:ne
            Φk[i] = XY[Φktr[i], p[i]]
        end
        if !(0 in Φk) && issorted(Φk) && Φk[1] != Φk[2] && Φk[ne-1] != Φk[ne]
            @views Φk1 = Φk[1:ne-1]
            @views Φk2 = Φk[2:ne]
            @. Φm = Φk1 - Φk2
            if !(0 in Φm)
                push!(ind, seq2num_ns(dof, ne, Φk))
                push!(val_H, W[ti])
                push!(val_M, Wm[ti])
            end
        end
        Φktr[1] += 1
        for ℓ = 1:ne-1
            if Φktr[ℓ] == 10
                Φktr[ℓ] = 1
                Φktr[ℓ+1] += 1
            end
        end
    end
    return ind, val_H, val_M
end

function ham_column_bs(ne::Int, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::hamBs2d)

    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    B = ham.Bee
    alpha = ham.alpha_lap

    dof = C.n
    N = Int(sqrt(dof))
    @assert dof > 2
    @assert length(Ψ) == length(combBasis)
    Φktensor = zeros(Float64, ntuple(x -> 9, ne))
    Φktensor1 = zeros(Float64, ntuple(x -> 9, ne))
    Φktensorm = zeros(Float64, ntuple(x -> 9, ne))
    ϕktensor = zeros(Float64, ntuple(x -> 9, ne))
    ϕktensorm = zeros(Float64, ntuple(x -> 9, ne))
    W = zeros(Float64, ntuple(x -> 9, ne))
    Wm = zeros(Float64, ntuple(x -> 9, ne))
    ind = Int64[]
    vals_H = Float64[]
    vals_M = Float64[]
    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    coulomb_which2 = collect(combinations(v, 2))

    aΔ = [AΔ[1+N, 2], AΔ[2+N, 2], AΔ[3+N, 2], AΔ[1, 2], AΔ[2, 2], AΔ[3, 2], AΔ[1+N, 2], AΔ[2+N, 2], AΔ[3+N, 2]]
    c = [C[1+N, 2] C[2+N, 2] C[3+N, 2] C[1, 2] C[2, 2] C[3, 2] C[1+N, 2] C[2+N, 2] C[3+N, 2]]
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
        broadcast!(+, ϕktensorm, ϕktensorm, reshape(M, ntuple(x -> 9, ne)))
    end

    Φk = zeros(Int64, ne)
    Φktr = ones(Int64, ne)
    kl = zeros(Int64, ne)
    lx = zeros(Int64, ne)
    ly = zeros(Int64, ne)
    pj = zeros(Int64, ne)
    av = zeros(9)
    b = zeros(81)
    XY = zeros(Int, 9, ne)
    M1 = zeros(9, 9^(ne - 1))
    M2 = zeros(81, 9^(ne - 2))
    Φm = zeros(Int64, ne - 1)
    for l = 1:length(combBasis)
        kl = combBasis[l]
        broadcast!(*, Φktensorm, ϕktensorm, Ψ[l])

        for i = 1:ne
            lx[i] = kl[i] % N == 0 ? N : kl[i] % N
            ly[i] = Int((kl[i] - lx[i]) / N) + 1
            @views XY[:, i] = con_ij(lx[i], ly[i], N, 0)
        end
        @. ϕktensor = 0.0
        for j = 1:ne# act act A on the j-th particle
            @. av = 0.0
            for i1 = 1:3, i2 = 1:3
                if 0 < lx[j] + i1 - 2 < N + 1 && 0 < ly[j] + i2 - 2 < N + 1
                    av[i1+(i2-1)*3] = AV[kl[j], lx[j]+i1-2+(ly[j]+i2-3)*N]
                end
            end
            # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
            @. av = 0.5 * alpha * aΔ + av
            mul!(M1, av, Cna)
            permutedims!(W, reshape(M1, ntuple(x -> 9, ne)), sortperm!(pj, append!([j], setdiff!(collect(1:ne), j))))
            broadcast!(+, ϕktensor, ϕktensor, W)
            #ϕktensor += permutedims(M,sortperm!(pj,append!([j],setdiff!(collect(1:ne),j))))
        end
        broadcast!(*, Φktensor1, ϕktensor, Ψ[l])

        @. ϕktensor = 0.0
        for j = 1:length(coulomb_which2)# act B on the p-th,q-th particle
            s = coulomb_which2[j][1]
            t = coulomb_which2[j][2]
            @. b = 0.0
            for i1 in 1:3, j1 in 1:3, i2 in 1:3, j2 in 1:3
                bx1 = lx[s] + i1 - 2
                by1 = ly[s] + j1 - 2
                bx2 = lx[t] + i2 - 2
                by2 = ly[t] + j2 - 2
                if 0 < bx1 < N + 1 && 0 < by1 < N + 1 && 0 < bx2 < N + 1 && 0 < by2 < N + 1
                    b[i1+(j1-1)*3+(i2+(j2-1)*3-1)*9] = B[kl[s]+(kl[t]-1)*dof, bx1+(by1-1)*N+(bx2+(by2-1)*N-1)*dof] #B[kl[s],kl[t],B3,B4]
                end
            end
            # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
            mul!(M2, b, Cn)
            permutedims!(W, reshape(M2, ntuple(x -> 9, ne)), sortperm!(pj, append!([s, t], setdiff!(collect(1:ne), s, t))))
            broadcast!(+, ϕktensor, ϕktensor, W)
            #ϕktensor += permutedims(M, sortperm(vcat(s,t,setdiff(1:ne,s,t))))
        end
        broadcast!(*, Φktensor, ϕktensor, Ψ[l])
        broadcast!(+, Φktensor, Φktensor, Φktensor1)

        #loop for the permutations
        for k = 1:length(p)
            @views pk = p[k]
            permutedims!(W, Φktensor, pk)
            permutedims!(Wm, Φktensorm, pk)

            @. Φktr = 1
            indk, val_H, val_M = t2v_full_bs_2d(ne, dof, Φk, Φktr, pk, XY, W, Wm, Φm)
            append!(ind, indk)
            append!(vals_H, val_H)
            append!(vals_M, val_M)
        end
    end

    ΦH = sparsevec(ind, vals_H, dof^ne)
    ΦM = sparsevec(ind, vals_M, dof^ne)
    return ΦH, ΦM
end
ham_column_bs(ne::Int, K::Array{Int64,1}, Ψ::Array{Float64,1}, ham::HamiltonianBoson) =
ham_column_bs(ne, map(x -> num2seq_ns(ham.C.n, ne, x), K), Ψ, ham)
