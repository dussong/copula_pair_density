
export ham_row
# PARAMETERS
# ne: number of electrons/particles
# k : the corresponding vector index of the index set of selected determinants
# combBasis : the index set of selected determinants
# Ψ : the coefficient of selected determinants
# A : 1-body operator, e.g., -Δ, v_ext
# B : 2-body operator, e.g., v_ee
# C : overlap
# RETURNS
# ΦH : the k-th row of H * Ψ
# ΦM : the k-th row of M * Ψ
#-------------------------------------------------------------------------------


function v2t_2d(ne::Int64, lx::Vector{Int64}, ly::Vector{Int64}, s::Vector{Int64},
    Nx::Int64, Ny::Int64, Ψ::SparseVector{Float64,Int64}, Φktensor)
    XY = zeros(Int, 9, ne)
    Φktr = ones(Int64, ne)
    Uk = zeros(Int64, ne)
    p = zeros(Int64, ne)
    for i = 1:ne
        @views XY[:, i] = con_ij(lx[i], ly[i], Nx, Ny, s[i])
    end

    while Φktr[ne] <= 9
        for i = 1:ne
            Uk[i] = XY[Φktr[i], i]
        end

        if !(0 in Uk) && allunique(Uk)
            sortperm!(p, Uk)
            sort!(Uk)
            ε = (-1)^parity(p)
            jk = seq2num_ns(2Nx*Ny, ne, Uk)
            w = Φktr[1]
            for j = 2:ne
                w += (Φktr[j] - 1) * 9^(j - 1)
            end
            Φktensor[w] = Ψ[jk] * ε
        end

        Φktr[1] += 1
        for ℓ = 1:ne-1
            if Φktr[ℓ] == 10
                Φktr[ℓ] = 1
                Φktr[ℓ+1] += 1
            end
        end
    end
    return Φktensor
end

function ham_row(k::Vector{Int64}, ne::Int, Ψ::SparseVector{Float64,Int64}, ham::ham2d)
    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    B = ham.Bee
    alpha = ham.alpha_lap
    Nx = ham.N[1] - 1
    Ny = ham.N[2] - 1

    dof = C.n
    n = length(k)
    ΦH = zeros(n)
    ΦM = zeros(n)
    Φktensor = zeros(Float64, ntuple(x -> 9, ne))

    Kl = map(x -> num2seq_ns(2dof, ne, x), k)
    v = 1:ne
    coulomb_which2 = collect(combinations(v, 2))

    c = [C[1+Nx, 2] C[2+Nx, 2] C[3+Nx, 2] C[1, 2] C[2, 2] C[3, 2] C[1+Nx, 2] C[2+Nx, 2] C[3+Nx, 2]]
    aΔ = [AΔ[1+Nx, 2] AΔ[2+Nx, 2] AΔ[3+Nx, 2] AΔ[1, 2] AΔ[2, 2] AΔ[3, 2] AΔ[1+Nx, 2] AΔ[2+Nx, 2] AΔ[3+Nx, 2]]
    am = c ./ ne

    kl = zeros(Int64, ne)
    lx = zeros(Int64, ne)
    ly = zeros(Int64, ne)
    s = zeros(Int64, ne)
    av = zeros(9)
    b = zeros(81)
    for t = 1:n
        # reshape the vector Ψ to the (antisymmetric) tensor
        Kt = Kl[t]
        for i = 1:ne
            kl[i] = Kt[i] > dof ? Kt[i] - dof : Kt[i]
            s[i] = Kt[i] > dof ? dof : 0
            lx[i] = kl[i] % Nx == 0 ? Nx : kl[i] % Nx
            ly[i] = Int((kl[i] - lx[i]) / Nx) + 1
        end
        broadcast!(*, Φktensor, Φktensor, 0.0)
        Φktensor = v2t_2d(ne, lx, ly, s, Nx, Ny, Ψ, Φktensor)

        for j = 1:ne# act A on the j-th particle
            @. av = 0.0
            for i1 = 1:3, i2 = 1:3
                if 0 < lx[j] + i1 - 2 < Nx + 1 && 0 < ly[j] + i2 - 2 < Ny + 1
                    av[i1+(i2-1)*3] = AV[kl[j], lx[j]+i1-2+(ly[j]+i2-3)*Nx]
                end
            end
            M = Φktensor
            Mc = Φktensor
            for i = 1:ne
                M = reshape(M, 9, 9^(ne - i))
                Mc = reshape(Mc, 9, 9^(ne - i))

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
            M = reshape(M, 9^2, 9^(ne - 2))
            @. b = 0.0
            for i1 in 1:3, j1 in 1:3, i2 in 1:3, j2 in 1:3
                bx1 = lx[p] + i1 - 2
                by1 = ly[p] + j1 - 2
                bx2 = lx[q] + i2 - 2
                by2 = ly[q] + j2 - 2
                if 0 < bx1 < Nx + 1 && 0 < by1 < Ny + 1 && 0 < bx2 < Nx + 1 && 0 < by2 < Ny + 1
                    b[i1+(j1-1)*3+(i2+(j2-1)*3-1)*9] = B[kl[p]+(kl[q]-1)*dof, bx1+(by1-1)*Nx+(bx2+(by2-1)*Nx-1)*dof] #B[kl[s],kl[t],B3,B4]
                end
            end
            M = b' * M
            M = reshape(M, ntuple(x -> 9, ne - 2))
            for i = 1:ne-2
                M = reshape(M, 9, 9^(ne - i - 2))
                M = c * M
            end
            ΦH[t] += M[1]
        end
    end
    return ΦH, ΦM
end
