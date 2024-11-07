#-------------------------------------------------------------------------------
#   compute 1-body and 2-body properties from the sparse ne-body wavefunction
#-------------------------------------------------------------------------------
export density_sp, pair_density_sp
# PARAMETERS
# ne : the number of electrons
# combBasis : the index set of selected determinants
# Ψ : the coefficient of selected determinants
# C : 1-body overlap
#
# RETURN
# ρ  : density
#-------------------------------------------------------------------------------

# density for 1D
function density_sp(x::Array{Float64}, ne::Int64, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::ham1d)
    L = ham.L
    C = ham.C
    N = C.n
    n = length(x)
    ρ = zeros(n)
    h = (2.0 * L) / (N + 1)
    m = floor.(Int64, (x .+ L) ./ h) # Index of the basis function on the left of x
    ϕleft = ((m .+ 1) .* h .- L .- x) ./ h
    ϕright = (x .+ L .- m .* h) ./ h
    fv = hcat(ϕleft, ϕright)

    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]

    index = tensor2vec(ne, N, combBasis)

    i = zeros(Int, ne)
    s = zeros(Int, ne)
    j = zeros(Int, ne)
    t = zeros(Int, ne)
    jp = zeros(Int, ne)
    tp = zeros(Int, ne)
    tj = zeros(Int, ne)
    si = zeros(Int, ne)
    Cp = zeros(ne)
    ϕv = zeros(n)
    val = zeros(n)
    X = zeros(Int, 3, ne)
    tjtr = ones(Int64, ne)

    for count1 = 1:length(combBasis)
        si = combBasis[count1]
        @. X = 0
        # find the connected index
        for l in 1:ne
            i[l] = si[l] > N ? si[l] - N : si[l]
            s[l] = si[l] > N ? 1 : 0
            for k = 1:3
                if 0 < i[l] + k - 2 < N + 1
                    X[k, l] = i[l] + k - 2 + s[l] * N
                end
            end
        end
        @. tjtr = 1
        count = Int[]
        while tjtr[ne] <= 3
            for l = 1:ne
                tj[l] = X[tjtr[l], l]
            end
            if !(0 in tj)
                sort!(tj)
                tj1 = 1
                for i = 1:ne
                    tj1 += (tj[i] - 1) * (2N)^(i - 1)
                end
                count2 = index[tj1]
                if count2 > 0 && !(count2 in count)
                    push!(count, count2)
                    @. val = 0.0
                    for l in 1:ne
                        j[l] = tj[l] > N ? tj[l] - N : tj[l]
                        t[l] = tj[l] > N ? 1 : 0
                    end

                    for k = 1:length(p)
                        Cv = 1.0
                        @. ϕv = 0.0
                        for l in 1:ne
                            tp[l] = t[p[k][l]]
                            jp[l] = j[p[k][l]]
                            Cp[l] = C[i[l], jp[l]]
                        end
                        if s == tp && !(0.0 in Cp)
                            for l in 1:ne
                                Cv *= Cp[l]
                                for xi = 1:n
                                    if 0 < i[l] - m[xi] + 1 < 3 && 0 < jp[l] - m[xi] + 1 < 3
                                        ϕv[xi] += fv[xi, i[l]-m[xi]+1] * fv[xi, jp[l]-m[xi]+1] / Cp[l]
                                    end
                                end
                            end
                            @. ϕv *= Cv
                            @. val += ε[k] * ϕv
                        end
                    end

                    @. ρ += Ψ[count1] * Ψ[count2] * val
                end
            end
            tjtr[1] += 1
            for ℓ = 1:ne-1
                if tjtr[ℓ] == 4
                    tjtr[ℓ] = 1
                    tjtr[ℓ+1] += 1
                end
            end
        end
    end

    return ρ
end

# density for 2D
function density_sp(xy::Vector{Vector{Float64}}, ne::Int64, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::ham2d)
    Lx = ham.L[1]
    Ly = ham.L[2]
    Nx = ham.N[1] - 1
    Ny = ham.N[2] - 1
    hx = (2.0 * Lx) / (Nx + 1)
    hy = (2.0 * Ly) / (Ny + 1)
    C = ham.C
    dof = C.n
    n1 = length(xy[1])
    n2 = length(xy[2])
    ρ = zeros(n2, n1)

    x = xy[1]
    ix = floor.(Int64, (x .+ Lx) ./ hx)
    ϕileft = ((ix .+ 1) .* hx .- Lx .- x) ./ hx
    ϕiright = (x .+ Lx .- ix .* hx) ./ hx

    y = xy[2]
    jy = floor.(Int64, (y .+ Ly) ./ hy)
    ϕjleft = ((jy .+ 1) .* hy .- Ly .- y) ./ hy
    ϕjright = (y .+ Ly .- jy .* hy) ./ hy

    fx = [ϕileft, ϕiright]
    fy = [ϕjleft, ϕjright]

    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]

    index = tensor2vec(ne, dof, combBasis)

    i = zeros(Int, ne)
    mi = zeros(Int, ne)
    ni = zeros(Int, ne)
    s = zeros(Int, ne)
    j = zeros(Int, ne)
    mj = zeros(Int, ne)
    nj = zeros(Int, ne)
    t = zeros(Int, ne)
    tp = zeros(Int, ne)
    tj = zeros(Int, ne)
    si = zeros(Int, ne)
    Cp = zeros(ne)
    ϕv = zeros(n2, n1)
    val = zeros(n2, n1)
    XY = zeros(Int, 9, ne)
    tjtr = ones(Int64, ne)
    mix = zeros(Int, n1, ne)
    niy = zeros(Int, n2, ne)
    mjx = zeros(Int, n1, ne)
    njy = zeros(Int, n2, ne)

    for count1 = 1:length(combBasis)
        si = combBasis[count1]
        for l in 1:ne
            i[l] = si[l] > dof ? si[l] - dof : si[l]
            mi[l] = i[l] % Nx == 0 ? Nx : i[l] % Nx
            ni[l] = Int((i[l] - mi[l]) / Nx) + 1
            s[l] = si[l] > dof ? 1 : 0
            @views XY[:, l] = con_ij(mi[l], ni[l], Nx, Ny, s[l] * dof)
            @. mix[:, l] = mi[l] - ix + 1
            @. niy[:, l] = ni[l] - jy + 1
        end
        @. tjtr = 1
        count = Int[]
        while tjtr[ne] <= 9
            for l = 1:ne
                tj[l] = XY[tjtr[l], l]
            end
            if !(0 in tj)
                sort!(tj)
                tj1 = 1
                for i = 1:ne
                    tj1 += (tj[i] - 1) * (2dof)^(i - 1)
                end
                count2 = index[tj1]
                if count2 > 0 && !(count2 in count)
                    push!(count, count2)
                    @. val = 0.0
                    for l in 1:ne
                        j[l] = tj[l] > dof ? tj[l] - dof : tj[l]
                        mj[l] = j[l] % Nx == 0 ? Nx : j[l] % Nx
                        nj[l] = Int((j[l] - mj[l]) / Nx) + 1
                        t[l] = tj[l] > dof ? 1 : 0
                        @. mjx[:, l] = mj[l] - ix + 1
                        @. njy[:, l] = nj[l] - jy + 1
                    end
                    for k = 1:length(p)
                        @. ϕv = 0.0
                        for l in 1:ne
                            tp[l] = t[p[k][l]]
                            Cp[l] = C[i[l], j[p[k][l]]]
                        end
                        Cv = prod(Cp)
                        if s == tp && Cv != 0
                            for l in 1:ne
                                for i1 = 1:n1
                                    for i2 = 1:n2
                                        if 0 < mix[i1, l] < 3 && 0 < niy[i2, l] < 3 && 0 < mjx[i1, p[k][l]] < 3 && 0 < njy[i2, p[k][l]] < 3
                                            ϕv[i2, i1] += fx[mix[i1, l]][i1] * fy[niy[i2, l]][i2] * fx[mjx[i1, p[k][l]]][i1] * fy[njy[i2, p[k][l]]][i2] / Cp[l]
                                        end
                                    end
                                end
                            end
                            @. ϕv *= Cv
                            @. val += ε[k] * ϕv
                        end
                    end
                    @. ρ += Ψ[count1] * Ψ[count2] * val
                end
            end

            tjtr[1] += 1
            for ℓ = 1:ne-1
                if tjtr[ℓ] == 10
                    tjtr[ℓ] = 1
                    tjtr[ℓ+1] += 1
                end
            end
        end
    end

    return ρ
end

density_sp(x, wf::WaveFunction_sp, ham::Hamiltonian) = density_sp(x, wf.ne, wf.combBasis, wf.val, ham)

# pair density for 1D
function pair_density_sp(xy::Array{Float64,2}, ne::Int64, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::ham1d)
    L = ham.L
    C = ham.C
    N = C.n
    h = (2.0 * L) / (N + 1)
    n = size(xy, 1)
    ρ2 = zeros(n, n)

    x = xy[:, 1]
    ix = floor.(Int64, (x .+ L) ./ h)# Index of the basis function on the left of x
    ϕileft = ((ix .+ 1) .* h .- L .- x) ./ h
    ϕiright = (x .+ L .- ix .* h) ./ h

    y = xy[:, 2]
    jy = floor.(Int64, (y .+ L) ./ h)# Index of the basis function on the left of y
    ϕjleft = ((jy .+ 1) .* h .- L .- y) ./ h
    ϕjright = (y .+ L .- jy .* h) ./ h

    fx = hcat(ϕileft, ϕiright)
    fy = hcat(ϕjleft, ϕjright)

    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    coulomb_which2 = collect(combinations(v, 2))

    index = tensor2vec(ne, N, combBasis)

    i = zeros(Int, ne)
    s = zeros(Int, ne)
    j = zeros(Int, ne)
    t = zeros(Int, ne)
    jp = zeros(Int, ne)
    tp = zeros(Int, ne)
    tj = zeros(Int, ne)
    si = zeros(Int, ne)
    Cp = zeros(ne)
    ϕv = zeros(n, n)
    val = zeros(n, n)
    X = zeros(Int, 3, ne)
    tjtr = ones(Int64, ne)

    for count1 = 1:length(combBasis)
        si = combBasis[count1]
        @. X = 0
        # find the connected index
        for l in 1:ne
            i[l] = si[l] > N ? si[l] - N : si[l]
            s[l] = si[l] > N ? 1 : 0
            for k = 1:3
                if 0 < i[l] + k - 2 < N + 1
                    X[k, l] = i[l] + k - 2 + s[l] * N
                end
            end
        end
        @. tjtr = 1
        count = Int[]
        while tjtr[ne] <= 3
            for l = 1:ne
                tj[l] = X[tjtr[l], l]
            end
            if !(0 in tj)
                sort!(tj)
                tj1 = 1
                for i = 1:ne
                    tj1 += (tj[i] - 1) * (2N)^(i - 1)
                end
                count2 = index[tj1]
                if count2 > 0 && !(count2 in count)
                    push!(count, count2)
                    @. val = 0.0
                    for l in 1:ne
                        j[l] = tj[l] > N ? tj[l] - N : tj[l]
                        t[l] = tj[l] > N ? 1 : 0
                    end

                    for k = 1:length(p)
                        Cv = 1.0
                        @. ϕv = 0.0
                        for l in 1:ne
                            tp[l] = t[p[k][l]]
                            jp[l] = j[p[k][l]]
                            Cp[l] = C[i[l], jp[l]]
                        end
                        Cv = prod(Cp)
                        if s == tp && Cv != 0.0
                            for l = 1:length(coulomb_which2)
                                la = coulomb_which2[l][1]
                                lb = coulomb_which2[l][2]
                                for i1 = 1:n, i2 = 1:n
                                    if 0 < i[la] - ix[i1] + 1 < 3 && 0 < i[lb] - jy[i2] + 1 < 3 && 0 < jp[la] - ix[i1] + 1 < 3 && 0 < jp[lb] - jy[i2] + 1 < 3
                                        ϕv[i1, i2] += fx[i1, i[la]-ix[i1]+1] * fy[i2, i[lb]-jy[i2]+1] * fx[i1, jp[la]-ix[i1]+1] * fy[i2, jp[lb]-jy[i2]+1] / (Cp[la] * Cp[lb])
                                    end
                                end
                            end
                            @. ϕv *= Cv
                            @. val += ε[k] * ϕv
                        end
                    end

                    @. ρ2 += Ψ[count1] * Ψ[count2] * val
                end
            end
            tjtr[1] += 1
            for ℓ = 1:ne-1
                if tjtr[ℓ] == 4
                    tjtr[ℓ] = 1
                    tjtr[ℓ+1] += 1
                end
            end
        end
    end

    return (ρ2 + ρ2') / 2
end

pair_density_sp(x, wf::WaveFunction_sp, ham::Hamiltonian) = pair_density_sp(x, wf.ne, wf.combBasis, wf.val, ham)
