export density_bs_sp

function density_bs_sp(x::Array{Float64}, ne::Int64, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::hamBs1d)
    L = ham.L
    C = ham.C
    N = C.n
    n = length(x)
    ρ = zeros(n)
    h = (2.0 * L) / (N + 1)
    m = floor.(Int64, (x .+ L) ./ h)
    ϕleft = ((m .+ 1) .* h .- L .- x) ./ h
    ϕright = (x .+ L .- m .* h) ./ h
    fv = hcat(ϕleft, ϕright)

    index = tensor2vec_bs(ne, N, combBasis)

    i = zeros(Int, ne)
    j = zeros(Int, ne)
    mi = zeros(Int, ne)
    mj = zeros(Int, ne)
    js = zeros(Int, ne)
    ϕv = zeros(n)
    X = zeros(Int, 3, ne)
    jtr = ones(Int64, ne)

    for count1 = 1:length(combBasis)
        i = combBasis[count1]
        @. X = 0
        for l in 1:ne
            for k = 1:3
                if 0 < i[l] + k - 2 < N + 1
                    X[k, l] = i[l] + k - 2
                end
            end
        end
        @. jtr = 1
        while jtr[ne] <= 3
            for l = 1:ne
                j[l] = X[jtr[l], l]
                js[l] = X[jtr[l], l]
            end
            if !(0 in j)
                sort!(js)
                j1 = 1
                for i = 1:ne
                    j1 += (js[i] - 1) * (N)^(i - 1)
                end
                count2 = index[j1]
                if count2 > 0
                    Cv = 1.0
                    for l in 1:ne
                        Cv *= C[i[l], j[l]]
                    end
                    if Cv != 0
                        @. ϕv = 0.0
                        for xi = 1:n
                            @. mi = i - m[xi] + 1
                            @. mj = j - m[xi] + 1
                            for k = 1:ne
                                if 0 < mi[k] < 3 && 0 < mj[k] < 3
                                    ϕv[xi] += fv[xi, mi[k]] * fv[xi, mj[k]] / C[i[k], j[k]]
                                end
                            end
                        end
                        @. ρ += Ψ[count1] * Ψ[count2] * Cv * ϕv
                    end
                end
            end
            jtr[1] += 1
            for ℓ = 1:ne-1
                if jtr[ℓ] == 4
                    jtr[ℓ] = 1
                    jtr[ℓ+1] += 1
                end
            end
        end
    end

    return ρ
end


function density_bs_sp(x::Array{Float64}, ne::Int64, combBasis::Array{Array{Int64,1},1}, Ψ::Array{Float64,1}, ham::hamBs2d)
    L = ham.L
    C = ham.C
    dof = C.n
    N = Int64(sqrt(dof))
    h = (2.0 * L) / (N + 1)
    n = length(x)
    ρ = zeros(n, n)

    ix = floor.(Int64, (x .+ L) ./ h)
    ϕileft = ((ix .+ 1) .* h .- L .- x) ./ h
    ϕiright = (x .+ L .- ix .* h) ./ h

    jy = floor.(Int64, (y .+ L) ./ h)
    ϕjleft = ((jy .+ 1) .* h .- L .- y) ./ h
    ϕjright = (y .+ L .- jy .* h) ./ h

    fx = hcat(ϕileft, ϕiright)
    fy = hcat(ϕjleft, ϕjright)

    index = tensor2vec_bs(ne, dof, combBasis)

    i = zeros(Int, ne)
    mi = zeros(Int, ne)
    ni = zeros(Int, ne)
    j = zeros(Int, ne)
    mj = zeros(Int, ne)
    nj = zeros(Int, ne)
    js = zeros(Int, ne)
    ϕv = zeros(n, n)
    XY = zeros(Int, 9, ne)
    jtr = ones(Int64, ne)
    mix = zeros(Int, n, ne)
    niy = zeros(Int, n, ne)
    mjx = zeros(Int, n, ne)
    njy = zeros(Int, n, ne)

    for count1 = 1:length(combBasis)
        i = combBasis[count1]
        for l in 1:ne
            mi[l] = i[l] % N == 0 ? N : i[l] % N
            ni[l] = Int((i[l] - mi[l]) / N) + 1
            @views XY[:, l] = con_ij(mi[l], ni[l], N, 0)
            @. mix[:, l] = mi[l] - ix + 1
            @. niy[:, l] = ni[l] - jy + 1
        end
        @. jtr = 1
        while jtr[ne] <= 9
            for l = 1:ne
                j[l] = XY[jtr[l], l]
                js[l] = XY[jtr[l], l]
            end
            if !(0 in j)
                sort!(js)
                j1 = 1
                for i = 1:ne
                    j1 += (js[i] - 1) * (dof)^(i - 1)
                end
                count2 = index[j1]
                if count2 > 0
                    for l in 1:ne
                        mj[l] = j[l] % N == 0 ? N : j[l] % N
                        nj[l] = Int((j[l] - mj[l]) / N) + 1
                        @. mjx[:, l] = mj[l] - ix + 1
                        @. njy[:, l] = nj[l] - jy + 1
                    end
                    Cv = 1.0
                    for l in 1:ne
                        Cv *= C[i[l], j[l]]
                    end
                    if Cv != 0.0
                        @. ϕv = 0.0
                        for l in 1:ne
                            for i1 = 1:n
                                for i2 = 1:n
                                    if 0 < mix[i1, l] < 3 && 0 < niy[i2, l] < 3 && 0 < mjx[i1, l] < 3 && 0 < njy[i2, l] < 3
                                        ϕv[i2, i1] += fx[i1, mix[i1, l]] * fy[i2, niy[i2, l]] * fx[i1, mjx[i1, l]] * fy[i2, njy[i2, l]] / C[i[l], j[l]]
                                    end
                                end
                            end
                        end
                        @. ρ += Ψ[count1] * Ψ[count2] * Cv * ϕv
                    end
                end
            end

            jtr[1] += 1
            for ℓ = 1:ne-1
                if jtr[ℓ] == 10
                    jtr[ℓ] = 1
                    jtr[ℓ+1] += 1
                end
            end
        end
    end

    return ρ
end

density_bs_sp(x, wf::WaveFunctionBs_sp, ham::HamiltonianBoson) = density_bs_sp(x, wf.ne, wf.combBasis, wf.val, ham)
