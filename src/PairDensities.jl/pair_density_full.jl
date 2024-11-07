#-------------------------------------------------------------------------------
#   compute 1-body and 2-body properties from the ne-body wavefunction
#-------------------------------------------------------------------------------
export density_coef, pair_density_coef, density, pair_density, pair_density_spin_coef, pair_density_spin
# PARAMETERS
# n : the number of electrons
# Ψ : ne-body wavefunction
# C : 1-body overlap
#
# RETURN
# ρ  : density
# ρ2 : pair density
# γ  : one-body reduced density matrix
# γ2 : two-body reduced density matrix
#-------------------------------------------------------------------------------


# return a tri-diagonal matrix that store the coefficients for ϕᵢ(x)⋅ϕⱼ(x)
function density_coef(n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
    basis1body = 1:2*N
    combBasis = collect(combinations(basis1body, n))
    # computate the permutations and paritiy
    v = 1:n
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            ik = seq2num_ns(2N, n, ij[p[k]])
            Ψtensor[ik] = Ψ[j] * ε[k]
        end
    end
    # integrate the n-1 variable out to obtain the coefficients of ρ
    mass = overlap(n - 1, C)
    indrow = [1:N; 1:N-1; 2:N]
    indcol = [1:N; 2:N; 1:N-1]
    val = zeros(Float64, 3 * N - 2)
    for k = 1:2*N
        sptr = zeros(Int, n - 1, 1)
        for s = 1:2^(n-1)
            sp = sptr * N
            uk = getindex(Ψtensor, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
            ukvec = reshape(uk, N^(n - 1), 1)[:]
            if k <= N
                val[k] += dot(ukvec, mass, ukvec)
            else
                val[k-N] += dot(ukvec, mass, ukvec)
            end
            if k < N
                uk_right = getindex(Ψtensor, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
                ukvec_right = reshape(uk_right, N^(n - 1), 1)[:]
                val[N+k] += dot(ukvec, mass, ukvec_right)
                val[2*N+k-1] += dot(ukvec, mass, ukvec_right)
            end
            if k > N && k < 2 * N
                uk_right = getindex(Ψtensor, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
                ukvec_right = reshape(uk_right, N^(n - 1), 1)[:]
                val[k] += dot(ukvec, mass, ukvec_right)
                val[N+k-1] += dot(ukvec, mass, ukvec_right)
            end
            # adjust sptr
            sptr[1] += 1
            if n >= 3
                for ℓ = 1:n-2
                    if sptr[ℓ] == 2
                        sptr[ℓ] = 0
                        sptr[ℓ+1] += 1
                    end
                end
            end # end if
        end
    end
    ρcoef = sparse(indrow, indcol, val)
    return ρcoef * n / length(p)
end

# compute the value ρ(x) with the coefficients
function density(x::Float64, L::Float64, coef::SparseMatrixCSC{Float64,Int64})
    val = 0.0
    N = coef.n
    h = (2.0 * L) / (N + 1)
    j = floor(Int64, (x + L) / h)
    ϕleft = ((j + 1) * h - L - x) / h
    ϕright = (x + L - j * h) / h
    if j > 0 && j < N + 1
        val += ϕleft^2 * coef[j, j]
    end
    if j < N
        val += ϕright^2 * coef[j+1, j+1]
    end
    if j > 0 && j < N
        val += ϕleft * ϕright * coef[j, j+1] + ϕleft * ϕright * coef[j+1, j]
    end
    return val
end

density(x::Array{Float64}, L::Float64, coef::SparseMatrixCSC{Float64,Int64}) =
    [density(x[i], L, coef) for i = 1:length(x)]

density(x::Array{Float64}, n::Int64, Ψ::Array{Float64,1}, ham::ham1d) = density(x, ham.L, density_coef(n, Ψ, ham.C))


# compute the coefficients for pair pair_density
function pair_density_coef(n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
    basis1body = [i for i = 1:2*N]
    combBasis = collect(combinations(basis1body, n))
    # computate the permutations and paritiy
    v = [i for i = 1:n]
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            ik = seq2num_ns(2N, n, ij[p[k]])
            Ψtensor[ik] = Ψ[j] * ε[k]
        end
    end
    # integrate the n-2 variable out to obtain the coefficients of ρ2
    # coefficients stored in a (2N+3)×(2N+3) matrix: N+2 for ϕ_iϕ_i and (N+1) for ϕ_iϕ_i+1
    coef = zeros(Float64, 2 * N + 3, 2 * N + 3)
    # only necessary to perform integration for more than 2-electron systems
    if n == 2
        for j = 1:2*N, k = 1:2*N
            jp = j < N + 1 ? j : j - N
            kp = k < N + 1 ? k : k - N
            coef[jp+1, kp+1] += Ψtensor[j, k]^2 #i1=i2,j1=j2
            if jp < N
                coef[N+2+jp+1, kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k] #i2=i1+1,j1=j2
            end
            if kp < N
                coef[jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j, k+1] #i1=i2,j2=j1+1
            end
            if jp < N && kp < N
                coef[N+2+jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k+1] #i2=i1+1,j2=j1+1
            end
        end
    else
        mass = overlap(n - 2, C)
        for j = 1:2*N, k = 1:2*N
            jp = j < N + 1 ? j : j - N
            kp = k < N + 1 ? k : k - N
            sptr = zeros(Int, n - 2, 1)
            for s = 1:2^(n-2)
                sp = sptr * N
                u = getindex(Ψtensor, j, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                uvec = reshape(u, N^(n - 2), 1)[:]
                coef[jp+1, kp+1] += dot(uvec, mass, uvec)
                if jp < N
                    u_xr = getindex(Ψtensor, j + 1, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_xr = reshape(u_xr, N^(n - 2), 1)[:]
                    coef[N+2+jp+1, kp+1] += dot(uvec_xr, mass, uvec)
                end
                if kp < N
                    u_yr = getindex(Ψtensor, j, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_yr = reshape(u_yr, N^(n - 2), 1)[:]
                    coef[jp+1, N+2+kp+1] += dot(uvec_yr, mass, uvec)
                end
                if jp < N && kp < N
                    coef[N+2+jp+1, N+2+kp+1] += dot(uvec_xr, mass, uvec_yr)
                end
                # adjust sptr
                sptr[1] += 1
                if n >= 4
                    for ℓ = 1:n-3
                        if sptr[ℓ] == 2
                            sptr[ℓ] = 0
                            sptr[ℓ+1] += 1
                        end
                    end
                end # end if
            end
        end
    end # end if n == 2
    return coef * n * (n - 1) / 2 / length(p)
end

# compute the value ρ2(x) with the coefficients
function pair_density(x::Float64, y::Float64, L::Float64, coef::Array{Float64,2})
    val = 0.0
    N = floor(Int, (size(coef)[1] - 3) / 2)
    h = (2.0 * L) / (N + 1)
    j = floor(Int64, (x + L) / h)
    ϕx_l = ((j + 1) * h - L - x) / h
    ϕx_r = (x + L - j * h) / h
    vecx = [ϕx_l * ϕx_l, ϕx_l * ϕx_r, ϕx_r * ϕx_l, ϕx_r * ϕx_r]
    k = floor(Int64, (y + L) / h)
    ϕy_l = ((k + 1) * h - L - y) / h
    ϕy_r = (y + L - k * h) / h
    vecy = [ϕy_l * ϕy_l, ϕy_l * ϕy_r, ϕy_r * ϕy_l, ϕy_r * ϕy_r]
    if j < N + 1 && k < N + 1
        mat_coef = [coef[j+1, k+1] coef[j+1, N+2+k+1] coef[j+1, N+2+k+1] coef[j+1, k+2]
            coef[N+2+j+1, k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, k+2]
            coef[N+2+j+1, k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, k+2]
            coef[j+2, k+1] coef[j+2, N+2+k+1] coef[j+2, N+2+k+1] coef[j+2, k+2]
        ]
        val += vecx' * mat_coef * vecy
    end
    return val
end

pair_density(xy::Array{Float64,2}, L::Float64, coef::Array{Float64,2}) =
    [pair_density(xy[i, 1], xy[j, 2], L, coef) for i = 1:size(xy, 1) for j = 1:size(xy, 1)]

pair_density(xy::Array{Float64,2}, L::Float64, n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64}) = pair_density(xy, L, pair_density_coef(n, Ψ, C))


# compute the one-body reduced density matrix from Ψ
function one_body_DM(Ψ::Array{Float64,1})

end

# compute the coefficients for pair pair_density_spin
function pair_density_spin_coef(n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64}, s1::Int64, s2::Int64)
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
    basis1body = [i for i = 1:2*N]
    combBasis = collect(combinations(basis1body, n))
    # computate the permutations and paritiy
    v = [i for i = 1:n]
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            ik = seq2num_ns(2N, n, ij[p[k]])
            Ψtensor[ik] = Ψ[j] * ε[k]
        end
    end
    # integrate the n-2 variable out to obtain the coefficients of ρ2
    # coefficients stored in a (2N+3)×(2N+3) matrix: N+2 for ϕ_iϕ_i and (N+1) for ϕ_iϕ_i+1
    coef = zeros(Float64, 2 * N + 3, 2 * N + 3)
    # only necessary to perform integration for more than 2-electron systems
    if n == 2
        for jp = 1:N, kp = 1:N
            j = jp + s1 * N
            k = kp + s2 * N
            coef[jp+1, kp+1] += Ψtensor[j, k]^2 #i1=i2,j1=j2
            if jp < N
                coef[N+2+jp+1, kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k] #i2=i1+1,j1=j2
            end
            if kp < N
                coef[jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j, k+1] #i1=i2,j2=j1+1
            end
            if jp < N && kp < N
                coef[N+2+jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k+1] #i2=i1+1,j2=j1+1
            end
        end
    else
        mass = overlap(n - 2, C)
        for jp = 1:N, kp = 1:N
            j = jp + s1 * N
            k = kp + s2 * N
            sptr = zeros(Int, n - 2, 1)
            for s = 1:2^(n-2)
                sp = sptr * N
                u = getindex(Ψtensor, j, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                uvec = reshape(u, N^(n - 2), 1)[:]
                coef[jp+1, kp+1] += dot(uvec, mass, uvec)
                if jp < N
                    u_xr = getindex(Ψtensor, j + 1, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_xr = reshape(u_xr, N^(n - 2), 1)[:]
                    coef[N+2+jp+1, kp+1] += dot(uvec_xr, mass, uvec)
                end
                if kp < N
                    u_yr = getindex(Ψtensor, j, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_yr = reshape(u_yr, N^(n - 2), 1)[:]
                    coef[jp+1, N+2+kp+1] += dot(uvec_yr, mass, uvec)
                end
                if jp < N && kp < N
                    coef[N+2+jp+1, N+2+kp+1] += dot(uvec_xr, mass, uvec_yr)
                end
                # adjust sptr
                sptr[1] += 1
                if n >= 4
                    for ℓ = 1:n-3
                        if sptr[ℓ] == 2
                            sptr[ℓ] = 0
                            sptr[ℓ+1] += 1
                        end
                    end
                end # end if
            end
        end
    end # end if n == 2
    return coef * n * (n - 1) / 2 / length(p)
end

pair_density_spin(xy::Array{Float64,2}, s1::Int64, s2::Int64, L::Float64, n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64}) = pair_density(xy, L, pair_density_spin_coef(n, Ψ, C, s1, s2))

pair_density_spin(xy::Array{Float64,2}, s1::Int64, s2::Int64, Ψ::WaveFunction_full, ham::ham1d) = pair_density_spin(xy, s1, s2, ham.L, Ψ.ne, Ψ.wf, ham.C)

#------------------------------------------------------------------------------------------
# density for 2D
# compute the value ρ(r), r=(x,y) with the coefficients
function density(x::Float64, y::Float64, Lx::Float64, Ly::Float64, Nx::Int64, Ny::Int64, coef::SparseMatrixCSC{Float64,Int64})
    val = 0.0
    Nx = Nx - 1
    Ny = Ny - 1
    hx = (2.0 * Lx) / (Nx + 1)
    hy = (2.0 * Ly) / (Ny + 1)

    i = floor(Int64, (x + Lx) / hx)
    ϕileft = ((i + 1) * hx - Lx - x) / hx
    ϕiright = (x + Lx - i * hx) / hx

    j = floor(Int64, (y + Ly) / hy)
    ϕjleft = ((j + 1) * hy - Ly - y) / hy
    ϕjright = (y + Ly - j * hy) / hy

    if i > 0 && i < Nx + 1
        if j > 0 && j < Ny + 1
            val += ϕileft^2 * ϕjleft^2 * coef[i+(j-1)*Nx, i+(j-1)*Nx]
        elseif j < Ny
            val += ϕileft^2 * ϕjright^2 * coef[i+j*Nx, i+j*Nx]
        elseif j > 0 && j < Ny
            val += ϕileft^2 * (ϕjleft * ϕjright * coef[i+(j-1)*Nx, i+j*Nx] + ϕjleft * ϕjright * coef[i+j*Nx, i+(j-1)*Nx])
        end
    elseif i < Nx
        if j > 0 && j < Ny + 1
            val += ϕiright^2 * ϕjleft^2 * coef[i+1+(j-1)*Nx, i+1+(j-1)*Nx]
        elseif j < Ny
            val += ϕiright^2 * ϕjright^2 * coef[i+1+j*Nx, i+1+j*Nx]
        elseif j > 0 && j < Ny
            val += ϕiright^2 * (ϕjleft * ϕjright * coef[i+1+(j-1)*Nx, i+1+j*Nx] + ϕjleft * ϕjright * coef[i+1+j*Nx, i+1+(j-1)*Nx])
        end
    elseif i > 0 && i < Nx
        if j > 0 && j < Ny + 1
            val += ϕileft * ϕiright * ϕjleft^2 * coef[i+(j-1)*Nx, i+1+(j-1)*Nx] + ϕileft * ϕiright * ϕjleft^2 * coef[i+1+(j-1)*Nx, i+(j-1)*Nx]
        elseif j < Ny
            val += ϕileft * ϕiright * ϕjright^2 * coef[i+j*Nx, i+1+j*Nx] + ϕileft * ϕiright * ϕjright^2 * coef[i+1+j*Nx, i+j*Nx]
        elseif j > 0 && j < Ny
            val += ϕileft * ϕiright * (ϕjleft * ϕjright * coef[i+(j-1)*Nx, i+1+j*Nx] + ϕjleft * ϕjright * coef[i+j*Nx, i+1+(j-1)*Nx]) + ϕileft * ϕiright * (ϕjleft * ϕjright * coef[i+1+(j-1)*Nx, i+j*Nx] + ϕjleft * ϕjright * coef[i+1+j*Nx, i+(j-1)*Nx])
        end
    end

    return val
end

density(xy::Vector{Vector{Float64}}, Lx::Float64, Ly::Float64, Nx::Int64, Ny::Int64, coef::SparseMatrixCSC{Float64,Int64}) =
    [density(xy[1][i], xy[2][j], Lx, Ly, Nx, Ny, coef) for i = 1:length(xy[1]) for j = 1:length(xy[2])]

density(xy::Vector{Vector{Float64}}, n::Int64, Ψ::Array{Float64,1}, ham::ham2d) = density(xy, ham.L[1], ham.L[2], ham.N[1], ham.N[2], density_coef(n, Ψ, ham.C))
