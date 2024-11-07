export HF_SCF, HF_12B

function gen_F_2d(Nx::Int64, Ny::Int64, fv::Array{Float64,2}, wt::Array{Float64,1}, ρ::Array{Float64,2})
    # Generate F

    F = zeros(Nx*Ny)

    for j = 1:Ny+1 #loop over y dof
        ylabel = [j - 1, j - 1, j, j] # label for the dof on the edges
        for i = 1:Nx+1 #loop over x dof
            xlabel = [i - 1, i, i - 1, i] # label for the dof on the edges
        
            # calculate stiff elements
            for t = 1:4
                jx = xlabel[t]
                jy = ylabel[t]
                mj = jx + (jy - 1) * Nx
                if 0 < jx < Nx+1 && 0 < jy < Ny+1
                    for k = 1:4, l = 1:4
                        kx = xlabel[k]
                        ky = ylabel[k]
                        lx = xlabel[l]
                        ly = ylabel[l]
                        mk = kx + (ky - 1) * Nx
                        ml = lx + (ly - 1) * Nx
                        if 0 < kx < Nx+1 && 0 < ky < Ny+1 && 0 < lx < Nx+1 && 0 < ly < Ny+1 #checking admissible basis functions
                            F[mj] += ρ[mk, ml] * sum(fv[:, k] .* fv[:, l] .* fv[:, t] .* wt)
                        end
                    end
                end
            end
        end
    end
    F = 4pi .* F

    return F
end

function gen_AH_2d(Nx::Int64, Ny::Int64, fv::Array{Float64,2}, wt::Array{Float64,1}, W::Array{Float64,1})
    # generate AH
    dof = Nx*Ny
    mat = zeros(dof, dof)

    for j = 1:Ny+1 #loop over y dof
        ylabel = [j - 1, j - 1, j, j] # label for the dof on the edges
        for i = 1:Nx+1 #loop over x dof
            xlabel = [i - 1, i, i - 1, i] # label for the dof on the edges

            # calculate stiff elements
            for k = 1:4, l = 1:4
                ix = xlabel[k]
                iy = ylabel[k]
                jx = xlabel[l]
                jy = ylabel[l]
                mi = ix + (iy - 1) * Nx
                mj = jx + (jy - 1) * Nx
                if 0 < ix < Nx+1 && 0 < iy < Ny+1 && 0 < jx < Nx+1 && 0 < jy < Ny+1 #checking admissible basis functions
                    for w = 1:4
                        wx = xlabel[w]
                        wy = ylabel[w]
                        mw = wx + (wy - 1) * Nx
                        if 0 < wx < Nx+1 && 0 < wy < Ny+1
                            mat[mi, mj] += W[mw] * sum(fv[:, k] .* fv[:, l] .* fv[:, w] .* wt)
                        end
                    end
                end
            end
        end
    end

    return sparse(mat)
end

function HF_SCF(ne::Int64, ham::ham2d; max_iter=100)
    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    alpha_lap = ham.alpha_lap
    Lx = ham.L[1]
    Ly = ham.L[2]
    Nx = ham.N[1] - 1
    Ny = ham.N[2] - 1

    AH = copy(C)
    dof = C.n
    hx = (2.0 * Lx) / (Nx + 1)
    hy = (2.0 * Ly) / (Ny + 1)
    W = zeros(dof)
    ρ1 = zeros(dof, dof)
    ρ2 = zeros(dof, dof)
    e = 1.0
    tol = 1e-8
    k1 = 0.7
    k2 = 1.0 - k1
    nx = 3
    ny = 3

    # Gauss-Legendre points
    p_x, w_x = gausslegendre(nx)
    p_y, w_y = gausslegendre(ny)

    w_X = [w_x[i] for i = 1:nx for j = 1:ny]
    w_Y = [w_y[j] for i = 1:nx for j = 1:ny]
    wt = w_X .* w_Y .* hx * hy / 4.0

    x = [p_x[i] for i = 1:nx for j = 1:ny]
    y = [p_y[j] for i = 1:nx for j = 1:ny]

    # value and gradient of basis functions at (x,y)
    # the hat functions are separated into 2 parts,
    # and we compute the 4 possibilities of product of 2
    fv = hcat((x .- 1.0) .* (y .- 1.0) / 4.0,
        .-(x .+ 1.0) .* (y .- 1.0) / 4.0,
        .-(x .- 1.0) .* (y .+ 1.0) / 4.0,
        (x .+ 1.0) .* (y .+ 1.0) / 4.0)

    m = cld(ne, 2)

    E1, U1 = eigs(AΔ, C, nev=m, which=:SR)
    U1 = Real.(U1)

    for i = 1:m
        ρ1 += U1[:, i] * U1[:, i]'
    end

    ρ = copy(ρ1)
    step = 0
    #Y = zeros(N,max_iter)
    for k = 1:max_iter
        if e < tol
            break
        end

        F = gen_F_2d(Nx, Ny, fv, wt, ρ)

        W = (AΔ + C) \ F

        AH = gen_AH_2d(Nx, Ny, fv, wt, W)

        H = 0.5 .* alpha_lap .* AΔ + AV + AH

        E, U2 = eigs(H, C, nev=m, which=:SR)
        U2 = Real.(U2)

        @. ρ2 = 0.0
        for i = 1:m
            ρ2 += U2[:, i] * U2[:, i]'
        end

        ρ = k1 .* ρ1 + k2 .* ρ2

        e = norm(ρ2 - ρ1)

        step += 1

        ρ1 = ρ
        #println("  step : $(step),    e : $(e)")
    end
    println("  step : $(step),    e : $(e)")
    return AH, e
end

function HF_12B(ne::Int64, U::Array{Float64,2}, ham::ham2d)
    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    B = ham.Bee
    Nx = ham.N[1] - 1
    Ny = ham.N[2] - 1
    alpha_lap = ham.alpha_lap

    tol = 1e-8
    dof = C.n
    m1 = cld(ne, 2)
    m2 = ne - m1

    A = 0.5 * alpha_lap * AΔ + AV
    Af = zeros(Float64, m1, m1)
    Cf = zeros(Float64, m1, m1)
    Bf = zeros(Float64, m1, m1, m1, m1)

    for i = 1:m1, j = 1:m1
        for k = 1:dof
            ki = k % Nx == 0 ? Nx : k % Nx
            kj = Int((k - ki) / Nx) + 1
            conk = con_ij(ki, kj, Nx, Ny, 0)
            for l in conk
                if l > 0
                    Af[i, j] += U[k, i] * U[l, j] * A[k, l]
                    Cf[i, j] += U[k, i] * U[l, j] * C[k, l]
                end
            end
        end
    end

    for i1 = 1:m1, i2 = 1:m1, j1 = 1:m1, j2 = 1:m1
        for k1 = 1:dof, k2 = 1:dof
            k1i = k1 % Nx == 0 ? Nx : k1 % Nx
            k1j = Int((k1 - k1i) / Nx) + 1
            conk1 = con_ij(k1i, k1j, Nx, Ny, 0)
            k2i = k2 % Nx == 0 ? Nx : k2 % Nx
            k2j = Int((k2 - k2i) / Nx) + 1
            conk2 = con_ij(k2i, k2j, Nx, Ny, 0)
            for l1 in conk1, l2 in conk2
                if l1 > 0 && l2 > 0
                    Bf[i1, i2, j1, j2] += U[k1, i1] * U[k2, i2] * U[l1, j1] * U[l2, j2] * B[k1+(k2-1)*dof, l1+(l2-1)*dof]
                end
            end
        end
    end

    return Af, Cf, Bf
end

