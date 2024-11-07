using FastGaussQuadrature
using StaticArrays
using SparseArrays

export oneB2nd_over, oneB2nd_V, oneB2nd_lap
export twoB2nd_Vee_sp
export matrix_1_2body_2ndfem

"""
One-body matrix with P2 finite elements basis:
numerical quadrature with Gauss-Legendre points.
"""

function oneB2nd_over(L::Float64, N::Int; nx=4)
    h = 2.0 * L / N
    x, w_x = gausslegendre(nx)
    wt = w_x .* h / 2.0
    indi1 = Int[]
    indi2 = Int[]
    oneB_vals = Float64[]
    xx = zeros(length(x))
    ψ = zeros(3, length(x))
    for k in 1:N
        @. xx = -L + h * (k - 0.5 + x / 2)
        xl = -L + h * (k - 1)
        xm = -L + h * (k - 0.5)
        xr = -L + h * k
        @. ψ[1, :] = (xx - xm) * (xx - xr) / (h^2 / 2)
        @. ψ[2, :] = (xx - xl) * (xx - xr) / (-h^2 / 4)
        @. ψ[3, :] = (xx - xl) * (xx - xm) / (h^2 / 2)
        for j1 = 1:3, j2 = 1:3
            i1 = 2 * (k - 1) + j1 - 1
            i2 = 2 * (k - 1) + j2 - 1
            if i1 > 0 && i1 < 2N && i2 > 0 && i2 < 2N
                push!(indi1, i1)
                push!(indi2, i2)
                val = sum(@. ψ[j1, :] * ψ[j2, :] * wt)
                push!(oneB_vals, val)
            end
        end
    end
    return sparse(indi1, indi2, oneB_vals)
end

function oneB2nd_V(L::Float64, N::Int, V::Function; nx=4)
    h = 2.0 * L / N
    x, w_x = gausslegendre(nx)
    wt = w_x .* h / 2.0
    indi1 = Int[]
    indi2 = Int[]
    oneB_vals = Float64[]
    xx = zeros(length(x))
    ψ = zeros(3, length(x))
    pot = zeros(length(x))
    for k in 1:N
        @. xx = -L + h * (k - 0.5 + x / 2)
        xl = -L + h * (k - 1)
        xm = -L + h * (k - 0.5)
        xr = -L + h * k
        @. ψ[1, :] = (xx - xm) * (xx - xr) / (h^2 / 2)
        @. ψ[2, :] = (xx - xl) * (xx - xr) / (-h^2 / 4)
        @. ψ[3, :] = (xx - xl) * (xx - xm) / (h^2 / 2)
        @. pot = V(xx)
        # if k==2 println(pot) end
        for j1 = 1:3, j2 = 1:3
            i1 = 2 * (k - 1) + j1 - 1
            i2 = 2 * (k - 1) + j2 - 1
            if i1 > 0 && i1 < 2N && i2 > 0 && i2 < 2N
                push!(indi1, i1)
                push!(indi2, i2)
                val = sum(@. ψ[j1, :] * ψ[j2, :] * pot * wt)
                push!(oneB_vals, val)
            end
        end
    end
    return sparse(indi1, indi2, oneB_vals)
end


function oneB2nd_lap(L::Float64, N::Int; nx=3)
    h = 2.0 * L / N
    x, w_x = gausslegendre(nx)
    wt = w_x .* h / 2.0
    indi1 = Int[]
    indi2 = Int[]
    oneB_vals = Float64[]
    xx = zeros(length(x))
    g = zeros(3, length(x))
    for k in 1:N
        @. xx = -L + h * (k - 0.5 + x / 2)
        xl = -L + h * (k - 1)
        xm = -L + h * (k - 0.5)
        xr = -L + h * k
        @. g[1, :] = ((xx - xm) + (xx - xr)) / (h^2 / 2)
        @. g[2, :] = ((xx - xl) + (xx - xr)) / (-h^2 / 4)
        @. g[3, :] = ((xx - xl) + (xx - xm)) / (h^2 / 2)
        for j1 = 1:3, j2 = 1:3
            i1 = 2 * (k - 1) + j1 - 1
            i2 = 2 * (k - 1) + j2 - 1
            if i1 > 0 && i1 < 2N && i2 > 0 && i2 < 2N
                push!(indi1, i1)
                push!(indi2, i2)
                val = sum(@.g[j1, :] * g[j2, :] * wt)
                push!(oneB_vals, val)
            end
        end
    end
    return sparse(indi1, indi2, oneB_vals)
end

function twoB2nd_Vee_sp(L::Float64, N::Int, Vee::Function; nx=4, ny=4)
    h = 2.0 * L / N
    dof2 = 2N - 1
    indi1 = Int[]
    indi2 = Int[]
    twoB_vals = Float64[]
    p_x, w_x = gausslegendre(nx)
    p_y, w_y = gausslegendre(ny)
    wt = [w_x[i] * w_y[j] * h^2 / 4.0 for i = 1:nx for j = 1:ny]
    x = [p_x[i] for i = 1:nx for j = 1:ny]
    y = [p_y[j] for i = 1:nx for j = 1:ny]
    xx = zeros(length(x))
    yy = zeros(length(y))
    ψ = zeros(3, length(x))
    ϕ = zeros(3, length(x))
    xy = zeros(length(x))
    vees = zeros(length(x))
    for k = 1:N, ℓ = 1:N
        @. xx = -L + h * (k - 0.5 + x / 2)
        xl = -L + h * (k - 1)
        xm = -L + h * (k - 0.5)
        xr = -L + h * k
        @. ψ[1, :] = (xx - xm) * (xx - xr) / (h^2 / 2)
        @. ψ[2, :] = (xx - xl) * (xx - xr) / (-h^2 / 4)
        @. ψ[3, :] = (xx - xl) * (xx - xm) / (h^2 / 2)
        @. yy = -L + h * (ℓ - 0.5 + y / 2)
        yl = -L + h * (ℓ - 1)
        ym = -L + h * (ℓ - 0.5)
        yr = -L + h * ℓ
        @. ϕ[1, :] = (yy - ym) * (yy - yr) / (h^2 / 2)
        @. ϕ[2, :] = (yy - yl) * (yy - yr) / (-h^2 / 4)
        @. ϕ[3, :] = (yy - yl) * (yy - ym) / (h^2 / 2)
        @. xy = abs(xx - yy)
        @. vees = Vee(xy)
        for jx1 = 1:3, jy1 = 1:3, jx2 = 1:3, jy2 = 1:3
            ix1 = 2(k - 1) + jx1 - 1
            iy1 = 2(ℓ - 1) + jy1 - 1
            ix2 = 2(k - 1) + jx2 - 1
            iy2 = 2(ℓ - 1) + jy2 - 1
            if 0 < ix1 < 2N && 0 < iy1 < 2N && 0 < ix2 < 2N && 0 < iy2 < 2N
                i1 = (iy1 - 1) * dof2 + ix1
                i2 = (iy2 - 1) * dof2 + ix2
                push!(indi1, i1)
                push!(indi2, i2)
                val = sum(@. (ψ[jx1, :] * ϕ[jy1, :]) * (ψ[jx2, :] * ϕ[jy2, :]) * vees * wt)
                push!(twoB_vals, val)
            end
        end
    end
    return sparse(indi1, indi2, twoB_vals)
end

function matrix_1_2body_2ndfem(L::Float64, N::Int, vext::Function, vee::Function, nx::Int, ny::Int)

    # construct the 1-body matrix for Δ, external potential and overlap
    AΔ = oneB2nd_lap(L, N)
    AV = oneB2nd_V(L, N, vext; nx=nx)
    C = oneB2nd_over(L, N)
    # construct the 2-body matrix for electron-electron interaction v_ee
    Bee = twoB2nd_Vee_sp(L, N, vee; nx=nx, ny=ny)
    return AΔ, AV, C, Bee
end
