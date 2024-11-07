export HF_SCF, HF_12B

function gen_F_1d(N::Int64, x::Array{Float64,1}, wt::Array{Float64,1}, ρ::Array{Float64,2})
    # Generate F

    F = zeros(N)

    fl1 = zeros(length(x))
    fr1 = zeros(length(x))
    Potl1 = zeros(length(x))
    Potr1 = zeros(length(x))

    for i in 1:N
        fl1 .= (x .+ 1.0) / 2.0
        fr1 .= .-(x .- 1.0) / 2.0
        if 1 < i < N
            Potl1 .= ρ[i-1, i-1] .* fr1 .* fr1 .+ 2 * ρ[i-1, i] .* fl1 .* fr1 .+ ρ[i, i] .* fl1 .* fl1
            Potr1 .= ρ[i, i] .* fr1 .* fr1 .+ 2 * ρ[i, i+1] .* fl1 .* fr1 .+ ρ[i+1, i+1] .* fl1 .* fl1
        elseif i == 1
            Potl1 .= ρ[i, i] .* fl1 .* fl1
            Potr1 .= ρ[i, i] .* fr1 .* fr1 .+ 2 * ρ[i, i+1] .* fl1 .* fr1 .+ ρ[i+1, i+1] .* fl1 .* fl1
        else
            Potl1 .= ρ[i-1, i-1] .* fr1 .* fr1 .+ 2 * ρ[i-1, i] .* fl1 .* fr1 .+ ρ[i, i] .* fl1 .* fl1
            Potr1 .= ρ[i, i] .* fr1 .* fr1
        end
        F[i] = 4pi * (sum(fl1 .* Potl1 .* wt) + sum(fr1 .* Potr1 .* wt))
    end

    return F
end

function gen_AH_1d(N::Int, x::Array{Float64,1}, wt::Array{Float64,1}, W::Array{Float64,1})
    # generate AH

    indi1 = Int[]
    indi2 = Int[]
    values = Float64[]

    fl1 = zeros(length(x))
    fr1 = zeros(length(x))
    Potl1 = zeros(length(x))
    Potr1 = zeros(length(x))

    # i2 = i1
    for i1 in 1:N
        fl1 .= (x .+ 1.0) / 2.0
        fr1 .= .-(x .- 1.0) / 2.0
        if 1 < i1 < N
            Potl1 .= W[i1-1] .* fr1 .+ W[i1] .* fl1
            Potr1 .= W[i1] .* fr1 .+ W[i1+1] .* fl1
        elseif i1 == 1
            Potl1 .= W[i1] .* fl1
            Potr1 .= W[i1] .* fr1 .+ W[i1+1] .* fl1
        else
            Potl1 .= W[i1-1] .* fr1 .+ W[i1] .* fl1
            Potr1 .= W[i1] .* fr1
        end

        v = sum(fl1 .* fl1 .* Potl1 .* wt) + sum(fr1 .* fr1 .* Potr1 .* wt)
        push!(indi1, i1)
        push!(indi2, i1)
        push!(values, v)

        if i1 < N
            v = sum(fl1 .* fr1 .* Potr1 .* wt)
            push!(indi1, i1)
            push!(indi2, i1 + 1)
            push!(values, v)

            push!(indi1, i1 + 1)
            push!(indi2, i1)
            push!(values, v)
        end
    end
    return sparse(indi1, indi2, values)
end

function HF_SCF(ne::Int64, ham::ham1d; max_iter=100)

    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    alpha_lap = ham.alpha_lap
    L = ham.L

    AH = copy(C)
    N = C.n
    F = zeros(N)
    W = zeros(N)
    ρ1 = zeros(N, N)
    h = 2.0 * L / (N + 1)
    e = 1.0
    tol = 1e-8
    k1 = 0.7
    k2 = 1.0 - k1
    nx = 3

    # Gauss-Legendre points
    x, w_x = gausslegendre(nx)
    wt = w_x .* h / 2.0
    m = cld(ne, 2)
    #m = ne

    E1, U1 = eigs(AΔ, C, nev=m, which=:SR)
    U1 = Real.(U1)

    for i = 1:m
        ρ1 += U1[:,i] * U1[:,i]'
    end

    ρ = copy(ρ1)
    step = 0
    #Y = zeros(N,max_iter)
    for k = 1:max_iter
        if e < tol
            break
        end

        F = gen_F_1d(N, x, wt, ρ)

        W = (AΔ + C) \ F

        AH = gen_AH_1d(N, x, wt, W)

        H = 0.5 .* alpha_lap .* AΔ + AV + AH
        #H = 0.5 * alpha_lap * AΔ + AH

        E, U2 = eigs(H, C, nev=m, which=:SR)

        ρ2 = zeros(N, N)
        for i = 1:m
            ρ2 += U2[:,i] * U2[:,i]'
        end

        ρ = k1 .* ρ1 + k2 .* ρ2

        e = norm(ρ2 - ρ1)

        step += 1
        #Y[:,k] = [ρ[i,i] for i = 1:N]

        ρ1 = ρ
        #println("  step : $(step),    e : $(e)")
    end
    println("  step : $(step),    e : $(e)")
    return AH, e
end

function HF_12B(ne::Int64, U::Array{Float64,2}, ham::ham1d)
    AΔ = ham.AΔ
    AV = ham.AV
    C = ham.C
    B = ham.Bee
    alpha_lap = ham.alpha_lap

    tol = 1e-8
    N = C.n
    m1 = cld(ne, 2)
    m2 = ne - m1

    A = 0.5 * alpha_lap * AΔ + AV
    Af = zeros(Float64, m1, m1)
    Cf = zeros(Float64, m1, m1)
    Bf = zeros(Float64, m1, m1, m1, m1)

    for i = 1:m1, j = 1:m1
        for k = 1:N, l in [k - 1, k, k + 1]
            if 0 < l < N + 1
                Af[i, j] += U[k, i] * U[l, j] * A[k, l]
                Cf[i, j] += U[k, i] * U[l, j] * C[k, l]
            end
        end
    end

    for i1 = 1:m1, i2 = 1:m1, j1 = 1:m1, j2 = 1:m1
        for k1 = 1:N, k2 = 1:N
            for l1 in [k1 - 1, k1, k1 + 1], l2 in [k2 - 1, k2, k2 + 1]
                if 0 < l1 < N + 1 && 0 < l2 < N + 1
                    Bf[i1, i2, j1, j2] += U[k2, i2] * U[l1, j1] * U[l2, j2] * B[k1, k2, l1, l2]
                end
            end
        end
    end

    return Af, Cf, Bf
end


