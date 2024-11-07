#-------------------------------------------------------------------------------
# Solving Schrödinger Equation with Hatree Fock Method Based on Finite Element Discretization
# ignore v_ee
# {ϕ_vs} v = 1,...,N;s = 0,1
# Ψ = |φ_1,0 ...φ_[ne/2],0 φ1,1...φ_ne-[ne/2],1>
# φ_ks = Σ_(v=1...N)u_v,ks * ϕ_vs
# Â*φi = ϵi*φi -> A*Ui = ϵi*C*Ui
#-------------------------------------------------------------------------------
# PARAMETERS
# ne: number of electrons/particles
# A : 1-body operator, e.g., -Δ, v_ext
# C : overlap
# RETURNS
# c0 : the ground state energy without v_ee
#-------------------------------------------------------------------------------
export HF

function HFtoFCI(ne::Int64, N::Int64, U::Array{Float64,2})
    ind = Int64[]
    val = Float64[]

    m = zeros(Int64, 2)
    m[1] = cld(ne, 2)
    m[2] = ne - m[1]

    basis1body = 1:N
    combBasis = map(x -> collect(combinations(basis1body, x)), m)
    b = zeros(length(combBasis[1]), 2)

    # loop for the spin
    for i = 1:2
        v = 1:m[i]
        p = collect(permutations(v))[:]
        ε = (-1) .^ [parity(p[l]) for l = 1:length(p)]
        ij = zeros(Int64, m[i])
        for j = 1:length(combBasis[i])
            @views ij = combBasis[i][j]
            for k = 1:length(p)
                a = ε[k]
                for l = 1:m[i]
                    ik = ij[p[k][l]]
                    a *= U[ik, l]
                end
                b[j, i] += a
            end
        end
    end

    # combine s=0 with s=1
    for i = 1:length(combBasis[1])
        for j = 1:length(combBasis[2])
            ij = vcat(combBasis[1][i], combBasis[2][j] .+ N)
            push!(ind, seq2num_ns(2N, ne, ij))
            push!(val, b[i, 1] * b[j, 2])
        end
    end

    c0 = sparsevec(ind, val, (2N)^ne)
    return c0
end

function HF_hm(ne::Int64, U::Array{Float64,2}, ham::Hamiltonian)
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    coulomb_which2 = collect(combinations(v, 2))
    valH = 0.0
    valM = 0.0
    m1 = cld(ne, 2)
    m2 = ne - m1

    i = vcat(collect(1:m1), collect(1:m2))
    s = vcat(zeros(Int, m1), ones(Int, m2))
    jp = zeros(Int, ne)
    tp = zeros(Int, ne)
    Cp = zeros(Float64, ne)

    Af, Cf, Bf = HF_12B(ne, U, ham)

    for k = 1:length(p)
        Av = 0.0
        Bv = 0.0
        for l in 1:ne
            tp[l] = s[p[k][l]]
            jp[l] = i[p[k][l]]
            Cp[l] = Cf[i[l], jp[l]]
        end
        Cv = prod(Cp)
        if s == tp && Cv != 0.0
            for l in 1:ne
                Av += Af[i[l], jp[l]] / Cp[l]
            end
            Av *= Cv

            for l = 1:length(coulomb_which2)
                ca = coulomb_which2[l][1]
                cb = coulomb_which2[l][2]
                Bv += Bf[i[ca], i[cb], jp[ca], jp[cb]] / (Cp[ca] * Cp[cb])
            end
            Bv *= Cv

            valH += ε[k] * (Av + Bv)
            valM += ε[k] * Cv
        end
    end

    return valH, valM
end

function HF(ne::Int64, ham::Hamiltonian; max_iter = 100)
    N = ham.C.n

    println("SCF time : ")
    @time AH, e = HF_SCF(ne, ham; max_iter = max_iter)
    m1 = cld(ne, 2)
    m2 = ne - m1

    Hf = 0.5 * ham.alpha_lap * ham.AΔ + ham.AV + AH
    E, U = eigs(Hf, ham.C, nev=m1, which=:SR)
    U = Real.(U)

    println("Turn into FCI time : ")
    @time c0 = HFtoFCI(ne, N, U)
    wfsp = WaveFunction_sp(ne, N, c0) 

    println("Compute energy time : ")
    @time valH, valM = HF_hm(ne, U, ham) ./ (norm(c0))^2

    return wfsp, U, valH, valM
end
