
export CDFCI_matfree_block_bs
# PARAMETERS
# A : 1-body operator, e.g., -Δ, v_ext
# B : 2-body operator, e.g., v_ee
# C : overlap
# c : initial vector
# k : length of block
# K : the corresponding vector index of the index set of selected determinants
# combBasis : the index set of selected determinants
# Ψ : the coefficient of selected determinants
# ϵ : tolerance of compression
# RETURNS
# E : the ground state energy of H
# Ψ : the ground state of H
#-------------------------------------------------------------------------------

# update several directions
function CDFCI_matfree_block_bs(Ψinit::WaveFunctionBs_sp, ham::HamiltonianBoson, k::Int64; max_iter=1000, ϵ=5.0e-7, tol=1.0e-3)

    ne = Ψinit.ne
    combBasis = Ψinit.combBasis
    Ψ = Ψinit.val
    c = copy(Ψinit.wfNP)
    K = c.nzind

    dof = ham.C.n
    n1 = binomial(dof, ne)
    n2 = (dof)^ne

    b, d = ham_column_bs(ne, combBasis, Ψ, ham) # H * c
    M1 = dot(c, d) # c' * d
    H1 = dot(c, b) # c' * b
    energy = H1 / M1
    j = sample(K, k, replace=false)# initial iteration coordinate 
    Dgk = zeros(k)
    DgE = zeros(2k)
    r = 0.0
    y = Float64[]
    num = Int64[]

    @printf("   step |     Energy  \n")

    for t = 1:max_iter

        # coordinate pick
        l = Int64[]
        for i = 1:k
            append!(l, ham_column_nonz_bs(ne, ham, j[i])) # Find the coordinates of the nonzero element of H[:,j[i]]
        end
        unique!(l)
        Dgl = zeros(length(l))
        @. Dgl = (b[l] / M1 - (d[l] * H1) / M1^2)
        Dgp = partialsortperm(abs.(Dgl), 1:k, rev=true)
        #Dgp = sortperm(abs.(Dgl), rev = true)[1:k]
        j = l[Dgp]
        Dgk = Dgl[Dgp]

        # test of convergence
        if norm(Dgk, Inf) < tol
            E = sample(1:n1, 2k, replace=false)
            E = map(x -> num_s2ns(dof, ne, x), E)
            @. DgE = abs(b[E] / M1 - (d[E] * H1) / M1^2)
            Ep = sortperm(abs.(DgE), rev=true)[1:k]
            j = E[Ep]
            e = DgE[Ep[1]]
            if abs(e) < tol && t > 10
                @printf "  %4.d  |   %.8f  \n" (t - 1) energy
                break
            end
            Dgk = DgE[Ep]
        end

        Dg = sparsevec(j, Dgk, n2)

        # update the stepsize
        # solve pβ^2 + qβ + s = 0 to get the minimizer
        Hg, Mg = ham_column_bs(ne, j, Dgk, ham) # H * Dg, M * Dg
        gH = dot(Dg, Hg) # Dg' * H * Dg
        gM = dot(Dg, Mg) # Dg' * M * Dg
        gd = dot(Dg, d) # Dg' * d
        gb = dot(Dg, b) # Dg' * b
        p = gH * gd - gM * gb
        q = gH * M1 - gM * H1
        s = gb * M1 - gd * H1
        if p == 0
            β = -s / q
        elseif q^2 - 4 * p * s >= 0
            β = (-q + sqrt(q^2 - 4 * p * s)) / 2p
        else
            β = -0.1
        end

        @. c[j] = c[j] + β * Dgk

        # compress b
        h = β .* Hg
        for i in h.nzind
            if b[i] != 0 || abs(h[i]) > ϵ
                b[i] += h[i]
            end
        end

        # compress d
        m = β .* Mg
        for i in m.nzind
            if d[i] != 0 || abs(m[i]) > ϵ
                d[i] += m[i]
            end
        end
        b[j], d[j] = ham_row_bs(j, ne, c, ham)

        H1 += 2 * β * dot(Dg, b) - β^2 * gH
        M1 += 2 * β * dot(Dg, d) - β^2 * gM

        energy = H1 / M1
        t % 10 == 0 && @printf "  %4.d  |   %.8f  \n" t energy

        push!(y, energy)
        push!(num, length(c.nzind))
    end
    @printf "  final iteration res : %.8f  \n" abs(y[end] - y[end-1])
    #return r, c#, step;
    #return y,num;
    return y, num, c
end