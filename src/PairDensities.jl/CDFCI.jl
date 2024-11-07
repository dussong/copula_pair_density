using PolynomialRoots
using StatsBase

export CDFCI, CDFCI_block
# PARAMETERS
# c : initial vector
# H : ne-body Hamiltonian -1/2⋅∑ᵢ Δ_{xᵢ} + ∑ᵢ V(xᵢ) + ∑_{i<j} v_ee(xᵢ-xⱼ)
# M : overlap
# RETURNS
# E : the ground state energy
# Ψ : the ground state eigenfunction
#-------------------------------------------------------------------------------

# coordinate pick
function det_select(B::Float64,D::Float64,b::Array{Float64,1},d::Array{Float64,1},l::Array{Int64,1})
    a0 = 0.
    j = 1

    for i = 1 : length(l)
        if b[l[i]] != 0 #|| d[l[i]] != 0
            a = 2*(b[l[i]]/D - (d[l[i]] * B)/D^2)
            if abs(a) > abs(a0)
                a0 = a
                j = l[i]
            end
        end
    end

    return j,a0;
end

# update one determinant every iteration
function CDFCI(c::Array{Float64,1}, H::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64,Int64}, max_iter::Int64, ϵ::Float64;tol = 1.0e-3)
    n = H.n
    H = Array(H)
    M = Array(M)
    b = zeros(n)
    d = zeros(n)

    b = H * c
    d = M * c
    M1 = c' * d
    H1 = c' * b
    energy = H1 / M1
    j = findmax(b)[2] # initial coordinate
    r = 0.
    step = 0
    j1 = 0
    y = Float64[]
    report(t, energy) = @info "$t-th iteration: energy = $energy"

    for t = 1 : max_iter
        # coordinate pick
        l = findall(!iszero,H[:,j])
        j,a0 = det_select(H1,M1,b,d,l)

        if j == j1
            j = rand(vcat(collect(2:j-1),collect(j+1:n)))
            a0 = 2*(b[j]/M1-(d[j] * H1)/M1^2)
        end

        # test of convergence
        if abs(a0) < tol
            E = sample(collect(1:n), 40, replace = false)
            j0,e = det_select(H1,M1,b,d,E)
            if abs(e) < tol && t > 10
                break;
            else
                j = j0;
            end
        end

        j1 = j

        # update the stepsize
        # solve pα^2 + qα + s = 0 to get the minimizer
        p = H[j,j] * d[j] - M[j,j] * b[j]
        q = H[j,j] * M1 - M[j,j] * H1
        s = b[j] * M1 - d[j] * H1

        if p == 0.
            α = -s / q
        elseif q^2 - 4 * p * s >= 0.
            α = (-q + sqrt(q^2 - 4 * p * s)) / 2p
        else
            α = -0.1
        end

        c[j] = c[j] + α

        # compress b
        for i = 1:n
            h = α * H[i,j]
            if h != 0.
                if b[i] != 0. || abs(h) > ϵ
                    b[i] += h
                end
            end
        end
        b[j] = dot(H[j,:],c)

        # compress d
        for i = 1:n
            m = α * M[i,j]
            if m != 0.
                if d[i] != 0. || abs(m) > ϵ
                    d[i] += m
                end
            end
        end
        d[j] = dot(M[j,:],c)

        #=
        b += α.* H[:,j]
        d += α.* M[:,j]
        =#

        H1 += 2α*b[j] - α^2*H[j,j] # c'*H*c
        M1 += 2α*d[j] - α^2*M[j,j] # c'*M*c

        energy = H1 / M1
        t % 10 == 0 && report(t, energy)

        #push!(y,H1/M1)
    end
    r = H1/M1

    return r, c;
end

# update k determinant every iteration
function CDFCI_block(k::Int64, c::Array{Float64,1},
            H::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64,Int64},
            max_iter::Int64, ϵ::Float64;tol = 1.0e-3)

    n = H.n
    H = Array(H)
    M = Array(M)
    b = zeros(n)
    d = zeros(n)

    b = H * c
    d = M * c
    M1 = c' * d
    H1 = c' * b
    energy = H1 / M1
    j = sample(findall(!iszero,c), k, replace = false)# initial coordinate
    DgE = zeros(2k)
    r = 0.
    step = 0
    y = Float64[]
    report(t, energy) = @info "$t-th iteration: energy = $energy"

    for t = 1 : max_iter
        # coordinate pick
        Dg = zeros(n) # Gradient of corresponding direction
        l = Int64[]
        for i = 1:k
            append!(l, findall(!iszero,H[:,j[i]])) # Find the coordinates of the nonzero element of H[:,j[i]]
        end
        unique!(l)
        Dgl = zeros(length(l))
        @. Dgl = (b[l] / M1 - (d[l] * H1) / M1^2)
        Dgp = partialsortperm(abs.(Dgl), 1:k, rev=true)
        #Dgp = sortperm(abs.(Dgl), rev = true)[1:k]
        j = l[Dgp]
        Dg[j] = Dgl[Dgp]

        # test of convergence
        if norm(Dg,Inf) < tol
            E = sample(1:n, 2k, replace = false)
            @. DgE = abs(b[E] / M1 - (d[E] * H1) / M1^2)
            Ep = partialsortperm(abs.(DgE), 1:k, rev=true)
            j = E[Ep]
            e = DgE[Ep[1]]
            if abs(e) < tol && t > 10
                report(t - 1, energy)
                break
            end
            Dg[j] = DgE[Ep]
        end

        # update the stepsizes
        # solve pα^2 + qα + s = 0 to get the minimizer
        gH = dot(Dg,H,Dg) # Dg' * H * Dg
        gM = dot(Dg,M,Dg) # Dg' * M * Dg
        gd = dot(Dg,d) # Dg' * d
        gb = dot(Dg,b) # Dg' * b
        p = gH * gd - gM * gb
        q = gH * M1 - gM * H1
        s = gb * M1 - gd * H1
        if p == 0.
            α = -s / q
        elseif q^2 - 4 * p * s >= 0.
            α = (-q + sqrt(q^2 - 4 * p * s)) / 2p
        else
            α = -0.1
        end
     
        @. c = c + α * Dg

        # compress b
        h = α * (H * Dg)
        for i = 1:n
            if h[i] != 0.
                if b[i] != 0. || abs(h[i]) > ϵ
                    b[i] += h[i]
                end
            end
        end
        b[j] = H[j,:] * c

        # compress d
        m = α * (M * Dg)
        for i = 1:n
            if m[i] != 0.
                if d[i] != 0. || abs(m[i]) > ϵ
                    d[i] += m[i]
                end
            end
        end
        d[j] = M[j,:] * c

        H1 += 2 * α * dot(Dg,b) - α^2 * gH
        M1 += 2 * α * dot(Dg,d) - α^2 * gM

        energy = H1 / M1
        t % 10 == 0 && report(t, energy)

        #push!(y,H1/M1)
    
    end
    r = H1/M1
    
    return r, c;
end


# c^(l+1) = c^l - α∇g
function full_g2(H::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64,Int64}, c::Array{Float64,1}, max_iter::Int64, α::Float64;tol = 1.0e-4)
    n = length(c)
    H = Array(H)
    M = Array(M)
    b = zeros(n)
    d = zeros(n)
    g = 1.
    step = 0

    for i = 1 : max_iter
        if norm(g) < tol
            break;
        end

        b = H * c
        d = M * c
        b1 = dot(c,b)
        d1 = dot(c,d)
        g = 2 .* (d1 .* b - b1 .* d) ./ d1^2 #gradient
        c -= α .* g

        step += 1
    end

    b = H * c
    d = M * c
    r = dot(c,b) / dot(c,d);
    println("  Iteration times : $(step)")

    return r, c;
end
