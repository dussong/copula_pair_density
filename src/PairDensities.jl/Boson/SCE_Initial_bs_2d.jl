
export sce_iv_bs_2d

function onebasis_bs_2d(x::Int64, y::Int64, N::Int64, ne::Int64, m1::Int64, m2::Int64)
    XY = Int[]
    for i = 1:length(x)
        for X in collect(x[i]-m1:x[i]+m2)
            for Y in collect(y[i]-m1:y[i]+m2)
                if X < N + 1 && Y < N + 1
                    push!(XY, X + (Y - 1) * N)
                else
                    push!(XY, 0)
                end
            end
        end
    end

    return XY
end


function sce_iv_bs_2d(ne::Int64, r0::Array{Int64,2}, ham::hamBs2d; m1=1, m2=1)

    dof = ham.C.n
    N = Int(sqrt(dof))
    h = 2L / N

    # Find the basis function near r0
    R0 = Vector[]
    for i = 1:ne
        push!(R0, map(x -> onebasis_ns(r0[x, i], r0[x, i+ne], N, ne, m1, m2), 1:length(r0[:, i])))
    end

    combBasis0 = Vector[]
    sq = zeros(Int, ne)
    sqtr = ones(Int, ne)
    l = (m1 + m2 + 1)^2
    for j = 1:length(R0[1])
        @. sqtr = 1
        while sqtr[ne] <= l
            for i = 1:ne
                sq[i] = R0[i][j][sqtr[i]]
            end
            if !(0 in sq) && allunique(sq)
                push!(combBasis0, sort(sq))
            end
            sqtr[1] += 1
            for ℓ = 1:ne-1
                if sqtr[ℓ] == l + 1
                    sqtr[ℓ] = 1
                    sqtr[ℓ+1] += 1
                end
            end
        end
    end
    unique!(combBasis0)
    combBasis0 = convert(Array{Array{Int,1},1}, combBasis0)

    @time SH, SM = hamiltonian_bs(ne, combBasis0, ham)

    @time E0, Ψ0 = inv_power(ne, SH, SM)

    #Ψ0 = real(Ψ0[:,1]/norm(Ψ0[:,1]))
    Ψ0 = Ψ0 / norm(Ψ0)
    H1 = dot(Ψ0, SH, Ψ0)
    M1 = dot(Ψ0, SM, Ψ0)

    wfsp = WaveFunctionBs_sp(ne, N, combBasis0, Ψ0)

    return wfsp, H1, M1
    #return c0;
end
