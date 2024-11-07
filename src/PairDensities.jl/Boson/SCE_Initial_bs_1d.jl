
export sce_iv_bs_1d

function onebasis_bs_1d(r0::Int64,N::Int64,ne::Int64,M::Int64)
    X = Int[]
    for j in collect(r0-M+1:r0+M)
        if 0 < j < N + 1
            push!(X, j)
        end
    end
    return X;
end

function sce_iv_bs_1d(ne::Int64, r0::Vector{Int64},
    ham::hamBs1d; M=2)

    N = ham.C.n
    # Find the basis function near r0
    R0 = Vector[]
    for i = 1:ne
        push!(R0, onebasis_bs_1d(r0[i], N, ne, M))
    end

    # generate combinations of the nearby basis function
    l = length.(R0)
    sq = zeros(Int, ne)
    sqtr = ones(Int, ne)
    combBasis0 = Vector[]
    while sqtr[ne] <= l[ne]
        for i = 1:ne
            sq[i] = R0[i][sqtr[i]]
        end
        if length(unique(sq)) == ne
            push!(combBasis0, sort(sq))
        end
        sqtr[1] += 1
        for ℓ = 1:ne-1
            if sqtr[ℓ] == l[ℓ] + 1
                sqtr[ℓ] = 1
                sqtr[ℓ+1] += 1
            end
        end
    end

    unique!(combBasis0)
    println("DOF = ", length(combBasis0))
    combBasis0 = convert(Array{Array{Int,1},1}, combBasis0)

    SH, SM = hamiltonian_bs(ne, combBasis0, ham)
    E0, Ψ0 = geneigsolve((SH, SM), 1, :SR)

    Ψ0 = Ψ0[1] / norm(Ψ0[1])
    H1 = dot(Ψ0, SH, Ψ0)
    M1 = dot(Ψ0, SM, Ψ0)

    wfsp = WaveFunctionBs_sp(ne, N, combBasis0, Ψ0)

    return wfsp, H1, M1
    #return c0;
end
