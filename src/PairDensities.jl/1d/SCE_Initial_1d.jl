# Generate initial wave_function by SCE 

export sce_iv

function onebasis_1d(r0::Int64,N::Int64,ne::Int64,M::Int64)
    X = Int[]
    for j in collect(r0-M+1:r0+M)
        if 0 < j < N + 1
            push!(X, j)
        end
    end
    return append!(X,X .+ N);
end

function sce_iv(ne::Int64, r0::Vector{Vector{Int64}}, ham::ham1d; M=2)

    N = ham.C.n
    combBasis0 = Vector[]
    # Find the basis function near r0
    for j = 1:length(r0)
        R0 = Vector[]
        for i = 1:ne
            push!(R0, onebasis_1d(r0[j][i], N, ne, M))
        end

        # generate combinations of the nearby basis function
        l = length.(R0)
        sq = zeros(Int, ne)
        sqtr = ones(Int, ne)
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
    end

    unique!(combBasis0)
    println("DOF = ", length(combBasis0))
    combBasis0 = convert(Array{Array{Int,1},1}, combBasis0)

    @time SH, SM = hamiltonian(ne, combBasis0, ham)
    @time E0, Ψ0 = geneigsolve((SH,SM), 1, :SR)

    Ψ0 = Real.(Ψ0[1]) ./ norm(Ψ0[1])

    H1 = dot(Ψ0, SH, Ψ0)
    M1 = dot(Ψ0, SM, Ψ0)

    wfsp = WaveFunction_sp(ne, N, combBasis0, Ψ0)

    return wfsp, H1, M1
    
end
