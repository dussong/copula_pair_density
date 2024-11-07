# Generate initial wave_function by SCE 

export sce_iv

function onebasis_2d(x::Vector{Int64},y::Vector{Int64},Nx::Int64,Ny::Int64,ne::Int64, m1::Int64, m2::Int64)
    XY = Int[]
    for i = 1:length(x)
        for X in collect(x[i]-m1:x[i]+m2)
            for Y in collect(y[i]-m1:y[i]+m2)
                if 0 < X < Nx + 1 && 0 < Y < Ny + 1
                    push!(XY, X + (Y - 1) * Nx)
                else
                    push!(XY, 0)
                end
            end
        end 
    end
    return [XY,XY .+ Nx*Ny];
end


function sce_iv(ne::Int64, r0::Vector{Vector{Int64}}, ham::ham2d; M=[1, 1])

    dof = ham.C.n
    Nx = ham.N[1] - 1
    Ny = ham.N[2] - 1

    r0_matrix = zeros(Int64, length(r0), 2ne)
    for i = 1:length(r0)
        r0_matrix[i, :] = r0[i]
    end

    # Find the basis function near r0
    m1 = M[1]
    m2 = M[2]
    R0 = Vector[]
    for i = 1:ne
        push!(R0, onebasis_2d(r0_matrix[:, i], r0_matrix[:, i+ne], Nx, Ny, ne, m1, m2))
    end
    # generate combinations of the nearby basis function
    basis1body0 = collect(Iterators.product(R0...))
    combBasis0 = Vector[]
    sq = zeros(Int, ne)
    sqtr = ones(Int, ne)
    l = (m1 + m2 + 1)^2
    for comb in basis1body0
        for j = 1:Int(length(comb[1]) / l)
            @. sqtr = 1
            while sqtr[ne] <= l
                for i = 1:ne
                    sq[i] = comb[i][sqtr[i]+(j-1)*l]
                end
                if !(0 in sq) && length(unique(sq)) == ne
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
    end
    unique!(combBasis0)
    println("DOF = ", length(combBasis0))
    combBasis0 = convert(Array{Array{Int,1},1}, combBasis0)

    @time SH, SM = hamiltonian(ne, combBasis0, ham)
    @time E0, Ψ0 = inv_power(ne, SH, SM)

    H1 = dot(Ψ0, SH, Ψ0)
    M1 = dot(Ψ0, SM, Ψ0)

    wfsp = WaveFunction_sp(ne, dof, combBasis0, Ψ0)

    return wfsp, H1, M1

end
