# Wave-function structure and related types
# using Arpack

#struct WaveFunction
#   V::Vector{Float64} #values
#   ne::Int
#   N::Int
#   L::Float64
#   # indexTensor2Vec::Array{Int64}
#   # indexVec2Tensor::Array{Int64}
#end

#----------------------------------------------------------------------------
# wavefunction structures:
# WaveFunction_full for full wavefunction; WaveFunction_sp for sparse wavefunction
#----------------------------------------------------------------------------
export WaveFunction, WaveFunction_full, WaveFunction_sp, density, pair_density, pair_density_spin

abstract type WaveFunction end

"""WaveFunction_full
ne : Nb of particles
wf : wavefunction 
"""
struct WaveFunction_full <: WaveFunction
   ne::Int
   wf::Vector{Float64}
end

# construct wave-function (full) by directly solving Schrodinger problem
function WaveFunction_FCI(ne::Int, ham::Hamiltonian; kdim=5, maxiter=100)
   dim = (binomial(2ham.C.n, ne))
   println("Dimension of the problem: $(dim)")
   H, M = hamiltonian(ne, ham)
   E, Ψt, cvinfo = geneigsolve((H, M), 1, :SR; krylovdim=kdim, maxiter=maxiter)
   @show cvinfo
   #solving the eigenvalue problem
   # eigs(H, M, nev = 1, which=:SR) #solving the eigenvalue problem
   println("Energy: $(E[1])\n")
   println("Norm of the residual: $(norm(H*Ψt[1]-E[1]*M*Ψt[1]))")
   return E[1], WaveFunction_full(ne, Real.(Ψt[1]) / norm(Ψt[1]))
end



# construct wave-function (full) by solving Schrodinger problem with matrix free
function WaveFunction_Matfree2(ne::Int, ham::Hamiltonian; kdim=10, maxiter=100)
   dim = (binomial(2ham.C.n, ne))
   x0 = ones(dim)
   N = ham.C.n
   println("Dimension of the problem: $(dim)")
   # combBasis = preallocate1(ne, N)
   Ψtensor, phihtensor, phimtensor, phiAtensor1, phiBtensor1, phiCtensor1 = preallocate2(ne, N)
   combBasis, coulomb_which2, ε, ik_ind, k_ind, j_ind = preallocate3(ne, N)
   A = 0.5 * ham.alpha_lap * ham.AΔ + ham.AV
   M_Ψ(Ψ::Array{Float64,1}) = ham_free_tensor!(
      ne, N, Ψ, A, ham.C, ham.Bee,
      combBasis, coulomb_which2, ε, ik_ind, k_ind, j_ind,
      Ψtensor, phihtensor, phimtensor, phiAtensor1, phiBtensor1, phiCtensor1)

   E, Ψt, cvinfo = geneigsolve(M_Ψ, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
      isposdef=true, verbosity=3)
   @show cvinfo
   HΨt, MΨt = M_Ψ(Ψt[1])
   #solving the eigenvalue problem
   # eigs(H, M, nev = 1, which=:SR) #solving the eigenvalue problem
   println("Energy: $(E[1])\n")
   println("Norm of the residual: $(norm(HΨt-E[1]*MΨt))")
   return E[1], WaveFunction_full(ne, Real.(Ψt[1]) / norm(Ψt[1]))
end


# construct wave-function (full) by solving Schrodinger problem with matrix free
function WaveFunction_Matfree(ne::Int, ham::Hamiltonian; kdim=5, maxiter=100)
   dim = (binomial(2ham.C.n, ne))
   x0 = rand(dim)
   println("Dimension of the problem: $(dim)")
   M_Ψ(Ψ::Array{Float64,1}) = ne == 1 ? ham_free_tensor_1ne(ne, Ψ, ham) : ham_free_tensor(ne, Ψ, ham)

   E, Ψt, cvinfo = geneigsolve(M_Ψ, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
      isposdef=true)
   @show cvinfo
   HΨt, MΨt = M_Ψ(Ψt[1])
   #solving the eigenvalue problem
   # eigs(H, M, nev = 1, which=:SR) #solving the eigenvalue problem
   println("Energy: $(E[1])\n")
   println("Norm of the residual: $(norm(HΨt-E[1]*MΨt))")
   return E[1], WaveFunction_full(ne, Real.(Ψt[1]) / norm(Ψt[1]))
end

#ne(Ψ::WaveFunction) = Ψ.ne

function density(Ψ::WaveFunction_full, ham::Hamiltonian; x=nothing)
   if typeof(ham) == ham1d
      L = ham.L
      N = ham.N
      xx = collect(range(-L, L, length=N + 1))[2:end-1]
      if x == nothing
         x = xx
      end
   elseif typeof(ham) == ham2d
      Lx = ham.L[1]
      Nx = ham.N[1]
      xx = collect(range(-Lx, Lx, length=Nx + 1))[2:end-1]
      Ly = ham.L[2]
      Ny = ham.N[2]
      yy = collect(range(-Ly, Ly, length=Ny + 1))[2:end-1]
      if x == nothing
         x = [xx, yy]
      end
   end
   return density(x, Ψ.ne, Ψ.wf, ham)
end

function pair_density(Ψ::WaveFunction_full, ham::ham1d; x=nothing)
   L = ham.L
   N = ham.N
   xx = collect(range(-L, L, length=N + 1))[2:end-1]
   @show length(xx)
   if x == nothing
      #x = zeros(length(xx)^2, 2)
      #count = 0
      #for xi in xx, xj in xx
      #   count += 1
      #   x[count, :] = [xi, xj]
      #end
      x = hcat(xx, xx)
   end
   return pair_density(x, L, Ψ.ne, Ψ.wf, ham.C)
end

function pair_density_spin(s1::Int64, s2::Int64, Ψ::WaveFunction_full, ham::ham1d; x=nothing)
   L = ham.L
   N = ham.N
   xx = collect(range(-L, L, length=N + 1))[2:end-1]
   @show length(xx)
   if x == nothing
      #x = zeros(length(xx)^2, 2)
      #count = 0
      #for xi in xx, xj in xx
      #   count += 1
      #   x[count, :] = [xi, xj]
      #end
      x = hcat(xx, xx)
   end
   return pair_density_spin(x, s1, s2, Ψ, ham)
end

"""WaveFunction_sp
ne : Nb of particles
dof : number of space basis function 
combBasis : combination bases corresponding to nonzero elements
val : values corresponding to nonzero elements
wfNP : sparse wavefunction without Pauli principle
"""
struct WaveFunction_sp <: WaveFunction
   ne::Int
   dof::Int
   combBasis::Vector{Vector{Int64}}
   val::Vector{Float64}
   wfP::SparseVector{Float64,Int64}
   wfNP::SparseVector{Float64,Int64}
end

function wfspGen(ne::Int, dof::Int, combBasis::Vector{Vector{Int64}}, val::Vector{Float64})

   indP = map(x -> seq2num(2dof, ne, x), combBasis)
   wfP = sparsevec(indP, val, binomial(2dof, ne))

   indNP = map(x -> seq2num_ns(2dof, ne, x), combBasis)
   wfNP = sparsevec(indNP, val, (2dof)^ne)

   return WaveFunction_sp(ne, dof, combBasis, val, wfP, wfNP)
end
WaveFunction_sp(ne, dof, combBasis, val) = wfspGen(ne, dof, combBasis, val)

function wfspGen(ne::Int, dof::Int, wfNP::SparseVector{Float64,Int64})

   indNP = wfNP.nzind
   val = wfNP.nzval

   combBasis = map(x -> num2seq_ns(2dof, ne, x), indNP)
   indP = map(x -> seq2num(2dof, ne, x), combBasis)
   wfP = sparsevec(indP, val, binomial(2dof, ne))

   return WaveFunction_sp(ne, dof, combBasis, val, wfP, wfNP)
end
WaveFunction_sp(ne, dof, wfNP) = wfspGen(ne, dof, wfNP)

# dvee : the derivative of Coulomb potential 1/|x|
# dvext : the derivative of external potential b1*x^2
function WaveFunction_SCI(ne::Int64, ham::Hamiltonian; a0=nothing, num=50, Nc=cld.(ham.N, 2), max_iter=3000, k=500, M=typeof(ham) == ham1d ? 2 : [1, 1], b1=1.0, ϵ=5.0e-7, tol=1e-6)

   d = InitPT(ne, ham; num=num, a0=a0) # find the minimizers
   r0 = typeof(ham) == ham1d ?
        [round.(Int, (d[i] .+ ham.L) .* ham.N ./ (2ham.L)) for i = 1:length(d)] :
        [vcat(round.(Int, (d[i][1:ne] .+ ham.L[1]) .* ham.N[1] ./ (2ham.L[1])), round.(Int, (d[i][ne+1:2ne] .+ ham.L[2]) .* ham.N[2] ./ (2ham.L[2]))) for i = 1:length(d)]
   unique!(r0)
   println("-------------------------------------------------------------------------------")
   @time wf, H1, M1 = sce_iv(ne, r0, ham; M=M) # wf : wave_function; H1/M1 : initial ground state energy
   println("  SCI:")
   @time y1, num1, c1 = SCI_matfree(wf, ham, k; max_iter=max_iter, ϵ=ϵ, tol=tol) # iteration by SCI algorithm
   c1 = c1 / norm(c1)
   wf = WaveFunction_sp(ne, ham.C.n, c1)
   return y1[end], wf
end

function WaveFunction_CDFCI(ne, ham; max_iter=3000, k=500, b1=1.0, ϵ=5.0e-7, tol=1e-6)

   println("-------------------------------------------------------------------------------")
   # generate HF initial state
   @time wfhf, U, Hv, Mv = HF(ne, ham)
   println("  CDFCI:")
   @time y1, num1, c1 = CDFCI_matfree_block(wfhf, ham, k; max_iter=max_iter, ϵ=ϵ, tol=tol) # iteration by CDFCI algorithm
   c1 = c1 / norm(c1)
   wf = WaveFunction_sp(ne, ham.C.n, c1)
   return y1[end], wf
end

function density(Ψ::WaveFunction_sp, ham::Hamiltonian; x=nothing)
   if typeof(ham) == ham1d
      L = ham.L
      N = ham.N
      xx = collect(range(-L, L, length=N + 1))[2:end-1]
      if x == nothing
         x = xx
      end
   elseif typeof(ham) == ham2d
      Lx = ham.L[1]
      Nx = ham.N[1]
      xx = collect(range(-Lx, Lx, length=Nx + 1))[2:end-1]
      Ly = ham.L[2]
      Ny = ham.N[2]
      yy = collect(range(-Ly, Ly, length=Ny + 1))[2:end-1]
      if x == nothing
         x = [xx, yy]
      end
   end
   return density_sp(x, Ψ, ham)
end

function pair_density(Ψ::WaveFunction_sp, ham::ham1d; x=nothing)
   L = ham.L
   N = ham.N
   xx = collect(range(-L, L, length=N + 1))[2:end-1]
   if x == nothing
      x = hcat(xx, xx)
   end
   return pair_density_sp(x, Ψ, ham)
end

function pair_density_spin(s1::Int64, s2::Int64, Ψ::WaveFunction_sp, ham::ham1d; x=nothing)

   Ψ_converted = WaveFunction_full(Ψ.ne, Array(Ψ.wfP))

   return pair_density_spin(s1, s2, Ψ_converted, ham; x=x)
end

# function WaveFunction(ne, ham, method::String;kwargs...) 
#    if method == "FCI_full"
#       return WaveFunction_full(ne, ham; kdim=5, maxiter=100) 
#    elseif method == "FCI_sparse"
#       return WaveFunction_sp(ne, ham; kdim=5, maxiter=100) 
#    end
# end

function WaveFunction(ne, ham, method::String; kwargs...)
   if method == "FCI_full"
      return WaveFunction_FCI(ne, ham; kwargs...)
   elseif method == "FCI_sparse"
      return WaveFunction_Matfree(ne, ham; kwargs...)
   elseif method == "FCI_sparse2"
      return WaveFunction_Matfree2(ne, ham; kwargs...)
   elseif method == "CDFCI_sparse" && ham.element == "P1"
      return WaveFunction_CDFCI(ne, ham; kwargs...)
   elseif method == "selected_CI_sparse" && ham.element == "P1"
      return WaveFunction_SCI(ne, ham; kwargs...)
   end
end


# # Wave-function structure and related types
# # using Arpack

# #struct WaveFunction
# #   V::Vector{Float64} #values
# #   ne::Int
# #   N::Int
# #   L::Float64
# #   # indexTensor2Vec::Array{Int64}
# #   # indexVec2Tensor::Array{Int64}
# #end

# #----------------------------------------------------------------------------
# # wavefunction structures:
# # WaveFunction_full for full wavefunction; WaveFunction_sp for sparse wavefunction
# #----------------------------------------------------------------------------
# export WaveFunction, WaveFunction_full, WaveFunction_sp, density, pair_density, pair_density_spin

# abstract type WaveFunction end

# """WaveFunction_full
# ne : Nb of particles
# wf : wavefunction 
# """
# struct WaveFunction_full <: WaveFunction
#    ne::Int
#    wf::Vector{Float64}
# end

# # construct wave-function (full) by directly solving Schrodinger problem
# function WaveFunction_FCI(ne::Int, ham::Hamiltonian; kdim=5, maxiter=100)
#    dim = (binomial(2ham.C.n, ne))
#    println("Dimension of the problem: $(dim)")
#    H, M = hamiltonian(ne, ham)
#    E, Ψt, cvinfo = geneigsolve((H, M), 1, :SR; krylovdim=kdim, maxiter=maxiter)
#    @show cvinfo
#    #solving the eigenvalue problem
#    # eigs(H, M, nev = 1, which=:SR) #solving the eigenvalue problem
#    println("Energy: $(E[1])\n")
#    println("Norm of the residual: $(norm(H*Ψt[1]-E[1]*M*Ψt[1]))")
#    return E[1], WaveFunction_full(ne, Real.(Ψt[1]) / norm(Ψt[1]))
# end

# # construct wave-function (full) by solving Schrodinger problem with matrix free
# function WaveFunction_Matfree(ne::Int, ham::Hamiltonian; kdim=5, maxiter=100)
#    dim = (binomial(2ham.C.n, ne))
#    x0 = rand(dim)
#    println("Dimension of the problem: $(dim)")
#    function M_Ψ(Ψ::Array{Float64,1})
#       HΨ, MΨ = ham_free_tensor(ne, Ψ, ham)
#       return HΨ, MΨ
#    end
#    E, Ψt, cvinfo = geneigsolve(M_Ψ, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
#       isposdef=true)
#    @show cvinfo
#    HΨt, MΨt = ham_free_tensor(ne, Ψt[1], ham)
#    #solving the eigenvalue problem
#    # eigs(H, M, nev = 1, which=:SR) #solving the eigenvalue problem
#    println("Energy: $(E[1])\n")
#    println("Norm of the residual: $(norm(HΨt-E[1]*MΨt))")
#    return E[1], WaveFunction_full(ne, Real.(Ψt[1]) / norm(Ψt[1]))
# end

# #ne(Ψ::WaveFunction) = Ψ.ne

# function density(Ψ::WaveFunction_full, ham::Hamiltonian; x=nothing)
#    if typeof(ham) == ham1d
#       L = ham.L
#       N = ham.N
#       xx = collect(range(-L, L, length=N + 1))[2:end-1]
#       if x == nothing
#          x = xx
#       end
#    elseif typeof(ham) == ham2d
#       Lx = ham.L[1]
#       Nx = ham.N[1]
#       xx = collect(range(-Lx, Lx, length=Nx + 1))[2:end-1]
#       Ly = ham.L[2]
#       Ny = ham.N[2]
#       yy = collect(range(-Ly, Ly, length=Ny + 1))[2:end-1]
#       if x == nothing
#          x = [xx, yy]
#       end
#    end
#    return density(x, Ψ.ne, Ψ.wf, ham)
# end

# function pair_density(Ψ::WaveFunction_full, ham::ham1d; x=nothing)
#    L = ham.L
#    N = ham.N
#    xx = collect(range(-L, L, length=N + 1))[2:end-1]
#    @show length(xx)
#    if x == nothing
#       #x = zeros(length(xx)^2, 2)
#       #count = 0
#       #for xi in xx, xj in xx
#       #   count += 1
#       #   x[count, :] = [xi, xj]
#       #end
#       x = hcat(xx, xx)
#    end
#    return pair_density(x, L, Ψ.ne, Ψ.wf, ham.C)
# end

# function pair_density_spin(s1::Int64, s2::Int64, Ψ::WaveFunction_full, ham::ham1d; x=nothing)
#    L = ham.L
#    N = ham.N
#    xx = collect(range(-L, L, length=N + 1))[2:end-1]
#    @show length(xx)
#    if x == nothing
#       #x = zeros(length(xx)^2, 2)
#       #count = 0
#       #for xi in xx, xj in xx
#       #   count += 1
#       #   x[count, :] = [xi, xj]
#       #end
#       x = hcat(xx, xx)
#    end
#    return pair_density_spin(x, s1, s2, Ψ, ham)
# end

# """WaveFunction_sp
# ne : Nb of particles
# dof : number of space basis function 
# combBasis : combination bases corresponding to nonzero elements
# val : values corresponding to nonzero elements
# wfns : sparse wavefunction without Pauli principle
# """
# struct WaveFunction_sp <: WaveFunction
#    ne::Int
#    dof::Int
#    combBasis::Vector{Vector{Int64}}
#    val::Vector{Float64}
#    wfP::SparseVector{Float64,Int64}
#    wfNP::SparseVector{Float64,Int64}
# end

# function wfspGen(ne::Int, dof::Int, combBasis::Vector{Vector{Int64}}, val::Vector{Float64})

#    indP = map(x -> seq2num(2dof, ne, x), combBasis)
#    wfP = sparsevec(indP, val, binomial(2dof, ne))

#    indNP = map(x -> seq2num_ns(2dof, ne, x), combBasis)
#    wfNP = sparsevec(indNP, val, (2dof)^ne)

#    return WaveFunction_sp(ne, dof, combBasis, val, wfP, wfNP)
# end
# WaveFunction_sp(ne, dof, combBasis, val) = wfspGen(ne, dof, combBasis, val)

# function wfspGen(ne::Int, dof::Int, wfNP::SparseVector{Float64,Int64})

#    indNP = wfNP.nzind
#    val = wfNP.nzval

#    combBasis = map(x -> num2seq_ns(2dof, ne, x), indNP)
#    indP = map(x -> seq2num(2dof, ne, x), combBasis)
#    wfP = sparsevec(indP, val, binomial(2dof, ne))

#    return WaveFunction_sp(ne, dof, combBasis, val, wfP, wfNP)
# end
# WaveFunction_sp(ne, dof, wfNP) = wfspGen(ne, dof, wfNP)

# # dvee : the derivative of Coulomb potential 1/|x|
# # dvext : the derivative of external potential b1*x^2
# function WaveFunction_SCI(ne::Int64, ham::Hamiltonian; a0=nothing, num=50, max_iter=3000, k=500, M=typeof(ham) == ham1d ? 2 : [1, 1], b1=1.0, ϵ=5.0e-7, tol=1e-6)

#    d = InitPT(ne, ham; num=num, a0=a0) # find the minimizers
#    r0 = typeof(ham) == ham1d ?
#         [round.(Int, (d[i] .+ ham.L) .* ham.N ./ (2ham.L)) for i = 1:length(d)] :
#         [vcat(round.(Int, (d[i][1:ne] .+ ham.L[1]) .* ham.N[1] ./ (2ham.L[1])), round.(Int, (d[i][ne+1:2ne] .+ ham.L[2]) .* ham.N[2] ./ (2ham.L[2]))) for i = 1:length(d)]
#    unique!(r0)
#    println("-------------------------------------------------------------------------------")
#    @time wf, H1, M1 = sce_iv(ne, r0, ham; M=M) # wf : wave_function; H1/M1 : initial ground state energy
#    println("  SCI:")
#    @time y1, num1, c1 = SCI_matfree(wf, ham, k; max_iter=max_iter, ϵ=ϵ, tol=tol) # iteration by SCI algorithm
#    wf = WaveFunction_sp(ne, ham.C.n, c1)
#    return y1[end], wf
# end

# function WaveFunction_CDFCI(ne, ham; max_iter=3000, k=500, b1=1.0, ϵ=5.0e-7, tol=1e-6)

#    println("-------------------------------------------------------------------------------")
#    # generate HF initial state
#    @time wfhf, U, Hv, Mv = HF(ne, ham)
#    println("  CDFCI:")
#    @time y1, num1, c1 = CDFCI_matfree_block(wfhf, ham, k; max_iter=max_iter, ϵ=ϵ, tol=tol) # iteration by CDFCI algorithm
#    wf = WaveFunction_sp(ne, ham.C.n, c1)
#    return y1[end], wf
# end

# function density(Ψ::WaveFunction_sp, ham::Hamiltonian; x=nothing)
#    if typeof(ham) == ham1d
#       L = ham.L
#       N = ham.N
#       xx = collect(range(-L, L, length=N + 1))[2:end-1]
#       if x == nothing
#          x = xx
#       end
#    elseif typeof(ham) == ham2d
#       Lx = ham.L[1]
#       Nx = ham.N[1]
#       xx = collect(range(-Lx, Lx, length=Nx + 1))[2:end-1]
#       Ly = ham.L[2]
#       Ny = ham.N[2]
#       yy = collect(range(-Ly, Ly, length=Ny + 1))[2:end-1]
#       if x == nothing
#          x = [xx, yy]
#       end
#    end
#    return density_sp(x, Ψ, ham)
# end

# function pair_density(Ψ::WaveFunction_sp, ham::ham1d; x=nothing)
#    L = ham.L
#    N = ham.N
#    xx = collect(range(-L, L, length=N + 1))[2:end-1]
#    if x == nothing
#       x = hcat(xx, xx)
#    end
#    return pair_density_sp(x, Ψ, ham)
# end

# function pair_density_spin(s1::Int64, s2::Int64, Ψ::WaveFunction_sp, ham::ham1d; x=nothing)

#    Ψ_converted = WaveFunction_full(Ψ.ne, Array(Ψ.wfP))

#    return pair_density_spin(s1, s2, Ψ_converted, ham; x=x)
# end

# # function WaveFunction(ne, ham, method::String;kwargs...) 
# #    if method == "FCI_full"
# #       return WaveFunction_full(ne, ham; kdim=5, maxiter=100) 
# #    elseif method == "FCI_sparse"
# #       return WaveFunction_sp(ne, ham; kdim=5, maxiter=100) 
# #    end
# # end

# function WaveFunction(ne, ham, method::String; kwargs...)
#    if method == "FCI_full"
#       return WaveFunction_FCI(ne, ham; kwargs...)
#    elseif method == "FCI_sparse"
#       return WaveFunction_Matfree(ne, ham; kwargs...)
#    elseif method == "CDFCI_sparse" && ham.element == "P1"
#       return WaveFunction_CDFCI(ne, ham; kwargs...)
#    elseif method == "selected_CI_sparse" && ham.element == "P1"
#       return WaveFunction_SCI(ne, ham; kwargs...)
#    end
# end