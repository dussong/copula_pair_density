
#-------------------------------------------------------------------------------
# Assemble n-body Hamiltonian matrices in a sparse form and map for indices
#-------------------------------------------------------------------------------
export overlap, ham_1B_sp, ham_2B_sp, index_VecTensor, tensor2vec

#-------------------------------------------------------------------------------
# Create the map from Tensor index to Vec index and the inverse
#-------------------------------------------------------------------------------
# PARAMETERS
# ne: number of electrons/particles
# M : spacial discretization parameter
# RETURNS
# indexTensor2Vec : [i1,⋯,i_n,s_1,⋯s_n] |-> 1≦i≦dof
# indexVec2Tensor : 1≦i≦dof |-> [i1,⋯,i_n,s_1,⋯s_n]

function index_VecTensor(ne::Int, N::Int)
  spin = [0,1];
  n = 2 * N
  dof = binomial(n, ne);
  # println("Degrees of Freedom = ", dof, "\n")
  dimIndex = ntuple(x -> x<=ne ? N : 2, 2*ne)
  indexTensor2Vec = - ones(Int, dimIndex)
  indexVec2Tensor = zeros(Int, dof, 2*ne)
  # index_mat  the loop index for particle coordinates
  # index_spin the loop index for particle spins
  i = ones(Int, ne)
  s = zeros(Int, ne)
  k = 0
  MAX_count = 2^ne * N^ne

  for count = 1 : MAX_count
    # check whether this index is well sorted
    j = [ i[ℓ] + N * s[ℓ] - ℓ/(ne+1) for ℓ = 1:ne ]   # in case j[ℓ] = j[ℓ+1]
    if issorted(j)
        k = k+1
        indexTensor2Vec[count] = k
        indexVec2Tensor[k,:] = [i;s]
    end
    # adjust the index
    i[1] += 1
    for ip = 1 : ne
        if i[ip] == N+1
            i[ip] = 1
            if ip == ne
                s[1] += 1
            else
                i[ip+1] += 1
            end
        end
    end
    for sp = 1 : ne
        if s[sp] == 2 && count < MAX_count
            s[sp] = 0
            s[sp+1] += 1
        end
    end
  end
  return [ indexTensor2Vec, indexVec2Tensor ]
end

function tensor2vec(ne::Int, N::Int, combBasis::Array{Array{Int,1},1})
    ind = ones(Int,length(combBasis))
    vals = collect(1:length(combBasis))
    for count = 1 : length(combBasis)
        for i = 1 : ne
            ind[count] += (combBasis[count][i]-1) * (2N)^(i-1)
        end
    end
    return sparsevec(ind,vals,(2N)^ne)
end

tensor2vec(ne::Int, N::Int) =  tensor2vec(ne, N, collect(combinations(1:2*N,ne)))
#-------------------------------------------------------------------------------
# Assemble n-body Hamiltonian matrices in a sparse form ---- for 1-body
# overlap
#-------------------------------------------------------------------------------

# PARAMETERS
# n : dimension of the tensor. When used in the code, it is usually
#     n = ne-2 for ρ₂  and  n = ne-1 for ρ and γ
# C : 1-body overlap
# RETURNS
# M : overlap for n-dimension tensor

function overlap(n::Int, C::SparseMatrixCSC{Float64,Int64})

  N = C.n
  # initialize the sparse array
  val = Float64[]
  indrow1body = Int64[]
  indcol1body = Int64[]

  lengthBasis = N^n
  i = ones(Int64, n)
  jptr = zeros(Int64, n)
  j = zeros(Int64, n)
  # loop for the 1-body matrix elements
  for count = 1:lengthBasis
      @. jptr = 0
      while jptr[n] < C.colptr[i[n]+1] - C.colptr[i[n]]
          Cv = 1.0
          # θ, ζ for indicies of the total Hamiltonian
          θ = 1
          ζ = 1
          for l in 1:n
              j[l] = C.rowval[C.colptr[i[l]]+jptr[l]]
              Cv *= C[i[l], j[l]]
              θ += (i[l] - 1) * N^(l - 1)
              ζ += (j[l] - 1) * N^(l - 1)
          end
          push!(indrow1body, θ)
          push!(indcol1body, ζ)
          push!(val, Cv)
          # adjust jptr
          jptr[1] += 1
          if n >= 2
              for ℓ = 1:n-1
                  if jptr[ℓ] == C.colptr[i[ℓ]+1] - C.colptr[i[ℓ]]
                      jptr[ℓ] = 0
                      jptr[ℓ+1] += 1
                  end
              end
          end # end if
      end # end while loop for jptr
      # adjust iptr
      i[1] += 1
      if n >= 2
          for ℓ = 1:n-1
              if i[ℓ] == N + 1
                  i[ℓ] = 1
                  i[ℓ+1] += 1
              end
          end
      end # end if
  end # end loop for Basis

  M = sparse(indrow1body, indcol1body, val);
  return M;
end

#-------------------------------------------------------------------------------
# Assemble n-body Hamiltonian matrices in a sparse form ---- for 1-body operator
#-------------------------------------------------------------------------------

# PARAMETERS
# ne: number of electrons/particles
# A : 1-body operator, e.g., -Δ, v_ext
# C : overlap
# RETURNS
# H : many-body Hamiltonian for the 1-body operator
#-------------------------------------------------------------------------------

function ham_1B_sp(ne::Int, combBasis::Array{Array{Int,1},1},
    A::SparseMatrixCSC{Float64,Int64}, C::SparseMatrixCSC{Float64,Int64})

    N = C.n
    # initialize the sparse array
    val = Float64[]
    indrow1body = Int64[]
    indcol1body = Int64[]
    # computate the permutations and paritiy
    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]

    # In the following, we use i, j for indicies for A, C
    # s, t for indicies of spin
    # θ, ζ for indicies of the total Hamiltonian
    #basis1body = 1:2*N
    #combBasis = collect(combinations(basis1body,ne))
    index = tensor2vec(ne, N, combBasis)

    i = zeros(Int, ne)
    s = zeros(Int, ne)
    j = zeros(Int, ne)
    t = zeros(Int, ne)
    jp = zeros(Int, ne)
    tj = zeros(Int, ne)
    jptr = zeros(Int64, ne)

    # loop for the 1-body matrix elements
    for count = 1:length(combBasis)
        si = combBasis[count]
        for l in 1:ne
            i[l] = si[l] > N ? si[l] - N : si[l]
            s[l] = si[l] > N ? 1 : 0
        end
        @. jptr = 0
        while jptr[ne] < C.colptr[i[ne]+1] - C.colptr[i[ne]]
            Cv = 1.0
            Av = 0.0
            for l in 1:ne
                j[l] = C.rowval[C.colptr[i[l]]+jptr[l]]
                Cv *= C[i[l], j[l]]
                Av += A[i[l], j[l]] / C[i[l], j[l]]
            end
            Av *= Cv
            # loop for the permutations
            for k = 1:length(p)
                for l in 1:ne
                    t[l] = s[p[k][l]]
                    jp[l] = j[p[k][l]]
                    tj[l] = jp[l] + N * t[l]
                end
                tj1 = 1
                for i = 1:ne
                    tj1 += (tj[i] - 1) * (2N)^(i - 1)
                end
                ζ = index[tj1]
                #ttj = ntuple(i -> tj[i], ne)
                #ζ = index[CartesianIndex(ttj)]
                if ζ > 0
                    push!(indrow1body, count)
                    push!(indcol1body, ζ)
                    push!(val, ε[k] * Av)
                end # end issorted(tj)
            end # end loop through permutation
            # adjust jptr
            jptr[1] += 1
            for ℓ = 1:ne-1
                if jptr[ℓ] == C.colptr[i[ℓ]+1] - C.colptr[i[ℓ]]
                    jptr[ℓ] = 0
                    jptr[ℓ+1] += 1
                end
            end
        end # end while loop for jptr
    end # end loop for combBasis

    H = sparse(indrow1body, indcol1body, val)
    return H
end

ham_1B_sp(ne::Int64, A::SparseMatrixCSC{Float64,Int64}, C::SparseMatrixCSC{Float64,Int64}) =
ham_1B_sp(ne,collect(combinations(1:2*C.n,ne)),A,C)
#-------------------------------------------------------------------------------
# Assemble n-body Hamiltonian matrices in a sparse form ---- for 2-body operator
#-------------------------------------------------------------------------------

# PARAMETERS
# ne: number of electrons/particles
# M : spacial discretization parameter
# B : 1/|r_i-r_j|
# C : overlap
# RETURNS
# HV : many-body Hamiltonian for 1/|r_i-r_j|
#-------------------------------------------------------------------------------

function ham_2B_sp(ne::Int, combBasis::Array{Array{Int,1},1}, B::Array{Float64,4},
          C::SparseMatrixCSC{Float64,Int64})

  N = C.n
  # initialize the sparse array
  indrow2body = Int64[]
  indcol2body = Int64[]
  valV = Float64[]

  # computate the permutations and paritiy
  v = 1:ne
  p = collect(permutations(v))[:]
  ε = (-1).^[parity(p[i]) for i=1:length(p)]
  # collect the pairs for Coulomb interactions
  coulomb_which2 = collect(combinations(v,2))

  # In the following, we use i, j for indicies for B, C
  # s, t for indicies of spin
  # θ, ζ for indicies of the total Hamiltonian
  #basis1body = 1:2*N
  #combBasis = collect(combinations(basis1body,ne))
  index = tensor2vec(ne, N, combBasis)

  i = zeros(Int,ne)
  s = zeros(Int,ne)
  j = zeros(Int,ne)
  t = zeros(Int,ne)
  jp = zeros(Int,ne)
  tj = zeros(Int,ne)

  # loop for the 1-body matrix elements
  for count = 1 : length(combBasis)
      si = combBasis[count]
      for l in 1:ne
          i[l]  = si[l] > N ? si[l] - N : si[l]
          s[l]  = si[l] > N ? 1 : 0
      end
      jptr = zeros(Int64, ne)
      while jptr[ne] < C.colptr[i[ne]+1] - C.colptr[i[ne]]
          Cv = 1.0
          for l in 1:ne
              j[l] = C.rowval[C.colptr[i[l]]+jptr[l]]
              Cv *= C[i[l],j[l]]
          end
          Bv = 0.0
          for l = 1 : length(coulomb_which2)
              ca = coulomb_which2[l][1]
              cb = coulomb_which2[l][2]
              Bv += Cv * B[ i[ca], i[cb], j[ca], j[cb] ] /
                   ( C[i[ca],j[ca]] * C[i[cb],j[cb]] )
          end
          # loop for the permutations
          for k = 1 : length(p)
              for l in 1:ne
                  t[l]  = s[p[k][l]]
                  jp[l] = j[p[k][l]]
                  tj[l] = jp[l] + N * t[l]
              end
              tj1 = 1
              for i = 1 : ne
                  tj1 += (tj[i]-1) * (2N)^(i-1)
              end
              ζ = index[tj1]
              if ζ > 0
                  push!(indrow2body, count)
                  push!(indcol2body, ζ)
                  push!(valV, ε[k] * Bv)
              end # end issorted(tj)
          end # end loop through permutation
          # adjust jptr
          jptr[1] += 1
          for ℓ = 1 : ne-1
              if jptr[ℓ] == C.colptr[i[ℓ]+1] - C.colptr[i[ℓ]]
                  jptr[ℓ] = 0
                  jptr[ℓ+1] += 1
              end
          end
      end # end while loop for jptr
  end # end loop for combBasis

  HV = sparse(indrow2body, indcol2body, valV);
  return HV;
end


function ham_2B_sp(ne::Int, combBasis::Array{Array{Int,1},1}, B::SparseMatrixCSC{Float64,Int64},
          C::SparseMatrixCSC{Float64,Int64})

  N = C.n
  # initialize the sparse array
  indrow2body = Int64[]
  indcol2body = Int64[]
  valV = Float64[]

  # computate the permutations and paritiy
  v = 1:ne
  p = collect(permutations(v))[:]
  ε = (-1).^[parity(p[i]) for i=1:length(p)]
  # collect the pairs for Coulomb interactions
  coulomb_which2 = collect(combinations(v,2))

  # In the following, we use i, j for indicies for B, C
  # s, t for indicies of spin
  # θ, ζ for indicies of the total Hamiltonian
  #basis1body = 1:2*N
  #combBasis = collect(combinations(basis1body,ne))
  index = tensor2vec(ne, N, combBasis)

  i = zeros(Int,ne)
  s = zeros(Int,ne)
  j = zeros(Int,ne)
  t = zeros(Int,ne)
  jp = zeros(Int,ne)
  tj = zeros(Int,ne)

  # loop for the 1-body matrix elements
  for count = 1 : length(combBasis)
      si = combBasis[count]
      for l in 1:ne
          i[l]  = si[l] > N ? si[l] - N : si[l]
          s[l]  = si[l] > N ? 1 : 0
      end
      jptr = zeros(Int64, ne)
      while jptr[ne] < C.colptr[i[ne]+1] - C.colptr[i[ne]]
          Cv = 1.0
          for l in 1:ne
              j[l] = C.rowval[C.colptr[i[l]]+jptr[l]]
              Cv *= C[i[l],j[l]]
          end
          Bv = 0.0
          for l = 1 : length(coulomb_which2)
              ca = coulomb_which2[l][1]
              cb = coulomb_which2[l][2]
              Bv += Cv * B[ i[ca] + (i[cb]-1)*N, j[ca] + (j[cb]-1)*N ] /
                   ( C[i[ca],j[ca]] * C[i[cb],j[cb]] )
          end
          # loop for the permutations
          for k = 1 : length(p)
              for l in 1:ne
                  t[l]  = s[p[k][l]]
                  jp[l] = j[p[k][l]]
                  tj[l] = jp[l] + N * t[l]
              end
              tj1 = 1
              for i = 1 : ne
                  tj1 += (tj[i]-1) * (2N)^(i-1)
              end
              ζ = index[tj1]
              if ζ > 0
                  push!(indrow2body, count)
                  push!(indcol2body, ζ)
                  push!(valV, ε[k] * Bv)
              end # end issorted(tj)
          end # end loop through permutation
          # adjust jptr
          jptr[1] += 1
          for ℓ = 1 : ne-1
              if jptr[ℓ] == C.colptr[i[ℓ]+1] - C.colptr[i[ℓ]]
                  jptr[ℓ] = 0
                  jptr[ℓ+1] += 1
              end
          end
      end # end while loop for jptr
  end # end loop for combBasis

  HV = sparse(indrow2body, indcol2body, valV);
  return HV;
end

ham_2B_sp(ne, B, C) = ham_2B_sp(ne,collect(combinations(1:2*C.n,ne)),B,C)