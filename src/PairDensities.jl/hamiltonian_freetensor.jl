
# Compute H*Ψ and M*Ψ with Ψ is full.
export ham_free_tensor, ham_free_tensor_1ne, ham_free_tensor!

function preallocate2(ne::Int, N::Int)
   Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
   Φhtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
   Φmtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
   φAtensor1 = zeros(Float64, ntuple(x -> N, ne))
   φBtensor1 = zeros(Float64, ntuple(x -> N, ne))
   φCtensor1 = zeros(Float64, ntuple(x -> N, ne))
   return Ψtensor, Φhtensor, Φmtensor, φAtensor1, φBtensor1, φCtensor1
end

function preallocate3(ne::Int, N::Int)
   combBasis = collect(combinations(1:2*N, ne))
   # computate the permutations and paritiy
   v = 1:ne
   p = collect(permutations(v))[:]
   ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
   coulomb_which2 = collect(combinations(v, 2))
   ik_ind = zeros(Int, length(combBasis) * length(p))
   k_ind = zeros(Int, length(combBasis) * length(p))
   j_ind = zeros(Int, length(combBasis) * length(p))
   # reshape the vector Ψ to the (antisymmetric) tensor
   count = 1
   for j = 1:length(combBasis)
      for k = 1:length(p)
         ik_ind[count] = combBasis[j][p[k][1]]
         for l = 2:ne
            ik_ind[count] += (combBasis[j][p[k][l]] - 1) * (2N)^(l - 1)
         end
         j_ind[count] = j
         k_ind[count] = k
         count += 1
      end
   end
   return combBasis, coulomb_which2, ε, ik_ind, k_ind, j_ind
end

function ham_free_tensor!(ne::Int, N::Int, Ψ::Array{Float64,1},
   A::SparseMatrixCSC{Float64,Int64},
   C::SparseMatrixCSC{Float64,Int64},
   B::Array{Float64,4},
   combBasis, coulomb_which2, ε, ik_ind, k_ind, j_ind,
   Ψtensor, phihtensor, phimtensor, phiAtensor1, phiBtensor1, phiCtensor1)

   Ψtensor .= 0.0
   phihtensor .= 0.0
   phimtensor .= 0.0
   phiAtensor1 .= 0.0
   phiBtensor1 .= 0.0
   phiCtensor1 .= 0.0

   @assert length(Ψ) == length(combBasis)
   @assert ne > 1
   phih = zeros(size(Ψ))
   phim = zeros(size(Ψ))

   # reshape the vector Ψ to the (antisymmetric) tensor
   for j in 1:length(k_ind)
      Ψtensor[ik_ind[j]] = Ψ[j_ind[j]] * ε[k_ind[j]]
   end

   # loop through different spin configurations
   sptr = zeros(Int64, ne)
   m1 = zeros(Int64, ne)
   mp1 = zeros(Int64, ne)
   m2 = zeros(Int64, ne)
   mp2 = zeros(Int64, ne)
   W = zeros(N, N, N^(ne - 2))
   M1 = zeros(N, N^(ne - 1))
   MA1 = zeros(N, N^(ne - 1))
   MC1 = zeros(N, N^(ne - 1))
   for s = 1:2^ne
      sp = sptr * N
      # @show sp
      np = ntuple(x -> sp[x]+1:sp[x]+N, ne)
      # @show np
      for j = 1:ne # act A on the j-th particle
         phiAtensor = getindex(Ψtensor, np...)
         phiCtensor = copy(phiAtensor)
         # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
         for i = 1:ne
            m1[1] = i
            m1[2:end] = setdiff(1:ne, i)
            MA = reshape(permutedims!(phiAtensor1, phiAtensor, m1), N, N^(ne - 1))
            MC = reshape(permutedims!(phiCtensor1, phiCtensor, m1), N, N^(ne - 1))
            if i == j
               mul!(MA1, A, MA)
            else
               mul!(MA1, C, MA)
            end
            mul!(MC1, C, MC)
            sortperm!(mp1, m1)
            permutedims!(phiAtensor, reshape(MA1, ntuple(x -> N, ne)), mp1)
            permutedims!(phiCtensor, reshape(MC1, ntuple(x -> N, ne)), mp1)
         end
         # assemble the value to phitensor
         phihtensor[np...] += phiAtensor
         phimtensor[np...] += phiCtensor
      end

      for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
         k = coulomb_which2[j][1]
         l = coulomb_which2[j][2]
         phiBtensor = getindex(Ψtensor, np...)
         # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
         for i = 1:ne
            if i != k && i != l
               m2[1] = i
               m2[2:end] = setdiff(1:ne, i)
               M = reshape(permutedims!(phiBtensor1, phiBtensor, m2), N, N^(ne - 1))
               mul!(M1, C, M)
               permutedims!(phiBtensor, reshape(M1, ntuple(x -> N, ne)), sortperm!(mp2, m2))
            elseif i == k
               m2[1] = k
               m2[2] = l
               m2[3:end] = setdiff(1:ne, k, l)
               M = reshape(permutedims!(phiBtensor1, phiBtensor, m2), N, N, N^(ne - 2))
               @. W = 0.0
               @tensor W[i, j, k] = B[i, j, a, b] * M[a, b, k]
               permutedims!(phiBtensor, reshape(W, ntuple(x -> N, ne)), sortperm!(mp2, m2))
            end
         end
         # assemble the value to phitensor
         phihtensor[np...] += phiBtensor
      end
      # adjust sptr
      sptr[1] += 1
      for ℓ = 1:ne-1
         if sptr[ℓ] == 2
            sptr[ℓ] = 0
            sptr[ℓ+1] += 1
         end
      end
   end

   for i = 1:length(combBasis)
      il = combBasis[i]
      l = il[1]
      for j = 2:ne
         l += (il[j] - 1) * (2N)^(j - 1)
      end
      phih[i] = phihtensor[l]
      phim[i] = phimtensor[l]
   end
   return phih, phim ./ ne
end

function ham_free_tensor(ne::Int, Ψ::Array{Float64,1},
   AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
   C::SparseMatrixCSC{Float64,Int64},
   B::Array{Float64,4}; alpha_lap=1.0)
   N = C.n
   Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
   Φhtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
   Φmtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
   φAtensor1 = zeros(Float64, ntuple(x -> N, ne))
   φBtensor1 = zeros(Float64, ntuple(x -> N, ne))
   φCtensor1 = zeros(Float64, ntuple(x -> N, ne))

   # indecies for the basis
   basis1body = 1:2*N
   combBasis = collect(combinations(basis1body, ne))
   # Φ = H⋅Ψ
   @assert length(Ψ) == length(combBasis)
   @assert ne > 1
   Φh = zeros(size(Ψ))
   Φm = zeros(size(Ψ))
   # computate the permutations and paritiy
   v = 1:ne
   p = collect(permutations(v))[:]
   ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
   coulomb_which2 = collect(combinations(v, 2))
   #ik = zeros(Int,ne)
   # reshape the vector Ψ to the (antisymmetric) tensor
   for j = 1:length(combBasis)
      ij = combBasis[j]
      for k = 1:length(p)
         ik = ij[p[k][1]]
         for l = 2:ne
            ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
         end
         Ψtensor[ik] = Ψ[j] * ε[k]
      end
   end

   A = 0.5 * alpha_lap * AΔ + AV

   # loop through different spin configurations
   sptr = zeros(Int64, ne)
   m1 = zeros(Int64, ne)
   mp1 = zeros(Int64, ne)
   m2 = zeros(Int64, ne)
   mp2 = zeros(Int64, ne)
   W = zeros(N, N, N^(ne - 2))
   M1 = zeros(N, N^(ne - 1))
   MA1 = zeros(N, N^(ne - 1))
   MC1 = zeros(N, N^(ne - 1))
   for s = 1:2^ne
      sp = sptr * N
      np = ntuple(x -> sp[x]+1:sp[x]+N, ne)
      for j = 1:ne # act A on the j-th particle
         φAtensor = getindex(Ψtensor, np...)
         φCtensor = copy(φAtensor)
         # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
         for i = 1:ne
            m1[1] = i
            m1[2:end] = setdiff(1:ne, i)
            MA = reshape(permutedims!(φAtensor1, φAtensor, m1), N, N^(ne - 1))
            MC = reshape(permutedims!(φCtensor1, φCtensor, m1), N, N^(ne - 1))
            if i == j
               mul!(MA1, A, MA)
            else
               mul!(MA1, C, MA)
            end
            mul!(MC1, C, MC)
            sortperm!(mp1, m1)
            permutedims!(φAtensor, reshape(MA1, ntuple(x -> N, ne)), mp1)
            permutedims!(φCtensor, reshape(MC1, ntuple(x -> N, ne)), mp1)
         end
         # assemble the value to Φtensor
         Φhtensor[np...] += φAtensor
         Φmtensor[np...] += φCtensor
      end

      for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
         k = coulomb_which2[j][1]
         l = coulomb_which2[j][2]
         φBtensor = getindex(Ψtensor, np...)
         # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
         for i = 1:ne
            if i != k && i != l
               m2[1] = i
               m2[2:end] = setdiff(1:ne, i)
               M = reshape(permutedims!(φBtensor1, φBtensor, m2), N, N^(ne - 1))
               mul!(M1, C, M)
               permutedims!(φBtensor, reshape(M1, ntuple(x -> N, ne)), sortperm!(mp2, m2))
            elseif i == k
               m2[1] = k
               m2[2] = l
               m2[3:end] = setdiff(1:ne, k, l)
               M = reshape(permutedims!(φBtensor1, φBtensor, m2), N, N, N^(ne - 2))
               @. W = 0.0
               @tensor W[i, j, k] = B[i, j, a, b] * M[a, b, k]
               permutedims!(φBtensor, reshape(W, ntuple(x -> N, ne)), sortperm!(mp2, m2))
            end
         end
         # assemble the value to Φtensor
         Φhtensor[np...] += φBtensor
      end
      # adjust sptr
      sptr[1] += 1
      for ℓ = 1:ne-1
         if sptr[ℓ] == 2
            sptr[ℓ] = 0
            sptr[ℓ+1] += 1
         end
      end
   end

   for i = 1:length(combBasis)
      il = combBasis[i]
      l = il[1]
      for j = 2:ne
         l += (il[j] - 1) * (2N)^(j - 1)
      end
      Φh[i] = Φhtensor[l]
      Φm[i] = Φmtensor[l]
   end
   return Φh, Φm ./ ne
end

# function ham_free_tensor(ne::Int, Ψ::Array{Float64,1},
#     AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
#     C::SparseMatrixCSC{Float64,Int64},
#     B::SparseMatrixCSC{Float64,Int64}; alpha_lap=1.0)
#     N = C.n
#     Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     Φhtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     Φmtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     φAtensor1 = zeros(Float64, ntuple(x -> N, ne))
#     φBtensor1 = zeros(Float64, ntuple(x -> N, ne))
#     φCtensor1 = zeros(Float64, ntuple(x -> N, ne))

#     # indecies for the basis
#     basis1body = 1:2*N
#     combBasis = collect(combinations(basis1body, ne))
#     # Φ = H⋅Ψ
#     @assert length(Ψ) == length(combBasis)
#     @assert ne > 1
#     Φh = zeros(size(Ψ))
#     Φm = zeros(size(Ψ))
#     # computate the permutations and paritiy
#     v = 1:ne
#     p = collect(permutations(v))[:]
#     ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
#     coulomb_which2 = collect(combinations(v, 2))
#     #ik = zeros(Int,ne)
#     # reshape the vector Ψ to the (antisymmetric) tensor
#     for j = 1:length(combBasis)
#         ij = combBasis[j]
#         for k = 1:length(p)
#             ik = ij[p[k][1]]
#             for l = 2:ne
#                 ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
#             end
#             Ψtensor[ik] = Ψ[j] * ε[k]
#         end
#     end

#     A = 0.5 * alpha_lap * AΔ + AV

#     # loop through different spin configurations
#     sptr = zeros(Int64, ne)
#     m1 = zeros(Int64, ne)
#     mp1 = zeros(Int64, ne)
#     m2 = zeros(Int64, ne)
#     mp2 = zeros(Int64, ne)
#     M1 = zeros(N, N^(ne - 1))
#     M2 = zeros(N^2, N^(ne - 2))
#     MA1 = zeros(N, N^(ne - 1))
#     MC1 = zeros(N, N^(ne - 1))
#     for s = 1:2^ne
#         sp = sptr * N
#         np = ntuple(x -> sp[x]+1:sp[x]+N, ne)
#         for j = 1:ne # act A on the j-th particle
#             φAtensor = getindex(Ψtensor, np...)
#             φCtensor = copy(φAtensor)
#             # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
#             for i = 1:ne
#                 m1[1] = i
#                 m1[2:end] = setdiff(1:ne, i)
#                 MA = reshape(permutedims!(φAtensor1, φAtensor, m1), N, N^(ne - 1))
#                 MC = reshape(permutedims!(φCtensor1, φCtensor, m1), N, N^(ne - 1))
#                 if i == j
#                     mul!(MA1, A, MA)
#                 else
#                     mul!(MA1, C, MA)
#                 end
#                 mul!(MC1, C, MC)
#                 sortperm!(mp1, m1)
#                 permutedims!(φAtensor, reshape(MA1, ntuple(x -> N, ne)), mp1)
#                 permutedims!(φCtensor, reshape(MC1, ntuple(x -> N, ne)), mp1)
#             end
#             # assemble the value to Φtensor
#             Φhtensor[np...] += φAtensor
#             Φmtensor[np...] += φCtensor
#         end

#         for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
#             k = coulomb_which2[j][1]
#             l = coulomb_which2[j][2]
#             φBtensor = getindex(Ψtensor, np...)
#             # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
#             for i = 1:ne
#                 if i != k && i != l
#                     m2[1] = i
#                     m2[2:end] = setdiff(1:ne, i)
#                     M = reshape(permutedims!(φBtensor1, φBtensor, m2), N, N^(ne - 1))
#                     mul!(M1, C, M)
#                     permutedims!(φBtensor, reshape(M1, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#                 elseif i == k
#                     m2[1] = k
#                     m2[2] = l
#                     m2[3:end] = setdiff(1:ne, k, l)
#                     M = reshape(permutedims!(φBtensor1, φBtensor, m2), N^2, N^(ne - 2))
#                     mul!(M2, B, M)
#                     permutedims!(φBtensor, reshape(M2, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#                 end
#             end
#             # assemble the value to Φtensor
#             Φhtensor[np...] += φBtensor
#         end
#         # adjust sptr
#         sptr[1] += 1
#         for ℓ = 1:ne-1
#             if sptr[ℓ] == 2
#                 sptr[ℓ] = 0
#                 sptr[ℓ+1] += 1
#             end
#         end
#     end

#     for i = 1:length(combBasis)
#         il = combBasis[i]
#         l = il[1]
#         for j = 2:ne
#             l += (il[j] - 1) * (2N)^(j - 1)
#         end
#         Φh[i] = Φhtensor[l]
#         Φm[i] = Φmtensor[l]
#     end
#     return Φh, Φm ./ ne
# end

ham_free_tensor(ne::Int, Ψ::Array{Float64,1}, ham::Hamiltonian) = ham_free_tensor(ne, Ψ, ham.AΔ, ham.AV, ham.C, ham.Bee; alpha_lap=ham.alpha_lap)


function ham_free_tensor_1ne(ne::Int, Ψ::Array{Float64,1},
   AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
   C::SparseMatrixCSC{Float64,Int64}; alpha_lap=1.0)
   N = C.n
   A = 0.5 * alpha_lap * AΔ + AV

   @views Ψ1 = Ψ[1:N]
   @views Ψ2 = Ψ[N+1:2N]
   Φh = vcat(A * Ψ1, A * Ψ2)
   Φm = vcat(C * Ψ1, C * Ψ2)

   return Φh, Φm
end

ham_free_tensor_1ne(ne::Int, Ψ::Array{Float64,1}, ham::Hamiltonian) = ham_free_tensor_1ne(ne, Ψ, ham.AΔ, ham.AV, ham.C; alpha_lap=ham.alpha_lap)



# # Compute H*Ψ and M*Ψ with Ψ is full.
# export ham_free_tensor

# function ham_free_tensor(ne::Int, Ψ::Array{Float64,1},
#     AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
#     C::SparseMatrixCSC{Float64,Int64},
#     B::Array{Float64,4}; alpha_lap=1.0)
#     N = C.n
#     Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     Φhtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     Φmtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     φAtensor1 = zeros(Float64, ntuple(x -> N, ne))
#     φBtensor1 = zeros(Float64, ntuple(x -> N, ne))
#     φCtensor1 = zeros(Float64, ntuple(x -> N, ne))

#     # indecies for the basis
#     basis1body = 1:2*N
#     combBasis = collect(combinations(basis1body, ne))
#     # Φ = H⋅Ψ
#     @assert length(Ψ) == length(combBasis)
#     Φh = zeros(size(Ψ))
#     Φm = zeros(size(Ψ))
#     # computate the permutations and paritiy
#     v = 1:ne
#     p = collect(permutations(v))[:]
#     ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
#     coulomb_which2 = collect(combinations(v, 2))
#     #ik = zeros(Int,ne)
#     # reshape the vector Ψ to the (antisymmetric) tensor
#     for j = 1:length(combBasis)
#         ij = combBasis[j]
#         for k = 1:length(p)
#             ik = ij[p[k][1]]
#             for l = 2:ne
#                 ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
#             end
#             Ψtensor[ik] = Ψ[j] * ε[k]
#         end
#     end

#     A = 0.5 * alpha_lap * AΔ + AV

#     # loop through different spin configurations
#     sptr = zeros(Int64, ne)
#     m1 = zeros(Int64, ne)
#     mp1 = zeros(Int64, ne)
#     m2 = zeros(Int64, ne)
#     mp2 = zeros(Int64, ne)
#     W = zeros(N, N, N^(ne - 2))
#     M1 = zeros(N, N^(ne - 1))
#     MA1 = zeros(N, N^(ne - 1))
#     MC1 = zeros(N, N^(ne - 1))
#     for s = 1:2^ne
#         sp = sptr * N
#         np = ntuple(x -> sp[x]+1:sp[x]+N, ne)
#         for j = 1:ne # act A on the j-th particle
#             φAtensor = getindex(Ψtensor, np...)
#             φCtensor = copy(φAtensor)
#             # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
#             for i = 1:ne
#                 m1[1] = i
#                 m1[2:end] = setdiff(1:ne, i)
#                 MA = reshape(permutedims!(φAtensor1, φAtensor, m1), N, N^(ne - 1))
#                 MC = reshape(permutedims!(φCtensor1, φCtensor, m1), N, N^(ne - 1))
#                 if i == j
#                     mul!(MA1, A, MA)
#                 else
#                     mul!(MA1, C, MA)
#                 end
#                 mul!(MC1, C, MC)
#                 sortperm!(mp1, m1)
#                 permutedims!(φAtensor, reshape(MA1, ntuple(x -> N, ne)), mp1)
#                 permutedims!(φCtensor, reshape(MC1, ntuple(x -> N, ne)), mp1)
#             end
#             # assemble the value to Φtensor
#             Φhtensor[np...] += φAtensor
#             Φmtensor[np...] += φCtensor
#         end

#         for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
#             k = coulomb_which2[j][1]
#             l = coulomb_which2[j][2]
#             φBtensor = getindex(Ψtensor, np...)
#             # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
#             for i = 1:ne
#                 if i != k && i != l
#                     m2[1] = i
#                     m2[2:end] = setdiff(1:ne, i)
#                     M = reshape(permutedims!(φBtensor1, φBtensor, m2), N, N^(ne - 1))
#                     mul!(M1, C, M)
#                     permutedims!(φBtensor, reshape(M1, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#                 elseif i == k
#                     m2[1] = k
#                     m2[2] = l
#                     m2[3:end] = setdiff(1:ne, k, l)
#                     M = reshape(permutedims!(φBtensor1, φBtensor, m2), N, N, N^(ne - 2))
#                     @. W = 0.0
#                     @tensor W[i, j, k] = B[i, j, a, b] * M[a, b, k]
#                     permutedims!(φBtensor, reshape(W, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#                 end
#             end
#             # assemble the value to Φtensor
#             Φhtensor[np...] += φBtensor
#         end
#         # adjust sptr
#         sptr[1] += 1
#         for ℓ = 1:ne-1
#             if sptr[ℓ] == 2
#                 sptr[ℓ] = 0
#                 sptr[ℓ+1] += 1
#             end
#         end
#     end

#     for i = 1:length(combBasis)
#         il = combBasis[i]
#         l = il[1]
#         for j = 2:ne
#             l += (il[j] - 1) * (2N)^(j - 1)
#         end
#         Φh[i] = Φhtensor[l]
#         Φm[i] = Φmtensor[l]
#     end
#     return Φh, Φm ./ ne
# end

# function ham_free_tensor(ne::Int, Ψ::Array{Float64,1},
#     AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
#     C::SparseMatrixCSC{Float64,Int64},
#     B::SparseMatrixCSC{Float64,Int64}; alpha_lap=1.0)
#     N = C.n
#     Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     Φhtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     Φmtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#     φAtensor1 = zeros(Float64, ntuple(x -> N, ne))
#     φBtensor1 = zeros(Float64, ntuple(x -> N, ne))
#     φCtensor1 = zeros(Float64, ntuple(x -> N, ne))

#     # indecies for the basis
#     basis1body = 1:2*N
#     combBasis = collect(combinations(basis1body, ne))
#     # Φ = H⋅Ψ
#     @assert length(Ψ) == length(combBasis)
#     Φh = zeros(size(Ψ))
#     Φm = zeros(size(Ψ))
#     # computate the permutations and paritiy
#     v = 1:ne
#     p = collect(permutations(v))[:]
#     ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
#     coulomb_which2 = collect(combinations(v, 2))
#     #ik = zeros(Int,ne)
#     # reshape the vector Ψ to the (antisymmetric) tensor
#     for j = 1:length(combBasis)
#         ij = combBasis[j]
#         for k = 1:length(p)
#             ik = ij[p[k][1]]
#             for l = 2:ne
#                 ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
#             end
#             Ψtensor[ik] = Ψ[j] * ε[k]
#         end
#     end

#     A = 0.5 * alpha_lap * AΔ + AV

#     # loop through different spin configurations
#     sptr = zeros(Int64, ne)
#     m1 = zeros(Int64, ne)
#     mp1 = zeros(Int64, ne)
#     m2 = zeros(Int64, ne)
#     mp2 = zeros(Int64, ne)
#     M1 = zeros(N, N^(ne - 1))
#     M2 = zeros(N^2, N^(ne - 2))
#     MA1 = zeros(N, N^(ne - 1))
#     MC1 = zeros(N, N^(ne - 1))
#     for s = 1:2^ne
#         sp = sptr * N
#         np = ntuple(x -> sp[x]+1:sp[x]+N, ne)
#         for j = 1:ne # act A on the j-th particle
#             φAtensor = getindex(Ψtensor, np...)
#             φCtensor = copy(φAtensor)
#             # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
#             for i = 1:ne
#                 m1[1] = i
#                 m1[2:end] = setdiff(1:ne, i)
#                 MA = reshape(permutedims!(φAtensor1, φAtensor, m1), N, N^(ne - 1))
#                 MC = reshape(permutedims!(φCtensor1, φCtensor, m1), N, N^(ne - 1))
#                 if i == j
#                     mul!(MA1, A, MA)
#                 else
#                     mul!(MA1, C, MA)
#                 end
#                 mul!(MC1, C, MC)
#                 sortperm!(mp1, m1)
#                 permutedims!(φAtensor, reshape(MA1, ntuple(x -> N, ne)), mp1)
#                 permutedims!(φCtensor, reshape(MC1, ntuple(x -> N, ne)), mp1)
#             end
#             # assemble the value to Φtensor
#             Φhtensor[np...] += φAtensor
#             Φmtensor[np...] += φCtensor
#         end

#         for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
#             k = coulomb_which2[j][1]
#             l = coulomb_which2[j][2]
#             φBtensor = getindex(Ψtensor, np...)
#             # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
#             for i = 1:ne
#                 if i != k && i != l
#                     m2[1] = i
#                     m2[2:end] = setdiff(1:ne, i)
#                     M = reshape(permutedims!(φBtensor1, φBtensor, m2), N, N^(ne - 1))
#                     mul!(M1, C, M)
#                     permutedims!(φBtensor, reshape(M1, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#                 elseif i == k
#                     m2[1] = k
#                     m2[2] = l
#                     m2[3:end] = setdiff(1:ne, k, l)
#                     M = reshape(permutedims!(φBtensor1, φBtensor, m2), N^2, N^(ne - 2))
#                     mul!(M2, B, M)
#                     permutedims!(φBtensor, reshape(M2, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#                 end
#             end
#             # assemble the value to Φtensor
#             Φhtensor[np...] += φBtensor
#         end
#         # adjust sptr
#         sptr[1] += 1
#         for ℓ = 1:ne-1
#             if sptr[ℓ] == 2
#                 sptr[ℓ] = 0
#                 sptr[ℓ+1] += 1
#             end
#         end
#     end

#     for i = 1:length(combBasis)
#         il = combBasis[i]
#         l = il[1]
#         for j = 2:ne
#             l += (il[j] - 1) * (2N)^(j - 1)
#         end
#         Φh[i] = Φhtensor[l]
#         Φm[i] = Φmtensor[l]
#     end
#     return Φh, Φm ./ ne
# end

# ham_free_tensor(ne::Int, Ψ::Array{Float64,1}, ham::Hamiltonian) = ham_free_tensor(ne, Ψ, ham.AΔ, ham.AV, ham.C, ham.Bee; alpha_lap=ham.alpha_lap)
