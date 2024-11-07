using JuMP, Ipopt
using LinearAlgebra
using TensorOperations

function dotF(f1, f2, x, nspatial)
   return dot(f1(x), f2(x)) / nspatial
end

function opti_coef(Ftrain, Ftest; ns=10000)
   # solve optimisation problem for best coefficients in W2 approx of density
   xf01 = range(0.0, 1.0, length=ns)
   k = length(Ftrain)
   model = Model(Ipopt.Optimizer)
   n = 1:k
   @variable(model, x[n], lower_bound = 0.0, upper_bound = 1.0)
   @constraint(model, sum(x) == 1.0)
   A = [dotF(Ftrain[i1], Ftrain[i2], xf01, ns) for i1 in 1:k, i2 in 1:k]
   b = [dotF(Ftest, Ftrain[i], xf01, ns) for i in 1:k]
   @objective(model, Min, sum(A[i, j] * x[i] * x[j] for i = 1:k, j = 1:k)
                          -
                          2 * sum(b[i] * x[i] for i = 1:k))
   status = optimize!(model)
   return value.(x)
end



# function logsumexp(mat, dims)
#    max_ = maximum(mat, dims=dims)
#    exp_mat = exp.(mat .- max_)
#    sum_exp_ = sum(exp_mat, dims=dims)
#    log.(sum_exp_) .+ max_
# end

# softmin(A, eps, axis) = -eps * logsumexp(-A / eps, axis)
# S(C, f, g) = C - f * (ones(size(f))') - ones(size(g)) * g'
# function S!(C, f, g, mat_temp)
#    for i in 1:size(C, 1)
#       for j in 1:size(C, 2)
#          mat_temp[i, j] = C[i, j] - f[i] - g[j]
#       end
#    end
#    return mat_temp
# end

# function barycenters_dual(a_vec, lambda_vec, C, eps, niter)
#    gl = zeros(size(a_vec))
#    fl = zeros(size(a_vec))
#    N, T = size(a_vec)
#    bary = zeros((N, 1))
#    temp = zeros((N, 1))
#    mat_temp = zeros(size(C))
#    logavec = log.(a_vec)
#    for i in 1:niter
#       @show i
#       for s in 1:T
#          fl[:, s] += eps * logavec[:, s] + softmin(S(C, fl[:, s], gl[:, s]), eps, 2)
#       end
#       temp = bary
#       for s in 1:T
#          # S!(C, fl[:,s], -bary, mat_temp)
#          # temp -= lambda_vec[s]*softmin(mat_temp, eps, 1)'
#          temp -= lambda_vec[s] * softmin(S(C, fl[:, s], -bary), eps, 1)'
#       end
#       bary = temp
#       for s in 1:T
#          # S!(C, fl[:,s], gl[:,s], mat_temp)
#          # gl[:,s] += bary + softmin(mat_temp, eps, 1)'
#          gl[:, s] += bary + softmin(S(C, fl[:, s], gl[:, s]), eps, 1)'
#       end
#    end
#    bartrue = exp.(bary / eps)
#    return fl, gl, bary, bartrue
# end


# # function barycenters_dual(a_vec, lambda_vec, C, eps, niter)
# #     gl = zeros(size(a_vec))
# #     fl = zeros(size(a_vec))
# #     mcv = Vector{Float64}() # Marginal constraint violation
# #     T = size(a_vec)[2]
# #     N = size(a_vec)[1]
# #     bary = zeros((N,1))
# #     temp = zeros((N,1))
# #     bartrue = ones((N,1))
# #     for i in 1:niter
# #         @show i
# #         for s in 1:T
# #             fl[:,s] = eps*log.(a_vec[:,s]) + fl[:,s] + softmin(S(C, fl[:,s], gl[:,s]), eps, 2)
# #         end
# #         temp = bary
# #         for s in 1:T
# #            temp = temp - lambda_vec[s]*softmin(S(C, fl[:,s],-bary), eps, 1)'
# #         end
# #         bary = temp
# #         for s in 1:T
# #             gl[:,s] = bary + gl[:,s] + softmin(S(C, fl[:,s], gl[:,s]), eps, 1)'
# #         end
# #         bartrue = exp.(bary/eps )
# #     end
# #     return fl, gl, bary, bartrue
# # end


# function W2_bar(mat_train, coef; eps=0.05, niter=4, L=5)
#    N = Int(sqrt(length(mat_train[1]))) + 1
#    xcLL = range(-L, L, length=N - 1)
#    x = collect(xcLL)
#    y = collect(xcLL)
#    un1d = ones(N - 1)
#    C1d = (x * un1d' - un1d * y') .^ 2
#    un2d = ones(N - 1, N - 1)
#    @tensor begin
#       Cmat[a, b, c, d] := C1d[a, c] * un2d[b, d] + C1d[b, d] * un2d[a, c]
#    end
#    C = reshape(Cmat, (N - 1) * (N - 1), (N - 1) * (N - 1))
#    K = exp.(-C / eps)

#    k = length(mat_train)
#    avec = zeros(((N - 1) * (N - 1)), k)
#    for i in 1:k
#       avec[:, i] = reshape(mat_train[i], (N - 1)^2, 1)
#    end
#    lambdavec = [coef[i] for i = 1:k]
#    fl, gl, bary, bartrue = barycenters_dual(avec, lambdavec, C, eps, niter)
#    bartruenorm = bartrue / sum(bartrue) * sum(mat_train[1])
#    mat_test = reshape(bartruenorm, N - 1, N - 1)
# end



# function sinkhorn_dual(a, b, C, eps, niter)
#     gl = zeros(size(a))
#     fl = zeros(size(b))
#     mcv = Vector{Float64}() #marginal constraint violation
#     for i in 1:niter
#         print("i = ")
#         println(i)
#         # Attempt with logsumexp trick in Peyre's book: subtract previously computed scalings
#         fl = eps*log.(a) + fl + softmin(S(C, fl, gl), eps, 2)
#         gl = eps*log.(b) + gl + softmin(S(C, fl, gl), eps, 1)'
#         Pi = exp.( -S(C, fl, gl)/eps )
#         push!(mcv, sum(abs.(sum(Pi, dims=1)'-b))) # Marginal constraint violation
#     end
#     P = exp.( -S(C, fl, gl)/eps )
#     # Transport plan P = np.diag(a)@ K @ np.diag(b)
#     cost = sum(P.*C)
#     return fl, gl, P, cost, mcv
# end
