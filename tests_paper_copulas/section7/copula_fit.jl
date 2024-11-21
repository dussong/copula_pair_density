using Plots
using Measures

using PyCall
ot = pyimport("ot")

using JSON
using LinearAlgebra
using Optim
using Dierckx
using DataFrames

include("../../src/PairDensitiesTests.jl")
include("../../src/PairDensities.jl/PairDensities.jl")

using Main.PairDensities
using Main.PairDensities: density, WaveFunction, pair_density
using Main.PairDensitiesTests: opti_coef

nb_ex = 5
ts = range(0.0, 1.0, length=nb_ex)

ne = 2; # Nb of particles
L = 5.0; #Interval length
N = 150; #Nb of discretization points of the interval [-L,L]
α = 1.0; #Parameter in the potential
A = range(1., 3., length=nb_ex) #spatial coef for the potential
α_lap = 1.0



nf = 45 #spatial fine discretisation parameter
nspatial = 1000

x01nf = range(0.0, 1.0, length=nf)
x01N = range(0.0, 1.0, length=N - 1)
x01nspatial = range(0, 1, length=nspatial)

xLLnf = range(-L, L, length=nf)
xLLN = range(-L, L; length=N - 1)
xLLnspatial = range(-L, L, length=nspatial) #fine spatial grid


X = [[x01nf[i], x01nf[j]] for i in 1:nf for j in 1:nf]
# Xf = [[x01N[i], x01N[j]] for i in 1:N-1 for j in 1:N-1]

# -------------------------------------------------
#
# Generate data
#
# -------------------------------------------------


rho = []
rho2 = []
copula = []
icdf = []
cdf = []

for (i, a) in enumerate(A)
   @show i

   vext(x) = -α * (1.0 ./ sqrt.((x - a) .^ 2 .+ 1) .+ 1.0 ./ sqrt.((x + a) .^ 2 .+ 1))
   #double well taken from Wagner, L.O., Stoudenmire, E.M., Burke, K., White, S.R.: Reference electronic structure calculations in one dimension. Phys. Chem. Chem. Phys. 14, 8581–8590 (2012). https://doi.org/10.1039/c2cp24118h
   vee(x) = 1.0 ./ sqrt.(x .^ 2 .+ 1)

   # Construct the Hamiltonian
   ham = ham1d(L, N; alpha_lap=α, vext=vext, vee)
   # Solve the eigenvalue problem
   _, Ψ = WaveFunction(ne, ham, "FCI_full"; maxiter=500, kdim=3)

   ρ = density(Ψ, ham)
   ρ = (ρ * N * ne) / (norm(ρ, 1) * 2L)
   ρ2 = reshape(pair_density(Ψ, ham), N - 1, N - 1)
   ρ2 = (ρ2 * N^2 * binomial(ne, 2)) / (norm(ρ2, 1) * 4 * L^2)

   push!(rho, ρ)
   push!(rho2, ρ2)
   # Mean-field pair density
   ρ2MF = 0.5 * ρ * ρ'

   # Ratio to mean-field
   ratio_MF = 2 * ne / (ne - 1) * ρ2 ./ (ρ * ρ')

   # Transport map for ρ
   T = 2 * L / N / ne * cumsum(ρ)

   ind = findall(x -> (x != 0), T[2:end] - T[1:end-1])
   push!(ind, N - 1)
   cop = Spline2D(T[ind], T[ind], ratio_MF[ind, ind])

   # cop = evalgrid(spline, x01nf, x01nf)
   # push!(copula, reshape(cop, nf^2, 1))
   push!(copula, cop)


   sρ = Spline1D(xLLN, ρ) #spline representation of density
   Tf = [integrate(sρ, -L, y) for y in xLLnspatial] ./ ne #T: cumulative distr. function
   Tf = Tf ./ maximum(Tf) #transport map
   indf = [findall(x -> (x != 0), Tf[2:end] - Tf[1:end-1]); length(Tf)] #remove singularity
   # push!(TT, Tf)
   sT = Spline1D(xLLnspatial, Tf) #spline representation of cdf
   push!(cdf, sT)
   sF = Spline1D(Tf[indf], xLLnspatial[indf]) #spline representation of F (inverse of cdf)
   push!(icdf, sF)
end


# -------------------------------------------------
#
# Plot exact copula
#
# -------------------------------------------------

for (i, t) in enumerate(ts)
   cop = evalgrid(copula[i], x01nf, x01nf)
   P = contour(x01nf, x01nf, cop, c=:viridis, clim=(0.0, 2.5), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20,
   lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   size=(500,489),
   margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/exact_copula$i.png")
end


# Evaluation of the copulas on the coarse grid (for OT calculations)
XY = [[x01nf[i]; x01nf[j]] for i = 1:nf for j = 1:nf]
cop1 = evalgrid(copula[1], x01nf, x01nf)
copend = evalgrid(copula[end], x01nf, x01nf)
a0dens = reshape(1e-16 .+ cop1, nf * nf)
a1dens = reshape(1e-16 .+ copend, nf * nf)
Amat = zeros(length(a0dens), 2)
Amat[:, 1] = a0dens
Amat[:, 2] = a1dens

M = zeros(nf^2, nf^2)
for i in 1:nf^2
   for j in 1:nf^2
      M[i, j] = norm(XY[i] - XY[j])^2
   end
end
M = M ./ maximum(M)
reg = 1e-3


# -------------------------------------------------
#
# Compute optimal W2 barycenter on the copula
#
# -------------------------------------------------

# true barycenter (W2, computed with a Sinkhorn algorithm)
tsfine = range(0.0, 1.0, length=20)
barycopfineW2 = [a0dens]
for (i, t) in enumerate(tsfine[2:end-1])
   @show t
   c = [1 - t, t]
   @time bt = ot.barycenter(Amat, M, reg, c, method="sinkhorn_log", numItermax=1000)
   push!(barycopfineW2, bt)
end
push!(barycopfineW2, a1dens)


# Compute optimal W2 barycenter on the copula
baryw2 = []
error_baryw2 = []
error_baryL2 = []
for (i, t) in enumerate(ts)
   @show t
   # compute optimal projection for the copula
   cop = evalgrid(copula[i], x01nf, x01nf)
   adens = reshape(1e-16 .+ cop, nf * nf)
   D = [ot.emd2(barycopfineW2[j] ./ sum(barycopfineW2[j]), adens ./ sum(adens), M) for j in 1:length(barycopfineW2)]
   # D = [norm(barycopfineW2[j] - adens) for j in 1:length(barycopfineW2)]
   @show D
   @show d, ind = findmin(D)
   push!(baryw2, barycopfineW2[ind])
   push!(error_baryw2, d)
   push!(error_baryL2, norm(barycopfineW2[ind] .- adens) / (nf))
end
@show error_baryw2
@show error_baryL2

for (i, t) in enumerate(ts)
   P = contour(x01nf, x01nf, (reshape(baryw2[i], nf, nf))', c=:viridis, clim=(0.0, 2.5), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20,
   lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   size=(500,489),
   margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/copula_w2_bary$i.png")
end


# -------------------------------------------------
#
# Compute optimal linear barycenter on the copula
#
# -------------------------------------------------


# linear interpolation on the copula
tsfine = range(0.0, 1.0, length=20)
barycopfinelinear = [(1 - t) * a0dens + t * a1dens for t in tsfine]

# Compute optimal linear barycenter on the copula
barylinear = []
error_barylinearW2 = []
error_barylinearL2 = []
for (i, t) in enumerate(ts)
   @show t
   # compute optimal projection for the copula
   cop = evalgrid(copula[i], x01nf, x01nf)
   adens = reshape(1e-16 .+ cop, nf * nf)
   D = [norm(barycopfinelinear[j] .- adens) / (nf) for j in 1:length(barycopfinelinear)]
   @show D
   @show dl2, ind = findmin(D)
   dw2 = ot.emd2(barycopfinelinear[ind] ./ sum(barycopfinelinear[ind]), adens ./ sum(adens), M)
   push!(barylinear, barycopfinelinear[ind])
   push!(error_barylinearW2, dw2)
   push!(error_barylinearL2, dl2)
end
@show error_barylinearW2
@show error_barylinearL2


for (i, t) in enumerate(ts)
   P = contour(x01nf, x01nf, (reshape(barylinear[i], nf, nf))', c=:viridis, clim=(0.0, 2.5), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20, lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   size=(500,489),
   margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/copula_linear_bary$i.png")
end


# -------------------------------------------------
#
# Compute optimal sigmoid fit on the copula
#
# -------------------------------------------------

# define cost function
σ(x, λ) = 1 / (1 + exp(-λ * x))
σ2d(x, λ) = 2 * (σ(x[1] - 0.5, λ) * (1 - σ(x[2] - 0.5, λ)) + (1 - σ(x[1] - 0.5, λ)) * σ(x[2] - 0.5, λ))

barysigma = []
sigmaopt = []
dW2 = []
dL2 = []
for i in 1:nb_ex
   t = ts[i]
   @show t
   copi = evalgrid(copula[i], x01nf, x01nf)
   adens = reshape(1e-16 .+ copi, nf * nf)
   function cost_l2(λ)
      sum(abs.(σ2d(x, λ) - copi[j])^2 for (j, x) in enumerate(X))
   end
   res = optimize(cost_l2, 10.0, 1000.0)
   @show λopt = res.minimizer
   opt_value = res.minimum
   push!(barysigma, [σ2d(x, λopt) for x in X])
   push!(sigmaopt, λopt)

   d = ot.emd2([σ2d(x, λopt) for x in X] ./ sum([σ2d(x, λopt) for x in X]), adens ./ sum(adens), M)
   push!(dW2, d)
   push!(dL2, norm([σ2d(x, λopt) for x in X] - adens) / (nf))
end

for (i, t) in enumerate(ts)
   P = contour(x01nf, x01nf, (reshape(barysigma[i], nf, nf))', c=:viridis, clim=(0.0, 2.5), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20,
   lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   size=(500,489),
   margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/copula_sigmoid$i.png")
end

# -------------------------------------------------
#
# Extract tables with errors
#
# -------------------------------------------------

# W2 errors
D_W2 = Dict()
D_W2["linearcop"] = error_barylinearW2[2:end-1]
D_W2["W2cop"] = error_baryw2[2:end-1]
D_W2["sigmoidcop"] = dW2[2:end-1]
print(DataFrame(D_W2))

# 
D_L2 = Dict()
D_L2["linearcop"] = error_barylinearL2[2:end-1]
D_L2["W2cop"] = error_baryL2[2:end-1]
D_L2["sigmoidcop"] = dL2[2:end-1]
print(DataFrame(D_L2))



# -------------------------------------------------
#
# Compute pair densities from approximations of the copula
#
# -------------------------------------------------

# For the computation of the Coulomb energy
vee = (x -> 1.0 ./ sqrt.((x) .^ 2 .+ 1))
B = zeros(N - 1, N - 1)
for i in 1:N-1
   for j in 1:N-1
      B[i, j] = vee(xLLN[i] - xLLN[j])
   end
end

# Compute exact Coulomb energy
# ec_exact = [sum(sum(1.0 ./ B .* rho2[i])) for i in 1:nb_ex]
ec_exact = [sum(sum(B .* rho2[i])) for i in 1:nb_ex]

# -------------------------------------------------
# Plot exact pair density
# -------------------------------------------------
for (i, t) in enumerate(ts)
   P = contour(xLLN, xLLN, rho2[i], c=:viridis,
      clim=(0.0, 0.1),
      aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20,lw=0,
      xticks = -5:5:5, yticks = -5:5:5,
      size=(500,493),
      margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/rho2_exact$i.png")
end

# Check reconstruction error
for (i, t) in enumerate(ts)
   Tgrid = evaluate(cdf[i], xLLN)
   cop_sp = evalgrid(copula[i], Tgrid, Tgrid)
   rho2_s = (ne - 1) / (2 * ne) * (rho[i] * rho[i]') .* cop_sp
   @assert norm(rho2_s - rho2[i])/N < 1e-3
end

# Construct
rho2_small = []
for i in 1:nb_ex
   spline = Spline2D(xLLN, xLLN, rho2[i])
   push!(rho2_small, evalgrid(spline, xLLnf, xLLnf))
end
rho2_small


# -------------------------------------------------
# Pair density from sigmoid fit
# -------------------------------------------------
rho2_sigma = []
for (i, t) in enumerate(ts)
   λ = sigmaopt[i]
   Tgrid = evaluate(cdf[i], xLLN)
   XTf = [[Tgrid[i], Tgrid[j]] for i in 1:N-1 for j in 1:N-1]
   cop_sp = reshape([σ2d(x, λ) for x in XTf], N - 1, N - 1)
   rho2_s = (ne - 1) / (2 * ne) * (rho[i] * rho[i]') .* cop_sp
   push!(rho2_sigma, rho2_s)
end
# push!(rho2_sigma, rho2[end])

for (i, t) in enumerate(ts)
   P = contour(xLLN, xLLN, rho2_sigma[i], c=:viridis,
      clim=(0.0, 0.1), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20, lw =0,
      xticks = -5:5:5, yticks = -5:5:5,
      size=(500,493),
      margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/rho2_sigmoid$i.png")
end


dW2_rho2_sigma = []
dL2_rho2_sigma = []
for (i, t) in enumerate(ts)
   spline = Spline2D(xLLN, xLLN, rho2_sigma[i]) #projection on smaller grid for fast evaluation of W2 distance
   r1 = reshape(evalgrid(spline, xLLnf, xLLnf), nf*nf) .+ 1e-16
   r1 = r1 ./ sum(r1)
   r2 = reshape(rho2_small[i], nf * nf) .+ 1e-16
   r2 = r2 ./ sum(r2)
   d = ot.emd2(r1,r2,M)
   push!(dL2_rho2_sigma, norm(rho2_sigma[i] - rho2[i]) / N * (2 * L))
   push!(dW2_rho2_sigma, d)
end
@show dW2_rho2_sigma
@show dL2_rho2_sigma

# ec_rho2_sigma = [sum(sum(1.0 ./ B .* rho2_sigma[i])) for i in 1:nb_ex]
ec_rho2_sigma = [sum(sum(B .* rho2_sigma[i])) for i in 1:nb_ex]
error_ec_rho2_sigma = abs.(ec_rho2_sigma - ec_exact)./ec_exact

# -------------------------------------------------
# Pair density from Wasserstein barycenter of copula
# -------------------------------------------------

rho2_w2_from_cop = []
for (i, t) in enumerate(ts)
   spline = Spline2D(x01nf, x01nf, reshape(baryw2[i], nf, nf))
   Tgrid = evaluate(cdf[i], xLLN)
   cop_sp = evalgrid(spline, Tgrid, Tgrid)
   rho2_s = (ne - 1) / (2 * ne) * (rho[i] * rho[i]') .* cop_sp
   push!(rho2_w2_from_cop, rho2_s)
end

dW2_rho2_cop = []
dL2_rho2_cop = []
for (i, t) in enumerate(ts)
   spline = Spline2D(xLLN, xLLN, rho2_w2_from_cop[i]) #projection on smaller grid for fast evaluation of W2 distance
   r1 = reshape(evalgrid(spline, xLLnf, xLLnf), nf * nf) .+ 1e-16
   r1 = r1 ./ sum(r1)
   r2 = reshape(rho2_small[i], nf * nf) .+ 1e-16
   r2 = r2 ./ sum(r2)
   # @show r2
   d = ot.emd2(r1, r2, M)
   push!(dL2_rho2_cop, norm(rho2_w2_from_cop[i] - rho2[i]) / N * (2 * L))
   push!(dW2_rho2_cop, d)
end
@show dW2_rho2_cop
@show dL2_rho2_cop

# ec_rho2_cop = [sum(sum(1.0 ./ B .* rho2_w2_from_cop[i])) for i in 1:nb_ex]
ec_rho2_cop = [sum(sum(B .* rho2_w2_from_cop[i])) for i in 1:nb_ex]
error_ec_rho2_cop = abs.(ec_rho2_cop - ec_exact) ./ ec_exact

for (i, t) in enumerate(ts)
   P = contour(xLLN, xLLN, max.(rho2_w2_from_cop[i],zeros(size(rho2_w2_from_cop[i]))), c=:viridis,
      clim=(0.0, 0.1), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20, lw =0,
      xticks = -5:5:5, yticks = -5:5:5,
      size=(500,493),
      margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/rho2_w2_from_cop$i.png")
end

# -------------------------------------------------
# Pair density from linear barycenter of copula
# -------------------------------------------------
rho2_linear_from_cop = []
for (i, t) in enumerate(ts)
   spline = Spline2D(x01nf, x01nf, reshape(barylinear[i], nf, nf))
   Tgrid = evaluate(cdf[i], xLLN)
   cop_sp = evalgrid(spline, Tgrid, Tgrid)
   rho2_s = (ne - 1) / (2 * ne) * (rho[i] * rho[i]') .* cop_sp
   push!(rho2_linear_from_cop, rho2_s)
end

dW2_rho2_linear_cop = []
dL2_rho2_linear_cop = []
for (i, t) in enumerate(ts)
   spline = Spline2D(xLLN, xLLN, rho2_linear_from_cop[i]) #projection on smaller grid for fast evaluation of W2 distance
   r1 = reshape(evalgrid(spline, xLLnf, xLLnf), nf * nf) .+ 1e-16
   r1 = r1 ./ sum(r1)
   r2 = reshape(rho2_small[i], nf * nf) .+ 1e-16
   r2 = r2 ./ sum(r2)
   d = ot.emd2(r1, r2, M)
   push!(dL2_rho2_linear_cop, norm(rho2_linear_from_cop[i] - rho2[i]) / N * (2 * L))
   push!(dW2_rho2_linear_cop, d)
end
@show dW2_rho2_linear_cop
@show dL2_rho2_linear_cop

# ec_rho2_linear_cop = [sum(sum(1.0 ./ B .* rho2_linear_from_cop[i])) for i in 1:nb_ex]
ec_rho2_linear_cop = [sum(sum(B .* rho2_linear_from_cop[i])) for i in 1:nb_ex]
error_ec_rho2_linear_cop = abs.(ec_rho2_linear_cop - ec_exact) ./ ec_exact

for (i, t) in enumerate(ts)
   P = contour(xLLN, xLLN, max.(rho2_linear_from_cop[i],zeros(size(rho2_linear_from_cop[i]))), c=:viridis,
      clim=(0.0, 0.1), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20,
      lw =0,
      xticks = -5:5:5, yticks = -5:5:5,
      size=(500,493),
      margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/rho2_w2_from_linear_cop$i.png")
end

# # -------------------------------------------------
# # Pair density from Wasserstein barycenter of pair density - removed
# # -------------------------------------------------

# XY = [[xLLN[i]; xLLN[j]] for i = 1:nf for j = 1:nf]
# a0dens = reshape(1e-16 .+ rho2_small[1], nf * nf)
# a1dens = reshape(1e-16 .+ rho2_small[end], nf * nf)
# Amat = zeros(length(a0dens), 2)
# Amat[:, 1] = a0dens
# Amat[:, 2] = a1dens

# M = zeros(nf^2, nf^2)
# for i in 1:nf^2
#    for j in 1:nf^2
#       M[i, j] = norm(XY[i] - XY[j])^2
#    end
# end
# M = M ./ maximum(M)
# reg = 1e-3

# # true barycenter (W2, computed with a Sinkhorn algorithm)
# icdftrain = [icdf[1], icdf[end]]
# baryrho2 = [a0dens]
# for (i, t) in enumerate(ts[2:end-1])
#    @show t
#    # compute optimal projection for the density
#    @show c = opti_coef(icdftrain, icdf[i+1]; ns=10000)
#    @time bt = ot.barycenter(Amat, M, reg, c, method="sinkhorn_log", numItermax=1000)
#    push!(baryrho2, bt)
# end
# push!(baryrho2, a1dens)

# # Compute pair density back on reference grid
# rho2_w2_back = []
# for i in 1:nb_ex
#    spline = Spline2D(xLLnf, xLLnf, reshape(baryrho2[i], nf, nf))
#    push!(rho2_w2_back, evalgrid(spline, xLLN, xLLN))
# end

# for (i, t) in enumerate(ts)
#    P = contour(xLLN, xLLN, rho2_w2_back[i], c=:viridis,
#       clim=(0.0, 0.1), aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20)
#    display(P)
#    sleep(0.5)
#    @show (sum(sum(rho2_w2_back[i])) * 2 * L) / N
#    savefig("plots_test_cases_paper_copulas/section7/rho2_w2_back$i.pdf")
# end

# # Computation of errors
# dW2_rho2_back = []
# dL2_rho2_back = []
# for (i, t) in enumerate(ts)
#    #projection on smaller grid for fast evaluation of W2 distance
#    r1 = reshape(baryrho2[i], nf * nf) .+ 1e-16
#    r1 = r1 ./ sum(r1)
#    r2 = reshape(rho2_small[i], nf * nf) .+ 1e-16
#    r2 = r2 ./ sum(r2)
#    d = ot.emd2(r1, r2, M)
#    push!(dL2_rho2_back, norm(rho2_w2_back[i] - rho2[i]) / N * (2 * L))
#    push!(dW2_rho2_back, d)
# end
# @show dW2_rho2_back
# @show dL2_rho2_back

# ec_rho2_back = [sum(sum(1.0 ./ B .* rho2_w2_back[i])) for i in 1:nb_ex]
# error_ec_rho2_sigma = abs.(ec_rho2_back - ec_exact) ./ ec_exact

# -------------------------------------------------
# Pair density using LDA approximation
# -------------------------------------------------
# rho2_lda0
h(s) = sin(s)/ s
# h(s) = (3 * (sin(s)- s*cos(s)))/ s^3
rho2lda = []
coplda = []
for i in 1:nb_ex
   @show i
   rho2ldai = zeros(N - 1, N - 1)
   for k in 1:N-1
      for l in 1:N-1
         if k == l
            rho2ldai[k, l] = 0.5 * rho[i][k] * rho[i][l] - 1 / 8 * rho[i][k]^2 - 1 / 8 * rho[i][l]^2
         else
            rho2ldai[k, l] = (0.5 * rho[i][k] * rho[i][l] 
                              - 1 / 8 * (rho[i][k] * h(pi/2 * rho[i][k] * (xLLN[k] - xLLN[l])))^2 
                              - 1 / 8 * (rho[i][l] * h(pi/2 * rho[i][l] * (xLLN[k] - xLLN[l])))^2
                              )  
         end
      end
   end
   rho2ldai = max.(rho2ldai, zeros(size(rho2ldai)))
   rhox = sum(rho2ldai[:,l] for l in 1:N-1)
   rhoy = sum(rho2ldai[k,:] for k in 1:N-1)

   # Ratio to mean-field
   ratio_MF = 2 * ne / (ne - 1) * rho2ldai ./ (rho[i] * rho[i]')
   # Transport map for ρ
   T = 2 * L / N / ne * cumsum(rho[i])

   ind = findall(x -> (x != 0), T[2:end] - T[1:end-1])
   push!(ind, N - 1)
   cop = Spline2D(T[ind], T[ind], ratio_MF[ind, ind])
   # cop2 = reshape(evalgrid(cop, x01nf, x01nf), nf, nf)
   # @show size(cop2)
   push!(coplda, cop)
   @show sum(sum(rho2ldai))*(2*L/N)^2
   push!(rho2lda, rho2ldai)
end

rho2lda
coplda

for (i, t) in enumerate(ts)
   cop = evalgrid(coplda[i], x01nf, x01nf)
   P = contour(x01nf, x01nf, max.(cop, zeros(size(cop))), c=:viridis, clim=(0.0, 2.5),  fill=:true, colorbar=:false, tickfontsize=20, lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   aspect_ratio=:equal,
   size=(500,489),
   margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/copula_lda$i.png")
end

ec_rho2_lda2 = [sum(sum(B .* rho2lda[i])) for i in 1:nb_ex]
error_ec_rho2_lda2 = abs.(ec_rho2_lda2 - ec_exact) ./ ec_exact


for (i, t) in enumerate(ts)
   P = contour(xLLN, xLLN, max.(rho2lda[i], zeros(size(rho2lda[i]))), c=:viridis,
      clim=(0.0, 0.1),
      aspect_ratio=:equal, fill=:true, colorbar=:false, tickfontsize=20,
      lw =0,
      xticks = -5:5:5, yticks = -5:5:5,
      size=(500,493),
      margin=4mm)
   display(P)
   sleep(0.5)
   savefig("plots_test_cases_paper_copulas/section7/rho2_lda$i.png")
end





# # rho2_lda
# h(s) = 3 * (sin(s) − s * cos(s)) / s^3
# rho2lda = []
# for i in 1:nb_ex
#    @show i
#    rho2ldai = zeros(N - 1, N - 1)
#    for k in 1:N-1
#       for l in 1:N-1
#          if k == l
#             rho2ldai[k, l] = 0.5 * rho[i][k] * rho[i][l] - 1 / 8 * rho[i][k] - 1 / 8 * rho[i][l]
#          else
#             rho2ldai[k, l] = 0.5 * rho[i][k] * rho[i][l] - 1 / 8 * rho[i][k] * (h((3 * pi^2 * rho[i][k])^(1 / 3) * abs.(xLLN[k] - xLLN[l])))^2 - 1 / 8 * rho[i][l] * (h((3 * pi^2 * rho[i][l])^(1 / 3) * abs.(xLLN[k] - xLLN[l])))^2
#          end
#       end
#    end
#    @show sum(sum(rho2ldai))*2*L/N
#    push!(rho2lda, rho2ldai)
# end

# cx = 3/4*(3/pi)^(1/3)
# ec_rho2_lda2 = (2 * L / N)^(-2)* [0.5 * (2 * L / N)^2 * sum(sum(1.0 ./ B .* (rho[i] * rho'[i]))) - cx * (2 * L / N) * sum((rho[i].^(4/3))) for i in 1:nb_ex]
# ec_rho2_lda2 = (2 * L / N)^(-2)* [0.5 * (2 * L / N)^2 * sum(sum(B .* (rho[i] * rho'[i]))) - cx * (2 * L / N) * sum((rho[i].^(4/3))) for i in 1:nb_ex]
ec_rho2_lda2 = [sum(sum(B .* rho2lda[i])) for i in 1:nb_ex]
error_ec_rho2_lda2 = abs.(ec_rho2_lda2 - ec_exact) ./ ec_exact

# -------------------------------------------------
#
# Extract tables with errors
#
# -------------------------------------------------

# W2 errors
D_W2_rho2 = Dict()
D_W2_rho2["linearcop"] = dW2_rho2_linear_cop[2:end-1]
D_W2_rho2["W2cop"] = dW2_rho2_cop[2:end-1]
D_W2_rho2["sigmoidcop"] = dW2_rho2_sigma[2:end-1]
print(DataFrame(D_W2_rho2))

# L2 erros
D_L2_rho2 = Dict()
D_L2_rho2["linearcop"] = dL2_rho2_linear_cop[2:end-1]
D_L2_rho2["W2cop"] = dL2_rho2_cop[2:end-1]
D_L2_rho2["sigmoidcop"] = dL2_rho2_sigma[2:end-1]
print(DataFrame(D_L2_rho2))

# Errors on Coulomb energy
D_C_rho2 = Dict()
D_C_rho2["linearcop"] = error_ec_rho2_linear_cop[2:end-1]
D_C_rho2["W2cop"] = error_ec_rho2_cop[2:end-1]
D_C_rho2["sigmoidcop"] = error_ec_rho2_sigma[2:end-1]
D_C_rho2["lda"] = error_ec_rho2_lda2[2:end-1]
print(DataFrame(D_C_rho2))

# # -------------------------------------------------
# #
# # Exchange-correlation hole
# #
# # -------------------------------------------------
sprho2 = []
sprho2_cop = []
sprho2_sigma = []
sprho2_linear_from_cop = []
sprho2_lda = []
for i in 1:length(rho2)
   push!(sprho2, Spline2D(collect(xLLN), collect(xLLN), rho2[i]))
   # push!(sprho2_cop, Spline2D(collect(xLLN), collect(xLLN), rho2_w2_from_cop[i]))
   # push!(sprho2_sigma, Spline2D(collect(xLLN), collect(xLLN), rho2_sigma[i]))
   # push!(sprho2_linear_from_cop, Spline2D(collect(xLLN), collect(xLLN), rho2_linear_from_cop[i]))
   push!(sprho2_lda, Spline2D(collect(xLLN), collect(xLLN), rho2lda[i]))
end

sprho = []
for i in 1:length(rho)
   push!(sprho, Spline1D(collect(xLLN), rho[i]))
end

function exch_hole(sprho2, sprho, x, r)
   # 1 / binomial(ne, 2) * (evalgrid(sprho2, [x], [x + r])[1] + evalgrid(sprho2, [x], [x - r])[1])
   0.5*(evalgrid(sprho2, [x], [x+r])[1] / sprho(x) - sprho(x+r)
       +evalgrid(sprho2, [x], [x-r])[1] / sprho(x) - sprho(x-r))
end

r = range(0, 2, length=50)
for i in 1:length(rho2)
   P = plot()
   P = plot!(r, [exch_hole(sprho2[i], sprho[i], A[i], rr) for rr in r], label="exact", legendfontsize=10, legend=:bottomright, color= :black, linewidth=2.5, tickfontsize=16)
   # P = plot!(r, [exch_hole(sprho2_linear_from_cop[i], sprho[i], A[i], rr) for rr in r], label="linear barycenter copula", color= :blue, linewidth=2.5, tickfontsize=16)
   # P = plot!(r, [exch_hole(sprho2_cop[i], sprho[i], A[i], rr) for rr in r], label="Wasserstein barycenter copula", color= :red, linewidth=2.5)
   # P = plot!(r, [exch_hole(sprho2_sigma[i], sprho[i], A[i], rr) for rr in r], label="one-parameter neural net copula", color= :green, linewidth=2.5)
   P = plot!(r, [exch_hole(sprho2_lda[i], sprho[i], A[i], rr) for rr in r], label="LDA-0", color= :purple, linewidth=2.5)
   P = xlabel!("r")
   display(P)
   savefig("plots_test_cases_paper_copulas/section7/exch_corrlda$i.pdf")
end


# --------------------------------
# compute relative integrals
# --------------------------------

cop = copula[4]
int_left_up = integrate(cop, 0., 0.25, 0.75, 1.)
int_left_down = integrate(cop, 0., 0.25, 0.5, 0.75)
int_right_down = integrate(cop, 0.25, 0.5, 0.5, 0.75)
int_right_up = integrate(cop, 0.25, 0.5, 0.75, 1.)

sum_square = int_left_up + int_left_down + int_right_down + int_right_up

rel_left_up = int_left_up / sum_square
rel_left_down = int_left_down / sum_square
rel_right_down = int_right_down / sum_square
rel_right_up = int_right_up / sum_square


# --------------------------------
# compute copula for SCE
# --------------------------------
# N = 2
nss = 500
XX = range(0, 1, length=nss)
g1 = [mod(x+ 1/2,1) for x in XX]

copsce2 = zeros(nss,nss)
for i in 1:nss
   for j in 1:nss
      if abs.(g1[i] - XX[j]) < 1e-2
         copsce2[i,j] = 1.
      end
   end
end

P = contour(XX, XX, copsce2, c=:viridis, clim=(0.0, 1.),  fill=:true, colorbar=:false, tickfontsize=20, lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   aspect_ratio=:equal,
   size=(500,489),
   margin=4mm)
   display(P)
   savefig("plots_test_cases_paper_copulas/section6/copsce2part.png")


# N = 3
nss = 500
XX = range(0, 1, length=nss)
g1 = [mod(x+ 1/3,1) for x in XX]
g2 = [mod(x+ 2/3,1) for x in XX]

copsce3 = zeros(nss,nss)
for i in 1:nss
   for j in 1:nss
      if (abs.(g1[i] - XX[j]) < 1e-2)||(abs.(g2[i] - XX[j]) < 1e-2)
         copsce3[i,j] = 1.
      end
   end
end

P = contour(XX, XX, copsce3, c=:viridis, clim=(0.0, 1.),  fill=:true, colorbar=:false, tickfontsize=20, lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   aspect_ratio=:equal,
   size=(500,489),
   margin=4mm)
   display(P)
   savefig("plots_test_cases_paper_copulas/section6/copsce3part.png")

   # N = 4
nss = 500
XX = range(0, 1, length=nss)
g1 = [mod(x+ 1/4,1) for x in XX]
g2 = [mod(x+ 2/4,1) for x in XX]
g3 = [mod(x+ 3/4,1) for x in XX]

copsce4 = zeros(nss,nss)
for i in 1:nss
   for j in 1:nss
      if (abs.(g1[i] - XX[j]) < 1e-2)||(abs.(g2[i] - XX[j]) < 1e-2)||(abs.(g3[i] - XX[j]) < 1e-2)
         copsce4[i,j] = 1.
      end
   end
end

P = contour(XX, XX, copsce4, c=:viridis, clim=(0.0, 1.),  fill=:true, colorbar=:false, tickfontsize=20, lw=0,
   xticks = 0:0.5:1, yticks = 0:0.5:1,
   aspect_ratio=:equal,
   size=(500,489),
   margin=4mm)
   display(P)
   savefig("plots_test_cases_paper_copulas/section6/copsce4part.png")



# # Plot copula gaussian distribution
# θ = 0.4
# Σ = [cos(θ) sin(θ); -sin(θ) cos(θ)]*[1 0; 0 5]*[cos(θ) -sin(θ); sin(θ) cos(θ)]
# isposdef(Σ)
# invΣ = inv(Σ)
# ρ2g = [ exp(-1/2* [x; y]'* invΣ * [x;y]) for x in xLLN, y in xLLN]

# ss = sum(sum(ρ2g))

# ρ2gn = ρ2g / ss

# P = contour(xLLN, xLLN, ρ2gn, c=:viridis, aspect_ratio=:equal, fill=:true, colorbar=:true, tickfontsize=20,
#       lw=0,
#       size=(500,489),
#       margin=4mm)
# savefig("plots_test_cases_paper_copulas/section7/gaussian.png")

# ρx = [sum(ρ2gn[:,i]) for i in 1:N-1]
# sum(ρx)
# ρy = [sum(ρ2gn[j,:]) for j in 1:N-1]
# sum(ρy)

# Tx = cumsum(ρx)
# Ty = cumsum(ρy)

# ratioMF = ρ2gn ./ (ρx * ρy')

# indx = findall(x -> (x != 0), Tx[2:end] - Tx[1:end-1])
# indy = findall(x -> (x != 0), Ty[2:end] - Ty[1:end-1])

# copg = Spline2D(Tx[indx], Ty[indy], ratioMF[indx, indy])

# cop = evalgrid(copg, x01nf, x01nf)
# P = contour(x01nf, x01nf, cop, c=:viridis, aspect_ratio=:equal, fill=:true, colorbar=:true, tickfontsize=20,
#       clims = (0., 2.),
#       lw=0,
#       xticks = 0:0.5:1, yticks = 0:0.5:1,
#       size=(500,489),
#       margin=4mm)
#       display(P)
# savefig("plots_test_cases_paper_copulas/section7/gaussian_copula.png")


