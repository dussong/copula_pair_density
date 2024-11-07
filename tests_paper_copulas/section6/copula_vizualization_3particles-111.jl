using Plots
pyplot()

using JSON
using LinearAlgebra
using Dierckx

include("../../src/PairDensitiesTests.jl")
include("../../src/PairDensities.jl/PairDensities.jl")

using Main.PairDensities
using Main.PairDensities: density, WaveFunction

ne = 3; # Nb of particles
L = 10.0; #Interval length
N = 50; #Nb of discretization points of the interval [-L,L]
α = 1.0; #Parameter in the potential
A = [2., 3.5, 7.] #spatial coef for the potential
α_lap = 1.0

xx = range(-L, L; length=N - 1)
xfine = range(-L, L; length=200)

for (i, a) in enumerate(A)
   @show i

   # β = α*(sqrt(a^2+1)*(1+1/sqrt(4*a^2+1))-2)/(sqrt(a^2 +1)-1) #guarantees that the values of the potential is equal in -a,0,and a.

   # vext(x) = -α * (1.0 ./ sqrt.((x - a) .^ 2 .+ 1) .+ 1.0 ./ sqrt.((x + a) .^ 2 .+ 1)) .- β ./ sqrt.(x.^ 2 .+ 1)
   vext(x) = -α * (1.0 ./ sqrt.((x - a) .^ 2 .+ 1) .+ 1.0 ./ sqrt.((x + a) .^ 2 .+ 1) .+ 1.0 ./ sqrt.(x.^ 2 .+ 1))

   @show (-a,0,a)
   # vext(x) = -α * (1.0 ./ abs.(x - a) .+ 1.0 ./ abs.(x + a) .+ 1.0 ./ abs.(x))
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

   ρsp = Spline1D(xx,ρ)

   # Mean-field pair density
   ρ2MF = 0.5 * ρ * ρ'

   # Ratio to mean-field
   ratio_MF = 2 * ne / (ne - 1) * ρ2 ./ (ρ * ρ')

   # Transport map for ρ
   T = 2 * L / N / ne * cumsum(ρ)

   # P = plot(xfine, vext.(xfine), label="", xlims=(-L, L), linewidth=4, tickfontsize=16)

   P = plot(xfine, vext.(xfine), label="", xlims=(-L, L), color=:blue, linewidth=4, tickfontsize=16, ylabel="Potential", yguidefontcolor=:blue, labelfontsize=16, legendfontsize=12)
   P = plot!(twinx(), xfine, ρsp.(xfine), label="", xlims=(-L, L), color=:red, linewidth=4, tickfontsize=16, ylabel="Density", yguidefontcolor=:red, labelfontsize=16, legendfontsize=12)
   savefig("plots_test_cases_paper_copulas/section5/dissociation111/external_pot_$i.pdf")

   display(P)
   sleep(0.5)

   P = plot(xfine, ρsp.(xfine), label="", xlims=(-L, L), linewidth=4, tickfontsize=16)
   savefig("plots_test_cases_paper_copulas/section5/dissociation111/density_$i.pdf")

   display(P)
   sleep(0.5)

   P = contour(xx, xx, ρ2, c=:viridis, aspect_ratio=:equal,
      colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)
   savefig("plots_test_cases_paper_copulas/section5/dissociation111/pair_density$i.pdf")

   display(P)
   sleep(0.5)

   P = contour(xx, xx, ρ2MF, c=:viridis, aspect_ratio=:equal, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)
   savefig("plots_test_cases_paper_copulas/section5/dissociation111/mean_field$i.pdf")

   display(P)
   sleep(0.5)

   P = contour(T, T, ratio_MF, c=:viridis, aspect_ratio=:equal, colorbar=:true, fill=:true, clim=(0.0, 2.2 * 2 / 3), tickfontsize=16, colorbar_tickfontsize=16)
   savefig("plots_test_cases_paper_copulas/section5/dissociation111/copula_$i.pdf")

   display(P)
   sleep(0.5)
end
