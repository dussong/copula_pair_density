# Few functions needed for the copula fits
# include("/Users/genevieve2/Gits/PairDensities.jl/src/PairDensities.jl")
# include("/Users/genevieve2/Research/01_En_cours/avec_Gero_Pair_densities/PairDensities.jl/src/PairDensities.jl")
# using PairDensities
# using Main.PairDensities: density, wavefunction, pair_density
using JSON

function run_PD_save(testcase, ne, N, L, vext_name, vext_type, folder;
   alpha_lap=1.0,
   vee_name="soft_coulomb",
   vext_param=[]
)
   titlejson = "$(folder)/$(testcase)-ne_$(ne)_L_$(L)_N_$(N)_$(vext_name).json"

   if vext_type == "soft_coulomb2"
      vext(x) = -vext_param[2] * (1.0 ./ sqrt.((x - vext_param[1]) .^ 2 .+ 1) .+ 1.0 ./ sqrt.((x + vext_param[1]) .^ 2 .+ 1))
      #double well taken from Wagner, L.O., Stoudenmire, E.M., Burke, K., White, S.R.: Reference electronic structure calculations in one dimension. Phys. Chem. Chem. Phys. 14, 8581–8590 (2012). https://doi.org/10.1039/c2cp24118h
   end

   if vee_name == "soft_coulomb"
      vee = (x -> 1.0 ./ sqrt.((x[1] - x[2]) .^ 2 .+ 1))
   end

   D = Dict()
   @time Ψ = wavefunction(ne, L, N; vext=vext, vee=vee, alpha=alpha_lap, krylovdim=3, maxiter=2000)
   ρ = density(Ψ)
   ρ2 = reshape(pair_density(Ψ), N - 1, N - 1)

   # Mean-field pair density
   ρ2MF = 0.5 * ρ * ρ'

   # Ratio to mean-field
   ratio_MF = 2 * ne / (ne - 1) * ρ2 ./ (ρ * ρ')

   # Transport map for ρ
   T = 2 * L / N / ne * cumsum(ρ)
   xx = range(-L, L; length=100)

   D["rho"] = ρ
   D["rho2"] = ρ2
   D["rho2MF"] = ρ2MF
   D["ratio_MF"] = ratio_MF
   D["T"] = T
   D["V"] = vext.(xx)
   D["xxV"] = xx
   D["N"] = N
   D["ne"] = ne
   D["L"] = L
   D["test_case"] = testcase
   D["vext_name"] = vext_name
   D["alpha"] = alpha_lap
   D["vext_param"] = vext_param
   D["vee_name"] = vee_name

   @show sum(ρ) * 2 * L / N

   json_string = JSON.json(D)
   open(titlejson, "w") do f
      JSON.print(f, json_string)
   end
end

function vect_2_mat(v)
   N = length(v)
   m = zeros(N, N)
   for i in 1:N
      for j in 1:N
         m[i, j] = v[i][j]
      end
   end
   return m
   STOP
end

function plots_from_dict(D; show=false)
   ne = D["ne"]
   N = D["N"]
   L = D["L"]
   vext = D["V"]
   vext_name = D["vext_name"]
   alpha_lap = D["alpha"]
   testcase = D["test_case"]
   xxV = D["xxV"]

   titlefig = "$(testcase)-ne_$(ne)_L_$(L)_N_$(N)_$(vext_name)"

   clf()
   xx = range(-L, L, length=N - 1)
   fig = figure("pyplot_surfaceplot", figsize=(4, 3))
   plot(xxV, vext, label="External potential")
   legend(loc="upper right", fancybox="true") # Create a legend of all the existing plots using their labels as names
   gcf()

   PyPlot.savefig("copula_fits/$(ne)particles_dissociation/figs/ext_pot.pdf")
   if show
      run(`open copula_fits/$(ne)particles_dissociation/figs/ext_pot.pdf`)
   end
   ρ = D["rho"]
   ρ2 = vect_2_mat(D["rho2"])

   # Mean-field pair density
   ρ2MF = vect_2_mat(D["rho2MF"])

   # Ratio to mean-field
   ratio_MF = vect_2_mat(D["ratio_MF"])

   # Transport map for ρ
   T = 2 * L / N / ne * cumsum(ρ)

   @show sum(ρ) * 2 * L / N

   clf()
   fig = figure("pyplot_surfaceplot3", figsize=(15, 8))


   subplot(231)
   plot(xxV, vext, label="External potential")
   legend(loc="upper right", fancybox="true") # Create a legend of all the existing plots using their labels as names

   subplot(234)
   P2 = plot(xx, ρ, label="density") #density
   legend(loc="upper right", fancybox="true") # Create a legend of all the existing plots using their labels as names

   subplot(232)
   ax = fig.add_subplot(2, 3, 2)
   surf = pcolor(xx, xx, ρ2MF)
   fig.colorbar(surf, shrink=1, aspect=5)
   axis("equal")
   # xlabel("X")
   ylabel("Y")
   PyPlot.title("Mean field pair density")

   subplot(235)
   ax = fig.add_subplot(2, 3, 5)
   surf = pcolor(xx, xx, ρ2)
   fig.colorbar(surf, shrink=1, aspect=5)
   axis("equal")
   xlabel("X")
   ylabel("Y")
   PyPlot.title("Pair density")

   subplot(233)
   ax = fig.add_subplot(2, 3, 3)
   surf = pcolor(xx, xx, ratio_MF, vmin=0, vmax=1.1 * ne / (ne - 1))
   # min(maximum(ratio_MF),)
   fig.colorbar(surf, shrink=1, aspect=5)
   axis("equal")
   # xlabel("X")
   ylabel("Y")
   PyPlot.title("Ratio to mean field")

   subplot(236)
   ax = fig.add_subplot(2, 3, 6)
   surf = pcolor(T, T, ratio_MF, vmin=0, vmax=1.1 * ne / (ne - 1))
   # min(maximum(ratio_MF),5))
   # ne/(ne-1)
   fig.colorbar(surf, shrink=1, aspect=5)
   axis("equal")
   xlabel("X")
   ylabel("Y")
   PyPlot.title("Copula")

   @show minimum(ratio_MF)

   tight_layout()
   PyPlot.savefig("copula_fits/$(ne)particles_dissociation/figs/full_plot_$(titlefig).pdf")
   if show
      run(`open copula_fits/$(ne)particles_dissociation/figs/full_plot_$(titlefig).pdf`)
   end
end
