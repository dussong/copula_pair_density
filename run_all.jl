using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("tests_paper_copulas/section6/copula_vizualization_2particles.jl")
include("tests_paper_copulas/section6/copula_vizualization_3particles-111.jl")
include("tests_paper_copulas/section6/copula_vizualization_3particles-21.jl")
include("tests_paper_copulas/section6/copula_vizualization_3particles-21-to-111.jl")
include("tests_paper_copulas/section6/copula_vizualization_4particles.jl")

include("tests_paper_copulas/section7/copula_fit.jl")