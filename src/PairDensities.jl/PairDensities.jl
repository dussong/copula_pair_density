"""
# `PairDensities.jl`

Package for solving the Schrodinger equation with density constraint.

"""
module PairDensities

using Printf, Plots, Plots.PlotMeasures
using FastGaussQuadrature
using StaticArrays, SparseArrays
using LinearAlgebra, KrylovKit, Arpack
using Combinatorics
using TensorOperations
using IterTools
using PolynomialRoots
using StatsBase
using ForwardDiff

# Basic functions
include("pert_comb_index.jl")
include("Nbody_MatFreeTensor.jl")
include("Nbody_matrices.jl")
include("hamiltonian.jl")
include("wave_function.jl")
include("hamiltonian_freetensor.jl")
include("pair_density_sp.jl")
include("pair_density_full.jl")
include("pair_density_hf.jl")
include("power_method.jl")
include("CDFCI.jl")
include("HF_Initial.jl")
include("CDFCI_MatFree.jl")
include("Initial_point.jl")
include("SCI_MatFree.jl")
include("Interpolate.jl")

# 1d system
include("1d/1_2body_matrices_1d.jl")
include("1d/1_2body_2ndfem.jl")
include("1d/HartreeFock_Initial_1d.jl")
include("1d/Nbody_Matfree_Column_1d.jl")
include("1d/Nbody_Matfree_Row_1d.jl")
include("1d/SCE_Initial_1d.jl")

# 2d system
include("2d/1_2body_matrices_2d.jl")
include("2d/HartreeFock_Initial_2d.jl")
include("2d/Nbody_Matfree_Column_2d.jl")
include("2d/Nbody_Matfree_Row_2d.jl")
include("2d/SCE_Initial_2d.jl")

# ignore the spin of electrons
include("Boson/hamiltonian_bs.jl")
include("Boson/wave_function_bs.jl")
include("Boson/Nbody_Matfree_Column_bs_2d.jl")
include("Boson/Nbody_Matfree_Column_bs_1d.jl")
include("Boson/Nbody_Matfree_Row_bs_2d.jl")
include("Boson/Nbody_Matfree_Row_bs.jl")
include("Boson/pair_density_sp_bs.jl")
include("Boson/SCE_Initial_bs_1d.jl")
include("Boson/SCE_Initial_bs_2d.jl")
include("Boson/CDFCI_MatFree_bs.jl")
include("Boson/SCI_MatFree_bs.jl")


end # module
