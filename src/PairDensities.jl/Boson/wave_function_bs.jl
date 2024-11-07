# Boson Wave-function structure and related types

"""WaveFunctionBs_sp
ne : Nb of particles
dof : number of space basis function 
combBasis : combination bases corresponding to nonzero elements
val : values corresponding to nonzero elements
wfns : sparse wavefunction without Pauli principle
"""

export WaveFunctionBs_sp

struct WaveFunctionBs_sp
    ne::Int
    dof::Int
    combBasis::Vector{Vector{Int64}}
    val::Vector{Float64}
    wfP::SparseVector{Float64,Int64}
    wfNP::SparseVector{Float64,Int64}
end

function wfspBsGen(ne::Int, dof::Int, combBasis::Vector{Vector{Int64}}, val::Vector{Float64})

    indP = map(x -> seq2num(dof, ne, x), combBasis)
    wfP = sparsevec(indP, val, binomial(dof, ne))

    indNP = map(x -> seq2num_ns(dof, ne, x), combBasis)
    wfNP = sparsevec(indNP, val, (dof)^ne)

    return WaveFunctionBs_sp(ne, dof, combBasis, val, wfP, wfNP)
end
WaveFunctionBs_sp(ne, dof, combBasis, val) = wfspBsGen(ne, dof, combBasis, val)

function wfspBsGen(ne::Int, dof::Int, wfNP::SparseVector{Float64,Int64})

    indNP = wfNP.nzind
    val = wfNP.nzval

    combBasis = map(x -> num2seq_ns(dof, ne, x), indNP)
    indP = map(x -> seq2num(dof, ne, x), combBasis)
    wfP = sparsevec(indP, val, binomial(dof, ne))

    return WaveFunctionBs_sp(ne, dof, combBasis, val, wfP, wfNP)
end
WaveFunctionBs_sp(ne, dof, wfNP) = wfspBsGen(ne, dof, wfNP)