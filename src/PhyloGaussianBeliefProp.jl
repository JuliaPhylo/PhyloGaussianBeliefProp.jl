module PhyloGaussianBeliefProp

using Graphs
using MetaGraphsNext
using Distributions: MvNormalCanon, MvNormal, AbstractMvNormal
import LinearAlgebra
using PDMats
using StaticArrays
using Tables

import PhyloNetworks as PN
import PhyloNetworks: HybridNetwork, getparents, getparent, getchild, getchildren


include("clustergraph.jl")
include("canonicalnormal.jl")

end
