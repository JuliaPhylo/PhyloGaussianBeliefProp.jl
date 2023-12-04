module PhyloGaussianBeliefProp

import Base: show
using Distributions: MvNormalCanon, MvNormal, AbstractMvNormal
using Graphs
import LinearAlgebra as LA
using MetaGraphsNext
using Optim, PreallocationTools, ForwardDiff
using PDMats
using StaticArrays
using StatsFuns
using Tables

import PhyloNetworks as PN
using PhyloNetworks: HybridNetwork, getparents, getparent, getparentedge,
        getchild, getchildren, getchildedge, hassinglechild


include("utils.jl")
include("clustergraph.jl")
include("evomodels/evomodels.jl") # abstract evomodel must be included before all other models
include("evomodels/brownianmotion.jl")
include("beliefs.jl")
include("beliefupdates.jl")
include("clustergraphbeliefs.jl")
include("calibration.jl")
include("score.jl")

end
