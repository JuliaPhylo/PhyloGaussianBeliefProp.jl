using PhyloGaussianBeliefProp

using DataFrames
using Graphs, MetaGraphsNext
using PhyloNetworks
using Test

@testset "PhyloGaussianBeliefProp.jl" begin
  include("test_clustergraph.jl")
end
