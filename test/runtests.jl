using PhyloGaussianBeliefProp

using DataFrames
using Graphs, MetaGraphsNext
using LinearAlgebra
using PhyloNetworks
using PreallocationTools
using Tables
using Test

const PGBP = PhyloGaussianBeliefProp

@testset "PhyloGaussianBeliefProp.jl" begin
  include("test_clustergraph.jl")
  include("test_evomodels.jl")
  include("test_canonicalform.jl")
  include("test_calibration.jl")
end
