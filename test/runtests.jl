using PhyloGaussianBeliefProp

using DataFrames
using Graphs, MetaGraphsNext
using LinearAlgebra
using Optim
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
  # include("test_optimization.jl") redundant with test_calibration.jl, but future tests could use networks in there to vary networks used in test
  include("test_exactBM.jl")
  include("test_generalized.jl")
end
