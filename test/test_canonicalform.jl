@testset "canonical form" begin
netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C)i2:1.0)i1:3.0);"
net = readTopology(netstr)
g = PhyloGaussianBeliefProp.moralize!(net)
PhyloGaussianBeliefProp.triangulate_minfill!(g)
ct = PhyloGaussianBeliefProp.cliquetree(g) # 6 sepsets, 7 cliques 
#=
ne(ct), nv(ct)
[ct[lab] for lab in labels(ct)]
[ct[lab...] for lab in edge_labels(ct)]
[(n.name, n.number) for n in net.nodes_changed]
=#

nm = BitArray(undef, 3,2); fill!(nm, true); nm[2,1]=false
b1 = PhyloGaussianBeliefProp.ClusterBelief(Int8[5,6], 3, nm, PhyloGaussianBeliefProp.bclustertype, 1)

@test PhyloGaussianBeliefProp.nodelabels(b1) == [5,6]
@test size(b1.nonmissing) == (3,2)
@test length(b1.μ) == 5
@test size(b1.J) == (5,5)

df = DataFrame(taxon=["A","B1","B2","C"],
  x=[10,  10,missing,  0],
  y=[1.0, 0.9,1.0,  -1.0])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
tbl_nonmissing = NamedTuple{keys(tbl)}(.!ismissing.(tbl[k]) for k in keys(tbl))

end