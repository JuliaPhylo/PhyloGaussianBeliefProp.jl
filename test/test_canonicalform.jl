@testset "canonical form" begin
netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C)i2:1.0)i1:3.0);"
net = readTopology(netstr)
@test_throws ErrorException PGBP.shrinkdegenerate_treeedges(net)
net.edge[8].length=0.0 # external edge
@test_throws ErrorException PGBP.shrinkdegenerate_treeedges(net)
net.edge[8].length=0.1
net.edge[4].length=0.0 # tree edge below hybrid
net_polytomy = PGBP.shrinkdegenerate_treeedges(net)
@test writeTopology(net_polytomy) == "((A:4.0,(B1:1.0,B2:1.0)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1;"
@test PGBP.isdegenerate.(net.node) == [0,0,0, 1,0,0,0,0,0]

g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g) # 6 sepsets, 7 cliques
#=
ne(ct), nv(ct)
[ct[lab] for lab in labels(ct)]
[ct[lab...] for lab in edge_labels(ct)]
[(n.name, n.number) for n in net.nodes_changed]
=#

nm = BitArray(undef, 3,2); fill!(nm, true); nm[2,1]=false
b1 = PGBP.ClusterBelief(Int8[5,6], 3, nm, PGBP.bclustertype, 1)

@test PGBP.nodelabels(b1) == [5,6]
@test size(b1.nonmissing) == (3,2)
@test length(b1.μ) == 5
@test size(b1.J) == (5,5)

df = DataFrame(taxon=["A","B1","B2","C"],
  x=[10,  10,missing,  0],
  y=[1.0, 0.9,1.0,  -1.0])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
# tbl_nonmissing = NamedTuple{keys(tbl)}(.!ismissing.(tbl[k]) for k in keys(tbl))

b = PGBP.init_beliefs_allocate(tbl, net, ct)
end
