@testset "canonical form" begin

netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C)i2:1.0)i1:3.0);"

df = DataFrame(taxon=["A","B1","B2","C"],
  x=[10,  10,missing,  0],
  y=[1.0, 0.9,1.0,  -1.0])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models

@testset "basics" begin
net = readTopology(netstr)
@test_throws ErrorException PGBP.shrinkdegenerate_treeedges(net)
net.edge[8].length=0.0 # external edge
@test_throws ErrorException PGBP.shrinkdegenerate_treeedges(net)
net.edge[8].length=0.1
net.edge[4].length=0.0 # tree edge below hybrid
net_polytomy = PGBP.shrinkdegenerate_treeedges(net)
@test writeTopology(net_polytomy) == "((A:4.0,(B1:1.0,B2:1.0)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1;"
@test PGBP.isdegenerate.(net.node) == [0,0,0, 1,0,0,0,0,0]
@test PGBP.hasdegenerate(net)
for i in [5,7] net.edge[i].length=0.0; end # hybrid edge: makes the hybrid degenerate
net.edge[4].length=0.6 # back to original
@test PGBP.isdegenerate.(net.node) == [0,0,0,0, 1,0,0,0,0]
@test PGBP.unscope(net.hybrid[1])

nm = trues(3,2); nm[2,1]=false
b1 = PGBP.ClusterBelief(Int8[5,6], 3, nm, PGBP.bclustertype, 1)
@test PGBP.nodelabels(b1) == [5,6]
@test size(b1.inscope) == (3,2)
@test length(b1.μ) == 5
@test size(b1.J) == (5,5)

end # of basic testset

@testset "degenerate hybrid: complex case" begin
net = readTopology(netstr)
net.edge[8].length=0.1 # was missing
for i in [5,7] net.edge[i].length=0.0; end # hybrid edge: makes the hybrid degenerate

g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g) # 6 sepsets, 7 cliques
#=
ne(ct), nv(ct)
[ct[lab] for lab in labels(ct)]
[ct[lab...] for lab in edge_labels(ct)]
[(n.name, n.number) for n in net.nodes_changed]
=#

# fixit: one clique should have both the child & parents of the degenerate hybrid

b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct)
beliefnodelabels = [[6,5], [7,6], [8,6], [5,4,2], [4,2,1], [3,2], [9,4], [6], [6], [5], [4,2], [2], [4]]
@test [PGBP.nodelabels(be) for be in b] == beliefnodelabels
@test PGBP.inscope(b[5]) == trues(2,3)
@test isempty(PGBP.scopeindex([5], b[1]))
@test PGBP.scopeindex([6], b[1]) == [1,2]
@test_throws ErrorException PGBP.scopeindex([2], b[1])
b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, true)
@test [PGBP.nodelabels(be) for be in b] == beliefnodelabels
@test PGBP.inscope(b[5]) == [true true false; true true false] # root not in scope

end # of degenerate testset

@testset "non-degenerate hybrid: simpler case" begin
net = readTopology(netstr)
net.edge[8].length=0.1 # was missing

g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g)

m = PGBP.UnivariateBrownianMotion(2, 3, 0) # 0 root prior variance: fixed root
b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, true);
PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed)
# ["$(be.type): $(be.nodelabel)" for be in b]

PGBP.propagate_belief!(b[1], b[7+1], b[2]) # 7 clusters, so b[7+1] = first sepset
PGBP.propagate_belief!(b[1], b[7+2], b[3])
PGBP.propagate_belief!(b[4], b[7+3], b[1])
PGBP.propagate_belief!(b[4], b[7+5], b[6])
PGBP.propagate_belief!(b[4], b[7+6], b[7])
PGBP.propagate_belief!(b[5], b[7+4], b[4]) # tree traversed once to cluster 5 as root
PGBP.integratebelief!(b[5])

#= likelihood using PN.vcv and matrix inversion
Σnet = Matrix(vcv(net)[!,Symbol.(df.taxon)])
loglikelihood(MvNormal(repeat([m.μ],4), m.σ2 .* Σnet), tbl.y) # -10.732857817537196
=#
end # of non-degenerate testset

end
