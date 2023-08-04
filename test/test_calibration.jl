@testset "calibration" begin

netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"

df = DataFrame(taxon=["A","B1","B2","C"], x=[10,10,missing,0], y=[1.0,.9,1,-1])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models

@testset "no optimization" begin
net = readTopology(netstr)

g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g)
spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)

m = PGBP.UnivariateBrownianMotion(2, 3, 0)
b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);

cgb = PGBP.ClusterGraphBelief(b)
PGBP.calibrate!(cgb, [spt])
@test PGBP.default_sepset1(cgb) == 8
llscore = -10.732857817537196
tmp1, tmp = PGBP.integratebelief!(cgb)
@test tmp1 ≈ [1.2633264344026676]
@test tmp ≈ llscore
for i in eachindex(cgb.belief)
    _, tmp = PGBP.integratebelief!(cgb, i)
    @test tmp ≈ llscore
end

#= likelihood using PN.vcv and matrix inversion, different params
σ2tmp = 1; μtmp = -2
Σnet = σ2tmp .* Matrix(vcv(net)[!,Symbol.(df.taxon)])
loglikelihood(MvNormal(repeat([μtmp],4), Σnet), tbl.y) # -8.091436736475565
=#
@test PGBP.calibrate_optimize_cliquetree!(cgb, ct, net.nodes_changed, tbl_y, df.taxon,
    PGBP.UnivariateBrownianMotion, (1,-2)) ≈ -8.091436736475565

end # of no-optimization

end
