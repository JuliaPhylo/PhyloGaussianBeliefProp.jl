@testset "calibration" begin

netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"

df = DataFrame(taxon=["A","B1","B2","C"], x=[10,10,missing,0], y=[1.0,.9,1,-1])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models
tbl_x = columntable(select(df, :x))

net = readTopology(netstr)
g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g)
spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)

@testset "no optimization" begin

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

@testset "Bethe: no optimization" begin
cg_bethe = PGBP.clustergraph!(net, PGBP.Bethe())
b_bethe = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg_bethe, m);
PGBP.init_beliefs_assignfactors!(b_bethe, m, tbl_y, df.taxon, net.nodes_changed);
cgb_bethe = PGBP.ClusterGraphBelief(b_bethe)
# schedule that updates beliefs of cluster graph, so that any schedule following
# is valid (i.e. all marginalization operations in the schedule are well-defined)
pa_lab, ch_lab, pa_j, ch_j =
    PGBP.minimal_valid_schedule(cg_bethe, [:Ai4, :B2i6, :B1i6, :Ci2])
for i in 1:length(pa_lab)
    ss_j = PGBP.sepsetindex(pa_lab[i], ch_lab[i], cgb_bethe)
    PGBP.propagate_belief!(b_bethe[ch_j[i]], b_bethe[ss_j], b_bethe[pa_j[i]])
end
# schedule that, though it may be inefficient for convergence, covers all edges
# of cluster graph
sch = [] # schedule that covers all edges of cluster graph
for n in net.nodes_changed
    ns = Symbol(n.name)
    sspt = PGBP.sub_spanningtree_clusterlist(cg_bethe, ns)
    isempty(spt[1]) && continue
    push!(sch, sspt)
end
PGBP.calibrate!(cgb_bethe, sch, 2)

# clique tree beliefs
ct_H5i4i2_var = cgb.belief[cgb.cdict[:H5i4i2]].J \ I
# 1.17972      0.371009     0.00843203
# 0.371009     0.753976    -0.00306619
# 0.00843203  -0.00306619   0.181748
ct_H5i4i2_mean = PGBP.integratebelief!(cgb.belief[cgb.cdict[:H5i4i2]])[1]
# 1.6393181556858691
# 2.5271166302556445
# -0.6420604806255028
ct_i6H5_var = cgb.belief[cgb.cdict[:i6H5]].J \ I
# 0.789199  0.536239
# 0.536239  1.17972
ct_i6H5_mean = PGBP.integratebelief!(cgb.belief[cgb.cdict[:i6H5]])[1]
# 1.2633264344026673
# 1.6393181556858687

# cluster graph beliefs
cg_bethe_H5i4i2_var = cgb_bethe.belief[cgb_bethe.cdict[:H5i4i2]].J \ I
cg_bethe_H5i4i2_mean = PGBP.integratebelief!(cgb_bethe.belief[cgb_bethe.cdict[:H5i4i2]])[1]
cg_bethe_i6H5_var = cgb_bethe.belief[cgb_bethe.cdict[:i6H5]].J \ I
cg_bethe_i6H5_mean = PGBP.integratebelief!(cgb_bethe.belief[cgb_bethe.cdict[:i6H5]])[1]
# Compare marginal estimates for i2, i4, H5, i6 with those from clique tree
@test all(ct_H5i4i2_var .== cg_bethe_H5i4i2_var)
@test all(ct_H5i4i2_mean .== cg_bethe_H5i4i2_mean)
@test all(ct_i6H5_var .== cg_bethe_i6H5_var)
@test all(ct_i6H5_mean .== cg_bethe_i6H5_mean)
end

#= likelihood using PN.vcv and matrix inversion, different params
σ2tmp = 1; μtmp = -2
Σnet = σ2tmp .* Matrix(vcv(net)[!,Symbol.(df.taxon)])
loglikelihood(MvNormal(repeat([μtmp],4), Σnet), tbl.y) # -8.091436736475565
=#

end # of no-optimization

@testset "with optimization" begin

# y: 1 trait, no missing values
m = PGBP.UnivariateBrownianMotion(2, 3, 0)
b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
cgb = PGBP.ClusterGraphBelief(b)

mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(cgb, ct, net.nodes_changed,
    tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1,-2))
@test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
@test llscore ≈ -5.174720533524127
@test mod.μ ≈ -0.26000871507162693
@test PGBP.varianceparam(mod) ≈ 0.35360518758586457

#= ML solution the matrix-way, analytical for BM:
# for y: univariate
Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])
n=4 # number of data points
i = ones(n) # intercept
μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * tbl.y) # -0.26000871507162693
r = tbl.y .- μhat
σ2hat_ML = (transpose(r) * inv(Σ) * r) / n # 0.35360518758586457
llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 # -5.174720533524127

# for x: third value is missing
xind = [1,2,4]; n = length(xind); i = ones(n) # intercept
Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])[xind,xind]
μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * tbl.x[xind]) # 3.500266520382341
r = tbl.x[xind] .- μhat
σ2hat_ML = (transpose(r) * inv(Σ) * r) / n # 11.257682945973125
llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 # -9.215574122592923
=#

# x: 1 trait, some missing values
b = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, m);
PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed);
cgb = PGBP.ClusterGraphBelief(b)

mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(cgb, ct, net.nodes_changed,
    tbl_x, df.taxon, PGBP.UnivariateBrownianMotion, (1,-2))
@test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
@test llscore ≈ -9.215574122592923
@test mod.μ ≈ 3.500266520382341
@test PGBP.varianceparam(mod) ≈ 11.257682945973125

# x,y: 2 traits, some missing values
m = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0,0))
b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
cgb = PGBP.ClusterGraphBelief(b)

mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(cgb, ct, net.nodes_changed,
    tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2,1), (1,-1)))
@test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
@test llscore ≈ -14.39029465611705 # -5.174720533524127 -9.215574122592923
@test mod.μ ≈ [3.500266520382341, -0.26000871507162693]
@test PGBP.varianceparam(mod) ≈ [11.257682945973125,0.35360518758586457]
end

end
