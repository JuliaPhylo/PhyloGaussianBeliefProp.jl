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
ctb = PGBP.ClusterGraphBelief(b)
PGBP.calibrate!(ctb, [spt])
@test PGBP.default_sepset1(ctb) == 8
llscore = -10.732857817537196
tmp1, tmp = PGBP.integratebelief!(ctb)
@test tmp1 ≈ [1.2633264344026676]
@test tmp ≈ llscore
for i in eachindex(ctb.belief)
    _, tmp = PGBP.integratebelief!(ctb, i)
    @test tmp ≈ llscore
end
# clique tree beliefs to compare marginal estimates for i2, i4, H5, i6 with
# those from cluster graph
ct_H5i4i2_var = ctb.belief[ctb.cdict[:H5i4i2]].J \ I
# 1.17972      0.371009     0.00843203
# 0.371009     0.753976    -0.00306619
# 0.00843203  -0.00306619   0.181748
ct_H5i4i2_mean = PGBP.integratebelief!(ctb.belief[ctb.cdict[:H5i4i2]])[1]
# 1.6393181556858691
# 2.5271166302556445
# -0.6420604806255028
ct_i6H5_var = ctb.belief[ctb.cdict[:i6H5]].J \ I
# 0.789199  0.536239
# 0.536239  1.17972
ct_i6H5_mean = PGBP.integratebelief!(ctb.belief[ctb.cdict[:i6H5]])[1]
# 1.2633264344026673
# 1.6393181556858687

@testset "Bethe: no optimization" begin
    cg = PGBP.clustergraph!(net, PGBP.Bethe())
    b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
    PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
    cgb= PGBP.ClusterGraphBelief(b)
    #= Modify beliefs of cluster graph so that any schedule is valid (i.e. all
    marginalization operations in the schedule are well-defined). =#
    PGBP.mod_beliefs_bethe!(cgb, PGBP.dimension(m), net)
    sch = [] # schedule that covers all edges of cluster graph
    for n in net.nodes_changed
        ns = Symbol(n.name)
        sspt = PGBP.sub_spanningtree_clusterlist(cg, ns)
        isempty(spt[1]) && continue
        push!(sch, sspt)
    end
    PGBP.calibrate!(cgb, sch, 2)
    cg_H5i4i2_var = cgb.belief[cgb.cdict[:H5i4i2]].J \ I
    cg_H5i4i2_mean = PGBP.integratebelief!(cgb.belief[cgb.cdict[:H5i4i2]])[1]
    cg_i6H5_var = cgb.belief[cgb.cdict[:i6H5]].J \ I
    cg_i6H5_mean = PGBP.integratebelief!(cgb.belief[cgb.cdict[:i6H5]])[1]
    @test ct_H5i4i2_var ≈ cg_H5i4i2_var rtol=1e-5
    @test ct_H5i4i2_mean ≈ cg_H5i4i2_mean rtol=1e-5
    @test ct_i6H5_var ≈ cg_i6H5_var rtol=1e-5
    @test ct_i6H5_mean ≈ cg_i6H5_mean rtol=1e-5
end
@testset "Join-graph: no optimization" begin
    cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
    b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
    PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
    cgb = PGBP.ClusterGraphBelief(b)
    pa_lab, ch_lab, pa_j, ch_j =
        PGBP.minimal_valid_schedule(cg, [:Ai4, :B2i6, :B1i6, :Ci2])
    for i in 1:length(pa_lab)
        ss_j = PGBP.sepsetindex(pa_lab[i], ch_lab[i], cgb)
        PGBP.propagate_belief!(b[ch_j[i]], b[ss_j], b[pa_j[i]])
    end
    sch = [] # schedule that covers all edges of cluster graph
    for n in net.nodes_changed
        ns = Symbol(n.name)
        sspt = PGBP.sub_spanningtree_clusterlist(cg, ns)
        isempty(spt[1]) && continue
        push!(sch, sspt)
    end
    PGBP.calibrate!(cgb, sch, 2)
    cg_H5i4i2_var = cgb.belief[cgb.cdict[:H5i4i2]].J \ I
    cg_H5i4i2_mean = PGBP.integratebelief!(cgb.belief[cgb.cdict[:H5i4i2]])[1]
    cg_i6H5_var = cgb.belief[cgb.cdict[:i6H5]].J \ I
    cg_i6H5_mean = PGBP.integratebelief!(cgb.belief[cgb.cdict[:i6H5]])[1]
    @test ct_H5i4i2_var == cg_H5i4i2_var
    @test ct_H5i4i2_mean == cg_H5i4i2_mean
    @test ct_i6H5_var == cg_i6H5_var
    @test ct_i6H5_mean != cg_i6H5_mean
    @test ct_i6H5_mean ≈ cg_i6H5_mean rtol=1e-5
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
