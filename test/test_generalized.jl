@testset "generalized beliefs" begin
    net1 = "((#H1:0.0::0.4,#H2:0.0::0.4)I1:1.0,(((A:1.0)#H1:0.0::0.6,#H3:0.0::0.4)#H2:0.0::0.6,(B:1.0)#H3:0.0::0.6)I2:1.0)I3;"
    # net2 is modified from mateescu_2010.phy with all hybrid edge lengths set to 0
    net2 = "((((g:1.0)#H4:0.0::0.6)#H2:0.0::0.6,(d:1.0,(#H2:0.0::0.4,#H4:0.0::0.4)#H3:0.0::0.6)D:1.0,(#H3:0.0::0.4)#H1:0.0::0.6)B:1.0,#H1:0.0::0.4)A;"
    # net3 is from SM section F of manuscript
    net3 = "((i1:1.0,(i2:1.0)#H1:0.0::0.5)i4:1.0, (#H1:0.0::0.5,i3:1.0)i6:1.0)i0;"
    net4 = "(((i2:0.0)#H1:0.0::0.5)i4:1.0, (#H1:0.0::0.5)i6:1.0)i0;"

@testset "degenerate extended family" begin
    netstr = "(((a:1.0)#H1:0.0::0.5)d:1.0, (((#H1:0.0::0.5)#H2:0.0::0.5)b:1.0, (#H2:0.0::0.5)c:1.0)e:1.0)f;"
    net = readnewick(netstr)
    df = DataFrame(taxon="a", x=[1.0])
    tbl_x = columntable(select(df, :x))
    # clusters: [[d, e, f], [H1, d, b, c, e], [H1, d, H2, b, c], [a, H1]]
    # In [H1, d, b, c, e], d is a parent of H1, and b, c are grandparents of H1
    clusters = [[6,2,1], [7,6,4,3,2], [7,6,5,4,3], [8,7]]
    cg = PGBP.clustergraph!(net, PGBP.LTRIP(clusters, net))
    # [[net.vec_node[i].name for i in c] for c in clusters]
    m = PGBP.UnivariateBrownianMotion(1, 0)
    b, (n2c, n2fam, n2fix, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.vec_node, cg, m)
    @test PGBP.isdegenerate_extendedfamily_covered(7, [8,7], n2fam, n2d, n2fix) == (false, true) # hybrid, no parent in cluster
    @test PGBP.isdegenerate_extendedfamily_covered(8, [8,7], n2fam, n2d, n2fix) == (false, true) # leaf
    @test PGBP.isdegenerate_extendedfamily_covered(1, [6,2,1], n2fam, n2d, n2fix) == (true, true) # fixed root
    @test PGBP.isdegenerate_extendedfamily_covered(5, [4,3,2], n2fam, n2d, n2fix) == (true, true)
    @test PGBP.isdegenerate_extendedfamily_covered(7, [4,3,2], n2fam, n2d, n2fix) == (false, true)
    @test PGBP.isdegenerate_extendedfamily_covered(7, [7,6,4,3], n2fam, n2d, n2fix) == (true, false)
    @test PGBP.isdegenerate_extendedfamily_covered([7,6,5,4,3], n2fam, n2d, n2fix)
    @test PGBP.isdegenerate_extendedfamily_covered([6,2,1], n2fam, n2d, n2fix)
    tmp = (@test_logs (:error, r"^cluster H1dbce is missing") PGBP.isdegenerate_extendedfamily_covered(cg, n2fam, n2d, n2fix))
    @test !tmp
end

@testset "Univariate BM. Clique tree" begin
@testset "1 tip. Leaf is degenerate child of hybrid" begin
    net = readnewick(net4)
    df = DataFrame(taxon=["i2"], x=[1.0])
    tbl_x = columntable(select(df, :x))
    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    m = PGBP.UnivariateBrownianMotion(1, 0)
    b, (n2c, n2fam, n2fix, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.vec_node, ct, m)
    ctb = PGBP.ClusterGraphBelief(b, n2c, n2fam, n2fix, c2n)
    PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.vec_node, n2c, n2fam, n2fix)
    spt = PGBP.spanningtree_clusterlist(ct, net.vec_node)
    @test_throws ("message from cluster H1i4i6 has larger constraint rank than " *
        "dimension of sepset (:H1i4i6, :i4i6i0), " *
        "represent cluster i4i6i0 and sepset (:H1i4i6, :i4i6i0) as GeneralizedBeliefs") PGBP.calibrate!(ctb, [spt])
    # b[3] and b[5] should also be represented as GeneralizedBeliefs
    b[3] = PGBP.GeneralizedBelief(b[3])
    b[5] = PGBP.GeneralizedBelief(b[5])
    PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.vec_node, n2c, n2fam, n2fix)
    PGBP.calibrate!(ctb, [spt])
    llscore = -1.5723649429247 # -0.5*(1-0)^2/0.5 - 0.5*logdet(2π*0.5)
    for i in eachindex(ctb.belief)
        _, tmp = PGBP.integratebelief!(ctb, i)
        @test tmp ≈ llscore
    end
end
# @testset "Level-1. 3 tips" begin
#     net = readnewick(net3)
#     df = DataFrame(taxon=["i1","i2","i3"], x=[1.0,1.0,1.0])
#     tbl_x = columntable(select(df, :x))
#     ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
#     m = PGBP.UnivariateBrownianMotion(1, 0)
#     b, (n2c, n2fam, n2fix, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.vec_node, ct, m)
#     PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.vec_node, n2c, n2fam, n2fix)
#     ctb = PGBP.ClusterGraphBelief(b, n2c, n2fam, n2fix, c2n)
#     spt = PGBP.spanningtree_clusterlist(ct, net.vec_node)
#     PGBP.calibrate!(ctb, [spt])
#     # μ = 0; σ2 = 1; Vy = sharedpathmatrix(net)[:Tips]; Y = [1.0,1.0,1.0]
#     # -0.5*transpose(Y .- μ)*inv(σ2*Vy)*(Y .- μ) - 0.5*logdet(2π*σ2*Vy) # -4.161534555831068
#     @test PGBP.integratebelief!(ctb, 6)[1] ≈ [0.6] # H1
#     llscore = -4.161534555831068
#     for i in eachindex(ctb.belief)
#         _, tmp = PGBP.integratebelief!(ctb, i)
#         @test tmp ≈ llscore
#     end
# end
@testset "Level-3. 2 tips" begin
    net = readnewick(net1)
    df = DataFrame(taxon=["A","B"], x=[2.11,2.15])
    tbl_x = columntable(select(df, :x))
    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    m = PGBP.UnivariateBrownianMotion(0.000325097529258775, 2.128439531859558)
    b, (n2c, n2fam, n2fix, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.vec_node, ct, m)
    ctb = PGBP.ClusterGraphBelief(b, n2c, n2fam, n2fix, c2n)
    PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.vec_node, n2c, n2fam, n2fix)
    spt = PGBP.spanningtree_clusterlist(ct, net.vec_node)
    PGBP.calibrate!(ctb, [spt])
    #= using StatsModels
    fitx_ml = phylolm(@formula(x ~ 1), df, net; tipnames=:taxon, reml=false)
    PhyloNetworks.loglikelihood(fitx_ml) # 4.73520292387366
    mu_phylo(fitx_ml) # 2.128439531859558
    sigma2_phylo(fitx_ml) # 0.000325097529258775
    ancestralStateReconstruction(net, [2.11,2.15], ParamsBM(2.128439531859558,0.000325097529258775))
    ──────────────────────────────────────────
    Node index    Pred.     Min.  Max. (95%)
    ──────────────────────────────────────────
    I1       3.0  2.12064  2.09036     2.15091
    H1       1.0  2.12625  2.10665     2.14586
    H2       2.0  2.13     2.11219     2.14781
    H3       5.0  2.13375  2.11183     2.15566
    I2       7.0  2.13624  2.10947     2.16301
    ────────────────────────────────────────── =#
    @test all(isapprox.(PGBP.integratebelief!(ctb,1)[1], [2.13375]; rtol=1e-5)) # H3
    @test all(isapprox.(PGBP.integratebelief!(ctb,2)[1], [2.12625,2.13,2.12064]; rtol=1e-5)) # H1,H2,I1
    @test all(isapprox.(PGBP.integratebelief!(ctb,3)[1], [2.12625]; rtol=1e-5)) # H1
    @test all(isapprox.(PGBP.integratebelief!(ctb,4)[1], [2.13375,2.13,2.13624]; rtol=1e-5)) # H3,H2,I2
    @test all(isapprox.(PGBP.integratebelief!(ctb,5)[1], [2.13,2.12064,2.13624]; rtol=1e-5)) # H2,I1,I2
    @test all(isapprox.(PGBP.integratebelief!(ctb,6)[1], [2.12064,2.13624]; rtol=1e-5)) # I1,I2
    llscore = 4.73520292387366
    for i in eachindex(ctb.belief)
        _, tmp = PGBP.integratebelief!(ctb, i)
        @test tmp ≈ llscore
    end
end
@testset "Level-4. 2 tips" begin
    #= root node is a parent of a degenerate hybrid, and also a parent of its
    coparent (for the hybrid) =#
    net = readnewick(net2)
    df = DataFrame(taxon=["d","g"], x=[1.0,-1.0])
    tbl_x = columntable(select(df, :x))
    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    m = PGBP.UnivariateBrownianMotion(1, 0)
    b, (n2c, n2fam, n2fix, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.vec_node, ct, m)
    PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.vec_node, n2c, n2fam, n2fix)
    ctb = PGBP.ClusterGraphBelief(b, n2c, n2fam, n2fix, c2n)
    spt = PGBP.spanningtree_clusterlist(ct, net.vec_node)
    PGBP.calibrate!(ctb, [spt])
    #= μ = 0; σ2 = 1; Vy = sharedpathmatrix(net)[:Tips]; Y = [-1.0,1.0]
    -0.5*transpose(Y .- μ)*inv(σ2*Vy)*(Y .- μ) - 0.5*logdet(2π*σ2*Vy) # -3.4486412230145387
    ancestralStateReconstruction(net, Y, ParamsBM(0,1))
    ───────────────────────────────────────────────
    Node index       Pred.       Min.  Max. (95%)
    ───────────────────────────────────────────────
    H4      2.0   0.0153366  -1.03755     1.06822
    H2      3.0  -0.04452    -1.18939     1.10035
    H3      5.0   0.105121   -0.927203    1.13745
    D       6.0   0.232915   -1.20313     1.66896
    H1      7.0  -0.0865686  -0.925765    0.752627
    B       8.0  -0.144281   -1.54294     1.25438
    ─────────────────────────────────────────────── =#
    @test all(isapprox.(PGBP.integratebelief!(ctb,5)[1], [-0.0865686,-0.144281]; rtol=1e-5)) # H1,B
    @test all(isapprox.(PGBP.integratebelief!(ctb,1)[1], [0.0153366,-0.04452,0.105121]; rtol=1e-5)) # H4,H2,H3
    @test all(isapprox.(PGBP.integratebelief!(ctb,6)[1], [0.232915]; rtol=1e-5)) # D
    llscore = -3.4486412230145387
    for i in eachindex(ctb.belief)
        _, tmp = PGBP.integratebelief!(ctb, i)
        @test tmp ≈ llscore
    end
end
end

@testset "Multivariate BM. Clique tree" begin
# @testset "1 tip. Leaf is non-degenerate child of hybrid" begin
#     netstr = "(((i2:1.0)#H1:0.0::0.5)i4:1.0, (#H1:0.0::0.5)i6:1.0)i0;" 
#     net = readnewick(netstr)
#     ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
#     df = DataFrame(taxon=["i2"], x=[1.0], y=[2.0])
#     df_var = select(df, Not(:taxon))
#     tbl = columntable(df_var)
#     m = PGBP.MvFullBrownianMotion([2.0 0.5; 0.5 1.0], [3.0, -3.0])
#     b, (n2c, n2fam, n2fix, n2d, c2n) = PGBP.allocatebeliefs(tbl, df.taxon,
#         net.vec_node, ct, m)
#     PGBP.assignfactors!(b, m, tbl, df.taxon, net.vec_node, n2c, n2fam, n2fix)
#     ctb = PGBP.ClusterGraphBelief(b, n2c, n2fam, n2fix, c2n)
#     spt = PGBP.spanningtree_clusterlist(ct, net.vec_node)
#     PGBP.calibrate!(ctb, [spt])
#     # Vy = sharedpathmatrix(net)[:Tips]
#     # μ = [3.0, -3.0]; σ2 = [2.0 0.5; 0.5 1.0]
#     # Y = [1.0, 2.0]
#     # -0.5*transpose(Y - μ)*inv(kron(Vy,σ2))*(Y - μ) - 0.5*logdet(2π*kron(Vy,σ2)) # -14.713626258961408
#     llscore = -14.713626258961408
#     for i in eachindex(ctb.belief)
#         _, tmp = PGBP.integratebelief!(ctb, i)
#         @test tmp ≈ llscore
#     end
#     end

@testset "Level-1. 3 tips" begin
# modified net3 by setting length of tree edge (i0,i4) to 0
netstr = "((i1:1.0,(i2:1.0)#H1:0.0::0.5)i4:0.0, (#H1:0.0::0.5,i3:1.0)i6:1.0)i0;"
net = readnewick(netstr)
ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
df = DataFrame(taxon=["i1","i2","i3"], x=[1.0,1.0,1.0], y=[2.0,2.0,2.0])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
m_biBM_fixedroot = PGBP.MvDiagBrownianMotion([2,1], [3,-3], [0,0])
b_xy_fixedroot = PGBP.allocatebeliefs(tbl, df.taxon, net.vec_node, ct,
    m_biBM_fixedroot)
PGBP.assignfactors!(b_xy_fixedroot[1], m_biBM_fixedroot, tbl, df.taxon,
    net.vec_node, b_xy_fixedroot[2][1], b_xy_fixedroot[2][2], b_xy_fixedroot[2][3]);
ctb = PGBP.ClusterGraphBelief(b_xy_fixedroot[1], b_xy_fixedroot[2][1],
    b_xy_fixedroot[2][2], b_xy_fixedroot[2][3], b_xy_fixedroot[2][5])
spt = PGBP.spanningtree_clusterlist(ct, net.vec_node)
PGBP.calibrate!(ctb, [spt])
# Vy = sharedpathmatrix(net)[:Tips];
# μ = repeat([3, -3],3); σ2 = [2 0; 0 1]; 
# Y = repeat([1.0,2.0],3)
# -0.5*transpose(Y - μ)*inv(kron(Vy,σ2))*(Y - μ) - 0.5*logdet(2π*kron(Vy,σ2)) # -34.364282186284285
llscore = -34.364282186284285
for i in eachindex(ctb.belief)
    _, tmp = PGBP.integratebelief!(ctb, i)
    @test tmp ≈ llscore
end

m = PGBP.MvFullBrownianMotion([2.0 0.5; 0.5 1.0], [3.0,-3.0])
PGBP.assignfactors!(b_xy_fixedroot[1], m, tbl, df.taxon,
    net.vec_node, b_xy_fixedroot[2][1], b_xy_fixedroot[2][2], b_xy_fixedroot[2][3]);
PGBP.calibrate!(ctb, [spt])
# Vy = sharedpathmatrix(net)[:Tips];
# μ = repeat([3, -3],3); σ2 = [2.0 0.5; 0.5 1.0]; 
# Y = repeat([1.0,2.0],3)
# -0.5*transpose(Y - μ)*inv(kron(Vy,σ2))*(Y - μ) - 0.5*logdet(2π*kron(Vy,σ2)) # -43.73541366877607
llscore = -43.73541366877607
for i in eachindex(ctb.belief)
    _, tmp = PGBP.integratebelief!(ctb, i)
    @test tmp ≈ llscore
end
end

end
end
