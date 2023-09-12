@testset "exact tree calibration for the BM" begin

netstr = "((A:1.5,B:1.5):1,(C:1,(D:0.5, E:0.5):0.5):1.5);"

df = DataFrame(taxon=["A","B","C","D","E"], x=[10,10,3,0,1], y=[1.0,.9,1,-1,-0.9])
n = 5
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models
tbl_x = columntable(select(df, :x))

net = readTopology(netstr)
g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g)
spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)

@testset "comparison with PhyloEM" begin # of PhyloEM

m = PGBP.UnivariateBrownianMotion(1, 0, 10000000000) # "infinite" root variance to match phyloEM
b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
cgb = PGBP.ClusterGraphBelief(b)
PGBP.calibrate!(cgb, [spt])
# Test conditional expectations and variances
# TODO: automatic ordering of nodes to match with ape order ?
@test PGBP.default_sepset1(cgb) == 9
llscore = -18.83505
condexp = [1,0.9,1,-1,-0.9,0.4436893,0.7330097,0.009708738,-0.6300971]
condexp = condexp[[6,8,9,5,4,3,7,2,1]]
tmp1, tmp = PGBP.integratebelief!(cgb)
@test tmp1 ≈ [condexp[cgb.belief[9].nodelabel[1]]] atol=1e-6
@test tmp ≈ llscore atol=1e-6
for i in eachindex(cgb.belief)
    tmp1, tmp = PGBP.integratebelief!(cgb, i)
    @test tmp1[end] ≈ condexp[cgb.belief[i].nodelabel[end]] atol=1e-6
    @test tmp ≈ llscore atol=1e-6
end
# Test conditional variances
condvar = [0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.9174757,0.5970874,0.3786408,0.2087379]
condvar = condvar[[6,8,9,5,4,3,7,2,1]]
for i in eachindex(cgb.belief)
    vv = inv(cgb.belief[i].J)[end,end]
    @test vv ≈ condvar[cgb.belief[i].nodelabel[end]] atol=1e-6
end
# Test conditional covariances
condcovar = [0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,NaN,0.3932039,0.2038835,0.1262136]
condcovar = condcovar[[6,8,9,5,4,3,7,2,1]]
for i in eachindex(cgb.belief)
    vv =  inv(cgb.belief[i].J)
    size(vv, 1) == 2 && @test vv[1,2] ≈ condcovar[cgb.belief[i].nodelabel[1]] atol=1e-6
end

#= likelihood and moments using PhylogeneticEM from R
library(PhylogeneticEM)
# tree and data
tree <- read.tree(text = "((A:1.5,B:1.5):1,(C:1,(D:0.5, E:0.5):0.5):1.5);")
tree <- reorder(tree, "postorder")
ntips <- length(tree$tip.label)
Y_data <- c(1.0,.9,1,-1,-0.9)
names(Y_data) <- c("A", "B", "C", "D", "E")
Y_data <- t(Y_data)
# tree traversal
theta_var <- 1
params_random <- params_BM(p = 1, variance = diag(theta_var, 1, 1), value.root = rep(0, 1),
                           random = TRUE, var.root = diag(10000000000, 1, 1))
resE <- PhylogeneticEM:::compute_E.upward_downward(phylo = tree,
                                                   Y_data = Y_data,
                                                   process = "BM",
                                                   params_old = params_random)
# likelihood
log_likelihood(params_random, phylo = tree, Y_data = Y_data) # -18.83505
# conditional expectations
resE$conditional_law_X$expectations # 1  0.9    1   -1 -0.9 0.4436893 0.7330097 0.009708738 -0.6300971
# conditional variances
resE$conditional_law_X$variances[1, 1, ] # 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.9174757 0.5970874 0.3786408 0.2087379
# conditional covariances
resE$conditional_law_X$covariances[1,1,] # 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000        NA 0.3932039 0.2038835 0.1262136
# mu hat
resE$conditional_law_X$expectations[ntips + 1] # 0.4436893
# sigma2 hat
num <- 0
den <- 0
for (i in 1:nrow(tree$edge)) {
  par <- tree$edge[i, 1]
  child <- tree$edge[i, 2]
  num <- num + (resE$conditional_law_X$expectations[par] - resE$conditional_law_X$expectations[child])^2 / tree$edge.length[i]
  den <- den + 1 - (resE$conditional_law_X$variances[,,par] + resE$conditional_law_X$variances[,,child] - 2 * resE$conditional_law_X$covariances[,,child]) / tree$edge.length[i] / theta_var
}
num / den
=#

end # of PhyloEM

@testset "update root status" begin
    # Check that an update from a fixed root to a random root is possible

    # y: 1 trait, no missing values
    m1 = PGBP.UnivariateBrownianMotion(1, 0, 0.9)
    b1 = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m1);
    PGBP.init_beliefs_assignfactors!(b1, m1, tbl_y, df.taxon, net.nodes_changed);

    m2 = PGBP.UnivariateBrownianMotion(1, 0, 0) # fixed root
    b2 = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m2);
    PGBP.init_beliefs_assignfactors!(b2, m2, tbl_y, df.taxon, net.nodes_changed);

    PGBP.update_root_inscope!(b1, m2)
    PGBP.init_beliefs_reset!(b1)
    PGBP.init_beliefs_assignfactors!(b1, m2, tbl_y, df.taxon, net.nodes_changed);
    
    for ind in eachindex(b1)
        @test b1[ind].nodelabel == b2[ind].nodelabel
        @test b1[ind].ntraits == b2[ind].ntraits
        @test b1[ind].inscope == b2[ind].inscope
        @test b1[ind].μ == b2[ind].μ
        @test b1[ind].h == b2[ind].h
        @test b1[ind].J == b2[ind].J
        @test b1[ind].type == b2[ind].type
        @test b1[ind].metadata == b2[ind].metadata
    end
    
    PGBP.update_root_inscope!(b1, m1)
    PGBP.init_beliefs_reset!(b1)
    PGBP.init_beliefs_assignfactors!(b1, m1, tbl_y, df.taxon, net.nodes_changed);
    PGBP.update_root_inscope!(b2, m1)
    PGBP.init_beliefs_reset!(b2)
    PGBP.init_beliefs_assignfactors!(b2, m1, tbl_y, df.taxon, net.nodes_changed);

    for ind in eachindex(b1)
        @test b1[ind].nodelabel == b2[ind].nodelabel
        @test b1[ind].ntraits == b2[ind].ntraits
        @test b1[ind].inscope == b2[ind].inscope
        @test b1[ind].μ == b2[ind].μ
        @test b1[ind].h == b2[ind].h
        @test b1[ind].J == b2[ind].J
        @test b1[ind].type == b2[ind].type
        @test b1[ind].metadata == b2[ind].metadata
    end

    # x,y: 2 traits, no missing values
    m1 = PGBP.MvDiagBrownianMotion((1,1), (0,0), (1.2,3))
    b1 = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m1);
    PGBP.init_beliefs_assignfactors!(b1, m1, tbl, df.taxon, net.nodes_changed);

    m2 = PGBP.MvDiagBrownianMotion((1,1), (0,0), (0,0))
    b2 = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m2);
    PGBP.init_beliefs_assignfactors!(b2, m2, tbl, df.taxon, net.nodes_changed);

    PGBP.update_root_inscope!(b1, m2)
    PGBP.init_beliefs_reset!(b1)
    PGBP.init_beliefs_assignfactors!(b1, m2, tbl, df.taxon, net.nodes_changed);
    
    for ind in eachindex(b1)
        @test b1[ind].nodelabel == b2[ind].nodelabel
        @test b1[ind].ntraits == b2[ind].ntraits
        @test b1[ind].inscope == b2[ind].inscope
        @test b1[ind].μ == b2[ind].μ
        @test b1[ind].h == b2[ind].h
        @test b1[ind].J == b2[ind].J
        @test b1[ind].type == b2[ind].type
        @test b1[ind].metadata == b2[ind].metadata
    end

end # of root update

@testset "exact formulas" begin

    # y: 1 trait, no missing values
    # exact REML
    m = PGBP.UnivariateBrownianMotion(1, 0, Inf) # infinite root variance
    b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
    PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
    cgb = PGBP.ClusterGraphBelief(b)
    PGBP.calibrate!(cgb, [spt])
    mod, llscore = PGBP.calibrate_exact_cliquetree!(cgb, ct, net.nodes_changed,
        tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1,0,0))

    @test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
    @test llscore ≈ -6.851098376474686
    @test mod.μ ≈ 0.4436893203883497
    @test PGBP.varianceparam(mod) ≈ 0.6235275080906149

    # numerical optim
    m = PGBP.UnivariateBrownianMotion(0.5,0.5,0) 
    b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
    PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
    cgb = PGBP.ClusterGraphBelief(b)
    PGBP.calibrate!(cgb, [spt])
    modopt, llscoreopt, opt = PGBP.calibrate_optimize_cliquetree!(cgb, ct, net.nodes_changed,
        tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (0.5,-3))

    @test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscoreopt
    @test llscoreopt ≈ -6.793239498189161
    @test modopt.μ ≈ mod.μ
    @test PGBP.varianceparam(modopt) ≈ PGBP.varianceparam(mod) * (n-1) / n

    # x,y: 2 traits, no missing values
    m = PGBP.MvDiagBrownianMotion((1,1), (0,0), (Inf,Inf))
    b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
    PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
    cgb = PGBP.ClusterGraphBelief(b)
    PGBP.calibrate!(cgb, [spt])
    mod, llscore = PGBP.calibrate_exact_cliquetree!(cgb, ct, net.nodes_changed,
       tbl, df.taxon, PGBP.MvFullBrownianMotion, ([1 0; 0 1], [0,0]))

    #@test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
    #@test llscore ≈  -6.851098376474686
    @test mod.μ ≈ [5.990291262135922 ; 0.4436893203883498]
    @test PGBP.varianceparam(mod) ≈ [5.970873786407767 1.3310679611650484 ; 1.3310679611650484 0.6235275080906149]

    #= ML solution the matrix-way, analytical for BM:
    # for y: univariate
    Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    n=5 # number of data points
    i = ones(n) # intercept
    μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * tbl.y) # 0.4436893203883497
    r = tbl.y .- μhat
    σ2hat_REML = (transpose(r) * inv(Σ) * r) / (n-1) # 0.6235275080906149
    σ2hat_ML = (transpose(r) * inv(Σ) * r) / n # 0.49882200647249186
    llscorereml = - (n-1)/2 - logdet(2π * σ2hat_REML .* Σ)/2 # -6.851098376474686
    llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 # -6.79323949818916
    # for x,y: multivariate
    Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    n=5 # number of data points
    datatbl = [tbl.x  tbl.y]
    i = ones(n) # intercept
    μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * datatbl) # 5.990291262135922 0.4436893203883498
    r = datatbl - i * μhat
    σ2hat_REML = (transpose(r) * inv(Σ) * r) / (n-1) # 2×2 Matrix{Float64}: 5.97087  1.33107 1.33107  0.623528
    llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 # -6.793239498189161
    =#

end # of exact formulas
    

end

@testset "exact network calibration for the BM" begin

    netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"

    df = DataFrame(taxon=["A","B1","B2","C"], x=[10,10,2,0], y=[1.0,.9,1,-1])
    n = 4
    df_var = select(df, Not(:taxon))
    tbl = columntable(df_var)
    tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models
    tbl_x = columntable(select(df, :x))
    
    net = readTopology(netstr)
    g = PGBP.moralize!(net)
    PGBP.triangulate_minfill!(g)
    ct = PGBP.cliquetree(g)
    spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
    
    @testset "exact formulas" begin
    
        # y: 1 trait, no missing values
        # exact REML
        m = PGBP.UnivariateBrownianMotion(1, 0, Inf) # infinite root variance
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        cgb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(cgb, [spt])
        mod, llscore = PGBP.calibrate_exact_cliquetree!(cgb, ct, net.nodes_changed,
            tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1,0,0))

        @test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
        @test llscore ≈ -5.250084678427689
        @test mod.μ ≈ -0.260008715071627
        @test PGBP.varianceparam(mod) ≈ 0.4714735834478194
    
        # numerical optim
        m = PGBP.UnivariateBrownianMotion(0.5,0.5,0) 
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        cgb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(cgb, [spt])
        modopt, llscoreopt, opt = PGBP.calibrate_optimize_cliquetree!(cgb, ct, net.nodes_changed,
            tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (0.5,-3))
    
        @test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscoreopt
        @test llscoreopt ≈ -5.174720533524127
        @test modopt.μ ≈ mod.μ
        @test PGBP.varianceparam(modopt) ≈ PGBP.varianceparam(mod) * (n-1) / n
    
        # x,y: 2 traits, no missing values
        m = PGBP.MvDiagBrownianMotion((1,1), (0,0), (Inf,Inf))
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        cgb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(cgb, [spt])
        mod, llscore = PGBP.calibrate_exact_cliquetree!(cgb, ct, net.nodes_changed,
           tbl, df.taxon, PGBP.MvFullBrownianMotion, ([1 0; 0 1], [0,0]))

        #@test PGBP.integratebelief!(cgb, spt[3][1])[2] ≈ llscore
        #@test llscore ≈  -6.851098376474686
        @test mod.μ ≈ [2.791001688545128 ; -0.260008715071627]
        @test PGBP.varianceparam(mod) ≈ [17.93326111121198 1.6089749098736517 ; 1.6089749098736517 0.4714735834478195]
    
        #= ML solution the matrix-way, analytical for BM:
        # for y: univariate
        Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])
        n=4 # number of data points
        i = ones(n) # intercept
        μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * tbl.y) 
        r = tbl.y .- μhat
        σ2hat_REML = (transpose(r) * inv(Σ) * r) / (n-1) 
        σ2hat_ML = (transpose(r) * inv(Σ) * r) / n 
        llscorereml = - (n-1)/2 - logdet(2π * σ2hat_REML .* Σ)/2 
        llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 
        # for x,y: multivariate
        Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])
        n=4 # number of data points
        datatbl = [tbl.x  tbl.y]
        i = ones(n) # intercept
        μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * datatbl) 
        r = datatbl - i * μhat
        σ2hat_REML = (transpose(r) * inv(Σ) * r) / (n-1) 
        llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 
        =#
    
    end # of exact formulas
        
    
    end