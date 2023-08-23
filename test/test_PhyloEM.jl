@testset "tree calibration BM" begin

# Comparison of results of BP with results from PhyloEM, for the BM on a tree.
# TODO: clean implementation of the mu hat and sigma2 hat estimations with BP (tree case / network case).
# TODO: multivariate BM.

netstr = "((A:1.5,B:1.5):1,(C:1,(D:0.5, E:0.5):0.5):1.5);"

df = DataFrame(taxon=["A","B","C","D","E"], x=[10,10,missing,0,1], y=[1.0,.9,1,-1,-0.9])
df_var = select(df, Not(:taxon))
tbl = columntable(df_var)
tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models
tbl_x = columntable(select(df, :x))

net = readTopology(netstr)
g = PGBP.moralize!(net)
PGBP.triangulate_minfill!(g)
ct = PGBP.cliquetree(g)
spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)

@testset "comparison with PhyloEM" begin

m = PGBP.UnivariateBrownianMotion(1, 0, 10000000000) # "infinite" root variance
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
    dim(vv) == 2 && @test vv[1,2] ≈ condcovar[cgb.belief[i].nodelabel[1]] atol=1e-6
end
# Test mu hat
muhat = 0.4436893
tmp1, tmp = PGBP.integratebelief!(cgb, 13)
@test tmp1[end] ≈ muhat atol=1e-6
# Test sigma2 hat
# TODO: as individual conditional moments match, this should also work.


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

end # of no-optimization

end
