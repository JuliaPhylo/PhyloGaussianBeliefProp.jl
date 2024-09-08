# ≈ 13.0s if run as the first/only testset
@testset "generalized beliefs" begin
    @testset "optimization" begin
        @testset "Level-3 w/ 3 degenerate hybrids. Clique tree + Optim / Exact" begin
            netstr = "((#H1:0.0::0.4,#H2:0.0::0.4)I1:1.0,(((A:1.0)#H1:0.0::0.6,#H3:0.0::0.4)#H2:0.0::0.6,(B:1.0)#H3:0.0::0.6)I2:1.0)I3;"
            net = readTopology(netstr)
            df = DataFrame(taxon=["A","B"], x=[2.11,2.15])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(1, 0)
            b, c2n = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            ctb = PGBP.ClusterGraphBelief(b, c2n)
            mod, ll = PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed, tbl_x,
                df.taxon, PGBP.UnivariateBrownianMotion, (1.0, 0))
            #=
            using StatsModels
            fitx_ml = phylolm(@formula(x ~ 1), df, net; tipnames=:taxon, reml=false)
            PhyloNetworks.loglikelihood(fitx_ml) # 4.73520292387366
            mu_phylo(fitx_ml) # 2.128439531859558
            sigma2_phylo(fitx_ml) # 0.000325097529258775
            =#
            @test ll ≈ 4.73520292387366 # maximum likelihood
            @test mod.μ ≈ 2.128439531859558
            @test mod.σ2 ≈ 0.000325097529258775 rtol=1e-6
        end
    end
end