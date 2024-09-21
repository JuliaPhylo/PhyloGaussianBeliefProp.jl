@testset "generalized beliefs" begin
    @testset "ancestral reconstruction" begin
        #=
        From example_networks/mateescu_2010.phy, but with hybrid edge lengths all set to 0.
        9 nodes: 2 tips, 4 hybrids. level-4, has hybrid ladder and deg-4 hybrid
        =#
        netstr = "((((g:1.0)#H4:0.0::0.6)#H2:0.0::0.6,(d:1.0,(#H2:0.0::0.4,#H4:0.0::0.4)#H3:0.0::0.6)D:1.0,(#H3:0.0::0.4)#H1:0.0::0.6)B:1.0,#H1:0.0::0.4)A;"
        net = readTopology(netstr)
        df = DataFrame(taxon=["d","g"], x=[1.0,-1.0])
        tbl_x = columntable(select(df, :x))
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        m = PGBP.UnivariateBrownianMotion(1, 0)
        b, c2n = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
        PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, c2n)
        ctb = PGBP.ClusterGraphBelief(b, c2n)
        spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
        PGBP.propagate_1traversal_postorder!(ctb, spt..., true, true, false)
        rootj = spt[3][1]
        res = PGBP.integratebelief!(ctb, rootj)
        #=
        using StatsModels
        fitx_ml = phylolm(@formula(x ~ 1), df, net; tipnames=:taxon, reml=false)
        PhyloNetworks.loglikelihood(fitx_ml) # -3.3793085481891216
        =#
        @test_broken res ≈ -3.3793085481891216
    end
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