@testset "generalized beliefs" begin
    net1 = "((#H1:0.0::0.4,#H2:0.0::0.4)I1:1.0,(((A:1.0)#H1:0.0::0.6,#H3:0.0::0.4)#H2:0.0::0.6,(B:1.0)#H3:0.0::0.6)I2:1.0)I3;"
    # net2 is modified from mateescu_2010.phy with all hybrid edge lengths set to 0
    net2 = "((((g:1.0)#H4:0.0::0.6)#H2:0.0::0.6,(d:1.0,(#H2:0.0::0.4,#H4:0.0::0.4)#H3:0.0::0.6)D:1.0,(#H3:0.0::0.4)#H1:0.0::0.6)B:1.0,#H1:0.0::0.4)A;"

    @testset "no optimization" begin
        @testset "Level-3 w/ 2 tips. Univariate. Clique tree" begin
            net = readTopology(net1)
            df = DataFrame(taxon=["A","B"], x=[2.11,2.15])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(0.000325097529258775, 2.128439531859558)
            b, c2n = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            ctb = PGBP.ClusterGraphBelief(b, c2n)
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, c2n)
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            PGBP.calibrate!(ctb, [spt])
            #=
            using StatsModels
            fitx_ml = phylolm(@formula(x ~ 1), df, net; tipnames=:taxon, reml=false)
            PhyloNetworks.loglikelihood(fitx_ml) # 4.73520292387366
            mu_phylo(fitx_ml) # 2.128439531859558
            sigma2_phylo(fitx_ml) # 0.000325097529258775
            =#
            llscore = 4.73520292387366
            for i in eachindex(ctb.belief)
                _, tmp = PGBP.integratebelief!(ctb, i) # llscore from norm constant
                if i in [1,3,7,8] # leaf clusters {1,3} and adjacent sepsets {7,8}
                    @test_broken tmp ≈ llscore
                else
                    @test tmp ≈ llscore
                end
            end
            # todo: compare ancestral reconstruction
        end
        @testset "Level-4 w/ 2 tips. Univariate. Clique tree" begin
            #= root node is a parent of a degenerate hybrid, and also a parent of its
            coparent (for the hybrid) =#
            net = readTopology(net2)
            df = DataFrame(taxon=["d","g"], x=[1.0,-1.0])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(1, 0)
            b, c2n = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, c2n)
            ctb = PGBP.ClusterGraphBelief(b, c2n)
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            PGBP.propagate_1traversal_postorder!(ctb, spt..., true, true, false)
            llscore = -3.4486412230145387
            #=
            using StatsModels
            fitx_ml = phylolm(@formula(x ~ 1), df, net; tipnames=:taxon, reml=false)
            μ = 0; σ2 = 1; Vy = fitx_ml.Vy; Y = fitx_ml.Y
            # llscore
            -0.5*transpose(Y .- μ)*inv(σ2*Vy)*(Y .- μ) - 0.5*logdet(2π*σ2*Vy) # -3.4486412230145387
            =#
            rootj = spt[3][1]
            _, res = PGBP.integratebelief!(ctb, rootj)
            @test_broken res ≈ llscore
        end
    end
end