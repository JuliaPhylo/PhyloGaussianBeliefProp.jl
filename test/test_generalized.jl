@testset "generalized beliefs" begin
    net1 = "((#H1:0.0::0.4,#H2:0.0::0.4)I1:1.0,(((A:1.0)#H1:0.0::0.6,#H3:0.0::0.4)#H2:0.0::0.6,(B:1.0)#H3:0.0::0.6)I2:1.0)I3;"
    # net2 is modified from mateescu_2010.phy with all hybrid edge lengths set to 0
    net2 = "((((g:1.0)#H4:0.0::0.6)#H2:0.0::0.6,(d:1.0,(#H2:0.0::0.4,#H4:0.0::0.4)#H3:0.0::0.6)D:1.0,(#H3:0.0::0.4)#H1:0.0::0.6)B:1.0,#H1:0.0::0.4)A;"
    # net3 is from SM section F of manuscript
    net3 = "((i1:1.0,(i2:1.0)#H1:0.0::0.5)i4:1.0, (#H1:0.0::0.5,i3:1.0)i6:1.0)i0;"
    net4 = "(((i2:0.0)#H1:0.0::0.5)i4:1.0, (#H1:0.0::0.5)i6:1.0)i0;"
    
    @testset "Univariate. No optimization. Clique tree" begin
        @testset "Hybrid leaf. 1 tip" begin
            net = readTopology(net4)
            df = DataFrame(taxon=["i2"], x=[1.0])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(1, 0)
            b, (n2c, n2f, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            ctb = PGBP.ClusterGraphBelief(b, n2c, n2f, c2n)
            # no method to multiply a Dirac measure into a CanonicalBelief
            @test_throws MethodError PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, n2c, n2f)
            # convert b[1] and b[4] into GeneralizedBelief
            b[1] = PGBP.GeneralizedBelief(b[1])
            b[4] = PGBP.GeneralizedBelief(b[4])
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, n2c, n2f)
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            # b[3] and b[5] should also be represented as GeneralizedBeliefs
            @test_throws ErrorException PGBP.calibrate!(ctb, [spt])
            b[3] = PGBP.GeneralizedBelief(b[3])
            b[5] = PGBP.GeneralizedBelief(b[5])
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, n2c, n2f)
            PGBP.calibrate!(ctb, [spt])
            llscore = -1.5723649429247 # -0.5*(1-0)^2/0.5 - 0.5*logdet(2π*0.5)
            for i in eachindex(ctb.belief)
                if i ∈ [1,4]
                    # belief is fully deterministic and not normalizable
                    @test_throws ErrorException PGBP.integratebelief!(ctb, i)
                else
                    _, tmp = PGBP.integratebelief!(ctb, i)
                    @test tmp ≈ llscore
                end
            end
        end
        @testset "Level-1. 3 tips" begin
            net = readTopology(net3)
            df = DataFrame(taxon=["i1","i2","i3"], x=[1.0,1.0,1.0])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(1, 0)
            b, (n2c, n2f, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, n2c, n2f)
            ctb = PGBP.ClusterGraphBelief(b, n2c, n2f, c2n)
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            PGBP.calibrate!(ctb, [spt])
            # μ = 0; σ2 = 1; Vy = sharedPathMatrix(net)[:Tips]; Y = [1.0,1.0,1.0]
            # -0.5*transpose(Y .- μ)*inv(σ2*Vy)*(Y .- μ) - 0.5*logdet(2π*σ2*Vy) # -4.161534555831068
            @test PGBP.integratebelief!(ctb, 6)[1] ≈ [0.6] # H1
            llscore = -4.161534555831068
            for i in eachindex(ctb.belief)
                _, tmp = PGBP.integratebelief!(ctb, i)
                @test tmp ≈ llscore
            end
        end
        @testset "Level-3. 2 tips" begin
            net = readTopology(net1)
            df = DataFrame(taxon=["A","B"], x=[2.11,2.15])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(0.000325097529258775, 2.128439531859558)
            b, (n2c, n2f, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            ctb = PGBP.ClusterGraphBelief(b, n2c, n2f, c2n)
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, n2c, n2f)
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
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
            net = readTopology(net2)
            df = DataFrame(taxon=["d","g"], x=[1.0,-1.0])
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(1, 0)
            b, (n2c, n2f, n2d, c2n) = PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m)
            PGBP.assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed, n2c, n2f)
            ctb = PGBP.ClusterGraphBelief(b, n2c, n2f, c2n)
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            PGBP.calibrate!(ctb, [spt])
            #= μ = 0; σ2 = 1; Vy = sharedPathMatrix(net)[:Tips]; Y = [-1.0,1.0]
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

    @testset "Multivariate. No optimization. Clique tree" begin
        net3 = "((i1:1.0,(i2:1.0)#H1:0.0::0.5)i4:1.0, (#H1:0.0::0.5,i3:1.0)i6:1.0)i0;"
        net = readTopology(net3)
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        df = DataFrame(taxon=["i1","i2","i3"], x=[1.0,1.0,1.0], y=[2.0,2.0,2.0])
        df_var = select(df, Not(:taxon))
        tbl = columntable(df_var)
        m_biBM_fixedroot = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0,0))
        b_xy_fixedroot = PGBP.allocatebeliefs(tbl, df.taxon, net.nodes_changed, ct,
            m_biBM_fixedroot)
        PGBP.assignfactors!(b_xy_fixedroot[1], m_biBM_fixedroot, tbl, df.taxon,
            net.nodes_changed, b_xy_fixedroot[2][1], b_xy_fixedroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_fixedroot[1], b_xy_fixedroot[2][1],
            b_xy_fixedroot[2][2], b_xy_fixedroot[2][4])
        spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
        PGBP.calibrate!(ctb, [spt])
        # Vy = sharedPathMatrix(net)[:Tips];
        # μ = repeat([3, -3],3); σ2 = [2 0; 0 1]; 
        # Y = repeat([1.0,2.0],3)
        # -0.5*transpose(Y - μ)*inv(kron(Vy,σ2))*(Y - μ) - 0.5*logdet(2π*kron(Vy,σ2)) # -24.362789882502057
        llscore = -24.362789882502057
        for i in eachindex(ctb.belief)
            _, tmp = PGBP.integratebelief!(ctb, i)
            @test tmp ≈ llscore
        end

        m = PGBP.MvFullBrownianMotion([2.0 0.5; 0.5 1.0], [3.0,-3.0])
        PGBP.assignfactors!(b_xy_fixedroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_fixedroot[2][1], b_xy_fixedroot[2][2]);
        PGBP.calibrate!(ctb, [spt])
        # Vy = sharedPathMatrix(net)[:Tips];
        # μ = repeat([3, -3],3); σ2 = [2.0 0.5; 0.5 1.0]; 
        # Y = repeat([1.0,2.0],3)
        # -0.5*transpose(Y - μ)*inv(kron(Vy,σ2))*(Y - μ) - 0.5*logdet(2π*kron(Vy,σ2)) # -29.905349936422414
        llscore = -29.905349936422414
        for i in eachindex(ctb.belief)
            _, tmp = PGBP.integratebelief!(ctb, i)
            if i ∈ [4,8]
                @test tmp ≈ llscore
            else
                @test_broken tmp ≈ llscore
            end
        end
    end
end