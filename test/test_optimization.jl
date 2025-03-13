@testset "calibration w/ optimization" begin
# univariate, multivariate, missing, fixed/proper/improper root
examplenetdir = joinpath(dirname(Base.find_package("PhyloGaussianBeliefProp")),
    "..", "test","example_networks")
    @testset "univariate, no missing, fixed root" begin
        net = readnewick(joinpath(examplenetdir, "mateescu_2010.phy"))
        # 9 nodes: 2 tips, 4 hybrids
        # level-4, not tree-child, has hybrid ladder, has deg-4 hybrid
        df = DataFrame(taxon=["d","g"], y=[1.0,-1.0])
        tbl_y = columntable(select(df, :y))
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        m = PGBP.UnivariateBrownianMotion(1.0, 0.0) # σ2 = 1.0, μ = 0.0
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        ## reference is from a previous call to `calibrate_optimize_cliquetree!`
        refμ = -0.07534357691418593
        refσ2 = 0.5932930079336234
        refll = -3.2763180687070053
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.vec_node, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1.0,0.0))
        @test mod.μ ≈ refμ
        @test mod.σ2 ≈ refσ2
        @test llscore ≈ refll
        ## test against `calibrate_optimize_cliquetree_autodiff!`
        lbc = GeneralLazyBufferCache(function (paramOriginal)
            mo = PGBP.UnivariateBrownianMotion(paramOriginal...)
            bel = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, mo)
            return PGBP.ClusterGraphBelief(bel)
        end)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct,
            net.vec_node, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1.0,0,0))
        @test mod.μ ≈ refμ rtol=4e-10
        @test mod.σ2 ≈ refσ2 rtol=3e-11
        @test llscore ≈ refll rtol=3e-16
        ## compare with Bethe
        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.vec_node, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1.0,0.0))
        @test mod.μ ≈ refμ rtol=2e-5
        @test mod.σ2 ≈ refσ2 rtol=2e-6
        @test fenergy ≈ refll rtol=3e-2
    end
    # norm(mod.μ-refμ)/max(norm(mod.μ),norm(refμ))
    # norm(mod.σ2-refσ2)/max(norm(mod.σ2),norm(refσ2))
    # norm(fenergy+refll)/max(norm(fenergy),norm(refll))
    # norm(llscore-refll)/max(norm(llscore),norm(refll))
    @testset "bivariate, no missing, improper root" begin
        net = readnewick(joinpath(examplenetdir, "sun_2023.phy"))
        # 42 nodes: 10 tips, 6 hybrids
        # level-6, not tree-child, has hybrid ladder, has deg-4 hybrid
        #= tip data simulated from ParamsMultiBM():
        rootmean = [0.0, 0.0]; rate = [2.0 1.0; 1.0 2.0]
        sim = simulate(net, ParamsMultiBM(rootmean, rate))
        y1 = sim[:Tips][1,:]; y2 = sim[:Tips][2,:] =#
        df = DataFrame(taxon=tiplabels(net),
                y1=[-1.001, 0.608, -3.606, -7.866, -5.977, -6.013, -7.774,
                    -5.511, -6.392, -6.471],
                y2=[0.262, 5.124, -5.076, -6.223, -7.033, -6.062, -6.42, -6.34,
                    -6.516, -6.501])
        df_var = select(df, Not(:taxon))
        tbl = columntable(df_var)
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        # min cluster size: 2, max cluster size: 5
        m = PGBP.MvFullBrownianMotion([2.0 1.0; 1.0 2.0], [0.0 0.0], [Inf 0.0; 0.0 Inf])
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.vec_node);
        ctb = PGBP.ClusterGraphBelief(b)
        # note: reaches max iterations before converging, so just save results for now
        # mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
        #     net.vec_node, tbl, df.taxon, PGBP.MvFullBrownianMotion,
        #     ([2.0 1.0; 1.0 2.0], [0.0 0.0], [Inf 0.0; 0.0 Inf]))
        
        # Multivariate Brownian motion
        # - evolutionary variance rate matrix: R:
        # [3.717085841556895 1.7464551312269698; 1.7464551312269698 2.0994767855707854]
        # - root mean: μ = [0.0, 0.0]
        # - root variance: v = [Inf 0.0; 0.0 Inf], -32.22404541422671,
        # * Status: failure (reached maximum number of iterations)
        #  * Candidate solution
        #     Final objective value:     3.222405e+01

        #  * Found with
        #     Algorithm:     L-BFGS

        #  * Convergence measures
        #     |x - x'|               = 4.05e-09 ≰ 0.0e+00
        #     |x - x'|/|x'|          = 6.16e-09 ≰ 0.0e+00
        #     |f(x) - f(x')|         = 5.68e-13 ≰ 0.0e+00
        #     |f(x) - f(x')|/|f(x')| = 1.76e-14 ≰ 0.0e+00
        #     |g(x)|                 = 9.97e-08 ≰ 1.0e-08

        #  * Work counters
        #     Seconds run:   248  (vs limit Inf)
        #     Iterations:    1000
        #     f(x) calls:    3180
        #     ∇f(x) calls:   3180

        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(4));
        # min cluster size: 1, max cluster size: 4
        b_jg = PGBP.init_beliefs_allocate(tbl, df.taxon, net, cg, m);
        PGBP.init_beliefs_assignfactors!(b_jg, m, tbl, df.taxon, net.vec_node);
        cgb = PGBP.ClusterGraphBelief(b_jg)
        # note: reaches max iterations before converging, so just save results for now
        # mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
        #     net.vec_node, tbl, df.taxon, PGBP.MvFullBrownianMotion,
        #     ([2.0 1.0; 1.0 2.0], [0.0 0.0], [Inf 0.0; 0.0 Inf]))
        # Multivariate Brownian motion
        # - evolutionary variance rate matrix: R:
        # [3.7170858696599423 1.7464551640805306; 1.7464551640805306 2.0994768084399875]
        # - root mean: μ = [0.0, 0.0]
        # - root variance: v = [Inf 0.0; 0.0 Inf], 32.270019493029075,
        # * Status: failure (reached maximum number of iterations)
        #  * Candidate solution
        #     Final objective value:     3.227002e+01
        
        #  * Found with
        #     Algorithm:     L-BFGS
        
        #  * Convergence measures
        #     |x - x'|               = 1.16e-10 ≰ 0.0e+00
        #     |x - x'|/|x'|          = 1.76e-10 ≰ 0.0e+00
        #     |f(x) - f(x')|         = 6.39e-14 ≰ 0.0e+00
        #     |f(x) - f(x')|/|f(x')| = 1.98e-15 ≰ 0.0e+00
        #     |g(x)|                 = 2.55e-07 ≰ 1.0e-08
        
        #  * Work counters
        #     Seconds run:   338  (vs limit Inf)
        #     Iterations:    1000
        #     f(x) calls:    3270
        #     ∇f(x) calls:   3270
    end
end