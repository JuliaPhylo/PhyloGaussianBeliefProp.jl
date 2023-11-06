#=
Test `calibrate_optimize...` methods over larger networks. For now:
(1) check that no exceptions are raised during the optimization for loopy cluster
graphs
(2) compare parameter estimates among cluster graphs and clique tree
=#
@testset "calibration w/ optimization" begin
    @testset "mateescu 2010" begin
        net = readTopology("test/example_networks/mateescu_2010.txt")
        df = DataFrame(taxon=["d","g"], y=[1.0,-1.0])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        m = PGBP.UnivariateBrownianMotion(5, 10, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (5,10))
        # σ2: 0.5932929749974909, μ: -0.0753438602908851, fenergy: 3.2476515320155706

        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (5,10))
        # σ2: 0.5932937632251198, μ: -0.07534469942447958, fenergy: 3.1875038192643235

        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (5,10))
        # σ2: 0.5932930079316453, μ: -0.07534357688052816, llscore: -3.276318068707007
    end
    @testset "lazaridis 2014" begin # 21 nodes: 7 tips, 4 hybrids
        net = readTopology("test/example_networks/lazaridis_2014.txt")
        tipnames = tipLabels(net) # A, B, C, D, E, F, G
        # Random.seed!(100); y = rand(length(tipnames))
        df = DataFrame(taxon=tipnames, y=[0.12364754150469992,
            0.17728426618671156, 0.5844262152209776, 0.9102808037366388,
            0.04170637082547557, 0.2882051781587147, 0.8650936306221244])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.05755704694375791, μ: 0.2667930364433495, fenergy: 2.7786627367693253

        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.057557046412633615, μ: 0.2667930347837426, fenergy: 2.739989632567589

        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1,0.5))
        # σ2: 0.0575570466385407, μ: 0.2667930359317636, llscore: -2.7983124321615196
    end
    @testset "nielsen 2023" begin # 25 nodes: 11 tips, 4 hybrids
        net = readTopology("test/example_networks/nielsen_2023.txt")
        tipnames = tipLabels(net) # Malta, Anzick, Aymara, USR1, Athabascan, H1,
            # Koryak, Saqqaq, H4, Ket, UstIshim
        # Random.seed!(302); y = rand(length(tipnames))
        df = DataFrame(taxon=tipnames, y=[0.20330328616895177, 0.4503698746530027,
            0.41801425177653495, 0.9600346330734624, 0.6790540382500203,
            0.5592384352556216, 0.0671986257495033, 0.32464497672593173,
            0.05002711799330517, 0.2623737748394136, 0.43366984099643113])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.05141066851135868, μ: 0.4230425215093448, fenergy: 3.4952232954589

        ## fixit: debug this case
        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        @test_broken mod, fenergy, opt =
            PGBP.calibrate_optimize_clustergraph!(cgb, cg, net.nodes_changed,
            tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1,0.5))

        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1,0.5))
        # σ2: 0.05141066842693317, μ: 0.42304252136242837, llscore: -3.5019465814003974
    end
    @testset "maier 2023" begin
        net = readTopology("test/example_networks/maier_2023.txt")
        tipnames = tipLabels(net) # D, E, A, B, H, F, G, C, OUT
        # Random.seed!(2439); y = rand(length(tipnames))
        df = DataFrame(taxon=tipnames, y=[0.9284240880503369, 0.6994810279219996,
            0.942636648810464, 0.32127909567594415, 0.9980270763251334,
            0.1479440266464357, 0.39852159496840733, 0.5103167425553811,
            0.045396526498428846])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.059637524451356276, μ: 0.20682634076421555, fenergy: 4.042161200358414
        
        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2 = 0.059637524149579464, μ = 0.20682634188672502, fenergy: 4.038876029644214
        
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1,0.5))
        # σ2: 0.05963752442674132, μ: 0.2068263407492875, llscore: -4.058399611427243
    end
    @testset "bergstrom 2020" begin # 19 nodes: 7 tips, 3 hybrids
        net = readTopology("test/example_networks/bergstrom_2020.txt")
        tipnames = tipLabels(net) # America_pool, New_Guinea, Baikal_pool,
            # Karelia_Mesolithic, Germany_EarlyNeolithic, Israel_7000BP, Andean_fox
        # Random.seed!(103); y = rand(length(tipnames))
        df = DataFrame(taxon=tipnames, y=[0.7211310054331097, 0.029622858442701072,
            0.5542917861393727, 0.716622917808044, 0.8389644993872266,
            0.15463907028759327, 0.5281502131861057])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.13135581848044076, μ: 0.48612938360704316, fenergy: 3.4623261393446825

        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.13135581895224993, μ: 0.48612938335848943, fenergy: 3.3812301723005884

        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1,0.5))
        # σ2: 0.13135581833904802, μ: 0.4861293844675344, llscore: -3.4623261393445737
    end
    @testset "sun 2023" begin # 42 nodes: 10 tips, 6 hybrids
        net = readTopology("test/example_networks/sun_2023.txt")
        tipnames = tipLabels(net) # PUN, PLE, TIG, SUM, JAX, COR, VIR, ALT, RUSA21, AMO
        # Random.seed!(3030); y = rand(length(tipnames))
        df = DataFrame(taxon=tipnames, y=[0.26435817547115015, 0.3118520778265339,
            0.5090517205125478, 0.32303037843523685, 0.22207876027252582,
            0.3477866367757225, 0.14482423074065276, 0.938668741962465,
            0.8496534329028661, 0.3251413542458399])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.15055281018552913, μ: 0.31482680770882715, fenergy: 4.832699815704785

        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0.5, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m)
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
                (1,0.5))
        # σ2: 0.15055281794771513, μ: 0.3148268242526379, fenergy: 4.844505357657141

        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion,
            (1,0.5))
        # σ2: 0.15055281928486133, μ: 0.3148268180453915, llscore: -4.8328379081101875
    end
end