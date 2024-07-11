# ≈ 13.0s if run as the first/only testset
@testset "generalized beliefs" begin
examplenetdir = joinpath(dirname(Base.find_package("PhyloGaussianBeliefProp")),
    "..", "test","example_networks")
    @testset "optimization" begin
        @testset "Level-1 w/ 3 degenerate hybrids. Clique tree + Optim / Exact" begin
            net = readTopology(joinpath(examplenetdir, "teo_2023.phy"))
            df = DataFrame(
                morph=["micranthum", "occidentale_occidentale", "reptans", "pectinatum",
                    "delicatum", "pulcherrimum_shastense", "elegans", "carneum", "eddyense",
                    "chartaceum", "aff._viscosum_sp._nov.", "elusum", "pauciflorum",
                    "confertum", "brandegeei", "foliosissimum", "apachianum"],
                x=[-0.865, 0.542, 0.747, 0.834, 0.0316, -1.04, -1.06, 0.757, -1.22, -1.61,
                    -1.37, 0.213, 0.222, -0.43, -0.675, 0.622, 0.289]) # leaflet log-length
            tbl_x = columntable(select(df, :x))
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            m = PGBP.UnivariateBrownianMotion(1, 0)
            b, c2n = PGBP.allocatebeliefs(tbl_x, df.morph, net.nodes_changed, ct, m)
            ctb = PGBP.ClusterGraphBelief(b, c2n)
            mod, ll = PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed, tbl_x,
                df.morph, PGBP.UnivariateBrownianMotion, (1.0, 0))
            #=
            using StatsModels
            fitx_ml = phylolm(@formula(x ~ 1), df, net; tipnames=:morph, reml=false)
            PhyloNetworks.loglikelihood(fitx_ml) # -20.73388059680892
            mu_phylo(fitx_ml) # -0.2903552966336476
            sigma2_phylo(fitx_ml) # 0.9052793896128867

            fitx_reml = phylolm(@formula(x ~ 1), df, net; tipnames=:morph, reml=true)
            PhyloNetworks.loglikelihood(fitx_reml) # -20.552504004459774
            # the above is the restricted maximum likelihood, to get the likelihood based on
            # the reml estimates, do:
            # -(16/2) - (17/2)*(log(sigma2_phylo(fitx_reml)) + log(2π)) - 
            #     0.5*logdet(fitx_reml.Vy)
            # which gives -20.749189882248615
            mu_phylo(fitx_reml) # -0.2903552966336476
            sigma2_phylo(fitx_reml) # 0.9618593514636921
            =#
            @test ll ≈ -20.73388059680892 # maximum likelihood
            @test mod.μ ≈ -0.2903552966336476
            @test mod.σ2 ≈ 0.9052793896128867

            sched = PGBP.spanningtrees_clusterlist(ct, net.nodes_changed);
            mod, ll = PGBP.calibrate_exact_cliquetree!(ctb, sched[1], net.nodes_changed, tbl_x,
                df.morph, PGBP.UnivariateBrownianMotion);
            @test ll ≈ -20.749189882248615 # likelihood with reml estimates
            @test mod.μ ≈ -0.2903552966336476
            @test mod.σ2 ≈ 0.9618593514636921
        end
    end
end