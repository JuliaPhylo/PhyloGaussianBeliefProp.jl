@testset "calibration" begin

#= Nest testsets by "optimization"/"no optimization", then network used, then
name by cluster graph method used =#
@testset "no optimization" begin
    @testset "4-taxon, level-1: #1" begin
        netstr = "(A:2.5,((B:1,#H1:0.5::0.1):1,(C:1,(D:0.5)#H1:0.5::0.9):1):0.5);"
        net = readTopology(netstr)
        df = DataFrame(y=[11.275034507978296, 10.032494469945764,
            11.49586603350308, 11.004447427824012], taxon=["A","B","C", "D"])
        tbl_y = columntable(select(df, :y))
        @testset "Bethe" begin
            m = PGBP.UnivariateBrownianMotion(1, 10, 0)
            cg = PGBP.clustergraph!(net, PGBP.Bethe())
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            cgb = PGBP.ClusterGraphBelief(b)
            PGBP.regularizebeliefs!(cgb, cg)
            schedule = PGBP.spanningtrees_cover_clusterlist(cg, net.nodes_changed)
            @test PGBP.calibrate!(cgb, schedule, 20; auto=true)

            # Compare posterior expectations with RxInfer
            posmeans = [PGBP.integratebelief!(cgb.belief[i])[1][1] for i in
                [PGBP.clusterindex(s, cgb) for s in [:I1,:H1,:I2,:I3]]]
            @test all([isapprox(m1, m2, rtol=1e-4) for (m1, m2) in
                zip(posmeans, [10.162833376432536, 10.932362054678077,
                    10.952187456727945, 10.27875520829012])])
            # Compare posterior variances with RxInfer
            posvars = [(cgb.belief[i].J \ I)[1] for i in
                [PGBP.clusterindex(s, cgb) for s in [:I1,:H1,:I2,:I3]]]
            @test all([isapprox(v1, v2, rtol=1e-4) for (v1, v2) in
                zip(posvars, [0.5768644029378899, 0.31991746006101257,
                    0.3847763603363032, 0.31694527029261793])])
        end
    end
    @testset "4-taxon, level-1: #2" begin
        netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"
        net = readTopology(netstr)
        df = DataFrame(taxon=["A","B1","B2","C"], y=[1.0,.9,1,-1])
        tbl_y = columntable(select(df, :y))
        m = PGBP.UnivariateBrownianMotion(2, 3, 0)

        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        @test PGBP.default_sepset1(ctb) == 8
        llscore = -10.732857817537196
        tmp1, tmp = PGBP.integratebelief!(ctb)
        @test tmp1 ≈ [1.2633264344026676]
        @test tmp ≈ llscore
        for i in eachindex(ctb.belief)
            _, tmp = PGBP.integratebelief!(ctb, i)
            @test tmp ≈ llscore
        end
        @test -PGBP.free_energy(ctb)[3] ≈ llscore
        #= likelihood using PN.vcv and matrix inversion, different params
        σ2tmp = 1; μtmp = -2
        Σnet = σ2tmp .* Matrix(vcv(net)[!,Symbol.(df.taxon)])
        loglikelihood(MvNormal(repeat([μtmp],4), Σnet), tbl.y) # -8.091436736475565
        =#
        # clique tree beliefs to compare marginal estimates for i2, i4, H5, i6 with
        # those from cluster graph
        ct_H5i4i2_ind = PGBP.clusterindex(:H5i4i2, ctb)
        ct_H5i4i2_var = ctb.belief[ct_H5i4i2_ind].J \ I
        ct_H5i4i2_mean = PGBP.integratebelief!(ctb.belief[ct_H5i4i2_ind])[1]
        ct_i6H5_ind = PGBP.clusterindex(:i6H5, ctb)
        ct_i6H5_var = ctb.belief[ct_i6H5_ind].J \ I
        ct_i6H5_mean = PGBP.integratebelief!(ctb.belief[ct_i6H5_ind])[1]

        @testset "Bethe" begin
            cg = PGBP.clustergraph!(net, PGBP.Bethe())
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            cgb = PGBP.ClusterGraphBelief(b)
            PGBP.regularizebeliefs!(cgb, cg)
            sch = [] # schedule that covers all edges of cluster graph
            for n in net.nodes_changed
                ns = Symbol(n.name)
                sspt = PGBP.sub_spanningtree_clusterlist(cg, ns)
                isempty(sspt[1]) && continue
                push!(sch, sspt)
            end
            PGBP.calibrate!(cgb, sch, 2)
            cg_H5i4i2_ind = PGBP.clusterindex(:H5i4i2, cgb)
            cg_H5i4i2_var = cgb.belief[cg_H5i4i2_ind].J \ I
            cg_H5i4i2_mean = PGBP.integratebelief!(cgb.belief[cg_H5i4i2_ind])[1]
            cg_i6H5_ind = PGBP.clusterindex(:i6H5, cgb)
            cg_i6H5_var = cgb.belief[cg_i6H5_ind].J \ I
            cg_i6H5_mean = PGBP.integratebelief!(cgb.belief[cg_i6H5_ind])[1]
            @test ct_H5i4i2_var ≈ cg_H5i4i2_var rtol=1e-5
            @test ct_H5i4i2_mean ≈ cg_H5i4i2_mean rtol=1e-5
            @test ct_i6H5_var ≈ cg_i6H5_var rtol=1e-5
            @test ct_i6H5_mean ≈ cg_i6H5_mean rtol=1e-5
        end
        @testset "Join-graph" begin
            cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            cgb = PGBP.ClusterGraphBelief(b)
            # pa_lab, ch_lab, pa_j, ch_j =
            #     PGBP.minimal_valid_schedule(cg, [:Ai4, :B2i6, :B1i6, :Ci2])
            # for i in 1:length(pa_lab)
            #     ss_j = PGBP.sepsetindex(pa_lab[i], ch_lab[i], cgb)
            #     PGBP.propagate_belief!(b[ch_j[i]], b[ss_j], b[pa_j[i]])
            # end
            PGBP.regularizebeliefs!(cgb, cg)
            sch = [] # schedule that covers all edges of cluster graph
            for n in net.nodes_changed
                ns = Symbol(n.name)
                sspt = PGBP.sub_spanningtree_clusterlist(cg, ns)
                isempty(spt[1]) && continue
                push!(sch, sspt)
            end
            PGBP.calibrate!(cgb, sch, 2)
            cg_H5i4i2_ind = PGBP.clusterindex(:H5i4i2, cgb)
            cg_H5i4i2_var = cgb.belief[cg_H5i4i2_ind].J \ I
            cg_H5i4i2_mean = PGBP.integratebelief!(cgb.belief[cg_H5i4i2_ind])[1]
            cg_i6H5_ind = PGBP.clusterindex(:i6H5, cgb)
            cg_i6H5_var = cgb.belief[cg_i6H5_ind].J \ I
            cg_i6H5_mean = PGBP.integratebelief!(cgb.belief[cg_i6H5_ind])[1]
            @test ct_H5i4i2_var ≈ cg_H5i4i2_var rtol=1e-5
            @test ct_H5i4i2_mean ≈ cg_H5i4i2_mean rtol=1e-5
            @test ct_i6H5_var ≈ cg_i6H5_var rtol=1e-5
            @test ct_i6H5_mean ≈ cg_i6H5_mean rtol=1e-5
        end
    end
    @testset "hybrid ladder #1" begin
        @testset "Bethe" begin
            netstr = "((#H1:0.1::0.4,#H2:0.1::0.4)I1:1.0,(((A:1.0)#H1:0.1::0.6,#H3:0.1::0.4)#H2:0.1::0.6,(B:1.0)#H3:0.1::0.6)I2:1.0)I3;"
            net = readTopology(netstr)
            df = DataFrame(taxon=["A","B"], y=[1.0,-1])
            tbl_y = columntable(select(df, :y))
            m = PGBP.UnivariateBrownianMotion(2, 3, 0)
    
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            ctb = PGBP.ClusterGraphBelief(b)
            PGBP.calibrate!(ctb, [spt])
            ct_H1H2I1_ind = PGBP.clusterindex(:H1H2I1, ctb)
            ct_H1H2I1_var = ctb.belief[ct_H1H2I1_ind].J \ I
            ct_H1H2I1_mean = PGBP.integratebelief!(ctb.belief[ct_H1H2I1_ind])[1]
            # PGBP.integratebelief!(ctb.belief[ct_H1H2I1_ind])[2] # -5.450872671913978
            ct_H3H2I2_ind = PGBP.clusterindex(:H3H2I2, ctb)
            ct_H3H2I2_var = ctb.belief[ct_H3H2I2_ind].J \ I
            ct_H3H2I2_mean = PGBP.integratebelief!(ctb.belief[ct_H3H2I2_ind])[1]
            # PGBP.integratebelief!(ctb.belief[ct_H3H2I2_ind])[2] # -5.450872671913977
    
            cg = PGBP.clustergraph!(net, PGBP.Bethe())
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            cgb = PGBP.ClusterGraphBelief(b)
            PGBP.regularizebeliefs!(cgb, cg)
            schedule = PGBP.spanningtrees_cover_clusterlist(cg, net.nodes_changed)
            PGBP.calibrate!(cgb, schedule, 5)
            cg_H1H2I1_ind = PGBP.clusterindex(:H1H2I1, cgb)
            cg_H1H2I1_var = cgb.belief[cg_H1H2I1_ind].J \ I
            cg_H1H2I1_mean = PGBP.integratebelief!(cgb.belief[cg_H1H2I1_ind])[1]
            # PGBP.integratebelief!(cgb.belief[cg_H1H2I1_ind])[2] # 3145.652666576807
            cg_H3H2I2_ind = PGBP.clusterindex(:H3H2I2, cgb)
            cg_H3H2I2_var = cgb.belief[cg_H3H2I2_ind].J \ I
            cg_H3H2I2_mean = PGBP.integratebelief!(cgb.belief[cg_H3H2I2_ind])[1]
            # PGBP.integratebelief!(cgb.belief[cg_H3H2I2_ind])[2] # 3145.6526665768074
    
            @test -PGBP.free_energy(ctb)[3] ≈ PGBP.integratebelief!(ctb)[2]
            @test ct_H1H2I1_mean ≈ cg_H1H2I1_mean rtol=1e-3
            @test ct_H3H2I2_mean ≈ cg_H3H2I2_mean rtol=2*1e-3
            @test diag(ct_H1H2I1_var) ≈ diag(cg_H1H2I1_var) rtol=2*1e-1
            @test diag(ct_H3H2I2_var) ≈ diag(cg_H3H2I2_var) rtol=3*1e-1
            @test PGBP.iscalibrated(cgb, collect(edge_labels(cg)), 1e-2)
        end
    end
    @testset "hybrid ladder #2" begin
        @testset "Bethe" begin
            netstr = "((#H2:0.1::0.4,((A:1.0)#H1:0.1::0.6)#H3:0.1::0.5)I1:1.0,((#H1:0.1::0.4)#H2:0.1::0.6,#H3:0.1::0.5)I2:1.0)I3;"
            net = readTopology(netstr)
            df = DataFrame(taxon=["A"], y=[1.0])
            tbl_y = columntable(select(df, :y))
            m = PGBP.UnivariateBrownianMotion(2, 3, 0)
        
            ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
            spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            ctb = PGBP.ClusterGraphBelief(b)
            PGBP.calibrate!(ctb, [spt])
            ct_I1I2I3_ind = PGBP.clusterindex(:I1I2I3, ctb)
            ct_I1I2I3_var = ctb.belief[ct_I1I2I3_ind].J \ I
            ct_I1I2I3_mean = PGBP.integratebelief!(ctb.belief[ct_I1I2I3_ind])[1]
            # PGBP.integratebelief!(ctb.belief[ct_I1I2I3_ind])[2] # -2.1270084292518145
            ct_H1H2H3_ind = PGBP.clusterindex(:H1H2H3, ctb)
            ct_H1H2H3_var = ctb.belief[ct_H1H2H3_ind].J \ I
            ct_H1H2H3_mean = PGBP.integratebelief!(ctb.belief[ct_H1H2H3_ind])[1]
            # PGBP.integratebelief!(ctb.belief[ct_H1H2H3_ind])[2] # -2.1270084292518163
    
            cg = PGBP.clustergraph!(net, PGBP.Bethe())
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            cgb = PGBP.ClusterGraphBelief(b)
            PGBP.regularizebeliefs!(cgb, cg)
            schedule = PGBP.spanningtrees_cover_clusterlist(cg, net.nodes_changed)
            PGBP.calibrate!(cgb, schedule, 5)
            cg_H1H2H3_ind = PGBP.clusterindex(:H1H2H3, cgb)
            cg_H1H2H3_var = cgb.belief[cg_H1H2H3_ind].J \ I
            cg_H1H2H3_mean = PGBP.integratebelief!(cgb.belief[cg_H1H2H3_ind])[1]
            PGBP.integratebelief!(cgb.belief[cg_H1H2H3_ind])[2] # -50834.081555779856
            cg_H3I1I2_ind = PGBP.clusterindex(:H3I1I2, cgb)
            cg_H3I1I2_var = cgb.belief[cg_H3I1I2_ind].J \ I
            cg_H3I1I2_mean = PGBP.integratebelief!(cgb.belief[cg_H3I1I2_ind])[1]
            PGBP.integratebelief!(cgb.belief[cg_H3I1I2_ind])[2] # -50834.08155577985
    
            @test -PGBP.free_energy(ctb)[3] ≈ PGBP.integratebelief!(ctb)[2]
            @test ct_I1I2I3_mean ≈ cg_H3I1I2_mean[2:3] rtol=1e-4
            @test ct_H1H2H3_mean ≈ cg_H1H2H3_mean rtol=1e-4
            @test diag(ct_I1I2I3_var) ≈ diag(cg_H3I1I2_var)[2:3] rtol=1e-1
            @test diag(ct_I1I2I3_var) ≈ diag(cg_H3I1I2_var)[2:3] rtol=1e-1
        end
    end
end
@testset "with optimization" begin
    @testset "4-taxon, level-1: #1" begin
        netstr = "(A:2.5,((B:1,#H1:0.5::0.1):1,(C:1,(D:0.5)#H1:0.5::0.9):1):0.5);"
        net = readTopology(netstr)
        df = DataFrame(y=[11.275034507978296, 10.032494469945764,
            11.49586603350308, 11.004447427824012], taxon=["A","B","C", "D"])
        tbl_y = columntable(select(df, :y))
        @testset "Bethe" begin
            cg = PGBP.clustergraph!(net, PGBP.Bethe())
            b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg,
                PGBP.UnivariateBrownianMotion(1, 0, 0));
            cgb = PGBP.ClusterGraphBelief(b)
            mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
                net.nodes_changed, tbl_y, df.taxon,
                PGBP.UnivariateBrownianMotion, (1,0))
            # Compare with RxInfer + Optim
            @test fenergy ≈ 3.4312133894974126 rtol=1e-4
            @test mod.μ ≈ 10.931640613828181 rtol=1e-4
            @test mod.σ2 ≈ 0.15239159696122745 rtol=1e-4
        end
    end
    @testset "4-taxon, level-1: #2" begin
        netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"
        net = readTopology(netstr)
        df = DataFrame(taxon=["A","B1","B2","C"], x=[10,10,missing,0], y=[1.0,.9,1,-1])
        df_var = select(df, Not(:taxon))
        tbl = columntable(df_var)
        tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models
        tbl_x = columntable(select(df, :x))
        m = PGBP.UnivariateBrownianMotion(2, 3, 0)
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)

        # y: 1 trait, no missing values
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed,
            tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1,-2))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ llscore
        @test llscore ≈ -5.174720533524127
        @test mod.μ ≈ -0.26000871507162693
        @test PGBP.varianceparam(mod) ≈ 0.35360518758586457
        PGBP.init_beliefs_reset!(ctb)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(ctb, ct,
            net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1,-2))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ -fenergy
        @test -fenergy ≈ -5.174720533524127
        @test mod.μ ≈ -0.26000871507162693
        @test PGBP.varianceparam(mod) ≈ 0.35360518758586457

        lbc = GeneralLazyBufferCache(function (paramOriginal)
            mo = PGBP.UnivariateBrownianMotion(paramOriginal...)
            bel = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, mo)
            return PGBP.ClusterGraphBelief(bel)
        end)
        mod2, llscore2, opt2 = PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct, net.nodes_changed,
            tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1, -2))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ llscore2
        @test llscore2 ≈ -5.174720533524127
        @test mod2.μ ≈ -0.26000871507162693
        @test PGBP.varianceparam(mod2) ≈ 0.35360518758586457

        #=
        using BenchmarkTools
        @benchmark PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct, net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1, -2))
        @benchmark PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed, tbl_y, df.taxon, PGBP.UnivariateBrownianMotion, (1, -2))
        =#

        #= ML solution the matrix-way, analytical for BM:
        # for y: univariate
        Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])
        n=4 # number of data points
        i = ones(n) # intercept
        μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * tbl.y) # -0.26000871507162693
        r = tbl.y .- μhat
        σ2hat_ML = (transpose(r) * inv(Σ) * r) / n # 0.35360518758586457
        llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 # -5.174720533524127

        # for x: third value is missing
        xind = [1,2,4]; n = length(xind); i = ones(n) # intercept
        Σ = Matrix(vcv(net)[!,Symbol.(df.taxon)])[xind,xind]
        μhat = inv(transpose(i) * inv(Σ) * i) * (transpose(i) * inv(Σ) * tbl.x[xind]) # 3.500266520382341
        r = tbl.x[xind] .- μhat
        σ2hat_ML = (transpose(r) * inv(Σ) * r) / n # 11.257682945973125
        llscore = - n/2 - logdet(2π * σ2hat_ML .* Σ)/2 # -9.215574122592923
        =#

        # x: 1 trait, some missing values
        m = PGBP.UnivariateBrownianMotion(2, 3, 0)
        b = (@test_logs (:error,"tip B2 in network without any data") PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, m))
        PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed,
            tbl_x, df.taxon, PGBP.UnivariateBrownianMotion, (1,-2))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ llscore
        @test llscore ≈ -9.215574122592923
        @test mod.μ ≈ 3.500266520382341
        @test PGBP.varianceparam(mod) ≈ 11.257682945973125
        PGBP.init_beliefs_reset!(ctb)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(ctb, ct,
            net.nodes_changed, tbl_x, df.taxon, PGBP.UnivariateBrownianMotion, (1,-2))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ -fenergy
        @test -fenergy ≈ -9.215574122592923
        @test mod.μ ≈ 3.500266520382341
        @test PGBP.varianceparam(mod) ≈ 11.257682945973125

        lbc = GeneralLazyBufferCache(function (paramOriginal)
            mo = PGBP.UnivariateBrownianMotion(paramOriginal...)
            bel = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, mo)
            return PGBP.ClusterGraphBelief(bel)
        end)
        mod2, llscore2, opt2 = (@test_logs (:error, "tip B2 in network without any data") PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct, net.nodes_changed,
            tbl_x, df.taxon, PGBP.UnivariateBrownianMotion, (1, -2)))
        @test llscore2 ≈ -9.215574122592923
        @test mod2.μ ≈ 3.500266520382341
        @test PGBP.varianceparam(mod2) ≈ 11.257682945973125

        # x,y: 2 traits, some missing values
        m = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0,0))
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed,
            tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2,1), (1,-1)))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ llscore
        @test llscore ≈ -14.39029465611705 # -5.174720533524127 -9.215574122592923
        @test mod.μ ≈ [3.500266520382341, -0.26000871507162693]
        @test PGBP.varianceparam(mod) ≈ [11.257682945973125,0.35360518758586457]
        PGBP.init_beliefs_reset!(ctb)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(ctb, ct,
            net.nodes_changed, tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2,1), (1,-1)))
        @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ -fenergy
        @test -fenergy ≈ -14.39029465611705
        @test mod.μ ≈ [3.500266520382341, -0.26000871507162693]
        @test PGBP.varianceparam(mod) ≈ [11.257682945973125, 0.35360518758586457]

        lbc = GeneralLazyBufferCache(function (paramOriginal)
            mo = PGBP.MvDiagBrownianMotion(paramOriginal...)
            bel = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, mo)
            return PGBP.ClusterGraphBelief(bel)
        end)
        mod2, llscore2, opt2 = PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct, net.nodes_changed,
            tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2, 1), (1, -1)))
        @test llscore2 ≈ -14.39029465611705 # -5.174720533524127 -9.215574122592923
        @test mod2.μ ≈ [3.500266520382341, -0.26000871507162693]
        @test PGBP.varianceparam(mod2) ≈ [11.257682945973125, 0.35360518758586457]

        #=
        using BenchmarkTools
        @benchmark PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct, net.nodes_changed, tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2, 1), (1, -1)))
        @benchmark PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed, tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2, 1), (1, -1)))
        =#
    end
end
end