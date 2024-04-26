@testset "calibration" begin

@testset "miscellaneous" begin
    ex = PGBP.BPPosDefException("belief 1, integrate 3,4", 1)
    io = IOBuffer()
    showerror(io, ex)
    @test String(take!(io)) == """BPPosDefException: belief 1, integrate 3,4
    matrix is not positive definite."""
end
@testset "residual_kldiv!" begin
    """
    ref distr (new belief): μ=[0, 1], Σ=[1 0; 0 1], J=[1 0; 0 1], h=[0, 1]
    approx distr (prev belief): μ=[1, 0], Σ=[2 1; 1 2], J=(1/3)[2 -1; -1 2],
        h=(1/3)[2, -1]
    resid distr (ref-approx): ΔJ=(1/3)[1 1; 1 1], Δh=(1/3)[-2, 4]
    ---
    Using R to compute KL divergence:
    > library(rags2ridges)
    > KLdiv(c(1,0),c(0,1),matrix(c(2,1,1,2),nrow=2),matrix(c(1,0,0,1),nrow=2))
    [1] 1.215973
    """
    # allocate memory for ::MessageResidual object (resid distr)
    res = PGBP.MessageResidual(zeros(2,2), zeros(2,))
    res.ΔJ .= ones(2,2)/3
    res.Δh .= [-2,4]/3
    # allocate memory for ::Belief object (ref distr)
    sepset = PGBP.Belief([1, 2], 1, BitArray([1 1]), PGBP.bsepsettype, (:A,:B))
    sepset.J .= [1 0; 0 1]
    sepset.h .= [0, 1]
    PGBP.residual_kldiv!(res, sepset)
    @test res.kldiv[1] ≈ 1.215973 rtol=1e-6
end
@testset "no optimization" begin
    @testset "Level-1 w/ 4 tips. Univariate. Clique tree" begin
        netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"
        net = readTopology(netstr)
        df = DataFrame(taxon=["A","B1","B2","C"], y=[1.0,.9,1,-1])
        tbl_y = columntable(select(df, :y))
        #= fitBM = phylolm(@formula(y ~ 1), df, net; tipnames=:taxon)
        sigma2_phylo(fitBM) # reml variance-rate: 0.4714735834478196
        # reconstructed root mean and variance
        coef(fitBM)[1] # reconstructed root mean: -0.26000871507162693
        ancestralStateReconstruction(fitBM).variances_nodes[5,5] # 0.33501871740664146
        loglikelihood(fitBM) # restricted likelihood: -4.877930583154144
        Assume reml variance-rate (0.471474) and compute posterior mean, posterior
        variance and likelihood for comparison. =#
        m = PGBP.UnivariateBrownianMotion(0.471474, 0, Inf) # 𝒩(0, ∞) prior on root mean
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
        spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        llscore = -4.877930583154144
        for i in eachindex(ctb.belief)
            _, tmp = PGBP.integratebelief!(ctb, i) # llscore from norm. constant
            @test tmp ≈ llscore
        end
        @test PGBP.factored_energy(ctb)[3] ≈ llscore
        root_ind = findfirst(be -> 1 ∈ PGBP.nodelabels(be), b) # 5
        @test PGBP.integratebelief!(b[root_ind])[1][end] ≈
            -0.26000871507162693 rtol=1e-5 # posterior root mean
        @test (b[root_ind].J \ I)[end,end] ≈
            0.33501871740664146 rtol=1e-5 # posterior root variance
        @testset "regularization by cluster" begin
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            PGBP.regularizebeliefs_bynodesubtree!(ctb, ct);
            PGBP.calibrate!(ctb, [spt]);
            _, tmp = PGBP.integratebelief!(ctb, 1)
            @test tmp ≈ llscore
            PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
            PGBP.regularizebeliefs_bycluster!(ctb, ct);
            PGBP.calibrate!(ctb, [spt]);
            _, tmp = PGBP.integratebelief!(ctb, 1)
            @test tmp ≈ llscore # graph invariant was preserved
        end
    end
    @testset "Level-1 w/ 4 tips. Univariate. Bethe, regularize on a schedule" begin
        netstr = "(A:2.5,((B:1,#H1:0.5::0.1):1,(C:1,(D:0.5)#H1:0.5::0.9):1):0.5);"
        net = readTopology(netstr)
        # tip data simulated from ParamsBM(0,1)
        df = DataFrame(y=[-1.81358, 0.468158, 0.658486, 0.643821],
                taxon=["A","B","C", "D"])
        tbl_y = columntable(select(df, :y))
        #= Reroot network at I3 for ancestral reconstruction.
        net0 = rootatnode!(deepcopy(net), "I3")
        PhyloNetworks.setBranchLength!(net0.edge[1], 0.5) # A now represents I4
        df = DataFrame(y=[0.0, 0.468158, 0.658486, 0.643821],
            taxon=["A","B","C", "D"])
        fitBM = phylolm(@formula(y ~ 1), df, net0; tipnames=:taxon)
        sigma2_phylo(fitBM) # reml variance-rate: 0.08612487128235946
        coef(fitBM)[1] # reconstructed root mean: 0.21511454631828986
        Compare with posterior mean for I3. =#
        m = PGBP.UnivariateBrownianMotion(0.0861249, 0) # 𝒩(0, 0) prior on root mean
        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        cgb = PGBP.ClusterGraphBelief(b)
        PGBP.regularizebeliefs_onschedule!(cgb, cg)
        sched = PGBP.spanningtrees_clusterlist(cg, net.nodes_changed)
        @test PGBP.calibrate!(cgb, sched, 20; auto=true)
        # [ Info: Calibration detected: iter 5, sch 1
        ind = PGBP.clusterindex(:I3, cgb)
        @test PGBP.integratebelief!(b[ind])[1][end] ≈
            0.21511454631828986 rtol=1e-5 # posterior root mean
    end
    @testset "level-3, 2 tips, 2 traits, 1 missing unscoped in 2 nodes. Join-graph, regularize by node subtree" begin
        netstr = "((#H1:0.1::0.4,#H2:0.1::0.4)I1:1.0,(((A:1.0)#H1:0.1::0.6,#H3:0.1::0.4)#H2:0.1::0.6,(B:1.0)#H3:0.1::0.6)I2:1.0)I3;"
        net = readTopology(netstr)
        # y1 simulated from ParamsBM(2,0.1)
        df = DataFrame(taxon=["A","B"], y1=[2.11,2.15], y2=[30.0,missing])
        tbl_y = columntable(select(df, :y1, :y2))
        m = PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [0,0], [Inf 0; 0 Inf]) # improper root
        cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        cgb = PGBP.ClusterGraphBelief(b)
        PGBP.regularizebeliefs_bynodesubtree!(cgb, cg)
        sch = [] # schedule based on 1 subtree per variable
        for n in net.nodes_changed
            subtree = PGBP.nodesubtree_clusterlist(cg, Symbol(n.name))
            isempty(subtree[1]) && continue
            push!(sch, subtree)
        end
        @test (@test_logs (:info, "calibration reached: iteration 4, schedule tree 1") PGBP.calibrate!(cgb, sch, 10; auto=true, info=true))
        #= Compare posterior means against clique tree estimates:
        ct = PGBP.clustergraph!(net, PGBP.Cliquetree());
        b_ct = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b_ct, m, tbl_y, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b_ct);
        PGBP.calibrate!(ctb, [PGBP.spanningtree_clusterlist(ct, net.nodes_changed)]);
        PGBP.integratebelief!(b_ct[6]) # cluster I1I2I3: PGBP.clusterindex(:I1I2I3, ctb)
        # ([2.121105154896223, 30.005552577448075, 2.1360649504455984, 30.013032475222563, 2.128585052670908, 30.00929252633532], -1.39059577242449)
        PGBP.integratebelief!(b_ct[2]) # cluster H1H2I1
        # ([2.125583120364, 30.007791560181964, 2.129918967774073, 30.009959483886966, 2.121105154896214, 30.00555257744811], -1.390595772423012)
        =#
        tmp = PGBP.integratebelief!(b[6]) # 6: PGBP.clusterindex(:I1I2I3, cgb)
        @test tmp[2] ≈ -1.390595772423
        @test all(tmp[1] .≈ [2.121105154896223, 30.005552577448075, 2.1360649504455984, 30.013032475222563, 2.128585052670943, 30.00929252633547])
        tmp = PGBP.integratebelief!(b[2]) # cluster 2: H1H2I1
        @test tmp[2] ≈ -1.390595772423
        @test all(tmp[1] .≈ [2.125583120364, 30.007791560181964, 2.129918967774073, 30.009959483886966, 2.121105154896214, 30.00555257744811])
        #= likelihood using PN.vcv and matrix inversion
        Σnet = kron(Matrix(vcv(net)[!,[:A,:B]]), [1 0.5; 0.5 1])
        Σnet_y2Bmissing = Σnet[1:3,1:3]; invΣ = inv(Σnet_y2Bmissing)
        yvec = [df[1,:y1], df[1,:y2], df[2,:y1]]
        X = [1 0; 0 1; 1 0]; Xt = transpose(X)
        μ = X * inv(Xt * invΣ * X) * Xt * invΣ * yvec
        # [2.128585052670943, 30.00929252633547, 2.128585052670943] # great: same as posterior at I3 = root
        loglikelihood(MvNormal(μ, Σnet_y2Bmissing), yvec) # -3.3498677834866997 but not same model: fixed root here
        =#
        m = PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [2.128585052670943,30.00929252633547]) # fixed root
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        cgb = PGBP.ClusterGraphBelief(b)
        PGBP.regularizebeliefs_bynodesubtree!(cgb, cg)
        @test (@test_logs PGBP.calibrate!(cgb, sch, 10; auto=true, info=false))
        tmp = PGBP.integratebelief!(b[6]) # I3 un-scoped bc fixed root, same posterior mean for I1 and I2
        @test tmp[2] ≈ -3.3498677834866997
        @test all(tmp[1] .≈ [2.121105154896223, 30.005552577448075, 2.1360649504455984, 30.013032475222563])
    end
end
@testset "with optimization" begin
    @testset "Level-1 w/ 4 tips. Univariate. Bethe + Optim." begin
        netstr = "(A:2.5,((B:1,#H1:0.5::0.1):1,(C:1,(D:0.5)#H1:0.5::0.9):1):0.5);"
        net = readTopology(netstr)
        df = DataFrame(y=[11.275034507978296, 10.032494469945764,
            11.49586603350308, 11.004447427824012], taxon=["A","B","C", "D"])
        tbl_y = columntable(select(df, :y))
        cg = PGBP.clustergraph!(net, PGBP.Bethe())
        m = PGBP.UnivariateBrownianMotion(1, 0)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, cg, m);
        cgb = PGBP.ClusterGraphBelief(b)
        mod, fenergy, opt = PGBP.calibrate_optimize_clustergraph!(cgb, cg,
            net.nodes_changed, tbl_y, df.taxon,
            PGBP.UnivariateBrownianMotion, (1,0))
        # Compare with RxInfer + Optim
        @test fenergy ≈ 3.4312133894974126 rtol=1e-4
        @test mod.μ ≈ 10.931640613828181 rtol=1e-4
        @test mod.σ2 ≈ 0.15239159696122745 rtol=1e-4
    end
    @testset "Level-1 w/ 4 tips. Univariate. Clique tree + Optim / Autodiff" begin
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
        
        # lbc = GeneralLazyBufferCache(function (paramOriginal)
        #     mo = PGBP.UnivariateBrownianMotion(paramOriginal...)
        #     bel = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, mo)
        #     return PGBP.ClusterGraphBelief(bel)
        # end)
        # mod2, llscore2, opt2 = (@test_logs (:error, "tip B2 in network without any data") PGBP.calibrate_optimize_cliquetree_autodiff!(lbc, ct, net.nodes_changed,
        #     tbl_x, df.taxon, PGBP.UnivariateBrownianMotion, (1, -2)))
        # @test llscore2 ≈ -9.215574122592923
        # @test mod2.μ ≈ 3.500266520382341
        # @test PGBP.varianceparam(mod2) ≈ 11.257682945973125

        # x,y: 2 traits, some missing values
        m = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0,0))
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        # mod, llscore, opt = PGBP.calibrate_optimize_cliquetree!(ctb, ct, net.nodes_changed,
        #     tbl, df.taxon, PGBP.MvDiagBrownianMotion, ((2,1), (1,-1)))
        # @test PGBP.integratebelief!(ctb, spt[3][1])[2] ≈ llscore
        # @test llscore ≈ -14.39029465611705 # -5.174720533524127 -9.215574122592923
        # @test mod.μ ≈ [3.500266520382341, -0.26000871507162693]
        # @test PGBP.varianceparam(mod) ≈ [11.257682945973125,0.35360518758586457]

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