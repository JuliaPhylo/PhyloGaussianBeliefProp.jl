@testset "evolutionary models parameters" begin
m = PGBP.MvDiagBrownianMotion([1,0.5], [-1,1]) # default 0 root variance
m = PGBP.MvDiagBrownianMotion([1,0.5], [-1,1], [0,1])
par = PGBP.params_optimize(m)
oripar = PGBP.params_original(m, par)
@test oripar == PGBP.params(m)
@test PGBP.dimension(m) == 2
m = PGBP.MvFullBrownianMotion([1.0, .5, 0.8660254037844386], [-1,1]) # default 0 root variance
@test PGBP.varianceparam(m) ≈ [1 0.5; 0.5 1]
@test PGBP.rootpriorvariance(m) == [0 0; 0 0]
m = PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [-1,1], [10^10 0; 0 10^10])
@test PGBP.dimension(m) == 2
m = PGBP.UnivariateBrownianMotion(2, 3)
par = PGBP.params_optimize(m)
oripar = PGBP.params_original(m, par)
@test oripar[1] ≈ PGBP.params(m)[1]
@test oripar[2] ≈ PGBP.params(m)[2]
h,J,g = PGBP.factor_treeedge(m, 1)
@test h == [0.0,0]
@test J == [.5 -.5; -.5 .5]
@test g ≈ -1.2655121234846454
m2 = PGBP.UnivariateBrownianMotion(2, 3, 0)
@test m == m2
m2 = PGBP.UnivariateBrownianMotion(2.0, 3, 0)
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], [3])
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], [3.0])
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], 3)
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], 3.0)
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], 3, 0)
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2.0], 3, 0)
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], 3, [0])
@test m == m2
m2 = PGBP.UnivariateBrownianMotion([2], 3.0, [0])
@test m == m2
@test_throws "scalars" PGBP.UnivariateBrownianMotion([2,2], [3], [0])
@test_throws "scalars" PGBP.UnivariateBrownianMotion([2,2], 3, 0)
@test_throws "scalars" PGBP.UnivariateBrownianMotion([2], [3,3])
@test_throws "scalars" PGBP.UnivariateBrownianMotion([2], 3, [0,0])
end

@testset "evolutionary models likelihood" begin
    netstr = "(((A:4.0,((B1:1.0,B2:1.0)i6:0.6)#H5:1.1::0.9)i4:0.5,(#H5:2.0::0.1,C:0.1)i2:1.0)i1:3.0);"
    net = readTopology(netstr)
    df = DataFrame(taxon=["A","B1","B2","C"], x=[10,10,missing,0], y=[1.0,.9,1,-1])
    df_var = select(df, Not(:taxon))
    tbl = columntable(df_var)
    tbl_y = columntable(select(df, :y)) # 1 trait, for univariate models
    tbl_x = columntable(select(df, :x))

    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    spt = PGBP.spanningtree_clusterlist(ct, net.nodes_changed)
    rootclusterindex = spt[3][1]
    # allocate beliefs to avoid re-allocation of same sizes for multiple tests
    m_uniBM_fixedroot = PGBP.UnivariateBrownianMotion(2, 3, 0)
    b_y_fixedroot = PGBP.allocatebeliefs(tbl_y, df.taxon, net.nodes_changed, ct, m_uniBM_fixedroot)
    m_uniBM_randroot = PGBP.UnivariateBrownianMotion(2, 3, Inf)
    b_y_randroot = PGBP.allocatebeliefs(tbl_y, df.taxon, net.nodes_changed, ct, m_uniBM_randroot)
    m_biBM_fixedroot = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0,0))
    b_xy_fixedroot = PGBP.allocatebeliefs(tbl, df.taxon, net.nodes_changed, ct, m_biBM_fixedroot)
    m_biBM_randroot = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0.1,10))
    b_xy_randroot = PGBP.allocatebeliefs(tbl, df.taxon, net.nodes_changed, ct, m_biBM_randroot)

    @testset "homogeneous univariate BM" begin
        @testset "Fixed Root, no missing" begin
        # y no missing, fixed root
        show(devnull, m_uniBM_fixedroot)
        PGBP.assignfactors!(b_y_fixedroot[1], m_uniBM_fixedroot, tbl_y, df.taxon,
            net.nodes_changed, b_y_fixedroot[2][1], b_y_fixedroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_y_fixedroot[1], b_y_fixedroot[2][1],
            b_y_fixedroot[2][2], b_y_fixedroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -10.732857817537196
        end
        @testset "Infinite Root, no missing" begin
        # y no missing, infinite root variance
        PGBP.assignfactors!(b_y_randroot[1], m_uniBM_randroot, tbl_y, df.taxon,
            net.nodes_changed, b_y_randroot[2][1], b_y_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_y_randroot[1], b_y_randroot[2][1],
            b_y_randroot[2][2], b_y_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -5.899094849099194
        end
        @testset "Random Root, with missing" begin
        # x with missing, random root
        m = PGBP.UnivariateBrownianMotion(2, 3, 0.4)
        b = (@test_logs (:error,"tip B2 in network without any data") PGBP.allocatebeliefs(tbl_x, df.taxon, net.nodes_changed, ct, m);)
        PGBP.assignfactors!(b[1], m, tbl_x, df.taxon, net.nodes_changed, b[2][1],
            b[2][2]);
        ctb = PGBP.ClusterGraphBelief(b[1], b[2][1], b[2][2], b[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -13.75408386332493
        end
    end
    @testset "homogeneous univariate OU" begin
        @testset "Random Root, no missing" begin
        m = PGBP.UnivariateOrnsteinUhlenbeck(2, 3, -2, 0.0, 0.4)
        show(devnull, m)
        PGBP.assignfactors!(b_y_randroot[1], m, tbl_y, df.taxon, net.nodes_changed,
            b_y_randroot[2][1], b_y_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_y_randroot[1], b_y_randroot[2][1],
            b_y_randroot[2][2], b_y_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -42.31401134496844
        #= code to compute the univariate OU likelihood by hand
        # 1. calculate vcv of all nodes in preorder
        V(t::Number) = (1-exp(-2m.α * t)) * m.γ2 # variance conditional on parent, one edge
        q(t::Number) = exp(-m.α * t) # actualization along tree edge: weight of parent value
        q(edge) = q(edge.length) * edge.gamma
        V(parentedges) = sum(V(e.length) * e.gamma * e.gamma for e in parentedges)
        net_vcv = zeros(9,9) # all 9 nodes
        net_vcv[1,1] = PGBP.rootpriorvariance(m)
        net_mean = zeros(9,1) # all 9 nodes expectations
        net_mean[1] = m.μ
        for i in 2:9 # non-root nodes
            n = net.nodes_changed[i]
            pae = [PhyloNetworks.getparentedge(n)]
            if n.hybrid push!(pae, PhyloNetworks.getparentedgeminor(n)); end
            nparents = length(pae)
            pa = [PhyloNetworks.getparent(e) for e in pae]
            pai = indexin(pa, net.nodes_changed)
            # var(Xi)
            net_vcv[i,i] = V(pae) # initialize
            for (j1,e1) in zip(pai, pae) for (j2,e2) in zip(pai, pae)
                net_vcv[i,i] += q(e1) * q(e2) * net_vcv[j1,j2]
            end; end
            # cov(Xi,Xj) for j<i in preorder
            for j in 1:(i-1)
                for (j1,e1) in zip(pai, pae)
                    net_vcv[i,j] += q(e1) * net_vcv[j1,j]
                end
                net_vcv[j,i] = net_vcv[i,j]
            end
            # E[Xi]
            for (j1,e1) in zip(pai, pae)
                net_mean[i] += q(e1) * net_mean[j1] + (e1.gamma - q(e1)) * m.θ
            end
        end
        net_vcv
        [n.name for n in net.nodes_changed] # i1,i2,C,i4,H5,i6,B2,B1,A
        df.taxon # A,B1,B2,C with preorder indices: 9,8,7,3
        taxon_ind = [findfirst(isequal(tax), n.name for n in net.nodes_changed) for tax in df.taxon]
        print(net_vcv[taxon_ind,taxon_ind]) # copy-pasted below
        Σnet = [
        0.3333333333334586 5.651295717406613e-10 5.651295717406613e-10 2.0226125393342098e-8;
        5.651295717406613e-10 0.33332926986180506 0.000822187254027199 1.4032919760109094e-6;
        5.651295717406613e-10 0.000822187254027199 0.33332926986180506 1.4032919760109094e-6;
        2.0226125393342098e-8 1.4032919760109094e-6 1.4032919760109094e-6 0.3334240245358365]
        print(net_mean[taxon_ind]) # copy-pasted below
        Mnet = [-1.9999972580818273, -1.9998778851480223, -1.9998778851480223, -1.92623366519752]
        loglikelihood(MvNormal(Mnet, Σnet), tbl.y) # -42.31401134496844
        =#
        end
    end
    @testset "Diagonal BM" begin
        @testset "homogeneous, fixed root" begin
        PGBP.assignfactors!(b_xy_fixedroot[1], m_biBM_fixedroot, tbl, df.taxon,
            net.nodes_changed, b_xy_fixedroot[2][1], b_xy_fixedroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_fixedroot[1], b_xy_fixedroot[2][1],
            b_xy_fixedroot[2][2], b_xy_fixedroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -24.8958130127972
        end
        @testset "homogeneous, random root" begin
        PGBP.assignfactors!(b_xy_randroot[1], m_biBM_randroot, tbl, df.taxon,
            net.nodes_changed, b_xy_randroot[2][1], b_xy_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_randroot[1], b_xy_randroot[2][1],
            b_xy_randroot[2][2], b_xy_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -21.347496753649892
        end
        @testset "homogeneous, improper root" begin
        m = PGBP.MvDiagBrownianMotion((2,1), (1,-3), (Inf,Inf))
        PGBP.assignfactors!(b_xy_randroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_randroot[2][1], b_xy_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_randroot[1], b_xy_randroot[2][1],
            b_xy_randroot[2][2], b_xy_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -17.66791635814575
        end
    end
    @testset "Full BM" begin
        @testset "homogeneous, fixed root" begin
        m = PGBP.MvFullBrownianMotion([2.0 0.5; 0.5 1.0], [3.0,-3.0])
        PGBP.assignfactors!(b_xy_fixedroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_fixedroot[2][1], b_xy_fixedroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_fixedroot[1], b_xy_fixedroot[2][1],
            b_xy_fixedroot[2][2], b_xy_fixedroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -24.312323855394055
        end
        @testset "homogeneous, random root" begin
        m = PGBP.MvFullBrownianMotion([2.0 0.5; 0.5 1.0], [3.0,-3.0],
                [0.1 0.01; 0.01 0.2])
        PGBP.assignfactors!(b_xy_randroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_randroot[2][1], b_xy_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_randroot[1], b_xy_randroot[2][1],
            b_xy_randroot[2][2], b_xy_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -23.16482738327936
        end
        @testset "homogeneous, improper root" begin
        m = PGBP.MvFullBrownianMotion([2.0 0.5; 0.5 1.0], [3.0,-3.0],
                [Inf 0; 0 Inf])
        PGBP.assignfactors!(b_xy_randroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_randroot[2][1], b_xy_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_randroot[1], b_xy_randroot[2][1],
            b_xy_randroot[2][2], b_xy_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -16.9626044836951
        end
    end
    @testset "heterogeneous BM" begin
        @testset "Fixed Root one mv rate" begin
        m = PGBP.HeterogeneousBrownianMotion([2.0 0.5; 0.5 1.0], [3.0, -3.0])
        show(devnull, m)
        PGBP.assignfactors!(b_xy_fixedroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_fixedroot[2][1], b_xy_fixedroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_fixedroot[1], b_xy_fixedroot[2][1],
            b_xy_fixedroot[2][2], b_xy_fixedroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -24.312323855394055
        end
        @testset "Random root several mv rates" begin
        rates = [[2.0 0.5; 0.5 1.0], [2.0 0.5; 0.5 1.0]]
        colors = Dict(9 => 2, 7 => 2, 8 => 2) # includes one hybrid edge
        pp = PGBP.PaintedParameter(rates, colors)
        show(devnull, pp)
        m = PGBP.HeterogeneousBrownianMotion(pp, [3.0, -3.0], [0.1 0.01; 0.01 0.2])
        show(devnull, m)
        PGBP.assignfactors!(b_xy_randroot[1], m, tbl, df.taxon, net.nodes_changed,
            b_xy_randroot[2][1], b_xy_randroot[2][2]);
        ctb = PGBP.ClusterGraphBelief(b_xy_randroot[1], b_xy_randroot[2][1],
            b_xy_randroot[2][2], b_xy_randroot[2][4])
        PGBP.propagate_1traversal_postorder!(ctb, spt...)
        _, tmp = PGBP.integratebelief!(ctb, rootclusterindex)
        @test tmp ≈ -23.16482738327936
        end
    end
    #= likelihood using PN.vcv and matrix inversion
    using Distributions
    Σnet = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    ## univariate y
    loglikelihood(MvNormal(repeat([3.0],4), Σnet), tbl.y) # -10.732857817537196
    ## univariate x
    xind = [1,2,4]; n = length(xind); i = ones(n) # intercept
    Σ = 2.0 * Σnet[xind,xind] .+ 0.4
    loglikelihood(MvNormal(repeat([3.0],3), Σ), Vector{Float64}(tbl.x[xind])) # -13.75408386332493
    ## Univariate y, REML
    ll(μ) = loglikelihood(MvNormal(repeat([μ],4), Σnet), tbl.y)
    using Integrals
    log(solve(IntegralProblem((x,p) -> exp(ll(x)), -Inf, Inf), QuadGKJL()).u) # -5.899094849099194
    ## Diagonal x, y
    loglikelihood(MvNormal(repeat([3],3), 2 .* Σnet[xind,xind]), Vector{Float64}(tbl.x[xind])) + 
    loglikelihood(MvNormal(repeat([-3],4), 1 .* Σnet), tbl.y) 
    ## Diagonal x y random root
    loglikelihood(MvNormal(repeat([3],3), 2 .* Σnet[xind,xind] .+ 0.1), Vector{Float64}(tbl.x[xind])) + 
    loglikelihood(MvNormal(repeat([-3],4), 1 .* Σnet .+ 10), tbl.y) 
    # Diagonal x y REML
    ll(μ) = loglikelihood(MvNormal(repeat([μ[1]],3), 2 .* Σnet[xind,xind]), Vector{Float64}(tbl.x[xind])) + loglikelihood(MvNormal(repeat([μ[2]],4), 1 .* Σnet), tbl.y) 
    using Integrals
    log(solve(IntegralProblem((x,p) -> exp(ll(x)), [-Inf, -Inf], [Inf, Inf]), HCubatureJL(), reltol = 1e-16, abstol = 1e-16).u) # -17.66791635814575
    # Full x y fixed root
    R = [2.0 0.5; 0.5 1.0]
    varxy = kron(R, Σnet)
    xyind = vcat(xind, 4 .+ [1,2,3,4])
    varxy = varxy[xyind, xyind]
    meanxy = vcat(repeat([3.0],3), repeat([-3.0],4))
    datxy = Vector{Float64}(vcat(tbl.x[xind], tbl.y))
    loglikelihood(MvNormal(meanxy, varxy), datxy) # -24.312323855394055
    # Full x y random root
    R = [2.0 0.5; 0.5 1.0]
    V = [0.1 0.01; 0.01 0.2]
    varxy = kron(R, Σnet) + kron(V, ones(4, 4))
    xyind = vcat(xind, 4 .+ [1,2,3,4])
    varxy = varxy[xyind, xyind]
    meanxy = vcat(repeat([3.0],3), repeat([-3.0],4))
    datxy = Vector{Float64}(vcat(tbl.x[xind], tbl.y))
    loglikelihood(MvNormal(meanxy, varxy), datxy) # -23.16482738327936
    # Full x y improper root
    R = [2.0 0.5; 0.5 1.0]
    varxy = kron(R, Σnet)
    xyind = vcat(xind, 4 .+ [1,2,3,4])
    varxy = varxy[xyind, xyind]
    meanxy = vcat(repeat([3.0],3), repeat([-3.0],4))
    datxy = Vector{Float64}(vcat(tbl.x[xind], tbl.y))
    ll(μ) = loglikelihood(MvNormal(vcat(repeat([μ[1]],3),repeat([μ[2]],4)), varxy), datxy)
    using Integrals
    log(solve(IntegralProblem((x,p) -> exp(ll(x)), [-Inf, -Inf], [Inf, Inf]),
        HCubatureJL(), reltol = 1e-16, abstol = 1e-16).u) # -16.9626044836951
    =#
end
