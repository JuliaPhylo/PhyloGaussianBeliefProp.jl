@testset "evolutionary models parameters" begin
m = PGBP.MvDiagBrownianMotion([1,0.5], [-1,1]) # default 0 root variance
m = PGBP.MvDiagBrownianMotion([1,0.5], [-1,1], [0,1])
par = PGBP.params_optimize(m)
oripar = PGBP.params_original(m, par)
@test oripar == PGBP.params(m)
@test PGBP.dimension(m) == 2
m = PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [-1,1]) # default 0 root variance
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

    @testset "homogeneous univariate BM" begin
        @testset "Fixed Root, no missing" begin
        # y no missing, fixed root
        m = PGBP.UnivariateBrownianMotion(2, 3, 0)
        show(devnull, m)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -10.732857817537196
        end
        @testset "Infinite Root, no missing" begin
        # y no missing, fixed root
        m = PGBP.UnivariateBrownianMotion(2, 3, Inf)
        b = PGBP.init_beliefs_allocate(tbl_y, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl_y, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -5.899094849099194
        end
        @testset "Random Root, with missing" begin
        # x with missing, random root
        m = PGBP.UnivariateBrownianMotion(2, 3, 0.4)
        b = (@test_logs (:error,"tip B2 in network without any data") PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, m);)
        PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -13.75408386332493
        end
    end
    @testset "Diagonal BM" begin
        @testset "homogeneous, fixed root" begin
        m = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0,0))
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -24.8958130127972
        end
        @testset "homogeneous, random root" begin
        m = PGBP.MvDiagBrownianMotion((2,1), (3,-3), (0.1,10))
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -21.347496753649892
        end
        @testset "homogeneous, improper root" begin
        m = PGBP.MvDiagBrownianMotion((2,1), (1,-3), (Inf,Inf))
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -17.66791635814575
        end
    end
    @testset "heterogeneous BM" begin
        @testset "Fixed Root one mv rate" begin
        m = PGBP.HeterogeneousBrownianMotion([2.0 0.5; 0.5 1.0], [3.0, -3.0])
        show(devnull, m)
        b = PGBP.init_beliefs_allocate(tbl, df.taxon, net, ct, m);
        PGBP.init_beliefs_assignfactors!(b, m, tbl, df.taxon, net.nodes_changed);
        ctb = PGBP.ClusterGraphBelief(b)
        PGBP.calibrate!(ctb, [spt])
        _, tmp = PGBP.integratebelief!(ctb)
        @test tmp ≈ -24.312323855394055
        end
    end
    #= likelihood using PN.vcv and matrix inversion
    using Distributions
    σ2tmp = 2; μtmp = 3
    Σnet = σ2tmp .* Matrix(vcv(net)[!,Symbol.(df.taxon)])
    # for y
    loglikelihood(MvNormal(repeat([μtmp],4), Σnet), tbl.y) # -10.732857817537196
    # for x: third value is missing
    xind = [1,2,4]; n = length(xind); i = ones(n) # intercept
    Σ = Σnet[xind,xind] .+ 0.4
    loglikelihood(MvNormal(repeat([μtmp],3), Σ), Vector{Float64}(tbl.x[xind])) # -13.75408386332493
    # For y, REML
    ll(μ) = loglikelihood(MvNormal(repeat([μ],4), Σnet), tbl.y)
    using Integrals
    log(solve(IntegralProblem((x,p) -> exp(ll(x)), -Inf, Inf), QuadGKJL()).u) # -5.899094849099194
    # For x, y, Diagonal
    Σnet = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    loglikelihood(MvNormal(repeat([3],3), 2 .* Σnet[xind,xind]), Vector{Float64}(tbl.x[xind])) + 
    loglikelihood(MvNormal(repeat([-3],4), 1 .* Σnet), tbl.y) 
    # For x, y, Diagonal, random root
    Σnet = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    loglikelihood(MvNormal(repeat([3],3), 2 .* Σnet[xind,xind] .+ 0.1), Vector{Float64}(tbl.x[xind])) + 
    loglikelihood(MvNormal(repeat([-3],4), 1 .* Σnet .+ 10), tbl.y) 
    # Diagonal, REML
    Σnet = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    ll(μ) = loglikelihood(MvNormal(repeat([μ[1]],3), 2 .* Σnet[xind,xind]), Vector{Float64}(tbl.x[xind])) + loglikelihood(MvNormal(repeat([μ[2]],4), 1 .* Σnet), tbl.y) 
    using Integrals
    log(solve(IntegralProblem((x,p) -> exp(ll(x)), [-Inf, -Inf], [Inf, Inf]), HCubatureJL(), reltol = 1e-16, abstol = 1e-16).u) # -17.66791635814575
    # For x, y, full
    Σnet = Matrix(vcv(net)[!,Symbol.(df.taxon)])
    R = [2.0 0.5; 0.5 1.0]
    varxy = kron(R, Σnet)
    xyind = vcat(xind, 4 .+ [1,2,3,4])
    varxy = varxy[xyind, xyind]
    meanxy = vcat(repeat([3.0],3), repeat([-3.0],4))
    datxy = Vector{Float64}(vcat(tbl.x[xind], tbl.y))
    loglikelihood(MvNormal(meanxy, varxy), datxy) # -24.312323855394055
    =#
end
