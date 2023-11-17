@testset "cluster graphs" begin
netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
# network from Mateescu et al. (2010) with 2 extra leaves
mateescu = "((((g:1)#H4:1)#H2:2.04,(d:1,(#H2:0.01::0.5,#H4:1::0.5)#H3:1)D:1,(#H3:1::0.5)#H1:0.01)B:1,#H1:1.01::0.5)A;"

@testset "Utilities" begin
    net = readTopology(netstr)
    g = PhyloGaussianBeliefProp.moralize!(net)
    @test nv(g) == net.numNodes
    @test ne(g) == net.numEdges + 1 # 1 extra: moralized
    @test PhyloGaussianBeliefProp.triangulate_minfill!(g) ==
    [:A,:B,:H1,:C,:C2,:D,:I5,:I1,:I2,:I3,:I4]
    @test ne(g) == 13 # 1 extra fill edge
end

#=
function metaplot(gr)
   elab = [gr[label_for(gr,src(e)),label_for(gr,dst(e))] for e in edges(gr)]
   gplothtml(gr, nodelabel=collect(labels(gr)), edgelabel=elab);
end
metaplot(ct)
=#

@testset "Bethe cluster graph" begin
    net = readTopology(netstr)
    cg = PGBP.clustergraph!(net, PGBP.Bethe())
    #= number of clusters:
    1. factor clusters:   1 / node family = 1 per non-root node, except when a family is a subset of another
    2. variable clusters: 1 / internal node (including the root, excluding leaves)
    =#
    numfactorclusters = net.numNodes-1
    numvarclusters = net.numNodes-net.numTaxa
    @test nv(cg) == numfactorclusters + numvarclusters
    #= number of edges in the cluster graph, assuming
        * leaves are not hybrids
        * bicombining: hybrid nodes have 2 parents
    1. external edge in net: 1 per leaf → 1 edge in cluster graph
    2. internal tree edge in net, e.g. internal tree node → 2 edges in graph
    3. hybrid node family, 1 per hybrid node in net → 3 edges in graph
    =#
    ninternal_tree = sum(!e.hybrid for e in net.edge) - net.numTaxa
    @test ne(cg) == (net.numTaxa + 2*ninternal_tree + 3*net.numHybrids)
    @test length(connected_components(cg)) == 1 # check for 1 connected component
    @test all(t[2] for t in PGBP.check_runningintersection(cg, net))
    cluster_properties = cg.vertex_properties
    clusters = [v[2][2] for v in values(cluster_properties)]
    # variable clusters: [1], [3], [4], [6], [8], [9]
    # factor clusters: [2, 1], [3, 1], [4, 3], [5, 4], [6, 4], [7, 6], [8, 3],
    #   [9, 8, 6], [10, 9], [11, 8]
    @test PGBP.isfamilypreserving(clusters, net)[1]

    # case with 3-cycle: one node family is a subset of another
    deleteleaf!(net, "C"); deleteleaf!(net, "C2"); removedegree2nodes!(net)
    preorder!(net)
    cg = PGBP.clustergraph!(net, PGBP.Bethe())
    @test nv(cg) == 4+3 # not (5-1) node families + 3 internal nodes
    @test ne(cg) == 6   # each internal node is 2 clusters
    clusters = [v[2][2] for v in values(cg.vertex_properties)]
    @test PGBP.isfamilypreserving(clusters, net)[1]
end

@testset "LTRIP cluster graph" begin
    net = readTopology(netstr)
    clusters = Vector{Int8}[ # node families, nodes specified as preorder indices
        [11, 8], [10, 9], [7, 6], [5, 4], [2, 1],
        [9, 8, 6], [8, 3], [6, 4], [4, 3], [3, 1]]
    # below: would error (test would fail) if `clusters` not family-preserving for net
    cg = PGBP.clustergraph!(net, PGBP.LTRIP(clusters, net))
    output_clusters = collect(v[2][2] for v in values(cg.vertex_properties))
    @test sort(clusters) == sort(output_clusters)
    @test is_connected(cg)
    @test all(t[2] for t in PGBP.check_runningintersection(cg, net))

    cg = PGBP.clustergraph!(net, PGBP.LTRIP(net))
    @test all(t[2] for t in PGBP.check_runningintersection(cg, net))
    clusters2 = [v[2][2] for v in values(cg.vertex_properties)] # has extra root cluster
    @test PGBP.isfamilypreserving(clusters2, net)[1]

    clusters3 = Vector{Int8}[
        [11, 8], [10, 9], [7, 6], [5, 4], [2, 1],
        [9, 8], [8, 3], [6, 4], [4, 3], [3, 1]] # not family-preserving
    @test_throws ErrorException PGBP.LTRIP(clusters3, net)
end

@testset "Join-graph structuring" begin
    net = readTopology(mateescu)
    # Mateescu network: 1 bucket has multiple minibuckets
    cg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(3))
    @test all(t[2] for t in PGBP.check_runningintersection(cg, net))
    @test !is_tree(cg)
    clusters = [[1],[2,1],[3,2,1],[4,3,2],[5,2],[5,4,3],[6,5,2],[7,6,5],[8,7],[9,4]]
    # clusters for netstr: [[2,1],[3],[3,1],[4,3],[5,4],[6,4,3],[7,6],[8,6,3],[9,8,6],[10,9],[11,8]]
    @test sort([v[2][2] for v in values(cg.vertex_properties)]) == clusters
    sepsets = [[1],[2],[2,1],[3,2],[4],[4,3],[5],[5,2],[6,5],[7]]
    # sepstes for netstr: [[1],[3],[3],[4],[4,3],[6],[6,3],[8],[8,6],[9]]
    @test sort([cg[l1,l2] for (l1,l2) in edge_labels(cg)]) == sepsets
    @test PGBP.isfamilypreserving(clusters, net)[1]
    # maxclustersize smaller than largest family:
    @test_throws ErrorException PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(2))
end

@testset "Clique tree" begin
    net = readTopology(netstr)
    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    
    @test ne(ct) == 8
    @test sort([ct[lab...] for lab in edge_labels(ct)]) == [[1],[3],[4],[6],
        [6,3],[8],[8,6],[9]]
    @test is_tree(ct)

    @test all(t[2] for t in PGBP.check_runningintersection(ct, net))
    cliques = [v[2][2] for v in values(ct.vertex_properties)]
    @test PGBP.isfamilypreserving(cliques, net)[1]

    net = readTopology(mateescu)
    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    @test is_tree(ct)
    @test ct[:H3DH1B][2] == [5,4,3,2] # largest clique
end

@testset "Traversal" begin
    net = readTopology(netstr)
    cg = PGBP.clustergraph!(net, PGBP.Bethe())
    clusterlabs = Set(labels(cg))
    nclusters = length(clusterlabs)
    edgenotused = Set(edge_labels(cg))
    schedule = PGBP.spanningtrees_cover_clusterlist(cg, net.nodes_changed)
    ntrees = length(schedule)
    # Check that `schedule` contains spanning trees of `cg`
    for (i, spt) in enumerate(schedule)
        spt_nedges = length(spt[1])
        @test spt_nedges == nclusters-1
        spt_edgecodes = [Edge(code_for(cg, spt[1][i]), code_for(cg, spt[2][i]))
            for i in 1:spt_nedges]
        sg, _ = induced_subgraph(cg, spt_edgecodes)
        # (1) `spt` spans the clusters of `cg`, (2) `spt` is a tree
        @test isequal(Set(labels(sg)), clusterlabs) && is_tree(sg)
        setdiff!(edgenotused,
            MetaGraphsNext.arrange(cg, spt[1][i], spt[2][i]) for i in 1:length(spt[1]))
    end
    # (3) Set of spanning trees covers all cluster graph edges
    @test isempty(edgenotused)
end

end