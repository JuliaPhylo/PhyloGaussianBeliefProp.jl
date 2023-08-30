@testset "cluster graphs" begin

@testset "Utilities" begin
    netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
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
    netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
    net = readTopology(netstr)
    cg = PGBP.clustergraph!(net, PGBP.Bethe())

    #=
    Count the no. of clusters:
    1. no. of factor clusters == no. of node families (== no. of nodes) except
    for the singleton {root}, which is counted as a variable cluster
    2. no. of variable clusters == no. of internal nodes
    =#
    numfactorclusters = net.numNodes-1
    numvarclusters = net.numNodes-net.numTaxa
    @test nv(cg) == numfactorclusters + numvarclusters
    
    #=
    Count the no. of cluster graph edges:
        * Assume no leaf node is a hybrid
        * Assume hybrid nodes have indegree 2
    1. network terminal edges correspond to clusters with 1 incident edge
    2. network non-terminal non-hybrid internal edges correspond to clusters
    with 2 incident edges
    3. hybrid node families correspond to clusters with 3 incident edges
    =#
    numterminaledges = net.numTaxa
    num_nonhybrid_internaledges =
        length(findall(!e.hybrid for e in net.edge)) - numterminaledges
    numhybridnodes = net.numHybrids
    @test ne(cg) == (numterminaledges + 2*num_nonhybrid_internaledges +
        3*numhybridnodes)
    
    # Check if connected
    @test length(connected_components(cg)) == 1

    #=
    Check the Running Intersection property -- the subgraph induced by clusters
    containing a specific variable must be a tree
    =#
    @test all(t[2] for t in PGBP.checkRI(cg, net))

    # Check that the set of clusters is family-preserving wrt the network
    cluster_properties = cg.vertex_properties
    clusters = [v[2][2] for v in values(cluster_properties)]
    # variable clusters: [1], [3], [4], [6], [8], [9]
    # factor clusters: [2, 1], [3, 1], [4, 3], [5, 4], [6, 4], [7, 6], [8, 3],
    #   [9, 8, 6], [10, 9], [11, 8]
    @test PGBP.isfamilypreserving!(clusters, net)[1]
end

@testset "LTRIP cluster graph" begin
    netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
    net = readTopology(netstr)
    T = PGBP.vgraph_eltype(net)
    # Clusters specified as vectors of preorder indices. The clusters used here
    # correspond to the node families in net, except {root}
    clusters = Vector{T}[
        [11, 8], [10, 9], [7, 6], [5, 4], [2, 1],
        [9, 8, 6], [8, 3], [6, 4], [4, 3], [3, 1]]

    # Check if input clusters are valid
    @test(try
        PGBP.LTRIP!(clusters, net); true
    catch
        false
    end)
    
    cg = PGBP.clustergraph!(net, PGBP.LTRIP!(clusters, net))

    # Compare input and output clusters (and hence check if family-preserving)
    cluster_properties = cg.vertex_properties
    output_clusters = collect(v[2][2] for v in values(cluster_properties))
    @test sort(clusters) == sort(output_clusters)

    @test length(connected_components(cg)) == 1
    @test all(t[2] for t in PGBP.checkRI(cg, net))

    cg2 = PGBP.clustergraph!(net, PGBP.LTRIP!(net))
    @test all(t[2] for t in PGBP.checkRI(cg2, net))
    clusters2 = [v[2][2] for v in values(cg2.vertex_properties)]
    @test PGBP.isfamilypreserving!(clusters2, net)[1]

    clusters3 = Vector{T}[
        [11, 8], [10, 9], [7, 6], [5, 4], [2, 1],
        [9, 8], [8, 3], [6, 4], [4, 3], [3, 1]] # not family-preserving
    @test_throws ErrorException PGBP.clustergraph!(net,
        PGBP.LTRIP!(clusters3, net))
end

@testset "Join-graph struturing" begin
    netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
    net = readTopology(netstr)
    cg = PGBP.clustergraph!(net, PGBP.JoinGraphStr(3))

    @test all(t[2] for t in PGBP.checkRI(cg, net))
    clusters = [v[2][2] for v in values(cg.vertex_properties)]
    @test PGBP.isfamilypreserving!(clusters, net)[1]
    @test maximum(cl -> length(cl), clusters) ≤ 3 # max cluster size is respected
    # catch invalid max cluster size
    @test_throws ErrorException PGBP.clustergraph!(net, PGBP.JoinGraphStr(2))
end

@testset "Clique tree" begin
    netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
    net = readTopology(netstr)
    ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
    
    @test ne(ct) == 8
    @test sort([ct[lab...] for lab in edge_labels(ct)]) == [[1],[3],[4],[6],
        [6,3],[8],[8,6],[9]]
    @test is_tree(ct)

    @test all(t[2] for t in PGBP.checkRI(ct, net))
    cliques = [v[2][2] for v in values(ct.vertex_properties)]
    @test PGBP.isfamilypreserving!(cliques, net)[1]
end

end