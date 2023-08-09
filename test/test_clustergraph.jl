@testset "cluster graphs" begin
netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
net = readTopology(netstr)
g = PhyloGaussianBeliefProp.moralize!(net)
@test nv(g) == net.numNodes
@test ne(g) == net.numEdges + 1 # 1 extra: moralized

@test PhyloGaussianBeliefProp.triangulate_minfill!(g) ==
    [:A,:B,:H1,:C,:C2,:D,:I5,:I1,:I2,:I3,:I4]
@test ne(g) == 13 # 1 extra fill edge

ct = PhyloGaussianBeliefProp.cliquetree(g)
@test ne(ct) == 8
@test sort([ct[lab...] for lab in edge_labels(ct)]) == [[1],[3],[4],[6],[6,3],[8],[8,6],[9]]
@test is_tree(ct)

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
    cg = PGBP.clustergraph(net, PGBP.Bethe(), nothing)
    # the singleton {root} cluster is counted as a variable cluster
    numfactorclusters = net.numNodes-1
    numvarclusters = net.numNodes-net.numTaxa
    @test nv(cg) == numfactorclusters + numvarclusters
    # terminal edges correspond to clusters with 1 edge (assuming no leaf node
    # is a hybrid)
    numterminaledges = net.numTaxa
    # non-hybrid internal edges correspond to clusters with 2 edges (assuming
    # all degree-2 internal non-root nodes have been suppressed)
    num_nonhybrid_internaledges =
        length(findall(!e.hybrid for e in net.edge)) - numterminaledges
    # hybrid nodes correspond to clusters with 3 edges (assuming hybrid nodes
    # have indegree 2)
    numhybridnodes = net.numHybrids
    @test ne(cg) == (numterminaledges + 2*num_nonhybrid_internaledges + 3*numhybridnodes)

    # sort cluster labels according to order in which clusters are added
    # factor clusters are added first, according to the preordering of their
    # child nodes. variable clusters are added next in postordering of their
    # nodes
    o = sortperm([v[1] for v in values(cg.vertex_properties)])
    # (n -> n.number).(net.nodes_changed) # node numbers arranged in preorder
    # [-2, 6, -3, -6, 5, -7, 4, -4, 3, 2, 1]
    # (n -> n.names).(net.nodes_changed) # node labels
    # ["I5", "D", "I4", "I3", "C2", "I2", "C", "I1", "H1", "B", "A"]
    @test collect(keys(cg.vertex_properties))[o] == [
        :DI5, :I4I5, :I3I4, :C2I3, :I2I3, :CI2, :I1I4, :H1I1I2, :BH1, :AI1,
        :H1, :I1, :I2, :I3, :I4, :I5
        ]
    @test all(e -> haskey(cg, e[1], e[2]),
    [(:H1, :H1I1I2), (:H1, :BH1),
    (:I1, :I1I4), (:I1, :H1I1I2), (:I1, :AI1),
    (:I2, :I2I3), (:I2, :CI2), (:I2, :H1I1I2),
    (:I3, :I3I4), (:I3, :C2I3), (:I3, :I2I3),
    (:I4, :I4I5), (:I4, :I3I4), (:I4, :I1I4),
    (:I5, :DI5), (:I5, :I4I5)])
end

@testset "LTRIP cluster graph" begin
    netstr = "(((A:4.0,(B:1.0)#H1:1.1::0.9):0.5,((#H1:1.0::0.1,C:0.6):1.0,C2):1.0):3.0,D:5.0);"
    net = readTopology(netstr)
    T = PGBP.vgraph_eltype(net)
    # each cluster is specified a vector of preorder indices corresponding to
    # the nodes of net
    # the clusters used here correspond to the node families in net, except {root}
    clusters = Vector{T}[
        [11, 8], [10, 9], [7, 6], [5, 4], [2, 1],
        [9, 8, 6], [8, 3], [6, 4], [4, 3], [3, 1]]
    cg = PGBP.clustergraph(net, PGBP.LTRIP(), clusters)
    @test nv(cg) == length(clusters)
    
    # arrange cluster labels in order of insertion (i.e. the order in `clusters`)
    o = sortperm([v[1] for v in values(cg.vertex_properties)])
    @test collect(keys(cg.vertex_properties))[o] == [
        :AI1, :BH1, :CI2, :C2I3, :DI5,
        :H1I1I2, :I1I4, :I2I3, :I3I4, :I4I5]

    # since the edges added for each (variable-specific) spanning tree depends
    # on how `kruskal_mst` is implemented, we focus on checking, for each
    # variable, if the subgraph induced by the clusters and edges containing that
    # variable is a tree

    # sub(cluster)graphs induced by clusters containing node label [...]
    sgI1, _ = induced_subgraph(cg, [1, 6, 7]) # :I1
    sgI2, _ = induced_subgraph(cg, [3, 6, 8]) # :I2
    sgI3, _ = induced_subgraph(cg, [4, 8, 9]) # :I3
    sgI4, _ = induced_subgraph(cg, [7, 9, 10]) # :I4
    sgI5, _ = induced_subgraph(cg, [5, 10]) # :I5
    sgH1, _ = induced_subgraph(cg, [2, 6]) # :H1

    res = falses(6)
    for (i, (nl, sg)) in enumerate([(:I1, sgI1), (:I2, sgI2), (:I3, sgI3),
        (:I4, sgI4), (:I5, sgI5), (:H1, sgH1)])
        # loop through edges of induced subgraph `sg`, and delete any that do not
        # contain the variable of interest `nl`
        for e in collect(keys(sg.edge_data))
            cl1, cl2 = e[1], e[2]
            nodelabs1 = sg[cl1][1]
            nodelabs2 = sg[cl2][1]
            if (nl ∉ nodelabs1) & (nl ∉ nodelabs2)
                delete!(sg, cl1, cl2)
            end
        end
        res[i] = is_tree(sg) # check if the subgraph remaining is a (spanning) tree
    end
    @test all(res)
end

end