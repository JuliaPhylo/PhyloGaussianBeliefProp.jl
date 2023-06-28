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

end
