@inline function vgraph_eltype(net::HybridNetwork)
    nn = 2max(length(net.node), length(net.edge))
    (nn < typemax(Int8) ? Int8 : (nn < typemax(Int16) ? Int16 : Int))
end
@enum EdgeType etreetype=1 ehybridtype=2 moralizedtype=3 filltype=4

"""
    moralize!(net::HybridNetwork, prefix="I")
    moralize(net)

Undirected graph `g` of type [MetaGraph](https://github.com/JuliaGraphs/MetaGraphsNext.jl)
with the same nodes as in `net`, labelled by their names in `net`, with extra
edges to moralize the graph, that is, to connect any nodes with a common child.
Node data, accessed as `g[:nodelabel]`, is their index in the network's preordering.
Edge data, accessed as `g[:label1, :label2]` is a type to indicate if the edge
was an original tree edge or hybrid edge, or added to moralize the graph.
Another type, not used here, if for fill edges that may need to be added to
triangulate `g` (make it chordal).

The first version modifies `net` to name its internal nodes (used as labels in `g`)
and to create or update its node preordering, then calls the second version.
"""
function moralize!(net::HybridNetwork, prefix="I")
    PN.preorder!(net)
    PN.nameinternalnodes!(net, prefix)
    moralize(net)
end
@doc (@doc moralize!) moralize
function moralize(net::HybridNetwork)
    T = vgraph_eltype(net)
    mg = MetaGraph(Graph{T}(0), # simple graph
        Symbol,   # label type: Symbol(original node name)
        T,        # vertex data type: store postorder
        EdgeType, # edge data type
        :moralized, # graph data
        edge_data -> one(T),  # weight function
        zero(T))
    # add vertices in preorder, which saves their index to access them in net.
    sym2code = Dict{Symbol,T}()
    for (code,n) in enumerate(net.nodes_changed)
        ns = Symbol(n.name)
        vt = T(code)
        push!(sym2code, ns => vt)
        add_vertex!(mg, ns, vt)
    end
    for e in net.edge
        et = (e.hybrid ? ehybridtype : etreetype)
        add_edge!(mg, Symbol(getparent(e).name), Symbol(getchild(e).name), et)
    end
    # moralize
    for n in net.node
        n.hybrid || continue
        plab = [Symbol(node.name) for node in getparents(n)] # parent labels
        npar = length(plab)
        for (i1,p1) in enumerate(plab), i2 in (i1+1):npar
            p2 = plab[i2]
            has_edge(mg.graph, sym2code[p1], sym2code[p2]) && continue
            add_edge!(mg, p1, p2, moralizedtype)
        end
    end
    return mg
end

#= todo fixit: add function to add more moralizing edges,
for a degenerate hybrid node with a single tree child to be remove from scope:
connect the tree child to all its grandparents
e.g. connect_degeneratehybridparents_treechild
=#

"""
    triangulate_minfill!(graph)

Ordering for node elimination, chosen to greedily minimize the number of fill edges
necessary to eliminate the node (to connect all its neighbors with each other).
Ties are broken by favoring the post-ordering of nodes.
`graph` is modified with these extra fill edges, making it chordal.
"""
function triangulate_minfill!(graph::AbstractGraph{T}) where T
    ordering = typeof(label_for(graph, one(T)))[]
    g2 = deepcopy(graph)
    fe = Tuple{T,T}[] # to reduce memory allocation
    scorefun = v -> (min_fill!(fe,v,g2), - g2[label_for(g2,v)]) # break ties using post-ordering
    while nv(g2) > 1
        i = argmin(scorefun, vertices(g2))
        # add fill edges in both graph and g2, then delete i from g2
        filledges!(fe, T(i), g2)
        for (v1,v2) in fe
            l1 = label_for(g2,v1); l2 = label_for(g2,v2)
            add_edge!(g2,    l1, l2, filltype)
            add_edge!(graph, l1, l2, filltype)
        end
        lab = label_for(g2,i)
        push!(ordering, lab)
        delete!(g2, lab)
    end
    push!(ordering, label_for(g2,one(T)))
    return ordering
end
function min_fill!(fe, vertex_code, graph::AbstractGraph)
    filledges!(fe, vertex_code, graph::AbstractGraph)
    return length(fe)
end
function filledges!(fe, vertex_code, graph::AbstractGraph)
    empty!(fe)
    neighb = inneighbors(graph, vertex_code)
    nn = length(neighb)
    for (i1,n1) in enumerate(neighb), i2 in (i1+1):nn
        n2 = neighb[i2]
        has_edge(graph, n1,n2) || push!(fe, (n1,n2))
    end
    return nothing # nn
end

"""
    cliquetree(chordal_graph)

Clique tree `U` for an input graph `g` assumed to be chordal (triangulated),
e.g. using [`triangulate_minfill!`](@ref).  
**Warning**: does *not* check that the graph is already triangulated.

- Each node in `U` is a maximal clique of `g` whose data is the tuple of vectors
  (node_labels, node_data) using the labels and data from `g`, with nodes sorted
  by decreasing data.
  If `g` was originally built from a phylogenetic network using [`moralize`](@ref),
  then the nodes' data are their preorder index, making them sorted in postorder
  within in each clique.
  The clique label is the concatenation of the node labels.
- For each edge (clique1, clique2) in `U`, the edge data hold the sepset
  (separating set) information as a vector of node data, for nodes shared by
  both clique1 and clique2. In this sepset, nodes are sorted by decreasing data.

Uses `maximal_cliques` and `kruskal_mst` (for min/maximum spanning trees) from
[Graphs.jl](https://juliagraphs.org/Graphs.jl/stable/).
"""
function cliquetree(graph::AbstractGraph{T}) where T
    mc = maximal_cliques(graph)
    mg = MetaGraph(Graph{T}(0),
        Symbol, # vertex label
        Tuple{Vector{Symbol},Vector{T}}, # vertex data: nodes in clique
        Vector{T}, # edge data: nodes in sepset
        :cliquetree,
        edge_data -> T(length(edge_data)),
        zero(T))
    node2clique = Dict{T,Vector{T}}() # to connect cliques faster later
    for (code, cl) in enumerate(mc)
        nodeindlist = [graph[label_for(graph,u)] for u in cl] # preorder index
        o = sortperm(nodeindlist, rev=true) # order to list nodes in postorder
        vdat = [label_for(graph,cl[i]) for i in o]
        nodeindlist .= nodeindlist[o]
        # clique data: (node labels, node preorder index), sorted in postorder
        add_vertex!(mg, Symbol(vdat...), (vdat, nodeindlist))
        for ni in nodeindlist
            if haskey(node2clique, ni)
                push!(node2clique[ni], T(code))
            else node2clique[ni] = [T(code)]
            end
        end
    end

    # create edges between pairs of cliques sharing the same node
    for node in sort!(collect(keys(node2clique)), rev=true) # sepsets will be sorted by nodes' postorder
        cliquelist = node2clique[node]
        # add node to the sepset between each pair of cliques that has that node
        for (i1, cl1) in enumerate(cliquelist)
        lab1 = label_for(mg, cl1)
          for i2 in 1:(i1-1)
            cl2 = cliquelist[i2]
            lab2 = label_for(mg, cl2)
            if has_edge(mg, cl1, cl2)
              elabs = MetaGraphsNext.arrange(mg, lab1, lab2)
              haskey(mg.edge_data, elabs) || error("hmm, mg.graph has the edge, but mg has no edge data")
              push!(mg.edge_data[elabs], node)
            else
              add_edge!(mg, lab1, lab2, [node])
            end
          end
        end
    end
    #= altenate way to create edges between cliques
    # would be faster for a complex graph with many nodes but few (large) cliques.
    for cl1 in vertices(mg)
        lab1 = label_for(mg, cl1)
        ni1 = mg[lab1][2] # node indices: preorder index of nodes in the clique
        for cl2 in Base.OneTo(cl1 - one(T))
            lab2 = label_for(mg, cl2)
            ni2 = mg[lab2][2]
            sepset = intersect(ni1, ni2) # would be nice to take advantage of the fact that ni1 and ni2 are both sorted (descending)
            isempty(sepset) && continue
            add_edge!(mg, lab1, lab2, sepset)
        end
    end =#

    # find maximum spanning tree: with maximum sepset sizes
    mst_edges = kruskal_mst(mg, minimize=false)
    # delete the other edges to get the clique tree
    # complication: edge iterator is invalidated by the first edge deletion
    todelete = setdiff!(collect(edges(mg)), mst_edges)
    for e in todelete
        rem_edge!(mg, src(e), dst(e))
    end
    return mg
end
