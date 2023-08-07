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

"""
    AbstractClusterGraphMethod

Abstract type for method of cluster graph construction.
"""
abstract type AbstractClusterGraphMethod end

# generic methods
clustergraphmethod(obj::AbstractClusterGraphMethod) = obj.code

"""
    Bethe

Bethe cluster graph (also known as factor graph).
"""
struct Bethe <: AbstractClusterGraphMethod
    code::Symbol
    Bethe(code) = (code != :Bethe) ? error("Set code to :Bethe") : new(:Bethe)
end
Bethe() = Bethe(:Bethe)

"""
    LTRIP

*Layered Trees Running Intersection Property* algorithm of Streicher & du Preez
(2017).

## References:

S. Streicher and J. du Preez. Graph Coloring: Comparing Cluster Graphs to Factor
Graphs. In *Proceedings of the ACM Multimedia 2017 Workshop on South African
Academic Participation, pages 35-42, 2017. doi: 10.1145/3132711.3132717.
"""
struct LTRIP <: AbstractClusterGraphMethod
    code::Symbol
    # todo: add another field for edge-weight function?
    LTRIP(code) = (code != :ltrip) ? error("Set code to :ltrip") : new(:ltrip)
end
LTRIP() = LTRIP(:ltrip)

"""
    JoinGraphStr

*Join-graph structuring* algorithm of Mateescu et al. (2010).

## References:

R. Mateescu, K. Kask, V.Gogate, and R. Dechter. Join-graph propagation algorithms.
*Journal of Artificial Intelligence Research*, 37:279-328,
2010. doi:10.1613/jair.2842.
"""
struct JoinGraphStr <: AbstractClusterGraphMethod
    code::Symbol
    JoinGraphStr(code) = (code != jgstr) ? error("Set code to :jgstr") : new(:jgstr)
end
JoinGraphStr() = JoinGraphStr(:jgstr)

"""
    clustergraph(graph, method, clusters)
    clustergraph(graph, clustergraph, method, clusters)

For Bethe, you don't need to specify anything (the clusters correspond to the
individual node families and variables).
(i) provide graphical model and indicate Bethe specification
(ii) generate clusters then add edges

For LTRIP, you must specify the clusters and the edge-weight function.
When initializing the cluster beliefs, one would need to check if the set of
clusters provided is family-preserving.
(i) provide graphical model, indicate LTRIP specification, provide clusters and
edge-weight function
(ii) add edges based on clusters and edge-weight function
"""
function clustergraph(graph::AbstractGraph{T},
    method::AbstractClusterGraphMethod,
    clusters::Union{Vector{Vector{T}},Nothing}=nothing) where T

    isnothing(clusters) && (clustergraphmethod(method) == LTRIP) ||
    error("LTRIP method requires user-specified clusters")
    
    clustergraph = MetaGraph(
        Graph{T}(0),
        Symbol, # vertex label
        Tuple{Vector{Symbol}, Vector{T}}, # vertex data: nodes in cluster
        Vector{T}, # edge data: nodes in sepset
        clustergraphmethod(method), # tag for the whole graph
        edge_data -> T(length(edge_data)),
        zero(T))

    clustergraph(graph, clustergraph, method, clusters)
end

function clustergraph(graph::AbstractGraph{T}, clustergraph::AbstractGraph{T},
    method::Bethe, _) where T

    is_directed(graph) || error("`graph` must be directed")

    node2cluster = Dict{T, (Symbol, Vector{T})}() # for joining clusters later
    for (code, lab) in enumerate(labels(graph))
        vdat = [lab; inneighbor_labels(graph, lab)]
        nodeindlist = [graph[l] for l in vdat] # preorder index
        o = sortperm(nodeindlist, rev=true) # order to list nodes in postorder
        vdat .= vdat[o]
        # cluster data: (node labels, node preorder index), sorted in postorder
        # add factor clusters
        add_vertex!(clustergraph, Symbol(vdat...), (vdat, nodeindlist))
        if length(nodeindlist) > 1 # non-root node of graph
            for ni in nodeindlist
                if haskey(node2cluster, ni)
                    push!(node2cluster[ni][2], T(code))
                else node2cluster[ni] = (lab, [T(code)])
                end
            end
        end
    end

    for node in sort!(collect(keys(node2cluster)), rev=true)
        # sepsets will be sorted by nodes' postorder
        lab, clusterlist = node2cluster[node]
        if length(clusterlist) > 1
            # add variable cluster only if there are > 1 clusters with this node
            add_vertex!(clustergraph, lab, ([lab], [node]))
            # connect variable cluster to factor clusters that contain this node
            for code in clusterlist
                lab2 = label_for(clustergraph, code)
                add_edge!(clustergraph, lab, lab2, [node])
            end
        end
    end
    return clustergraph
end

function clustergraph(graph::AbstractGraph{T}, clustergraph::AbstractGraph{T},
    method::LTRIP, clusters::Vector{Vector{T}}) where T
    
    is_directed(graph) || error("`graph` must be directed")
    # todo: check if family-preserving

    node2cluster = Dict{T, Vector{T}}() # for joining clusters later
    for (code, cl) in enumerate(clusters)
        nodeindlist = [graph[label_for(graph, u)] for u in cl] # preorder index
        o = sortperm(nodeindlist, rev=true) # order to list nodes in postorder
        vdat = [label_for(graph, cl[i]) for i in o]
        nodeindlist .= nodeindlist[o]
        # cluster data: (node labels, node preorder index), sorted in postorder
        add_vertex!(clustergraph, Symbol(vdat...), (vdat, nodeindlist))
        for ni in nodeindlist
            if haskey(node2cluster, ni)
                push!(node2cluster[ni], T(code))
            else node2cluster[ni] = [T(code)]
            end
        end
    end

    # compute edge weights on copy of cluster graph
    cg = deepcopy(clustergraph)
    cg.weight_function = (edge_data -> edge_data) # todo: check if this works
    for node in sort!(collect(keys(node2cluster)), rev=true)
        clusterlist = node2cluster[node]
        for (i1, cl1) in enumerate(clusterlist)
            lab1 = label_for(cg, cl1)
            for i2 in 1:(i1-1)
                cl2 = clusterlist[i2]
                lab2 = label_for(cg, cl2)
                # todo: change with ltrip weight
                w = length(intersect(cg[lab1], cg[lab2]))
                add_edge!(cg, lab1, lab2, w)
            end
        end
        mst_edges = kruskal_mst(cg, minimize=false)
        
        # overlay spanning trees on original cluster graph
        for e in mst_edges
            cl1, cl2 = src(e), dst(e)
            lab1, lab2 = label_for(clustergraph, cl1), label_for(clustergraph, cl2)
            if has_edge(clustergraph, cl1, cl2)
                elabs = MetaGraphsNext.arrange(clustergraph, lab1, lab2)
                haskey(clustergraph.edge_data, elabs) ||
                    error("clustergraph.graph has the edge, but clustergraph has
                    no edge data")
                push!(clustergraph.edge_data[elabs], node)
            else
                add_edge!(clustergraph, lab1, lab2, [node])
            end
        end
    end
    return clustergraph
end

"""
    spanningtree_clusterlist(clustergraph, root_index)
    spanningtree_clusterlist(clustergraph, nodevector_preordered)

Build the depth-first search spanning tree of the cluster graph, starting from
the node indexed `root_index` in the underlying simple graph;
find the associated topological ordering of the clusters (preorder); then
return a tuple of these four vectors:
1. `parent_labels`: labels of the parents' child clusters. The first one is the root.
2. `child_labels`: labels of clusters in pre-order, except for the cluster
    choosen to be the root.
3. `parent_indices`: indices of the parent clusters
4. `child_indices`: indices of the child clusters, listed in preorder as before.

In the second version in which `root_index` is not provided, the root of the
spanning tree is chosen to be a cluster that contains the network's root. If
multiple clusters contain the network's root, then one is chosen containing the
smallest number of taxa: see [`default_rootcluster`](@ref).
"""
function spanningtree_clusterlist(cgraph::MetaGraph, prenodes::Vector{PN.Node})
    rootj = default_rootcluster(cgraph, prenodes)
    spanningtree_clusterlist(cgraph, rootj)
end
function spanningtree_clusterlist(cgraph::MetaGraph, rootj::Integer)
    par = dfs_parents(cgraph.graph, rootj)
    spt = Graphs.tree(par) # or directly: spt = dfs_tree(cgraph.graph, rootj)
    # spt.fadjlist # forward adjacency list: sepsets, but edges not indexed
    childclust_j = topological_sort(spt)[2:end] # cluster in preorder, excluding the root cluster
    parentclust_j = par[childclust_j] # parent of each cluster in spanning tree
    childclust_lab  = [cgraph.vertex_labels[j] for j in childclust_j]
    parentclust_lab = [cgraph.vertex_labels[j] for j in parentclust_j]
    return parentclust_lab, childclust_lab, parentclust_j, childclust_j
end

"""
    default_rootcluster(clustergraph, nodevector_preordered)

Index of a cluster that contains the network's root, whose label is assumed to
be `1` (preorder index). If multiple clusters contain the network's root,
then one is chosen with the smallest number of taxa (leaves in the network).

For cluster with label `:lab`, its property `clustergraph[:lab][2]`
should list the nodes in the cluster, by the index of each node in
`nodevector_preordered` such that `1` corresponds to the network's root.
Typically, this vector is `net.nodes_changed` after the network is preordered.
"""
function default_rootcluster(cgraph::MetaGraph, prenodes::Vector{PN.Node})
    hasroot = lab -> begin   # Inf if the cluster does not contain the root 1
        nodelabs = cgraph[lab][2]  # number of taxa in the cluster otherwise
        (1 ∈ nodelabs ? sum(prenodes[i].leaf for i in nodelabs) : Inf)
    end
    rootj = argmin(hasroot(lab) for lab in labels(cgraph))
    return rootj
end
