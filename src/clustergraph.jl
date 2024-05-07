@inline function vgraph_eltype(net::HybridNetwork)
    nn = 2max(length(net.node), length(net.edge))
    (nn < typemax(Int8) ? Int8 : (nn < typemax(Int16) ? Int16 : Int))
end
@enum EdgeType etreetype=1 ehybridtype=2 moralizedtype=3 filltype=4

"""
    preprocessnet!(net::HybridNetwork, prefix="I")

Create or update the pre-ordering of nodes in `net` using `PhyloNetworks.preorder!`,
then name unnamed internal nodes, with names starting with `prefix`.
Nodes in phylogenetic networks need to have names to build cluster graphs,
in which a cluster contains network nodes. Pre-ordering is also used to traverse
the network for building cluster graphs.

See [`clustergraph!`](@ref).
"""
function preprocessnet!(net::HybridNetwork, prefix="I")
    PN.preorder!(net)
    PN.nameinternalnodes!(net, prefix)
end

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
    preprocessnet!(net, prefix)
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
    nodefamilies(net)

Vector `v` with elements of type `Vector{T}`.
`v[i]` first lists `i`, the preorder index of node `net.nodes_changed[i]`,
followed by the preorder index of all of this node's parents in `net`,
sorted in decreasing order. Due to pre-ordering,
all of the parents' indices are listed after the node (their child) index.
A given node and its parents is called a "node family".

**Warning**: `net` is assumed preordered, see [`preprocessnet!`](@ref) and
[`PhyloNetworks.preorder!`](https://crsl4.github.io/PhyloNetworks.jl/latest/lib/public/#PhyloNetworks.preorder!)).
"""
function nodefamilies(net::HybridNetwork)
    T = vgraph_eltype(net)
    prenodes = net.nodes_changed
    node2family = Vector{Vector{T}}(undef, length(prenodes))
    for (code, n) in enumerate(prenodes)
        o = sort!(indexin(getparents(n), prenodes), rev=true)
        pushfirst!(o, code)
        node2family[code] = o # then node2family[code][1] = code
    end
    return node2family
end

"""
    isfamilypreserving(clusters, net)

Tuple `(ispreserving, isfamily_incluster)`:
1. `ispreserving`: true (false) if `clusters` is (is not) family-preserving
   with respect to `net`, that is: if each node family (a node and all of its parents)
   in `net` is contained in at least 1 cluster in `clusters`.
   `clusters` should be a vector, where each element describes one cluster,
   given as a vector of preorder indices. Index `i` corresponds to node number `i`
   in `net` according the node pre-ordering: `net.nodes_changed[i]`.
2. `isfamily_incluster`: vector of `BitVector`s. `isfamily_incluster[i][j]` is
   true (false) if the family of node `i` is (is not) fully contained in cluster [j].
   `i` is taken as the preorder index of a node in `net`.
   `ispreserving` is true if every element (bit vector) in `isfamily_incluster`
    contains at least 1 true value.

**Warning**: assumes that `net` is preordered, see
[`PhyloNetworks.preorder!`](https://crsl4.github.io/PhyloNetworks.jl/latest/lib/public/#PhyloNetworks.preorder!)).

See also [`nodefamilies`](@ref) to get node families.
"""
function isfamilypreserving(clusters::Vector{Vector{T}},
                            net::HybridNetwork) where {T <: Integer}
    node2family = nodefamilies(net) # vectors of type vgraph_eltype(net)
    isfamilyincluster = Vector{BitVector}(undef, length(node2family))
    for nf in node2family
        ch = nf[1] # preorder index of family's child. also: node2family[ch] = nf
        isfamilyincluster[ch] = BitArray(nf ⊆ cl for cl in clusters)
    end
    ifp = all(any.(isfamilyincluster)) # *every* node family is in *some* cluster
    return (ifp, isfamilyincluster)
end

"""
    check_runningintersection(clustergraph, net)

Vector of tuples. Each tuple is of the form `(nodelabel, istree)`, where
`nodelabel::Symbol` is the label of a node in `net` and `istree` is true (false)
if the node's cluster subgraph is (is not) a tree.
This "cluster subgraph" for a given node is the subgraph of `clustergraph`
induced by the clusters containing the node and by the edges whose sepset
contain the node: see [`nodesubtree`](@ref).

`clustergraph` satisfies the generalized *running-intersection* property if
`istree` is true for all nodes in `net`.

**Warning**:
- assumes that `net` has been preordered, and
- does *not* check if `clustergraph` has been correctly constructed.
"""
function check_runningintersection(clustergraph::MetaGraph, net::HybridNetwork)
    res = Tuple{Symbol, Bool}[]
    for (nod_ind, n) in enumerate(net.nodes_changed)
        nodelab = Symbol(n.name)
        sg, _ = nodesubtree(clustergraph, nodelab, nod_ind)
        push!(res, (nodelab, is_tree(sg)))
    end
    return res
end

"""
    nodesubtree(clustergraph::MetaGraph, node_symbol)
    nodesubtree(clustergraph::MetaGraph, node_symbol, node_preorderindex)

MetaGraph subgraph of `clustergraph` induced by the clusters and sepsets containing
the node labelled `node_symbol` (of specified preorder index if known, not checked).
If `clustergraph` satisfies the generalized running-intersection property,
then this subgraph should be a tree.
"""
function nodesubtree(cgraph::MetaGraph, ns::Symbol)
    # find 1 cluster containing the node of interest
    clu = findfirst(ns ∈ cgraph[clab][1] for clab in labels(cgraph))
    isnothing(clu) && error("no cluster with node labelled $ns")
    # find preorder index of node of interest: stored in cluster data
    n1data = cgraph[label_for(cgraph,clu)]
    node_ind = n1data[2][findfirst(isequal(ns), n1data[1])]
    nodesubtree(cgraph, ns, node_ind)
end
function nodesubtree(cgraph::MetaGraph, ns::Symbol, node_ind)
    # indices of clusters containing the node of interest
    clusters_i = findall(ns ∈ cgraph[clab][1] for clab in labels(cgraph))
    isempty(clusters_i) && error("no cluster with node labelled $ns")
    sg, vmap = induced_subgraph(cgraph, clusters_i)
    # in subgraph, delete any edge whose sepset lacks node_ind, the node's preorder index
    for e in edge_labels(sg)
        if node_ind ∉ sg[e...] # edge data = vector of nodes preorder indices
            delete!(sg, e...)
        end
    end
    return sg, vmap
end

"""
    AbstractClusterGraphMethod

Abstract type for cluster graph construction algorithms.
"""
abstract type AbstractClusterGraphMethod end

getclusters(obj::AbstractClusterGraphMethod) =
    hasfield(typeof(obj), :clusters) ? obj.clusters : nothing
getmaxclustersize(obj::AbstractClusterGraphMethod) =
    hasfield(typeof(obj), :maxclustersize) ? obj.maxclustersize : nothing

"""
    Bethe

Subtype of [`AbstractClusterGraphMethod`](@ref).

## Algorithm

A Bethe cluster graph (also known as factor graph) has:
- a factor-cluster `{v, parents(v}}` for each node-family in the network, that is,
  for each non-root node `v` (a family is a child node and all of its parents)
  * with one exception: if `v`'s family is included in another family,
    then no factor-cluster is created for `v`.
- a variable-cluster `{v}` for each non-leaf node `v` in the network,
  or more specifically, for each node `v` that belongs in more than 1 factor.

Each variable-cluster `{v}` is joined to the factor-clusters that contain `v`,
by an edge labelled with sepset `{v}`.

## References

D. Koller and N. Friedman.
*Probabilistic graphical models: principles and techniques*.
MIT Press, 2009. ISBN 9780262013192.
"""
struct Bethe <: AbstractClusterGraphMethod end

"""
    LTRIP{T<:Integer}

Subtype of [`AbstractClusterGraphMethod`](@ref).
A HybridNetwork and a valid LTRIP are passed to [`clustergraph!`](@ref) to
construct a cluster graph from the user-provided clusters based on the *Layered
Trees Running Intersection Property* algorithm of Streicher & du Preez (2017).

## Fieldnames

- clusters: vector of clusters, required to be family-preserving
  with respect to some HybridNetwork -- see [`isfamilypreserving`](@ref).
  Within each cluster, nodes (identified by their preorder index in the network)
  are required to be sorted in decreasing order (for postorder)

## Constructors

- `LTRIP(net)`: uses [`nodefamilies(net)`](@ref) as input clusters,
  which are guaranteed to be family-preserving
- `LTRIP(clusters, net)`: checks if that clusters provided are family-preserving,
   then sorts each cluster in decreasing order (modifying them in place!)
   before creating the LTRIP object.

They assume, *with no check*, that `net` already has a preordering.

## Algorithm

1. An initial graph G is considered, in which each input cluster is a node.
   An edge (C1,C2) is added if clusters C1 and C2 share at least 1 node (in `net`).
   The weight of edge (C1,C2) is defined as the size of the intersection C1 ∩ C2.
2. For each node `n` in `net`,
   the subgraph of G induced by the clusters containing `n`, G_n, has its
   weights adjusted as follows:
     * the edges of maximum weight (within G_n) are identified, then
     * the weight of each edge is increased by the number of max-weight edges
       that either of its endpoints adjacent to.
   Then, LTRIP finds a maximum-weight spanning tree of G_n. The edges of this
   tree are all labelled with `{n}` (or its label or preorder index).
3. The spanning trees for each node are layered on one another to form a cluster
   graph. In other words, an edge (C1,C2) is added if it is present is any
   spanning tree. If so, its sepset is the union of its labels across the
   different spanning trees.

## References

S. Streicher and J. du Preez. Graph Coloring: Comparing Cluster Graphs to Factor
Graphs. In *Proceedings of the ACM Multimedia 2017 Workshop on South African
Academic Participation, pages 35-42, 2017.
doi: [10.1145/3132711.3132717](https://doi.org/10.1145/3132711.3132717).
"""
struct LTRIP{T<:Integer} <: AbstractClusterGraphMethod
    clusters::Vector{Vector{T}}
end
function LTRIP(net::HybridNetwork)
    clusters = nodefamilies(net)
    return LTRIP(clusters)
end
function LTRIP(clusters::Vector{Vector{T}}, net::HybridNetwork) where {T <: Integer}
    isfamilypreserving(clusters, net)[1] ||
    error("`clusters` is not family preserving with respect to `net`")
    for cl in clusters
        issorted(cl, rev=true) || sort!(cl, rev=true)
    end
    return LTRIP(clusters)
end

"""
    JoinGraphStructuring

Subtype of [`AbstractClusterGraphMethod`](@ref).

## Fieldnames

- `maxclustersize`: upper limit for cluster size.
  This value must be at least the size of the largest node family in
  the input phylogenetic network, normally 3 if the network is bicombining
  (each hybrid node has 2 parents, never more). See [`nodefamilies(net)`](@ref).

## Constructors

- `JoinGraphStructuring(maxclustersize, net)`:
   checks that the input `maxclustersize` is valid for `net`

## Algorithm, by Mateescu et al. (2010)

Requires:
- a user-specified maximum cluster size
- an elimination order for the nodes in the HybridNetwork, hopefully yielding
  a small induced-width, e.g. from a heuristic such as greedy min-fill
  (see [`triangulate_minfill!(graph)`](@ref)).

1. Each node in `net` labels a "bucket", and these buckets are ordered according
   to the elimination order, e.g. the highest-priority bucket is labelled by the
   first node in the elimination order.
2. Node families are assigned to buckets based on the highest-priority node they
   contain. For example, if {1,4,9} forms a node family and if node 4 is
   higher in the elimination order than nodes 1 or 9, then {1,4,9} gets assigned
   to bucket 4.
3. Node families are clusters (of nodes), and we refer to the clusters within a
   bucket as "minibuckets". Minibuckets within a bucket can be merged as long as
   the size of their union does not exceed the maximum cluster size allowed.
   Sometimes, this can be done in multiple ways.
4. Starting with the highest-priority bucket, we create new minibuckets by
   "marginalizing out" the bucket label from each existing minibucket
   (these are left unchanged in the process).
5. Each new minibucket is joined to its "originator" minibucket by an edge that
   is labeled by their intersection (i.e. the variables in the new minibucket).
   Each new minibucket is then reassigned to a new (and necessarily lower-priority)
   bucket based on its highest priority node. Merging can take place during
   reassignment as long as the the maximum cluster size is respected.
   The union of 2 minibuckets retains the edges of each of minibucket.
6. Steps 4 & 5 are carried out for each bucket in order of priority.
7. The resulting minibuckets in each bucket are then joined in a chain (there is
   some degree of freedom for how this can be done),
   where each edge is labelled by the bucket label.

## References

R. Mateescu, K. Kask, V.Gogate, and R. Dechter. Join-graph propagation algorithms.
*Journal of Artificial Intelligence Research*, 37:279-328, 2010
doi: [10.1613/jair.2842](https://doi.org/10.1613/jair.2842).
"""
struct JoinGraphStructuring <: AbstractClusterGraphMethod
    maxclustersize::Integer
end
function JoinGraphStructuring(maxclustersize::Integer, net::HybridNetwork)
    maxindegree = maximum(n -> length(getparents(n)), net.hybrid)
    maxclustersize ≥ (maxindegree + 1) ||
        error("maxclustersize $maxclustersize is smaller than the size of largest node family $(maxindegree+1).")
    return JoinGraphStructuring(maxclustersize)
end

"""
    Cliquetree

Subtype of [`AbstractClusterGraphMethod`](@ref).

## Algorithm

1. [`moralize`](@ref) the network (connect partners that share a child).
2. triangulate the resulting undirected graph using greedy min-fill,
   see [`triangulate_minfill!(graph)`](@ref).
3. extract the maximal cliques of the resulting chordal graph.
4. calculate the edge weight between each pair of maximal cliques as the size of
   their intersection
5. find a maximum-weight spanning tree
6. label the retained edges (in the spanning tree) by the intersection of
   the two cliques they connect.

## References

D. Koller and N. Friedman.
*Probabilistic graphical models: principles and techniques*.
MIT Press, 2009. ISBN 9780262013192.
"""
struct Cliquetree <: AbstractClusterGraphMethod end

"""
    clustergraph!(net, method)
    clustergraph( net, method)

Cluster graph `U` for an input network `net` and a `method` of cluster graph
construction. The following methods are supported:
- [`Bethe`](@ref)
- [`LTRIP`](@ref)
- [`JoinGraphStructuring`](@ref)
- [`Cliquetree`](@ref)

The first method pre-processes `net`, which may modify it in place,
see [`preprocessnet!`](@ref).
The second method assumes that `net` is already pre-processed.
"""
function clustergraph!(net::HybridNetwork, method::AbstractClusterGraphMethod)
    preprocessnet!(net)
    return clustergraph(net, method)
end

@doc (@doc clustergraph!) clustergraph
clustergraph(net::HybridNetwork, ::Bethe)       = betheclustergraph(net)
clustergraph(net::HybridNetwork, method::LTRIP) = ltripclustergraph(net, method)
clustergraph(net::HybridNetwork, method::JoinGraphStructuring) =
    joingraph(net, JoinGraphStructuring(getmaxclustersize(method), net))
function clustergraph(net::HybridNetwork, ::Cliquetree)
    g = moralize(net)
    triangulate_minfill!(g)
    return cliquetree(g)
end

"""
    betheclustergraph(net)

See [`Bethe`](@ref)
"""
function betheclustergraph(net::HybridNetwork)
    T = vgraph_eltype(net)
    clustergraph = init_clustergraph(T, :Bethe)

    node2cluster = Dict{T, Tuple{Symbol, Vector{Symbol}}}() # for joining clusters later
    node2code = Dict{T,T}() # node preorder index -> code of family(node) in cluster graph
    code = zero(T)
    prenodes = net.nodes_changed
    prenodes_names = [Symbol(n.name) for n in prenodes]
    # add a factor-cluster for each non-root node
    for noi in reverse(eachindex(prenodes)) # postorder: see fam(h) before fam(p) in case fam(p) ⊆ fam(h)
        n = prenodes[noi]
        vt = T(noi)
        o = sort!(indexin(getparents(n), prenodes), rev=true) # for postorder
        nodeind = pushfirst!(T.(o), vt)   # preorder indices of nodes in factor
        nodesym = prenodes_names[nodeind] # node symbol      of nodes in factor
        # (nodesym, nodeind) = factor-cluster data, its nodes listed in postorder
        length(nodeind) > 1 || continue # skip the root
        # if n's family ⊆ another family: would have in one of its children's
        isfamilysubset = false
        for ch in getchildren(n)
            ch_code = node2code[findfirst(x -> x===ch, prenodes)]
            if nodeind ⊆ clustergraph[label_for(clustergraph, ch_code)][2]
                isfamilysubset = true
                node2code[noi] = ch_code # family(n) assigned to its child's cluster
                break # do not check other children's families
            end
        end
        isfamilysubset && continue # to next node: do *not* create a new cluster in graph
        factorCname = Symbol(nodesym...) # factor-cluster name: label in metagraph
        code += one(T)
        node2code[noi] = code
        add_vertex!(clustergraph, factorCname, (nodesym, nodeind))
        for (nns, nni) in zip(nodesym, nodeind)
            if haskey(node2cluster, nni)
                push!(node2cluster[nni][2], factorCname)
            else node2cluster[nni] = (nns, [factorCname])
            end
        end
    end
    # add a variable-cluster for each non-leaf node, and its adjacent edges (sepsets)
    for ni in sort!(collect(keys(node2cluster)), rev=true) # add nodes in postorder
        ns, clusterlist = node2cluster[ni]
        length(clusterlist) > 1 || continue # skip leaves: in only 1 factor-cluster
        add_vertex!(clustergraph, ns, ([ns], [ni]))
        for lab in clusterlist # lab: factor-cluster name, for each factor that contains the node
            add_edge!(clustergraph, ns, lab, [ni]) # sepset: singleton {ni}
        end
    end
    return clustergraph
end

"""
    ltripclustergraph(net, method::LTRIP)

See [`LTRIP`](@ref)
"""
function ltripclustergraph(net::HybridNetwork, method::LTRIP)
    T = vgraph_eltype(net)
    clustg = init_clustergraph(T, :ltrip) # 'clustergraph' is a function already
    cg = MetaGraph(Graph{T}(0), # auxiliary graph to hold connection weights
        Symbol, # vertex label
        Tuple{Vector{Symbol}, Vector{T}}, # vertex data: nodes in cluster
        T, # edge data holds edge weight
        :connectionweights, # tag for the whole graph
        edge_data -> edge_data,
        zero(T)) # default weight

    node2cluster = Dict{T, Vector{T}}() # for joining clusters later
    clusters = getclusters(method)
    prenodes_names = [Symbol(n.name) for n in net.nodes_changed]
    # build nodes in clustg and in auxiliary cg
    for (code, nodeindlist) in enumerate(clusters) # nodeindlist assumed sorted, decreasing
        cdat = prenodes_names[nodeindlist]
        cname = Symbol(cdat...)
        # cluster data: (node labels, node preorder index), sorted in postorder
        add_vertex!(clustg, cname, (cdat, nodeindlist))
        add_vertex!(cg,     cname, (cdat, nodeindlist))
        for ni in nodeindlist
            if haskey(node2cluster, ni)
                push!(node2cluster[ni], T(code))
            else node2cluster[ni] = [T(code)]
            end
        end
        # add edges in auxiliary cg: to calculate intersection size only once
        for code2 in 1:(code-1)
            c2name = label_for(cg, code2)
            c2nodeindlist = clusters[code2]
            w = length(intersect(nodeindlist, c2nodeindlist))
            w > 0 && add_edge!(cg, cname, c2name, w)
        end
    end

    clustweight = Dict(lab => 0 for lab in labels(cg)) # clusters score in cg's subgraph sg
    for ni in sort!(collect(keys(node2cluster)), rev=true)
        # sepsets will be sorted by nodes' postorder
        clusterindlist = node2cluster[ni]
        # build subgraph sg of auxiliary cg, then adjust its edge weights
        sg, _ = induced_subgraph(cg, clusterindlist)
        ne(sg) > 0 || continue # e.g leaves: only 1 cluster, no edge to add
        maxw = maximum(sg[e...] for e in edge_labels(sg))
        for e in edge_labels(sg)
            if sg[e...] == maxw # if a sepset edge has maximum weigh, then
                clustweight[e[1]] += 1 # add 1 to each adjacent cluster's score
                clustweight[e[2]] += 1
            end
        end # separate loop below: adjust weights *after* testing for max weight
        for e in edge_labels(sg)
            sg[e...] += clustweight[e[1]] + clustweight[e[2]]
        end
        # reset clustweight before next cluster
        for cl in keys(clustweight) clustweight[cl]=0; end
        mst_edges = kruskal_mst(sg, minimize=false) # mst: maximum spanning tree
        # augment sepsets in clustg, based on edges in max spanning tree
        for e in mst_edges
            lab1 = label_for(sg, src(e))
            lab2 = label_for(sg, dst(e))
            if haskey(clustg, lab1, lab2) # if (lab1, lab2) is an edge
                clustg[lab1, lab2] = push!(clustg[lab1, lab2], ni) # augment sepset
            else # create the edge, initialize sepset
                add_edge!(clustg, lab1, lab2, [ni])
            end
        end
    end
    return clustg
end

"""
    joingraph(net, method::JoinGraphStructuring)

See [`JoinGraphStructuring`](@ref)
"""
function joingraph(net::HybridNetwork, method::JoinGraphStructuring)
    g = moralize(net)
    ordering = triangulate_minfill!(g) # node labels in elimination order
    T = vgraph_eltype(net)
    # preorder indices sorted in elimination order
    eliminationorder2preorder = [g[ns] for ns in ordering]

    #= steps 1-3: initialize `buckets` and their minibuckets
    i: elimination order for the node labeling the bucket
    buckets[i][1]: node symbol
    buckets[i][2]: dictionary minibucket size => minibucket(s)
    =#
    buckets = Dict{T, Tuple{Symbol, Dict{T, Vector{Vector{T}}}}}(
        i => (ns, Dict()) for (i, ns) in enumerate(ordering)
    )
    # node families (represented as vectors of preorder indices) = initial factors
    node2family = nodefamilies(net)
    maxclustersize = T(getmaxclustersize(method)) # size limit for minibuckets
    cg = init_clustergraph(T, :auxiliary)
    for nf in node2family
        # minibucket: node family, represented then sorted by elimination order
        mb = Vector{T}(indexin(nf, eliminationorder2preorder))
        sort!(mb)
        # first node has lowest elimination order: highest priority
        bi = mb[1] # assign `mb` to bucket corresponding to this first node
        di = buckets[bi][2] # dictionary minibucket size => minibucket(s)
        # merge `mb` with existing minibucket or add as new minibucket, depending on max size
        assign!(di, mb, maxclustersize)
    end
    # steps 4-6: marginalize the bucket label from each minibucket
    for i in eachindex(ordering)
        _, bd = buckets[i] # bucket symbol, bucket dictionary
        # @info "node $i, bd has $(sum(length(mb) for mb in values(bd))) minibuckets"
        bi = eliminationorder2preorder[i] # preorder index of bucket labeling node
        previous_mb_label = nothing
        for minibuckets in values(bd), mb in minibuckets
            # create cluster in `cg` corresponding to minibucket `mb`
            nodeindlist = eliminationorder2preorder[mb]
            o = sortperm(nodeindlist, rev=true)
            nodeindlist .= nodeindlist[o]
            vdat = ordering[mb][o]
            lab = Symbol(vdat...)
            add_vertex!(cg, lab, (vdat, nodeindlist)) # nothing happens if cg already has lab
            # chain minibuckets: sepset = bucket labeling node, whose preorder is bi
            isnothing(previous_mb_label) || # connect `mb` to chain of current bucket
                # sepset is [bi] even if minibuckets have a larger intersection
                add_edge!(cg, previous_mb_label, lab, [bi])
            previous_mb_label = lab # `mb` becomes new tail of "chain"
            # new minibucket: marginalize bucket labeling node bi
            mb_new = copy(mb)
            popfirst!(mb_new)
            isempty(mb_new) && continue # below: mb_new is not empty
            #= Assign to bucket labeled by highest priority element mb_new[1]:
               merge mb_new with an existing minibucket (mb2) to get mb1.
               - if possible: great. It may be that mb1=mb2, if mb_new is a subset of mb2.
               - else: mb1=mb_new and mb2 is empty.
            =#
            (mb1, mb2) = assign!(buckets[mb_new[1]][2], mb_new, maxclustersize)
            # create cluster corresponding to marginalized & merged minibucket
            nodeindlist1 = eliminationorder2preorder[mb1]
            o1 = sortperm(nodeindlist1, rev=true)
            nodeindlist1 .= nodeindlist1[o1]
            vdat1 = ordering[mb1][o1]
            lab1 = Symbol(vdat1...)
            add_vertex!(cg, lab1, (vdat1, nodeindlist1)) # nothing happens if cg already has lab1
            # connect mb to mb1 in cluster graph cg
            add_edge!(cg, lab, lab1, filter(ni -> ni != bi, nodeindlist))
            # if mb2 ⊂ mb1 strictly and if mb2 was already a cluster in cg
            # then contract mb1-mb2, leaving mb1 only
            if length(mb1) != length(mb2) # mb1 ≠ mb2 bc mb2 ⊆ mb1
                o2 = sortperm(eliminationorder2preorder[mb2], rev=true)
                vdat2 = ordering[mb2][o2]
                lab2 = Symbol(vdat2...)
                if haskey(cg, lab2) # then replace mb2 by mb1:
                    for labn in neighbor_labels(cg, lab2) # connect mb2's neibhbors to mb1
                        add_edge!(cg, lab1, labn, cg[lab2, labn])
                    end
                    delete!(cg, lab2) # delete mb2 from cg
                end
            end
        end
    end
    return cg
end

"""
    assign!(bucket, new_minibucket, max_minibucket_size)

Merge `new_minibucket` with one of the minibuckets contained in `bucket`
(in order of decreasing size) subject to the constraint that the resulting
minibucket does not exceed `max_minibucket_size`. If this is not possible,
then `new_minibucket` is added to `bucket` as a new minibucket.
The bucket should be represented as a dictionary:
minibucket_size => vector of minibuckets of that size,
where each minibucket is itself represented as a vector of indices.

Output:
- (`resulting_minibucket`, `minibucket_merged_into`) if a successful merge is found
- (`new_minibucket`, []) otherwise
"""
function assign!(bucket::Dict{T, Vector{Vector{T}}},
                 new::Vector{T}, maxsize::T) where {T <: Integer}
    for sz in sort(collect(keys(bucket)), rev=true) # favor merging with large minibuckets
        minibuckets = bucket[sz] # minibuckets of size `sz`
        for (i, mb) in enumerate(minibuckets)
            merged = sort(union(new, mb))
            mergedsz = length(merged)
            if mergedsz ≤ maxsize
                popat!(minibuckets, i) # remove minibucket being merged with
                isempty(minibuckets) && pop!(bucket, sz) # remove any empty size categories
                # insert result of merge into appropriate size category
                if haskey(bucket, mergedsz)
                    push!(bucket[mergedsz], merged)
                else
                    bucket[mergedsz] = [merged]
                end
                return (merged, mb)
            end
        end
    end
    # no merging: insert `new` to `bucket`
    sz = length(new)
    if haskey(bucket, sz)
        push!(bucket[sz], new)
    else
        bucket[sz] = [new]
    end
    return (new, T[])
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
    mg = init_clustergraph(T, :cliquetree)
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
    init_clustergraph(T::Type{<:Integer}, clustergraph_method::Symbol)

`MetaGraph` with an empty base graph (0 vertices, 0 edges),
meta-graph data `clustergraph_method`,
edge-weight function counting the length of the edge-data vector,
and the following types:
- vertex indices in the base graph: `T`
- vertex labels: `Symbol`
- vertex data: `Tuple{Vector{Symbol}, Vector{T}}`
  to hold information about the variables (nodes in phylogenetic network)
  in the cluster (vertex in cluster graph):
  node names as symbols, and node preorder index
- edge data: `Vector{T}` to hold information about the sepset:
  preorder index of nodes in the sepset.

See packages [MetaGraphsNext](https://juliagraphs.org/MetaGraphsNext.jl/dev/)
and [Graphs](https://juliagraphs.org/Graphs.jl/dev/).

The empty graph above is of type `Graphs.SimpleGraphs.SimpleGraph{T}`:
undirected, with vertex indices of type `T`. After addition of `n` vertices,
the vertex indices range from 1 to `n`, technically in `Base.OneTo{T}(n)`.
"""
function init_clustergraph(T::Type{<:Integer}, method::Symbol)
    clustergraph = MetaGraph(
        Graph{T}(0),
        Symbol, # vertex label
        Tuple{Vector{Symbol}, Vector{T}}, # vertex data: nodes in cluster
        Vector{T}, # edge data: nodes in sepset
        method, # tag for the whole graph
        edge_data -> T(length(edge_data)),
        zero(T)) # default weight
    return clustergraph
end
function get_nodesymbols2index(clustergraph::MetaGraph)
    Dict(ns => ni
        for l in labels(clustergraph)
        for (ns, ni) in zip(clustergraph[l][1], clustergraph[l][2]) )
end

"""
    spanningtree_clusterlist(clustergraph, root_index)
    spanningtree_clusterlist(clustergraph, nodevector_preordered)

Build the depth-first search spanning tree of the cluster graph, starting from
the cluster indexed `root_index` in the underlying simple graph;
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
    spanningtrees_clusterlist(clustergraph, nodevector_preordered)

Vector of spanning trees for `clustergraph`, that together cover all edges.
Each spanning tree is specified as a tuple of 4 vectors describing a
depth-first search traversal of the tree, starting from a cluster that contains
the network's root, as in [`spanningtree_clusterlist`](@ref).

The spanning trees are iteratively obtained using Kruskal's minimum-weight
spanning tree algorithm, with edge weights defined as the number of previous
trees covering the edge.
"""
function spanningtrees_clusterlist(cgraph::MetaGraph{T}, prenodes::Vector{PN.Node}) where T
    # graph with same clusters and edges, but different edge data & edge weights
    cg = MetaGraph(Graph{T}(0), Symbol, Tuple{Vector{Symbol}, Vector{T}},
        T, # edge data type: edge data = number of spanning trees containing the edge
        :edgeweights, # graph tag: hold edge weights to compute min spanning tree
        edge_data -> edge_data, # edge data holds edge weight
        typemax(T)) # default weight for absent edges
    # copy clusters (same order) and edges from cgraph, but set edge data to 0
    for l in labels(cgraph) add_vertex!(cg, l, cgraph[l]); end
    for (l1,l2) in edge_labels(cgraph) add_edge!(cg, l1, l2, 0); end
    # schedule: initialize vector of spanning trees
    sched = Tuple{Vector{Symbol}, Vector{Symbol}, Vector{T}, Vector{T}}[]
    # repeat until every edge has data > 0: used in ≥1 spanning tree
    while any(cg[l1,l2] == 0 for (l1,l2) in edge_labels(cg))
        mst_edges = kruskal_mst(cg) # vector of edges in min spanning tree
        sg, vmap = induced_subgraph(cg, mst_edges) # spanning tree as `metagraph`
        spt = spanningtree_clusterlist(sg, prenodes)
        spt[3] .= vmap[spt[3]] # code i in `sg` maps to code vmap[i] in `cg`
        spt[4] .= vmap[spt[4]]
        push!(sched, spt)
        # update edge data: +1 for each edge in spanning tree, to prioritize
        # unused (or rarely used) in future spanning trees
        for e in mst_edges
            parentlab = label_for(cg, src(e))
            childlab = label_for(cg, dst(e))
            cg[parentlab, childlab] += 1
        end
    end
    return sched
end

"""
    nodesubtree_clusterlist(clustergraph::MetaGraph, nodesymbol)

Spanning tree of the subgraph of `clustergraph` induced by the clusters and
sepsets that contain the node labelled `nodesymbol`, see [`nodesubtree`](@ref).
If `clustergraph` satisfies the generalized running-intersection property,
then this subgraph should be a tree anyway, but this is not assumed
(via extracting a spanning tree).

Output: tuple of 4 vectors describing a depth-first search traversal of the tree,
starting from a cluster containing the node of smallest preorder index, as
determined by [`default_rootcluster`](@ref).
Each element in this tuple is a vector: see [`spanningtree_clusterlist`](@ref).
"""
function nodesubtree_clusterlist(cgraph::MetaGraph, ns::Symbol)
    sg, vmap = nodesubtree(cgraph, ns)
    # to root sg: pick a cluster containing a node with smallest preorder index
    rootj = default_rootcluster(sg)
    spt = spanningtree_clusterlist(sg, rootj)
    # map cluster indices back to those for `cgraph`
    spt[3] .= vmap[spt[3]]
    spt[4] .= vmap[spt[4]]
    return spt
end

"""
    minimal_valid_schedule(clustergraph, clusterswithevidence)

Generate a minimal valid schedule of messages to be computed on a initialized
Bethe cluster graph, so that any schedule of messages following is valid.
Return the schedule as a tuple of four vectors: (`parent_labels`, `child_labels`,
`parent_indices`, `child_indices`) as in [`spanningtree_clusterlist`](@ref).
"""
function minimal_valid_schedule(cgraph::MetaGraph, wevidence::Vector{Symbol})
    !isempty(wevidence) || error("`wevidence` cannot be empty")
    #= `received` tracks clusters that have received evidence (through messages
    during initialization). Only clusters that have received evidence can
    transmit this (such clusters get added to `cansend`) to neighbor clusters
    through a message. =#
    received = Set{Symbol}(wevidence)
    cansend = copy(wevidence)
    T = typeof(cgraph[cansend[1]][2][1])
    childclust_j = T[] # child cluster indices
    parentclust_j = T[] # parent cluster indices
    childclust_lab = Symbol[] # child cluster labels
    parentclust_lab = Symbol[] # parent cluster labels
    while !isempty(cansend)
        #= For each cluster in `cansend`, send a message to any neighbors that
        have not received evidence (all such neighbors get added to `cansend`),
        then remove it from `cansend`. Since the cluster graph represented by
        `cgraph` is connected, all clusters will eventually be added to `cansend`
        and processed in order. Hence, this generates a minimal sequence of
        messages that can be computed, so that the updated cluster beliefs will
        be non-degenerate wrt any messages they are allowed to compute (i.e.
        any schedule of messages following is valid). =#
        cl = popfirst!(cansend) # remove node to be processed
        nb = neighbor_labels(cgraph, cl)
        for cl2 in nb
            if cl2 ∉ received
                push!(childclust_j, code_for(cgraph, cl2))
                push!(parentclust_j, code_for(cgraph, cl))
                push!(childclust_lab, cl2)
                push!(parentclust_lab, cl)
                push!(received, cl2) # `cl2` can no longer receive messages
                push!(cansend, cl2) # `cl2` can now send messages
            end
        end
    end
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

"""
    default_rootcluster(clustergraph)

Index of a cluster that contains the node with the smallest preorder index.
If multiple clusters contain that node, then one is chosen that *only* has
that node. If all clusters containing that node have more than 1 node,
then a cluster is chosen containing a node with the second-smallest-preorder index.

For cluster with label `:lab`, its property `clustergraph[:lab][2]`
should list the nodes in the cluster by their preorder index,
sorted in decreasingly order (so the smallest is at the end).
"""
function default_rootcluster(cgraph::MetaGraph)
    i0 = minimum(cgraph[lab][2][end] for lab in labels(cgraph)) # smallest preorder index
    rootscore = lab -> begin   # Inf if the cluster does not contain i0
        nodelabs = cgraph[lab][2]  # second smallest index otherwise
        (i0 ∈ nodelabs ?
            (length(nodelabs)==1 ? 0 : nodelabs[end-1]) :
            Inf)
    end
    rootj = argmin(rootscore(lab) for lab in labels(cgraph))
    return rootj
end
