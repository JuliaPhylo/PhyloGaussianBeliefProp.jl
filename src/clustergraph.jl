@inline function vgraph_eltype(net::HybridNetwork)
    nn = 2max(length(net.node), length(net.edge))
    (nn < typemax(Int8) ? Int8 : (nn < typemax(Int16) ? Int16 : Int))
end
@enum EdgeType etreetype=1 ehybridtype=2 moralizedtype=3 filltype=4

"""
    preprocessnet!(net::HybridNetwork, prefix="I")
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
    nodefamilies(net)

A dictionary of type `Dict{T, Vector{T}}` that maps each node's preorder index
to a vector containing its preorder index and those of its parents (i.e. the
preorder indices of the node family that it is the child in). Within each vector,
the indices are sorted in decreasing order so that the parents' indices come
after the child's.

**Warning**: assumes that `net` is preordered ([`preorder!`](@ref)).
"""
function nodefamilies(net::HybridNetwork)
    # preprocessnet!(net)
    T = vgraph_eltype(net)
    node2family = Dict{T, Vector{T}}()
    preordernames = [n.name for n in net.nodes_changed]
    for (code, n) in enumerate(net.nodes_changed)
        o = sort!(indexin([p.name for p in getparents(n)], preordernames),
            rev=true)
        node2family[code] = [code; o]
    end
    return node2family
end

"""
    isfamilypreserving(clusters, net)

Tuple:
1. A boolean: true (false) if `clusters` (each cluster is given as a vector of
   preorder indices) is (is not) family-preserving with respect to `net`, that
   is: if each node family (a node and all of its parents) in `net` is contained
   in at least 1 cluster in `clusters`.
2. A dictionary of type `Dict{T, BitVector}` that maps each family (represented
   by the preorder index for its child node) to a vector of booleans, where each
   boolean indicates if the family is included in the associated cluster or not.

**Warning**: assumes that `net` is preordered ([`preorder!`](@ref)).

See also [`nodefamilies`](@ref) to get node families.
"""
function isfamilypreserving(clusters::Vector{Vector{T}},
    net::HybridNetwork) where {T <: Integer}
    T1 = vgraph_eltype(net)
    isa(T, Type{T1}) || error("Supply `clusters` as Vector{Vector{$T1}} object")
    node2family = nodefamilies(net)
    family2cluster = Dict{T, BitVector}()
    ifp = true # is family-preserving
    for ni in keys(node2family)
        nf = node2family[ni]
        ch = nf[1] # index family by child node
        # mark which clusters (if any) each node family is contained in
        family2cluster[ch] = BitArray(nf ⊆ cl for cl in clusters)
        # check for potential violation only if family-preserving so far
        ifp && (ifp = any(family2cluster[ch]))
    end
    return (ifp, family2cluster)
end

"""
    check_runningintersection(clustergraph, net)

Vector of tuples: `[(t1::Symbol, t2::Bool), ...]`, where `t1` is a node label
and `t2` indicates if the subgraph of `clustergraph` induced by the clusters
containing that node is a tree.

**Warning**:
- Assumes that `net` has been preordered ([`preorder!`](@ref)).
- Does *not* check if `clustergraph` has been correctly constructed.

`clustergraph` satisfies the *running-intersection* (RI) property if for each
node in `net`, the subgraph induced by the clusters containing that node is a
tree (i.e. if `t2` is `true` for all nodes).
"""
function check_runningintersection(clustergraph::MetaGraph, net::HybridNetwork)
    cluster_properties = clustergraph.vertex_properties
    subgraphs = Tuple{Symbol, MetaGraph}[]
    for n in net.nodes_changed
        # vector of cluster symbols
        clusters_s = filter(cl -> Symbol(n.name) ∈ cluster_properties[cl][2][1],
            keys(cluster_properties))
        # vector of cluster indices
        clusters_i = [cluster_properties[cl][1] for cl in clusters_s]
        sg_n, _ = induced_subgraph(clustergraph, clusters_i)
        push!(subgraphs, (Symbol(n.name), sg_n))
    end
    res = Tuple{Symbol, Bool}[]
    for (i, (nl, sg)) in enumerate(subgraphs)
        # loop through edges of the induced subgraph `sg`, and delete any that
        # do not contain the variable of interest `nl`
        for e in collect(keys(sg.edge_data))
            cl1, cl2 = e[1], e[2]
            nodelabs1 = sg[cl1][1]
            nodelabs2 = sg[cl2][1]
            if (nl ∉ nodelabs1) & (nl ∉ nodelabs2)
                delete!(sg, cl1, cl2)
            end
        end
        push!(res, (nl, is_tree(sg))) # check if the subgraph remaining is a tree
    end
    return res
end

"""
    AbstractClusterGraphMethod

Abstract type for method of cluster graph construction.
"""
abstract type AbstractClusterGraphMethod end

getclusters(obj::AbstractClusterGraphMethod) =
    hasfield(typeof(obj), :clusters) ? obj.clusters : nothing
getmaxclustersize(obj::AbstractClusterGraphMethod) =
    hasfield(typeof(obj), :maxclustersize) ? obj.maxclustersize : nothing

"""
    Bethe

Subtype of [`AbstractClusterGraphMethod`](@ref).

## Algorithm:
A Bethe cluster graph (also known as factor graph) has:
- a factor-cluster for each node-family (except for the singleton containing the
root node) in the network, and
- a variable-cluster for each node in the network

Each variable-cluster is joined to the factor-clusters that contain it. The
sepset for each edge is the variable-cluster that it is incident to.

## References:
D. Koller and N. Friedman. *Probabilistic graphical models: principles and
techniques*. MIT Press, 2009. ISBN 9780262013192. doi:
[10.5555/1795555](https://doi.org/10.5555/1795555).
"""
struct Bethe <: AbstractClusterGraphMethod end

"""
    LTRIP{T<:Integer}

Subtype of [`AbstractClusterGraphMethod`](@ref).
A HybridNetwork and a valid LTRIP are passed to [`clustergraph!`](@ref) to
construct a cluster graph from the user-provided clusters based on the *Layered
Trees Running Intersection Property* algorithm of Streicher & du Preez (2017).

## Fieldnames:
- clusters: a vector of user-specified clusters that has to be family-preserving
([`isfamilypreserving`](@ref)) with respect to some HybridNetwork

## Constructors:
- `LTRIP!(net)`: uses minimal clusters ([`nodefamilies(net)`](@ref)), which
are guaranteed to be family-preserving
- `LTRIP!(clusters, net)`: checks if that clusters provided are family-preserving

## Algorithm:
1. For each node, LTRIP finds a spanning-tree for the clusters that contain that
node. The edges of this spanning-tree are all labelled with the singleton
containing that node.
    1. For the clusters containing a particular node, the edge weight for any
    pair of clusters is set to the size of their intersection.
    2. The max-weight edges are identified, then the weight of each edge is
    increased by the number of max-weight edges that either of its endpoints
    adjacent to.
    3. A maximum spanning-tree based on the set of augmented edge weights is
    returned.
2. The spanning-trees for each node are layered on one another so that the
sepset for any edge is the union of its labels across the different
spanning-trees.

## References:
S. Streicher and J. du Preez. Graph Coloring: Comparing Cluster Graphs to Factor
Graphs. In *Proceedings of the ACM Multimedia 2017 Workshop on South African
Academic Participation, pages 35-42, 2017. doi:
[10.1145/3132711.3132717](https://doi.org/10.1145/3132711.3132717).
"""
struct LTRIP{T<:Integer} <: AbstractClusterGraphMethod
    clusters::Vector{Vector{T}}
end
function LTRIP!(net::HybridNetwork)
    node2family = nodefamilies(net)
    clusters = collect(values(node2family))
    return LTRIP(clusters)
end
function LTRIP!(clusters::Vector{Vector{T}}, net::HybridNetwork) where {T <: Integer}
    isfamilypreserving(clusters, net)[1] ||
    error("`clusters` is not family preserving with respect to `net`")
    return LTRIP(clusters)
end

"""
    JoinGraphStr

Subtype of [`AbstractClusterGraphMethod`](@ref).

*Join-graph structuring* algorithm of Mateescu et al. (2010).

## Fieldnames:
- maxclustersize: upper limit for cluster size that must be ≥ size of the
largest minimal cluster ([`nodefamilies(net)`](@ref))

## Constructors:
- `JoinGraphStr(maxclustersize, net)`: checks that max cluster size provided is
valid

## Algorithm:
Requires:
- user-specified max cluster size
- an elimination order for the nodes in the HybridNetwork that is expected to
yield a small induced-width (e.g. from a heuristic such as greedy min-fill
([`triangulate_minfill!(graph)`](@ref)))

1. Each node in the HybridNetwork labels a "bucket", and these buckets are ordered
according to the elimination order (e.g. the highest-priority bucket is labelled
by the first node in the elimination order).
2. Node families are assigned to buckets based on the highest-priority node they
contain (e.g. if {1, 4, 9} forms a node family and node 4 is higher in the
elimination order than nodes 1 or 9, then {1, 4, 9} gets assigned to bucket 4).
3. Node families are clusters (of nodes), and we refer to the clusters within a
bucket as "minibuckets". Minibuckets within a bucket can be merged as long as
the size of their union does not exceed the max cluster size provided (there is
some degree of freedom for how this can be done).
4. Starting with the highest-priority bucket, we create new minibuckets by
"marginalizing out" the bucket label from each existing minibucket (these are
left unchanged in the process).
5. Each new minibucket is joined to its "originator" minibucket by an edge that
is labeled by their intersection (i.e. the variables in the new minibucket).
Each new minibucket is then reassigned to a new (and necessarily lower-priority)
bucket based on its highest priority node. Merging can take place during
reassignment as long as the the max cluster size is respected. The union of 2
minibuckets retains the edges of each of minibucket.
6. Steps 4 & 5 are carried out for each bucket in order of priority.
7. The resulting minibuckets in each bucket are then joined in a chain (there is
some degree of freedom for how this can be done), where each edge is labelled by
the bucket label.

## References:
R. Mateescu, K. Kask, V.Gogate, and R. Dechter. Join-graph propagation algorithms.
*Journal of Artificial Intelligence Research*, 37:279-328, 2010. doi:
[10.1613/jair.2842](https://doi.org/10.1613/jair.2842).
"""
struct JoinGraphStr <: AbstractClusterGraphMethod
    maxclustersize::Integer
end
function JoinGraphStr(maxclustersize::Integer, net::HybridNetwork)
    maxindegree = maximum(n -> length(getparents(n)), net.node)
    maxclustersize ≥ (maxindegree + 1) || error("`maxclustersize` is smaller
    than size of largest minimal cluster")
    return JoinGraphStr(maxclustersize)
end

"""
    Cliquetree

## Algorithm:

1. Moralizes ([`moralize`](@ref)) the HybridNetwork.
2. Triangulates the resulting undirected graph using greedy min-fill
([`triangulate_minfill!(graph)`](@ref)).
3. Extracts the maximal cliques of the resulting chordal graph.
4. Defines the edge weight between each pair of maximal cliques as the size of
their intersection, then finds a maximum spanning tree. Edges between cliques
are labelled by their intersection.

## References:
D. Koller and N. Friedman. *Probabilistic graphical models: principles and
techniques*. MIT Press, 2009. ISBN 9780262013192. doi:
[10.5555/1795555](https://doi.org/10.5555/1795555).
"""
struct Cliquetree <: AbstractClusterGraphMethod end

"""
    clustergraph!(net, method)

Cluster graph `U` for an input network `net`, a `method` of cluster graph
construction.

The following methods for cluster graph construction are supported:
    - [`Bethe`](@ref)
    - [`LTRIP`](@ref)
    - [`JoinGraphStr`](@ref)
    - [`Cliquetree`](@ref)
"""
function clustergraph!(net::HybridNetwork, method::AbstractClusterGraphMethod)
    preprocessnet!(net)
    return clustergraph(net, method)
end

clustergraph(net::HybridNetwork, ::Bethe) = betheclustergraph(net)
clustergraph(net::HybridNetwork, method::LTRIP) = ltripclustergraph(net, method)

function clustergraph(net::HybridNetwork, method::JoinGraphStr)
    joingraph(net, JoinGraphStr(getmaxclustersize(method), net))
end

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
    preordernames = [n.name for n in net.nodes_changed]
    # iterate through network nodes in preorder
    for (code, n) in enumerate(net.nodes_changed)
        ns = Symbol(n.name)
        vt = T(code)
        o = sort!(indexin([p.name for p in getparents(n)], preordernames), rev=true)
        # vdat and nodeindlist are sorted in postorder
        vdat = [ns; [Symbol(n.name) for n in net.nodes_changed[o]]] # node symbols
        nodeindlist = [vt; Vector{T}(o)] # preorder indices
        if length(nodeindlist) > 1 # non-root node of graph
            # cluster data: (node labels, node preorder index), sorted in postorder
            # add factor clusters
            add_vertex!(clustergraph, Symbol(vdat...), (vdat, nodeindlist))
            for ni in nodeindlist
                if haskey(node2cluster, ni)
                    push!(node2cluster[ni][2], Symbol(vdat...))
                else node2cluster[ni] = (Symbol(preordernames[ni]), [Symbol(vdat...)])
                end
            end
        end
    end

    for ni in sort!(collect(keys(node2cluster)), rev=true)
        # sepsets will be sorted by nodes' postorder
        ns, clusterlist = node2cluster[ni]
        if length(clusterlist) > 1
            # add variable cluster only if there are > 1 clusters with this node
            add_vertex!(clustergraph, ns, ([ns], [ni]))
            # connect variable cluster to factor clusters that contain this node
            for lab in clusterlist
                add_edge!(clustergraph, ns, lab, [ni])
            end
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
    clustergraph = init_clustergraph(T, :ltrip)
    
    # only used for ltrip, so doesn't count as code duplication
    # auxiliary metagraph 
    cg = MetaGraph(Graph{T}(0),
        Symbol, # vertex label
        Tuple{Vector{Symbol}, Vector{T}}, # vertex data: nodes in cluster
        T, # edge data holds edge weight
        :auxiliary, # tag for the whole graph
        edge_data -> edge_data,
        zero(T)) # default weight

    node2cluster = Dict{T, Vector{T}}() # for joining clusters later
    clusters = getclusters(method)
    for (code, nodeindlist) in enumerate(clusters)
        o = sortperm(nodeindlist, rev=true) # order to list nodes in postorder
        nodeindlist .= nodeindlist[o] # preorder indices
        vdat = [Symbol(n.name) for n in net.nodes_changed[nodeindlist]]
        # cluster data: (node labels, node preorder index), sorted in postorder
        add_vertex!(clustergraph, Symbol(vdat...), (vdat, nodeindlist))
        add_vertex!(cg, Symbol(vdat...), (vdat, nodeindlist))
        for ni in nodeindlist
            if haskey(node2cluster, ni)
                push!(node2cluster[ni], T(code))
            else node2cluster[ni] = [T(code)]
            end
        end
    end

     # compute edge weights using auxiliary metagraph
     for ni in sort!(collect(keys(node2cluster)), rev=true)
        # sepsets will be sorted by nodes' postorder
        clusterindlist = node2cluster[ni]
        sg, _ = induced_subgraph(cg, clusterindlist)
        node2edge = Dict{Symbol, Vector{Symbol}}() # for updating edge weights later
        topscoring = Symbol[] # track strongly-connected clusters in `sg`
        topscore = 0 # track top score for cluster "connectivity"
        for (i1, cl1) in enumerate(clusterindlist)
            lab1 = label_for(clustergraph, cl1)
            maxw = 0 # track the max no. of elements `cl1` shares with its neighbors
            for i2 in 1:(i1-1)
                cl2 = clusterindlist[i2]
                lab2 = label_for(clustergraph, cl2)
                w = length(intersect(sg[lab1], sg[lab2]))
                maxw = (w > maxw) ? w : maxw
                add_edge!(sg, lab1, lab2, w)
            end
            # mark clusters that are incident to a max-weight edge
            if maxw > topscore # mark cluster `cl1` and unmark all others
                topscoring, topscore = [lab1], maxw
            elseif maxw == topscore # mark cluster `cl1`
                push!(topscoring, lab1)
            end
        end
        # update edge weights
        for cl in topscoring # topscoring nodes have incident max-weight edges
            neighborlabs = neighbor_labels(sg, cl)
            # count no. of incident max-weight edges and add this no. to weights
            # for all incident edges
            Δw = length([sg[cl, ncl] == topscore for ncl in neighborlabs])
            for ncl in neighborlabs
                sg[cl, ncl] += Δw
            end
        end
        mst_edges = kruskal_mst(sg, minimize=false)

        for e in mst_edges
            lab1 = label_for(sg, src(e))
            lab2 = label_for(sg, dst(e))
            if haskey(clustergraph, lab1, lab2) # if has edge {lab1, lab2}
                clustergraph[lab1, lab2] = push!(clustergraph[lab1, lab2], ni)
            else
                add_edge!(clustergraph, lab1, lab2, [ni])
            end
        end
    end
    return clustergraph
end

"""
    joingraph(net, method)

See [`JoinGraphStr`](@ref)

1. Sort node families into buckets. These are the initial clusters for each
bucket.
2. Loop through buckets
    i. For each bucket, partition clusters into minibuckets (subject to
    maxclustersize) and connect minibuckets to one another. The sepset for this
    connection is the bucket variable. There is some degree of freedom in how
    clusters are partitioned into minibuckets.
    ii. For each minibucket in a bucket, create a subminibucket (by
    marginalizing out the bucket variable) and assign it to the relevant bucket.
    Connect each minibucket to its corresponding subminibucket. The sepset for
    this connection is their full intersection (all nodes except for the bucket
    variable).
"""
function joingraph(net::HybridNetwork, method::JoinGraphStr)
    g = moralize(net)
    ordering = triangulate_minfill!(g) # node labels in elimination order
    T = vgraph_eltype(net)
    # preorder indices sorted in elimination order
    eliminationorder2preorder = [g[ns] for ns in ordering]

    # initialize `buckets`
    buckets = Dict{T, Tuple{Symbol, Dict{T, Vector{Vector{T}}}}}(
        i => (ns, Dict()) for (i, ns) in enumerate(ordering)
    )
    # node families (represented as vectors of preorder indices) correspond to
    # initial factors
    node2family = nodefamilies(net)
    maxclustersize = T(getmaxclustersize(method)) # size limit for minibuckets
    cg = init_clustergraph(T, :auxiliary)
    for ni in keys(node2family)
        # minibucket: node family specified in elimination order indices and
        # sorted in elimination order
        mb = Vector{T}(indexin(node2family[ni], eliminationorder2preorder))
        sort!(mb)
        # first element in minibucket always has highest priority wrt elimination
        # order. assign `mb` to bucket corresponding to that element
        bi = mb[1]
        # dictionary mapping minibucket size to minibucket(s) within bucket
        di = buckets[bi][2]
        # merge `mb` (if within size limits) with existing minibuckets within
        # bucket, otherwise just add to bucket
        assign!(di, mb, maxclustersize)
    end

    for i in 1:length(ordering)
        bs, bd = buckets[i] # bucket symbol, bucket dictionary
        bi = eliminationorder2preorder[i] # preorder index for `bs`
        for minibuckets in values(bd)
            # connect minibuckets in a "chain", where the sepset is the bucket
            # variable `bs`, which corresponds to preorder index `bi`
            for (j, mb) in enumerate(minibuckets)
                nodeindlist = eliminationorder2preorder[mb]
                o = sortperm(nodeindlist, rev=true)
                nodeindlist .= nodeindlist[o]
                vdat = ordering[mb][o]
                lab = Symbol(vdat...)
                # add cluster corresponding to `mb` to `cg`
                add_vertex!(cg, lab, (vdat, nodeindlist))
                if j > 1
                    # connect `mb` to current tail of "chain"
                    add_edge!(cg, prev, lab, bi)
                end
                prev = lab # `mb` becomes new tail of "chain"
                mb_new = copy(mb) # initialize new minibucket
                popfirst!(mb_new) # remove ("marginalize") bucket variable
                if !isempty(mb_new)
                    #= Assign to bucket corresponding to highest priority element
                    and check if `mb_new` can be merged with an existing minibucket
                    (starting with the smallest) in the assigned bucket. If so,
                    complete the merge and update the size => minibucket
                    dictionary (no. of minibuckets decreases by 1) =#

                    #= If `mb2` is empty, then `mb1` == `mb_new`. Otherwise, it
                    # is merged with `mb2` (an existing minibucket) to produce
                    `mb1`. It may be that `mb2` == `mb1` =#
                    (mb1, mb2) = assign!(buckets[mb_new[1]][2], mb_new,
                        maxclustersize)
                    nodeindlist1 = eliminationorder2preorder[mb1]
                    o1 = sortperm(nodeindlist1, rev=true)
                    nodeindlist1 .= nodeindlist1[o1]
                    vdat1 = ordering[mb1][o1]
                    lab1 = Symbol(vdat1...)
                    # add new cluster for `mb1`
                    add_vertex!(cg, lab1, (vdat1, nodeindlist1))
                    add_edge!(cg, lab, lab1,
                        filter(ni -> ni != bi, nodeindlist))
                    if length(mb1) != length(mb2) # mb1 != mb2
                        # cluster for `mb2` is replaced by new cluster for `mb1`
                        o2 = sortperm(eliminationorder2preorder[mb2], rev=true)
                        vdat2 = ordering[mb2][o2]
                        lab2 = Symbol(vdat2...)
                        if haskey(cg, lab2)
                            # connect edges into `mb2` to `mb1`
                            for labn in neighbor_labels(cg, lab2)
                                add_edge!(cg, lab1, labn,
                                    cg[lab2, labn])
                            end
                            delete!(cg, lab2) # delete `mb2`
                        end
                    end
                end
            end
        end
    end

    # copy `cg` onto `clustergraph`. Though `cg` stores the correct cluster and
    # sepset information, due to alternating edge/vertex additions in its
    # construction, this messes with edge (and consequently edge metadata) access 
    # fixit: rewrite in a more efficient and stable way
    clustergraph = init_clustergraph(T, :jgstr)
    for lab in values(cg.vertex_labels)
        add_vertex!(clustergraph, lab, cg[lab])
    end
    edges = cg.edge_data
    for (lab1, lab2) in keys(edges)
        add_edge!(clustergraph, lab1, lab2, edges[(lab1, lab2)])
    end
    return clustergraph
end
"""
    assign!(bucket, new_minibucket, max_minibucket_size)

Attempts to merge `new_minibucket` with one of the minibuckets contained in
`bucket` (in order of increasing size), subject to the constraint that the
resulting minibucket does not exceed `max_minibucket_size`. If a successful
merge is found, then (`resulting_minibucket`, `minibucket_merged_into`) is
returned. Otherwise, (`new_minibucket`, []) is returned.
"""
function assign!(bucket::Dict{T, Vector{Vector{T}}},
    new::Vector{T}, maxsize::T) where {T <: Integer}
    if !isempty(bucket) # there exist potential minibuckets to merge into
        # loop through different size collections of minibuckets, in order of
        # decreasing size
        for sz in sort(collect(keys(bucket)), rev=true)
            minibuckets = bucket[sz] # minibuckets with size `sz`
            for (i, mb) in enumerate(minibuckets)
                merged = sort(union(new, mb))
                mergedsz = length(merged)
                if mergedsz ≤ maxsize
                    popat!(minibuckets, i) # remove minibucket being merged with
                    if isempty(minibuckets) # remove any empty size categories
                        pop!(bucket, sz)
                    end
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
    end
    # `new` cannot be merged with any existing minibucket within `bucket`, then
    # insert it whole into the appropriate size collection in `bucket`
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
    init_clustergraph(vertex_type, clustergraph_method)

Construct empty `MetaGraph` based on an empty `Graph`. Metadata types are
initialized as follows:
    - `label_type::Symbol`
    - `vertex_data_type::Tuple{Vector{Symbol}, Vector{T}}`
    - `edge_data_type::Vector{T}`
Metadata for `Graph` as a whole is `method::Symbol`.

Note: const Graph = Graphs.SimpleGraphs.SimpleGraph
Note: SimpleGraph{T}(n=0) constructs an empty SimpleGraph{T} with n vertices and
0 edges. If not specified, the element type `T` is the type of `n`.
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
    sub_spanningtree_clusterlist(clustergraph, nodesymbol)

Extract the subgraph (of `clustergraph`) induced by the clusters that contain
the node labelled `nodesymbol` and build a depth-first search spanning tree of
this subgraph starting from an arbitrary cluster. Return the spanning tree as a
schedule of edges expressed as a tuple of four vectors: (`parent_labels`,
`child_labels`, `parent_indices`, `child_indices`), as in
[`spanningtree_clusterlist`](@ref).

Each edge of `clustergraph` is contained in ≥1 such (subgraph)spanning trees, so
iterating over all nodes generates a sequence of schedules that collectively
covers all edges of `clustergraph`.
"""
function sub_spanningtree_clusterlist(cgraph::MetaGraph, ns::Symbol)
    cluster_properties = cgraph.vertex_properties
    # labels for clusters that contain node `ns`
    clusters_s = filter(cl -> ns ∈ cgraph[cl][1], collect(labels(cgraph)))
    # indices for clusters that contain node `ns`
    clusters_i = [code_for(cgraph, cl) for cl in clusters_s]
    sg, vmap = induced_subgraph(cgraph, clusters_i)
    # pick any node to be the root: rootj can be any value between 1 and
    # length(cg.vertex_labels)
    rootj = 1 # fixit: fix to 1 for now
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
