"""
    ClusterGraphBelief(belief_vector)

Structure to hold a vector of beliefs, with cluster beliefs coming first and
sepset beliefs coming last. Fields:
- `belief`: vector of beliefs
- `nclusters`: number of clusters
- `cdict`: dictionary to get the index of a cluster belief from its node labels
- `sdict`: dictionary to get the index of a sepset belief from the labels of
   its two incident clusters.

Assumptions:
- For a cluster belief, the cluster's nodes are stored in the belief's `metadata`.
- For a sepset belief, its incident clusters' nodes are in the belief's metadata.
"""
struct ClusterGraphBelief{T<:AbstractBelief}
    "vector of beliefs, cluster beliefs first and sepset beliefs last"
    belief::Vector{T}
    "number of clusters"
    nclusters::Int
    "dictionary: cluster label => cluster index"
    cdict::Dict{Symbol,Int}
    "dictionary: cluster neighbor labels => sepset index"
    sdict::Dict{Set{Symbol},Int}
end
nbeliefs(obj::ClusterGraphBelief)  = length(obj.belief)
nclusters(obj::ClusterGraphBelief) = obj.nclusters
nsepsets(obj::ClusterGraphBelief)  = nbeliefs(obj) - nclusters(obj)
function Base.show(io::IO, b::ClusterGraphBelief)
    disp = "beliefs for $(nclusters(b)) clusters and $(nsepsets(b)) sepsets.\nclusters:\n"
    for (k,v) in b.cdict disp *= "  $(rpad(k,10)) => $v\n";end
    disp *= "sepsets:\n"
    for (k,v) in b.sdict disp *= "  $(rpad(join(k,", "),20)) => $v\n";end
    print(io, disp)
end

sepsetindex(c1, c2, obj::ClusterGraphBelief) = sepsetindex(c1, c2, obj.sdict)
function sepsetindex(clustlabel1, clustlabel2, sepsetdict)
    sepsetdict[Set((clustlabel1, clustlabel2))]
end

integratebelief!(obj::ClusterGraphBelief, rootindex) = integratebelief!(obj.belief[rootindex])

function ClusterGraphBelief(beliefs)
    i = findfirst(b -> b.type == bsepsettype, beliefs)
    nc = (isnothing(i) ? length(beliefs) : i-1)
    all(beliefs[i].type == bclustertype for i in 1:nc) ||
        error("clusters are not consecutive")
    all(beliefs[i].type == bsepsettype for i in (nc+1):length(beliefs)) ||
        error("sepsets are not consecutive")
    cdict = get_clusterindexdictionary(beliefs, nc)
    sdict = get_sepsetindexdictionary(beliefs, nc)
    return ClusterGraphBelief{eltype(beliefs)}(beliefs,nc,cdict,sdict)
end
function get_clusterindexdictionary(beliefs, nclusters)
    Dict(beliefs[j].metadata => j for j in 1:nclusters)
end
function get_sepsetindexdictionary(beliefs, nclusters)
    Dict(Set(beliefs[j].metadata) => j for j in (nclusters+1):length(beliefs))
end


"""
    calibrate!(beliefs::ClusterGraphBelief, clustergraph,
              nodevector_preordered, niterations=10)

fixit
`niterations` is ignored and set to 1 if the cluster graph is a clique tree.
"""
function calibrate!(beliefs::ClusterGraphBelief, cgraph, prenodes::Vector{PN.Node},
                    niter=10)
    niter = (cgraph.graph_data == :cliquetree || is_tree(cgraph) ? 1 : niter)
    for _ in 1:niter
        spt = spanningtree_clusterlist(cgraph, prenodes)
        # fixit: pick a *random* spanning tree instead, for a general cluster graph
        propagate_1treetraversal_postorder!(beliefs, spt)
        propagate_1treetraversal_preorder!(beliefs, spt)
    end
end

"""
    propagate_1treetraversal!(beliefs::ClusterGraphBelief, spanningtree,
                              postorder=true::Bool)

Messages are propagated from the tips to the root of the tree by default,
or from the root to the tips if `postorder` is false.

All nodes (resp. edges) in the `spanningtree` should correspond to clusters
(resp. sepsets) in `beliefs`: labels and indices in the spanning tree information
should correspond to indices in `beliefs`.
This condition holds if beliefs are produced on a given cluster graph and if the
tree is produced by [`spanningtree_clusterlist`](@ref) on the same graph.

fixit API, to:
do postorder only, to get loglikelihood at the root *without* the conditional
distribution at all nodes.
No need to recalculate the spanning tree for different parameter values.
get the log-likelihood
"""
function propagate_1treetraversal_postorder!(beliefs, spt)
    pa_lab, ch_lab, pa_j, ch_j = spt
    b = beliefs.belief
    # (parent <- sepset <- child) in postorder
    for i in reverse(1:length(pa_lab))
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        propagate_belief!(b[pa_j[i]], b[ss_j], b[ch_j[i]])
    end
end
function propagate_1treetraversal_postorder!(beliefs::ClusterGraphBelief, cgraph, prenodes::Vector{PN.Node})
    # cgraph.graph_data == :cliquetree || error("the graph is not a clique tree")
    spt = spanningtree_clusterlist(cgraph, prenodes)
    propagate_1treetraversal_postorder!(beliefs, spt)
end

function propagate_1treetraversal_preorder!(beliefs, spt)
    pa_lab, ch_lab, pa_j, ch_j = spt
    b = beliefs.belief
    # (child <- sepset <- parent) in preorder
    for i in 1:length(pa_lab)
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        propagate_belief!(b[ch_j[i]], b[ss_j], b[pa_j[i]])
    end
end
