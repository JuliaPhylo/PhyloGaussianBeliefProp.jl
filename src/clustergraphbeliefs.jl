"""
    ClusterGraphBelief{B<:Belief, F<:FamilyFactor, M<:MessageResidual}

Structure to hold a vector of beliefs, with cluster beliefs coming first and
sepset beliefs coming last. Fields:
- `belief`: vector of beliefs
- `factor`: vector of initial cluster beliefs after factor assignment
- `nclusters`: number of clusters
- `cdict`: dictionary to get the index of a cluster belief from its node labels
- `sdict`: dictionary to get the index of a sepset belief from the labels of
   its two incident clusters.
- `messageresidual`: dictionary to log information about sepset messages,
  which can be used to track calibration or help adaptive scheduling with
  residual BP. See [`MessageResidual`](@ref).
  The keys of `messageresidual` are tuples of cluster labels, similarly to a
  sepset's metadata. Each edge in the cluster graph has 2 messages corresponding
  to the 2 directions in which a message can be passed, with keys:
  `(label1, label2)` and `(label2, label1)`.
  The cluster receiving the message is the first label in the tuple,
  and the sending cluster is the second.

Assumptions:
- For a cluster belief, the cluster's nodes are stored in the belief's `metadata`.
- For a sepset belief, its incident clusters' nodes are in the belief's metadata.
"""
struct ClusterGraphBelief{B<:Belief, F<:FamilyFactor, M<:MessageResidual}
    "vector of beliefs, cluster beliefs first and sepset beliefs last"
    belief::Vector{B}
    """vector of initial factors from the graphical model, one per cluster.
    Each node family defines the conditional probability of the node
    conditional to its parent, and the root has its prior probability.
    Each such density is assigned to 1 cluster.
    A cluster belief can be assigned 0, 1 or more such density.
    Cluster beliefs are modified during belief propagation, but factors are not.
    They are useful to approximate the likelihood by the factored energy."""
    factor::Vector{F}
    "number of clusters"
    nclusters::Int
    "dictionary: cluster label => cluster index"
    cdict::Dict{Symbol,Int}
    "dictionary: cluster neighbor labels => sepset index"
    sdict::Dict{Set{Symbol},Int}
    "dictionary: message labels (cluster_to, cluster_from) => residual information"
    messageresidual::Dict{Tuple{Symbol,Symbol}, M}
end
nbeliefs(obj::ClusterGraphBelief) = length(obj.belief)
nclusters(obj::ClusterGraphBelief) = obj.nclusters
nsepsets(obj::ClusterGraphBelief) = nbeliefs(obj) - nclusters(obj)
function Base.show(io::IO, b::ClusterGraphBelief)
    disp = "beliefs for $(nclusters(b)) clusters and $(nsepsets(b)) sepsets.\nclusters:\n"
    for (k, v) in b.cdict
        disp *= "  $(rpad(k,10)) => $v\n"
    end
    disp *= "sepsets:\n"
    for (k, v) in b.sdict
        disp *= "  $(rpad(join(k,", "),20)) => $v\n"
    end
    print(io, disp)
end

clusterindex(c, obj::ClusterGraphBelief) = clusterindex(c, obj.cdict)
function clusterindex(clusterlabel, clusterdict)
    clusterdict[clusterlabel]
end
sepsetindex(c1, c2, obj::ClusterGraphBelief) = sepsetindex(c1, c2, obj.sdict)
function sepsetindex(clustlabel1, clustlabel2, sepsetdict)
    sepsetdict[Set((clustlabel1, clustlabel2))]
end

"""
    ClusterGraphBelief(belief_vector::Vector{B})

Constructor of a `ClusterGraphBelief` with belief `belief_vector` and all other
fields constructed accordingly. New memory is allocated for these other fields,
e.g. for factors (with data copied from cluster beliefs) and message residuals
(with data initialized to 0 but of size matching that from sepset beliefs)

To construct the input vector of beliefs, see [`init_beliefs_allocate`](@ref)
and [`init_beliefs_assignfactors!`](@ref)
"""
function ClusterGraphBelief(beliefs::Vector{B}) where B<:Belief
    i = findfirst(b -> b.type == bsepsettype, beliefs)
    nc = (isnothing(i) ? length(beliefs) : i - 1)
    all(beliefs[i].type == bclustertype for i in 1:nc) ||
        error("clusters are not consecutive")
    all(beliefs[i].type == bsepsettype for i in (nc+1):length(beliefs)) ||
        error("sepsets are not consecutive")
    cdict = get_clusterindexdictionary(beliefs, nc)
    sdict = get_sepsetindexdictionary(beliefs, nc)
    mr = init_messageresidual_allocate(beliefs, nc)
    factors = init_factors_allocate(beliefs, nc)
    return ClusterGraphBelief{B,eltype(factors),valtype(mr)}(beliefs,factors,nc,cdict,sdict,mr)
end

function get_clusterindexdictionary(beliefs, nclusters)
    Dict(beliefs[j].metadata => j for j in 1:nclusters)
end
function get_sepsetindexdictionary(beliefs, nclusters)
    Dict(Set(beliefs[j].metadata) => j for j in (nclusters+1):length(beliefs))
end

"""
    init_beliefs_reset!(beliefs::ClusterGraphBelief)

Reset
- cluster beliefs to existing factors
- sepset beliefs to h=0, J=0, g=0

fixit: is this ever used?
fixit: also reset message residuals to 0 and their flags to false?
fixit: rename? init_clustergraphbeliefs_reset! or init_beliefs_fromfactors! ?
"""
function init_beliefs_reset!(beliefs::ClusterGraphBelief)
    nc, nb = nclusters(beliefs), length(beliefs.belief)
    b, f = beliefs.belief, beliefs.factor
    for i in 1:nc
        b[i].h   .= f[i].h
        b[i].J   .= f[i].J
        b[i].g[1] = f[i].g[1]
    end
    for i in (nc+1):nb
        b[i].h   .= 0.0
        b[i].J   .= 0.0
        b[i].g[1] = 0.0
    end
end

"""
    iscalibrated_residnorm(beliefs::ClusterGraphBelief)
    iscalibrated_kl(beliefs::ClusterGraphBelief)

True if all edges in the cluster graph have calibrated messages in both directions,
in that their latest message residuals have norm close to 0 (`residnorm`)
or KL divergence close to 0 between the message received and prior sepset belief.
False if not all edges have calibrated messages.

This condition is sufficient but not necessary for calibration.

Calibration was determined for each individual message residual by
[`iscalibrated_residnorm!`](@ref) and [`iscalibrated_kl!`](@ref) using some
tolerance value.
"""
iscalibrated_residnorm(cb::ClusterGraphBelief) =
    all(x -> iscalibrated_residnorm(x), values(cb.messageresidual))

iscalibrated_kl(cb::ClusterGraphBelief) =
    all(x -> iscalibrated_kl(x), values(cb.messageresidual))

"""
    integratebelief!(obj::ClusterGraphBelief, beliefindex)
    integratebelief!(obj::ClusterGraphBelief)
    integratebelief!(obj::ClusterGraphBelief, clustergraph, nodevector_preordered)

`(μ,g)` from fully integrating the object belief indexed `beliefindex`. This
belief is modified, with its `belief.μ`'s values updated to those in `μ`.

The second method uses the first sepset containing a single node. This is valid
if the beliefs are fully calibrated (including a pre-order traversal), but
invalid otherwise.
The third method uses the default cluster containing the root,
see [`default_rootcluster`](@ref). This is valid if the same cluster was used
as the root of the cluster graph, if this graph is a clique tree, and after
a post-order traversal to start the calibration.
"""
function integratebelief!(obj::ClusterGraphBelief, cgraph::MetaGraph, prenodes)
    integratebelief!(obj, default_rootcluster(cgraph, prenodes))
end
integratebelief!(b::ClusterGraphBelief) = integratebelief!(b, default_sepset1(b))
integratebelief!(b::ClusterGraphBelief, j::Integer) = integratebelief!(b.belief[j])

# first sepset containing a single node
default_sepset1(b::ClusterGraphBelief) = default_sepset1(b.belief, nclusters(b) + 1)
function default_sepset1(beliefs::AbstractVector, n::Integer)
    j = findnext(b -> length(nodelabels(b)) == 1, beliefs, n)
    isnothing(j) && error("no sepset with a single node") # should not occur: degree-1 taxa
    return j
end

"""
    regularizebeliefs_bycluster!(beliefs::ClusterGraphBelief, clustergraph)
    regularizebeliefs_bycluster!(beliefs::ClusterGraphBelief, clustergraph, cluster_label)

Modify beliefs of cluster graph by adding positive values to some diagonal
elements of precision matrices `J`, while preserving the full graphical model
(product of cluster beliefs over product of sepset beliefs,
invariant during belief propagation) so that all beliefs are non-degenerate.

This regularization could be done after initialization with
[`init_beliefs_assignfactors!`](@ref) for example.

The goal is that at each later step of belief propagation, the sending cluster
has a non-degenerate (positive definite) precision matrix for the variables to be
integrated, so that the message to be sent is well-defined (i.e. can be computed)
and positive semidefinite.

## Algorithm

For each cluster Ci (or for only for 1 cluster, labeled `cluster_label`):
1. Find a regularization parameter adaptively for that cluster:
   ϵ = maximum absolute value of all entries in Ci's precision matrix J, and
   of the machine epsilon.  
   Then loop through its incident edges:
2. For each neighbor cluster Cj and associated sepset Sij,
   add ϵ > 0 to the diagonal entries of Ci's precision matrix `J`
   corresponding to the traits in Sij.
3. To preserve the graphical model's joint distribution for the full set of
   variables (invariant during BP), the same ϵ is added to each diagonal entry
   of Sij's precision matrix.
"""
function regularizebeliefs_bycluster!(beliefs::ClusterGraphBelief, cgraph::MetaGraph)
    for lab in labels(cgraph)
        regularizebeliefs_bycluster!(beliefs, cgraph, lab)
    end
end
function regularizebeliefs_bycluster!(beliefs::ClusterGraphBelief{B},
        cgraph::MetaGraph, clusterlab) where B<:Belief{T} where T
    b = beliefs.belief
    cluster_to = b[clusterindex(clusterlab, beliefs)] # receiving-cluster
    ϵ = max(eps(T), maximum(abs, cluster_to.J)) # regularization constant
    for nblab in neighbor_labels(cgraph, clusterlab)
        sepset = b[sepsetindex(clusterlab, nblab, beliefs)]
        regularizebeliefs_1clustersepset!(cluster_to, sepset, ϵ)
    end
end

"""
    regularizebeliefs_1clustersepset!(cluster::AbstractBelief, sepset::AbstractBelief, ϵ)

Modify beliefs of 1 cluster and 1 of sepset, assumed to be neighbors in a
cluster graph (such that the sepset's scope is a subset of the cluster's scope)
by adding ϵ to all diagonal elements of the sepset precision matrice `J`
and to the corresponding diagonal elements of the cluster precision,
so as to preserve the full graphical model.

Used by
[`regularizebeliefs_bycluster!`](@ref) and
[`regularizebeliefs_onschedule!`](@ref).
"""
function regularizebeliefs_1clustersepset!(cluster::AbstractBelief, sepset::AbstractBelief, ϵ)
    upind = scopeindex(sepset, cluster)  # indices to be updated
    isempty(upind) && return
    ΔJ = ϵ*LA.I(length(upind))
    view(cluster.J, upind, upind) .+= ΔJ # regularize cluster precision
    sepset.J .+= ΔJ                      # preserve the graph invariant
    return
end

"""
    regularizebeliefs_bynodesubtree!(beliefs::ClusterGraphBelief, clustergraph)

Modify beliefs of cluster graph by adding positive values to some diagonal
elements of precision matrices `J`, while preserving the full graphical model
(product of cluster beliefs over product of sepset beliefs,
invariant during belief propagation) so that all beliefs are non-degenerate.

The goal is the same as [`regularizebeliefs_bycluster!`](@ref) and
[`regularizebeliefs_onschedule!`](@ref), but the algorithm is different.


## Algorithm

For each node (or variable) v:
1. Consider the subgraph T of clusters & edges that have v. If `clustergraph`
   has the generalized running-intersection property, this subgraph is a tree.
2. Root T at a cluster containing a node with the largest postorder index.
3. Find a regularization parameter adaptively for that node:
   ϵ = maximum absolute value of all entries in Ci's precision matrix J, and
   of the machine epsilon, over clusters Ci in T.
4. For each trait j, find the subtree Tj of clusters and sepsets that
   have trait j of node v in their scope.
5. For each cluster and sepset in Tj, except at its cluster root:
   add ϵ on the diagonal of their belief precision matrix `J` corresponding to
   trait j of node v.
6. Check that graphical model invariant is preserved, that is: for each trait j,
   ϵ was added to the same number of clusters as number of sepsets.
"""
function regularizebeliefs_bynodesubtree!(beliefs::ClusterGraphBelief, cgraph::MetaGraph)
    for (node_symbol, node_ind) in get_nodesymbols2index(cgraph)
        regularizebeliefs_bynodesubtree!(beliefs, cgraph, node_symbol, node_ind)
    end
end
function regularizebeliefs_bynodesubtree!(beliefs::ClusterGraphBelief{B},
        cgraph::MetaGraph, node_symbol, node_ind) where B<:Belief{T} where T
    b = beliefs.belief
    sg, _ = nodesubtree(cgraph, node_symbol, node_ind)
    nv(sg) <= 1 && return nothing
    is_tree(sg) || error("running intersection violated for node / variable $node_symbol")
    rootj = argmax(sg[l][2][1] for l in labels(sg)) # cluster with largest preorder index
    # degree(sg, rootj) == 1 || @warn "cluster $(label_for(sg, rootj)) not a leaf in subtree for $node_symbol, I had hoped so."
    spt = spanningtree_clusterlist(sg, rootj) # ! cluster indices refer to those in sg, not cgraph
    ϵ = eps(T)
    for l in labels(sg)
        ϵ = max(ϵ, maximum(abs, b[clusterindex(l, beliefs)].J))
    end
    for (par_l, chi_l, _, _) in zip(spt...) # traverse node subtree
        childbelief = b[clusterindex(chi_l, beliefs)]
        sepsetbelief = b[sepsetindex(par_l, chi_l, beliefs)]
        s_ind, c_ind = scopeindex(node_ind, sepsetbelief, childbelief)
        d = length(s_ind)
        view(childbelief.J,  c_ind, c_ind) .+= ϵ*LA.I(d)
        view(sepsetbelief.J, s_ind, s_ind) .+= ϵ*LA.I(d)
    end
    return nothing
end

"""
    regularizebeliefs_onschedule!(beliefs::ClusterGraphBelief, clustergraph)

Modify beliefs of the cluster graph so that all beliefs are non-degenerate, by
(1) adding positive values to some diagonal elements of precision matrices `J`
while preserving the full graphical model, and
(2) propagating messages by marginalizing cluster beliefs.

The goal is the same as [`regularizebeliefs_bycluster!`](@ref) and
[`regularizebeliefs_bynodesubtree!`](@ref), but the algorithm is different.

## Algorithm

The outcomes depends on a "schedule", that is, on an order in which clusters
are considered. Here, the method simply takes the order given by `labels(clustergraph)`.
Mark all messages Ci → Cj as un-sent.
For each cluster Ci in order:

1. For each neighbor cluster Cj of Ci, if the message Cj → Ci has not been sent,
   then:
   - add ϵ_i to the diagonal entries of sepset Sij's precision matrix Jij
   - add ϵ_i to the diagonal entries of Ci's precision matrix Ji that correspond to
     the inscope variables of Sij
   - mark Cj → Ci as sent.
   Notes:
   + ϵ_i is taken as the maximum of all absolute values of entries in Ji,
     and of the square root of machine epsilon.
   + This is equivalent to "sending" a default message, that is not computed
     from Cj's belief, with diagonal precision matrix ϵ⋅I and other canonical
     parameters h=0, g=0.
2. For each neighbor cluster Cj of Ci, if Ci → Cj has not been sent then
   - send a message from Ci to Cj by marginalizing Ci's belief (that is, using belief propagation).
   - mark the message Ci → Cj as sent.
"""
function regularizebeliefs_onschedule!(beliefs::ClusterGraphBelief, cgraph::MetaGraph)
    # (cluster1, cluster2) ∈ messagesent if cluster1 has sent a message to cluster2
    messagesent = Set{NTuple{2,Symbol}}()
    tosend = Tuple{Tuple{Symbol,Int},Int}[] # indices for step 2, reset at each cluster
    ϵ0 = sqrt(eps(Float64))
    b = beliefs.belief
    mr = beliefs.messageresidual
    for clusterlab in labels(cgraph)
        empty!(tosend)
        ci = clusterindex(clusterlab, beliefs) # cluster index
        ϵ = max(maximum(abs, b[ci].J), ϵ0)     # regularization parameter
        for nblab in neighbor_labels(cgraph, clusterlab)
            nb_i = clusterindex(nblab, beliefs) # neighbor cluster index
            ss_i = sepsetindex(clusterlab, nblab, beliefs) # sepset index
            if (nblab, clusterlab) ∉ messagesent # step 1: regularize precisions
                regularizebeliefs_1clustersepset!(b[ci], b[ss_i], ϵ)
                push!(messagesent, (nblab, clusterlab))
            end
            if (clusterlab, nblab) ∉ messagesent # store indices for step 2 later
                push!(tosend, ((nblab, nb_i), ss_i))
                push!(messagesent, (clusterlab, nblab))
            end
        end
        for ((nblab, nb_i), ss_i) in tosend # step 2: pass message with BP
            propagate_belief!(b[nb_i], b[ss_i], b[ci], mr[(nblab, clusterlab)])
        end
    end
end
