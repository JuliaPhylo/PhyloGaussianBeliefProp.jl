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
    They are useful to aproximate the likelihood by the free energy."""
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


#=
"""
    mod_beliefs_bethe!(beliefs::ClusterGraphBelief, traitdimension,
        net, ridgeconstant::Float=1.0)

Modify naively initialized beliefs (see [`init_beliefs_assignfactors!`](@ref))
of a Bethe cluster graph so that:
1. messages sent/received between neighbor clusters are well-defined, and
2. messages sent from a hybrid cluster are non-degenerate.

For each hybrid cluster belief, add `ridgeconstant` to the diagonal elements of
its precision matrix that correspond to the parent nodes in the cluster, so that
the affected principal submatrix is stably invertible (the first `traitdimension`
rows/columns of the precision matrix correspond to the hybrid node).
For the edge beliefs associated with the parent nodes in a hybrid cluster, add
`ridgeconstant` to the diagonal elements of its precision matrix so as to
preserve the probability model of the full data (the product of cluster beliefs
divided by the product of edge beliefs, which is invariant throughout belief propagation).

Send a message along each affected edge from hybrid cluster to variable cluster,
so that all subsequent messages received by these hybrid clusters are Gaussians
with positive semi-definite variance/precision.
"""
function mod_beliefs_bethe!(beliefs::ClusterGraphBelief, numt::Integer, net::HybridNetwork, ϵ::Float64=1.0)
    # fixit: set ϵ adaptively
    prenodes = net.nodes_changed
    b = beliefs.belief
    for n in net.hybrid
        o = sort!(indexin(getparents(n), prenodes), rev=true)
        parentnames = [pn.name for pn in prenodes[o]]
        clustlabel = Symbol(n.name, parentnames...)
        cbi = clusterindex(clustlabel, beliefs) # cluster belief index
        sb_idx = [sepsetindex(clustlabel, clustlabel2, beliefs) for clustlabel2
            in Symbol.(parentnames)] # sepset belief indices for parent nodes
        # Add ϵ to diagonal entries of principal submatrix (for parent nodes) of
        # cluster belief precision. The first `numt` coordinates are for the hybrid node
        b[cbi].J[(numt+1):end, (numt+1):end] .+= ϵ * LA.I(length(sb_idx)*numt)
        for (i, sbi) in enumerate(sb_idx)
            # add ϵ to diagonal of sepset's J to preserve cluster graph invariant
            b[sbi].J .+= ϵ*LA.I(numt)
            # send non-degenerate message: hybrid cluster → variable cluster
            propagate_belief!(b[clusterindex(Symbol(parentnames[i]), beliefs)], b[sbi], b[cbi])
        end
    end
end
=#

"""
    init_messages!(beliefs::ClusterGraphBelief, clustergraph)

Modify naively assigned beliefs (see [`init_beliefs_assignfactors!`](@ref)) of a
cluster graph (while preserving the cluster graph invariant) so that all cluster
beliefs are non-degenerate, and for any subsequent schedule of messages:

        (1) cluster/sepset beliefs stay non-degenerate (i.e. positive definite)
        (2) all received messages are well-defined (i.e. positive semi-definite)

## Algorithm
1. All clusters are considered unprocessed and no messages have been sent.
2. Pick an arbitary unprocessed cluster and let it receive default messages from
all neighbor clusters that have not sent it a message.
3. Compute and send a message from this cluster to any neighbor cluster that has
not received a message from it. The selected cluster is now marked as processed.
4. Repeat steps 2-3 until all clustes have been processed.

Step 2 (the receipt of non-degenerate messages from all neighbors) guarantees
that all cluster beliefs will be non-degenerate, while step 3 (the use, where
possible, of messages that can be computed instead of default messages)
guarantees that all subsequent received messages are well-defined.
"""
function init_messages!(beliefs::ClusterGraphBelief, cgraph::MetaGraph)
    # (clust1, clust2) ∈ messagesent => clust1 has sent a message to clust2
    messagesent = Set{NTuple{2,Symbol}}()
    b = beliefs.belief
    for clusterlab in labels(cgraph)
        tosend = NTuple{3,Int}[] # track messages to send after updating belief
        from = clusterindex(clusterlab, beliefs) # sending-cluster index
        for nblab in neighbor_labels(cgraph, clusterlab)
            to = clusterindex(nblab, beliefs) # receiving-cluster index
            by = sepsetindex(clusterlab, nblab, beliefs) # sepset index
            if (nblab, clusterlab) ∉ messagesent
                propagate_belief!(b[from], b[by]) # receive default message
                push!(messagesent, (nblab, clusterlab))
            end
            if (clusterlab, nblab) ∉ messagesent
                push!(tosend, (to, by, from))
                push!(messagesent, (clusterlab, nblab))
            end
        end
        for (to, by, from) in tosend
            #= `false`: raise error if message is ill-defined instead of
            handling it by sending a default message =#
            propagate_belief!(b[to], b[by], b[from], false)
        end
    end
end

"""
    regularizebeliefs!(beliefs::ClusterGraphBelief, clustergraph)

Modify naively assigned beliefs (see [`init_beliefs_assignfactors!`](@ref)) of a
cluster graph (while preserving the cluster graph invariant) so that all cluster
beliefs are non-degenerate, and for any subsequent schedule of messages:

        (1) cluster/sepset beliefs stay non-degenerate (i.e. positive definite)
        (2) all messages sent are well-defined (i.e. can be computed) and are
        positive semidefinite

This modification is described as "regularization" in the sense that the initial
cluster/sepset precisions are perturbed from their actual value.

## Algorithm
1. For each cluster, loop through its incident edges. For each sepset, add ϵ > 0
to the diagonal entries of its precision (i.e. the `J` parameter).
2. To preserve the cluster graph invariant, each time the precision of a sepset
belief is modified, make an equivalent change to the precision of the cluster
belief.
For example, if the diagonal entry for variable `x` is incremented in the sepset
precision, then increment the diagonal entry for variable `x` in the cluster
precision by the same amount.
"""
function regularizebeliefs!(beliefs::ClusterGraphBelief, cgraph::MetaGraph)
    #= Notes:
    Guarantees (1) and (2) stated above are different from those described in
    Lemma 2 of Du et al. (2017): "Convergence Analysis of Distributed Inference
    with Vector-Valued Gaussian Belief Propagation". They have that all
    Sum-Product messages after a specific initialization are positive definite.
    In contrast, translating our setting to the Sum-Product framework, we may
    have Sum-Product messages that are positive semidefinite but not definite,
    though we show that the cluster beliefs will still remain positive definite
    so that messages sent are always well-defined.
    * Todo (Ben): I'm formalizing the explanation of this. I also think that a
    sufficient condition for this may be the absence of deterministic factors. =#
    b = beliefs.belief
    for clusterlab in labels(cgraph)
        cluster_to = b[clusterindex(clusterlab, beliefs)] # receiving-cluster
        ϵ = maximum(abs, cluster_to.J) # regularization constant
        for nblab in neighbor_labels(cgraph, clusterlab)
            sepset = b[sepsetindex(clusterlab, nblab, beliefs)]
            upind = scopeindex(sepset, cluster_to) # indices to be updated
            d = length(upind)
            view(cluster_to.J, upind, upind) .+= ϵ*LA.I(d) # regularize cluster precision
            sepset.J .+= ϵ*LA.I(d) # preserve cluster graph invariant
        end
    end
end