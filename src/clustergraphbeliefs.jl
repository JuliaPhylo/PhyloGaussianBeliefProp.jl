"""
    ClusterGraphBelief{B<:Belief}
    ClusterGraphBelief(belief_vector::Vector{B})

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
struct ClusterGraphBelief{B<:Belief, M<:MessageResidual}
    "vector of beliefs, cluster beliefs first and sepset beliefs last"
    belief::Vector{B}
    "vector of factors from the graphical model, used to initialize cluster beliefs"
    factors::Vector{B} # fixit: review
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

# fixit: review. Assumes that `beliefs` modified by `init_beliefs_assignfactors!`
function ClusterGraphBelief(beliefs::Vector{B}) where B<:Belief
    i = findfirst(b -> b.type == bsepsettype, beliefs)
    nc = (isnothing(i) ? length(beliefs) : i - 1)
    all(beliefs[i].type == bclustertype for i in 1:nc) ||
        error("clusters are not consecutive")
    all(beliefs[i].type == bsepsettype for i in (nc+1):length(beliefs)) ||
        error("sepsets are not consecutive")
    cdict = get_clusterindexdictionary(beliefs, nc)
    sdict = get_sepsetindexdictionary(beliefs, nc)
    mr = init_messageresidual(beliefs, nc)
    factors = [deepcopy(beliefs[i]) for i in 1:nc] # beliefs[1:nc] makes a copy, but not deep
    return ClusterGraphBelief{B,valtype(mr)}(beliefs,factors,nc,cdict,sdict,mr)
end
function get_clusterindexdictionary(beliefs, nclusters)
    Dict(beliefs[j].metadata => j for j in 1:nclusters)
end
function get_sepsetindexdictionary(beliefs, nclusters)
    Dict(Set(beliefs[j].metadata) => j for j in (nclusters+1):length(beliefs))
end
function init_messageresidual(beliefs::Vector{B}, nclusters) where B<:Belief{T} where T<:Real
    messageresidual = Dict{Tuple{Symbol,Symbol}, MessageResidual{T}}()
    for j in (nclusters+1):length(beliefs)
        ssbe = beliefs[j] # sepset belief
        (clustlab1, clustlab2) = ssbe.metadata
        messageresidual[(clustlab1, clustlab2)] = MessageResidual(ssbe.J, ssbe.h)
        messageresidual[(clustlab2, clustlab1)] = MessageResidual(ssbe.J, ssbe.h)
    end
    return messageresidual
end

"""
    init_beliefs_reset!(beliefs::ClusterGraphBelief)

Reset cluster beliefs to factors and sepset beliefs to h=0, J=0, g=0.
"""
function init_beliefs_reset!(beliefs::ClusterGraphBelief)
    # fixit: change name of method?
    nc, nb = nclusters(beliefs), length(beliefs.belief)
    b, f = beliefs.belief, beliefs.factors
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
    factors_reset!(beliefs::ClusterGraphBelief)

Reset factors to initial cluster beliefs for a different instantiation of model
parameters (i.e. after [`init_beliefs_reset!`](@ref) and
[`init_beliefs_assignfactors!`](@ref) are run for different parameter values).
"""
function factors_reset!(beliefs::ClusterGraphBelief)
    b, f = beliefs.belief, beliefs.factors
    for i in 1:nclusters(beliefs)
        f[i].h   .= b[i].h
        f[i].J   .= b[i].J
        f[i].g[1] = b[i].g[1]
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
    integratebelief!(obj, beliefindex)
    integratebelief!(obj)
    integratebelief!(obj::ClusterGraphBelief, clustergraph, nodevector_preordered)

(μ,g) from fully integrating the object belief indexed `beliefindex`.
The second form uses the first sepset containing a single node. This is valid
if the beliefs are fully calibrated (including a pre-order traversal), but
invalid otherwise.
The third form uses the default cluster containing the root,
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
