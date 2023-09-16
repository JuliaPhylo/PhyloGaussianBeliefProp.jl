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

clusterindex(c, obj::ClusterGraphBelief) = clusterindex(c, obj.cdict)
function clusterindex(clusterlabel, clusterdict)
    clusterdict[clusterlabel]
end

sepsetindex(c1, c2, obj::ClusterGraphBelief) = sepsetindex(c1, c2, obj.sdict)
function sepsetindex(clustlabel1, clustlabel2, sepsetdict)
    sepsetdict[Set((clustlabel1, clustlabel2))]
end

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
default_sepset1(b::ClusterGraphBelief) = default_sepset1(b.belief, nclusters(b)+1)
function default_sepset1(beliefs::AbstractVector, n::Integer)
    j = findnext(b -> length(nodelabels(b)) == 1, beliefs, n)
    isnothing(j) && error("no sepset with a single node") # should not occur: degree-1 taxa
    return j
end

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
preserve the cluster graph invariant (i.e. the product of cluster beliefs
divided by the product of edge beliefs is invariant throughout belief
propagation).

Send a message along each affected edge from hybrid cluster to variable cluster,
so that all subsequent messages received by these hybrid clusters are valid
Gaussians (i.e. with positive semi-definite variance/precision).
"""
function mod_beliefs_bethe!(beliefs::ClusterGraphBelief,
    numt::Integer, net::HybridNetwork, ϵ::Float64=1.0)
    # fixit: set ϵ adaptively
    prenodes = net.nodes_changed
    b = beliefs.belief
    # modify beliefs of hybrid clusters and their incident sepsets
    for n in net.hybrid
        o = sort!(indexin(getparents(n), prenodes), rev=true)
        parentnames = [pn.name for pn in prenodes[o]]
        clustlabel = Symbol(n.name, parentnames...)
        cbi = clusterindex(clustlabel, beliefs) # cluster belief index
        sb_idx = [sepsetindex(clustlabel, clustlabel2, beliefs) for clustlabel2
            in Symbol.(parentnames)] # sepset belief indices for parent nodes
        #= Add ϵ to diagonal entries of principal submatrix (for parent nodes)
        of cluster belief precision. The first `numt` coordinates are for the
        hybrid node. =#
        b[cbi].J[(numt+1):end, (numt+1):end] .+= ϵ*LA.I(length(sb_idx)*numt)
        for (i, sbi) in enumerate(sb_idx)
            #= Add ϵ to diagonal entries of sepset belief precision to preserve
            the cluster graph invariant. =#
            b[sbi].J .+= ϵ*LA.I(numt)
            #= Send non-degenerate message from hybrid cluster to neighbor
            variable clusters for each parent node. =#
            propagate_belief!(b[clusterindex(Symbol(parentnames[i]), beliefs)],
                b[sbi], b[cbi])
        end
    end
end

"""
    mod_beliefs!(beliefs::ClusterGraphBelief, traitdimension, net,
        ridgeconstant::Float=1.0)
"""
function mod_beliefs!(beliefs::ClusterGraphBelief, numt::Integer,
    net::HybridNetwork, ϵ::Float64=1.0)
    # (1) Pick a spanning tree of the cluster graph. Select a root cluster.

    # (2) Do a post-order traversal of spanning tree edges

    # (3) Do a pre-order traversal of spanning tree edges

    #= (4) Repeat the above, but with a different spanning tree. We want the set
    of spanning trees to cover the edges of the cluster graph. =#
end

"""
    calibrate!(beliefs::ClusterGraphBelief, schedule, niterations=1)

Propagate messages for each tree in the `schedule` list,
in postorder then in preorder, and repeats this for `niterations`.
Each schedule "tree" should be a tuple of 4 vectors as output by
[`spanningtree_clusterlist`](@ref), where each vector provides the
parent/child label/index of an edge along which to pass a message,
and where these edges are listed in preorder. For example, the parent
of the first edge is taken to be the root of the schedule tree.

The default of 1 iteration is sufficient for exact calibration if
the schedule tree is a clique tree for the graphical model.
"""
function calibrate!(beliefs::ClusterGraphBelief, schedule::AbstractVector, niter=1::Integer)
    for _ in 1:niter
        # spt = spanningtree_clusterlist(cgraph, prenodes)
        for spt in schedule
            calibrate!(beliefs, spt)
        end
    end
end
function calibrate!(beliefs::ClusterGraphBelief, spt::Tuple)
    propagate_1traversal_postorder!(beliefs, spt...)
    propagate_1traversal_preorder!(beliefs, spt...)
end

"""
    propagate_1traversal_postorder!(beliefs::ClusterGraphBelief, spanningtree...)
    propagate_1traversal_preorder!(beliefs::ClusterGraphBelief,  spanningtree...)

Messages are propagated from the tips to the root of the tree by default,
or from the root to the tips if `postorder` is false.

The "spanning tree" should be a tuple of 4 vectors as output by
[`spanningtree_clusterlist`](@ref), meant to list edges in preorder.
Its nodes (resp. edges) should correspond to clusters (resp. sepsets) in
`beliefs`: labels and indices in the spanning tree information
should correspond to indices in `beliefs`.
This condition holds if beliefs are produced on a given cluster graph and if the
tree is produced by [`spanningtree_clusterlist`](@ref) on the same graph.
"""
function propagate_1traversal_postorder!(beliefs::ClusterGraphBelief,
            pa_lab, ch_lab, pa_j, ch_j)
    b = beliefs.belief
    # (parent <- sepset <- child) in postorder
    for i in reverse(1:length(pa_lab))
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        propagate_belief!(b[pa_j[i]], b[ss_j], b[ch_j[i]])
    end
end

function propagate_1traversal_preorder!(beliefs::ClusterGraphBelief,
            pa_lab, ch_lab, pa_j, ch_j)
    b = beliefs.belief
    # (child <- sepset <- parent) in preorder
    for i in 1:length(pa_lab)
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        propagate_belief!(b[ch_j[i]], b[ss_j], b[pa_j[i]])
    end
end

#------ parameter optimization. fixit: place in some other file? ------#
"""
    calibrate_optimize_cliquetree!(beliefs::ClusterGraphBelief, clustergraph,
        nodevector_preordered, tbl::Tables.ColumnTable, taxa::AbstractVector,
        evolutionarymodel_name, evolutionarymodel_startingparameters)

Optimize model parameters using belief propagation along `clustergraph`,
assumed to be a clique tree for the input network, whose nodes in preorder are
`nodevector_preordered`. Optimization aims to maximize the likelihood
of the data in `tbl` at leaves in the network. The taxon names in `taxa`
should appear in the same order as they come in `tbl`.
The parameters being optimized are the variance rate(s) and prior mean(s)
at the root. The prior variance at the root is fixed.

The calibration does a postorder of the clique tree only, to get the likelihood
at the root *without* the conditional distribution at all nodes, modifying
`beliefs` in place. Therefore, if the distribution of ancestral states is sought,
an extra preorder calibration would be required.
Warning: there is *no* check that the cluster graph is in fact a clique tree.
"""
function calibrate_optimize_cliquetree!(beliefs::ClusterGraphBelief,
        cgraph, prenodes::Vector{PN.Node},
        tbl::Tables.ColumnTable, taxa::AbstractVector,
        evomodelfun, # constructor function
        evomodelparams)
    spt = spanningtree_clusterlist(cgraph, prenodes)
    rootj = spt[3][1] # spt[3] = indices of parents. parent 1 = root
    mod = evomodelfun(evomodelparams...) # model with starting values
    function score(θ) # θ: unconstrained parameters, e.g. log(σ2)
        model = evomodelfun(params_original(mod, θ)...)
        init_beliefs_reset!(beliefs.belief)
        init_beliefs_assignfactors!(beliefs.belief, model, tbl, taxa, prenodes)
        propagate_1traversal_postorder!(beliefs, spt...)
        _, res = integratebelief!(beliefs, rootj) # drop conditional mean
        return -res # score to be minimized (not maximized)
    end
    # autodiff does not currently work with ForwardDiff, ReverseDiff of Zygote,
    # because they cannot differentiate array mutation, as in: view(be.h, factorind) .+= h
    opt = Optim.optimize(score, params_optimize(mod), Optim.LBFGS())
    # fixit: if BM and fixed root, avoid optimization bc there exists an exact alternative
    loglikscore = - Optim.minimum(opt)
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, loglikscore, opt
end
