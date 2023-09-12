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

sepsetindex(c1, c2, obj::ClusterGraphBelief) = sepsetindex(c1, c2, obj.sdict)
function sepsetindex(clustlabel1, clustlabel2, sepsetdict)
    sepsetdict[Set((clustlabel1, clustlabel2))]
end

function ClusterGraphBelief(beliefs)
    i = findfirst(b -> b.type == bsepsettype, beliefs)
    nc = (isnothing(i) ? length(beliefs) : i - 1)
    all(beliefs[i].type == bclustertype for i in 1:nc) ||
        error("clusters are not consecutive")
    all(beliefs[i].type == bsepsettype for i in (nc+1):length(beliefs)) ||
        error("sepsets are not consecutive")
    cdict = get_clusterindexdictionary(beliefs, nc)
    sdict = get_sepsetindexdictionary(beliefs, nc)
    return ClusterGraphBelief{eltype(beliefs)}(beliefs, nc, cdict, sdict)
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
default_sepset1(b::ClusterGraphBelief) = default_sepset1(b.belief, nclusters(b) + 1)
function default_sepset1(beliefs::AbstractVector, n::Integer)
    j = findnext(b -> length(nodelabels(b)) == 1, beliefs, n)
    isnothing(j) && error("no sepset with a single node") # should not occur: degree-1 taxa
    return j
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
    loglikscore = -Optim.minimum(opt)
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, loglikscore, opt
end

"""
    calibrate_exact_cliquetree!(beliefs::ClusterGraphBelief, clustergraph,
        nodevector_preordered, tbl::Tables.ColumnTable, taxa::AbstractVector,
        evolutionarymodel_name)

For a Brownian Motion with a fixed root, compute the exact maximum likelihood
parameters using closed form formulas relying on belief propagation.
The  `clustergraph` is assumed to be a clique tree for the input tree,
whose nodes in preorder are `nodevector_preordered`.
Optimization aims to maximize the likelihood
of the data in `tbl` at leaves in the network.
The taxon names in `taxa` should appear in the same order as they come in `tbl`.
The parameters being optimized are the variance rate(s) and prior mean(s)
at the root. The prior variance at the root is fixed to zero.

The calibration does a postorder of the clique tree only, using the optimal parameters,
to get the likelihood
at the root *without* the conditional distribution at all nodes, modifying
`beliefs` in place. Therefore, if the distribution of ancestral states is sought,
an extra preorder calibration would be required.
Warning: there is *no* check that the cluster graph is in fact a clique tree.
"""
# TODO deal with missing values (only completelly missing tips)
# TODO network case
function calibrate_exact_cliquetree!(beliefs::ClusterGraphBelief,
    cgraph, prenodes::Vector{PN.Node},
    tbl::Tables.ColumnTable, taxa::AbstractVector,
    evomodelfun, # constructor function
    evomodelparams
)
    evomodelfun == UnivariateBrownianMotion || evomodelfun == MvFullBrownianMotion || error("Exact optimization is only implemented for the univariate or full Brownian Motion.")
    model = evomodelfun(evomodelparams...)
    isrootfixed(model) || error("Exact optimization is only implemented for the BM with fixed root.")
    ## TODO: check that the tree was calibrated with the right model MvDiagBrownianMotion((1,1), (0,0), (Inf,Inf)) ?
    ## TODO: or do this first calibration directly in the function ? (API change)
    p = dimension(model)

    ## Compute mu_hat from root belief
    ## Root is the last node of the root cluster
    spt = spanningtree_clusterlist(cgraph, prenodes)
    rootj = spt[3][1] # spt[3] = indices of parents. parent 1 = root
    exp_root, _ = integratebelief!(beliefs, rootj)
    mu_hat = exp_root[(end-p+1):end]

    ## Compute sigma2_hat from conditional moments
    tmp_num = zeros(p, p)
    tmp_den = 0
    # loop over all nodes
    for i in eachindex(prenodes)
        # TODO: is it the correct way to iterate over the graph ?
        # remove the root which is first in pre-order
        i == 1 && continue
        # find associated cluster
        nodechild = prenodes[i]
        clusterindex = findClusterIndex(nodechild, beliefs.belief)
        b = beliefs.belief[clusterindex]
        dimclus = length(b.nodelabel)
        # child ind in the cluster
        childind = findfirst(b.nodelabel .== i)
        # find parents: assumes that a cluster has all the parents of a node, which should be the case thanks to findClusterIndex
        parind = findall([PN.isconnected(prenodes[nl], nodechild) && nl != i for nl in b.nodelabel])
        all_parent_edges = [PN.getConnectingEdge(prenodes[b.nodelabel[d]], nodechild) for d in parind]
        all_gammas = zeros(dimclus)
        all_gammas[parind] = [ee.gamma for ee in all_parent_edges]
        # parent(s) edge length
        edge_length = 0.0
        for ee in all_parent_edges
            edge_length += ee.gamma * ee.gamma * ee.length
        end
        edge_length == 0.0 && continue # if edge has length zero, then the parameter R does not occur in the factor
        # moments
        exp_be, _ = integratebelief!(b)
        vv = inv(b.J)
        # tip node
        if (nodechild.leaf) # tip node
            # TODO: deal with missing data
            # TODO: is there a more simple way to do that ? Record of data in belief object ?
            size(vv, 1) == p || error("A leaf node should have only on non-degenerate factor.")
            # find tip data
            nodelab = nodechild.name
            i_row = findfirst(isequal(nodelab), taxa)
            !isnothing(i_row) || error("A node with data does not match any taxon")
            tipvalue = [tbl[v][i_row] for v in eachindex(tbl)]
            # parent node moments are the p first
            indpar = 1:p
            diffExp = view(exp_be, indpar) - tipvalue
            tmp_num += diffExp * transpose(diffExp) ./ edge_length
            # assumes that vv is a scalar times R_test
            tmp_den += 1 - vv[1, 1] / edge_length
        else # internal node
            # init with child node
            begic = (childind - 1) * p + 1
            endic = childind * p
            diffExp = view(exp_be, begic:endic)
            diffVar = vv[begic, begic]
            # sum over parent nodes
            for d in parind
                # indexes
                begi = (d - 1) * p + 1
                endi = d * p
                # exp and covar with child
                diffExp -= all_gammas[d] .* view(exp_be, begi:endi)
                diffVar -= 2 * all_gammas[d] * vv[begic, begi]
                # parents var covar
                for d2 in parind
                    begi2 = (d2 - 1) * p + 1
                    diffVar += all_gammas[d] * all_gammas[d2] * vv[begi, begi2]
                end
            end
            tmp_num += diffExp * transpose(diffExp) ./ edge_length
            tmp_den += 1 - diffVar / edge_length
        end
    end
    sigma2_hat = tmp_num ./ tmp_den
    ## TODO: This is the REML estimate. Should we get ML instead ?

    ## Get optimal paramters
    bestθ = (sigma2_hat, mu_hat, zeros(p, p))
    bestmodel = evomodelfun(bestθ...)
    ## Get associated likelihood
    ## TODO: likelihood for the full BM (not implemented)
    loglikscore = NaN
    if (evomodelfun == UnivariateBrownianMotion)
        init_beliefs_reset!(beliefs.belief)
        update_root_inscope!(beliefs.belief, model)
        init_beliefs_assignfactors!(beliefs.belief, bestmodel, tbl, taxa, prenodes)
        propagate_1traversal_postorder!(beliefs, spt...)
        _, loglikscore = integratebelief!(beliefs, rootj)
    end

    return bestmodel, loglikscore
end

"""
    findClusterIndex(node::PN.Node, belief_vector)

In the belief in the vector that contains both the node and all its parents.
Throws an error if this cluster does not ex

"""
function findClusterIndex(node::PN.Node, belief_vector)
    nodelab = node.name
    for i in eachindex(belief_vector)
        b = belief_vector[i]
        # only cluster beliefs
        b.type == bclustertype || continue
        # label should match
        occursin(nodelab, String(b.metadata)) || continue
        # node should be in a cluster with all its parents
        parentlabels = [nn.name for nn in PN.getparents(node)]
        all([occursin(ll, String(b.metadata)) for ll in parentlabels]) || continue
        # if still here, we found the cluster
        return i
    end
    error("Could not find a cluster with the node and all its parents.")
end