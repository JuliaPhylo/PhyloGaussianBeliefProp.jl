"""
    calibrate!(beliefs::ClusterGraphBelief, schedule, niterations=1;
        auto::Bool=false, info::Bool=true)

Propagate messages in postorder then preorder for each tree in the `schedule`
list, and loop through for `niterations`. Each schedule "tree" should be a tuple
of 4 vectors as output by [`spanningtree_clusterlist`](@ref), where each vector
provides the parent/child label/index of an edge along which to pass a message,
and where these edges are listed in preorder. For example, the parent of the
first edge is taken to be the root of the schedule tree.

Calibration is flagged when after a schedule "tree" is run, it is found that each
possible message sent in the cluster graph was last equal to its moderating
sepset belief (within some tolerance). If `info=true`, then prints the
iteration number and tree index when calibration is flagged, or that `niterations`
have passed without calibration being flagged.

If `auto=true`, then `true` if calibration is flagged within `niterations`
(returns upon detection) and `false` otherwise.

The conditions for flagging calibration are sufficient but not necessary (e.g.
the default of 1 iteration is sufficient for exact calibration if the schedule
tree is a clique tree for the graphical model, but an additional iteration is
required to detect that all messages have not changed).

See also: [`iscalibrated_residnorm`](@ref)
"""
function calibrate!(beliefs::ClusterGraphBelief, schedule::AbstractVector,
    niter::Integer=1; auto::Bool=false, info::Bool=false)
    # fixit: input checks?
    i = 0
    iscal = false
    while i < niter
        i += 1
        for (j, spt) in enumerate(schedule)
            if calibrate!(beliefs, spt) && (iscal = true)
                info && @info "Calibration detected: iter $i, sch $j"
                auto && return true # stop if calibration is detected
            end
        end
    end
    iscal || info && @info "`niter` reached before calibration detected"
    # fixit: check calibration properly at the end?
    auto && return false
end
function calibrate!(beliefs::ClusterGraphBelief, spt::Tuple)
    propagate_1traversal_postorder!(beliefs, spt...)
    propagate_1traversal_preorder!(beliefs, spt...)
    return iscalibrated_residnorm(beliefs)
end

"""
    iscalibrated(beliefs::ClusterGraphBelief, edge_labels, rtol::Float64=1e-2)
    iscalibrated(residual::Tuple, atol::Float64=1e-5)

`true` if each edge belief is *marginally consistent* with the beliefs of the
pair of clusters that it spans (i.e. their marginal means and variances are
approximately equal with relative tolerance `rtol`), `false` otherwise.

Mean vectors are compared by their 2-norm and variance matrices are compared by
the Frobenius norm.

The second form checks if the residual (in terms of canonical parameters: h, J)
between a new message from a cluster through an edge, and the current edge
belief is within `atol` of 0, elementwise.
"""
function iscalibrated(beliefs::ClusterGraphBelief,
        edgelabs::Vector{Tuple{Symbol, Symbol}}, rtol=1e-2)
    # fixit: discuss tolerance for comparison
    #= fixit: this will probably be moved to `test_calibration.jl` since it is
    an inefficient way to check if the cluster graph is calibrated on the fly. =#
    b = beliefs.belief
    belief2meanvar = Dict{Symbol, Tuple{AbstractVector, AbstractMatrix}}()
    for (clustlab1, clustlab2) in edgelabs
        clust1 = b[clusterindex(clustlab1, beliefs)]
        clust2 = b[clusterindex(clustlab2, beliefs)]
        if haskey(belief2meanvar, clustlab1)
            clustmean1, clustvar1 = belief2meanvar[clustlab1]
        else
            clustmean1, clustvar1 = integratebelief!(clust1)[1], clust1.J \ LA.I
            belief2meanvar[clustlab1] = (clustmean1, clustvar1)
        end
        if haskey(belief2meanvar, clustlab2)
            clustmean2, clustvar2 = belief2meanvar[clustlab2]
        else
            clustmean2, clustvar2 = integratebelief!(clust2)[1], clust2.J \ LA.I
            belief2meanvar[clustlab2] = (clustmean2, clustvar2)
        end
        sepset = b[sepsetindex(clustlab1, clustlab2, beliefs)]
        sepsetmean, sepsetvar = integratebelief!(sepset)[1], sepset.J \ LA.I
        ind1 = scopeindex(sepset, clust1) # indices to be compared
        sepsetmean .== clustmean1[ind1]
        sepsetvar .== clustvar1[ind1, ind1]
        ind2 = scopeindex(sepset, clust2)
        sepsetmean .== clustmean2[ind2]
        sepsetvar .== clustvar2[ind2, ind2]
        # check the agreement between cluster and sepset beliefs
        isapprox(sepsetmean, clustmean1[ind1], rtol=rtol) &&
        isapprox(sepsetmean, clustmean2[ind2], rtol=rtol) &&
        isapprox(sepsetvar, clustvar1[ind1, ind1], rtol=rtol) &&
        isapprox(sepsetvar, clustvar2[ind2, ind2], rtol=rtol) || return false
    end
    return true
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
    mr = beliefs.messageresidual
    # (parent <- sepset <- child) in postorder
    for i in reverse(1:length(pa_lab))
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        sepset, residual = propagate_belief!(b[pa_j[i]], b[ss_j], b[ch_j[i]])
        mrss = mr[(pa_lab[i], ch_lab[i])]
        mrss.ΔJ .= residual[1]
        mrss.Δh .= residual[2]
        iscalibrated_residnorm!(mrss)
        approximate_kl!(mrss, sepset) # KL div between new and old sepset belief
    end
end

function propagate_1traversal_preorder!(beliefs::ClusterGraphBelief,
                                        pa_lab, ch_lab, pa_j, ch_j)
    b = beliefs.belief
    mr = beliefs.messageresidual
    # (child <- sepset <- parent) in preorder
    for i in eachindex(pa_lab)
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        sepset, residual = propagate_belief!(b[ch_j[i]], b[ss_j], b[pa_j[i]])
        mrss = mr[(ch_lab[i], pa_lab[i])]
        mrss.ΔJ .= residual[1]
        mrss.Δh .= residual[2]
        iscalibrated_residnorm!(mrss)
        approximate_kl!(mrss, sepset)
    end
end

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
        # below: includes a reset
        init_beliefs_assignfactors!(beliefs.belief, model, tbl, taxa, prenodes)
        init_factors_frombeliefs!(beliefs.factor, beliefs.belief)
        propagate_1traversal_postorder!(beliefs, spt...)
        _, res = integratebelief!(beliefs, rootj) # drop conditional mean
        return -res # score to be minimized (not maximized)
    end
    # autodiff does not currently work with ForwardDiff, ReverseDiff of Zygote,
    # because they cannot differentiate array mutation, as in: view(be.h, factorind) .+= h
    # consider solutions suggested here: https://fluxml.ai/Zygote.jl/latest/limitations/
    # Could this cache technique be used ?
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/136#issuecomment-237941790
    # https://juliadiff.org/ForwardDiff.jl/dev/user/limitations/
    # See PreallocationTools.jl package (below)
    opt = Optim.optimize(score, params_optimize(mod), Optim.LBFGS())
    # fixit: if BM and fixed root, avoid optimization bc there exists an exact alternative
    loglikscore = -Optim.minimum(opt)
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, loglikscore, opt
end

function calibrate_optimize_cliquetree_autodiff!(bufferbeliefs::GeneralLazyBufferCache,
        cgraph, prenodes::Vector{PN.Node},
        tbl::Tables.ColumnTable, taxa::AbstractVector,
        evomodelfun, # constructor function
        evomodelparams)
    spt = spanningtree_clusterlist(cgraph, prenodes)
    rootj = spt[3][1] # spt[3] = indices of parents. parent 1 = root
    mod = evomodelfun(evomodelparams...) # model with starting values
    #= 
    TODO: externalize the cache to avoid re-alocation (as done here) ?
    Or define cache inside of the function ?
    Note that the second option needs net in the arguments.
    lbc = PreallocationTools.GeneralLazyBufferCache(function (paramOriginal)
         model = evomodelfun(paramOriginal...)
         belief = init_beliefs_allocate(tbl, taxa, net, cgraph, model);
         return ClusterGraphBelief(belief)
     end)
    =#
    #= 
    TODO: GeneralLazyBufferCache is the "laziest" solution from PreallocationTools
    there might be more efficient solutions using lower level caches. 
    =#
    # score function using cache
    function score(θ) # θ: unconstrained parameters, e.g. log(σ2)
        paramOriginal = params_original(mod, θ)
        model = evomodelfun(paramOriginal...)
        dualBeliefs = bufferbeliefs[paramOriginal]
        # below: includes a reset
        init_beliefs_assignfactors!(dualBeliefs.belief, model, tbl, taxa, prenodes)
        init_factors_frombeliefs!(dualBeliefs.factor, dualBeliefs.belief)
        propagate_1traversal_postorder!(dualBeliefs, spt...)
        _, res = integratebelief!(dualBeliefs, rootj) # drop conditional mean
        return -res # score to be minimized (not maximized)
    end
    # optim using autodiff
    od = OnceDifferentiable(score, params_optimize(mod); autodiff = :forward);
    opt = Optim.optimize(od, params_optimize(mod), Optim.LBFGS())
    # fixit: if BM and fixed root, avoid optimization bc there exists an exact alternative
    loglikscore = -Optim.minimum(opt)
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, loglikscore, opt
end

"""
    calibrate_optimize_clustergraph!(beliefs::ClusterGraphBelief, clustergraph,
        nodevector_preordered, tbl::Tables.ColumnTable, taxa::AbstractVector,
        evolutionarymodel_name, evolutionarymodel_startingparameters,
        max_iterations)

Same as [`calibrate_optimize_cliquetree!`](@ref) above, except that the user can
supply an arbitrary `clustergraph` (including a clique tree) for the input
network. Optimization aims to maximize a free energy approximation (the negative
Bethe [`free_energy`](@ref)) to the ELBO for the log-likelihood of the data.
When `clustergraph` is a clique tree, the free energy approximation is exactly
equal to the ELBO and the log-likelihood.

The calibration repeatedly loops through a minimal set of spanning trees (see
[`spanningtrees_cover_clusterlist`](@ref)) that covers all edges in the cluster
graph, and does a postorder-preorder traversal for each tree. The loop runs till
calibration is detected or till `max_iterations` have passed, whichever occurs
first.
"""
function calibrate_optimize_clustergraph!(beliefs::ClusterGraphBelief,
        cgraph, prenodes::Vector{PN.Node},
        tbl::Tables.ColumnTable, taxa::AbstractVector,
        evomodelfun, # constructor function
        evomodelparams, maxiter::Integer=100)
    sch = spanningtrees_cover_clusterlist(cgraph, prenodes)
    mod = evomodelfun(evomodelparams...) # model with starting values
    function score(θ)
        model = evomodelfun(params_original(mod, θ)...)
        init_beliefs_assignfactors!(beliefs.belief, model, tbl, taxa, prenodes)
        init_factors_frombeliefs!(beliefs.factor, beliefs.belief)
        # fixit: raise warning if calibration is not attained within `maxiter`?
        regularizebeliefs!(beliefs, cgraph)
        calibrate!(beliefs, sch, maxiter, auto=true)
        return free_energy(beliefs)[3] # minimize Bethe free energy
    end
    opt = Optim.optimize(score, params_optimize(mod), Optim.LBFGS())
    fenergy = Optim.minimum(opt) 
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, fenergy, opt
end

"""
    calibrate_exact_cliquetree!(beliefs::ClusterGraphBelief, clustergraph,
        nodevector_preordered, node2belief,
        tbl::Tables.ColumnTable, taxa::AbstractVector,
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
    node2belief::AbstractVector{<:Integer},
    tbl::Tables.ColumnTable, taxa::AbstractVector,
    evomodelfun, # constructor function
    evomodelparams
)
    evomodelfun ∈ (UnivariateBrownianMotion, MvFullBrownianMotion) ||
        error("Exact optimization is only implemented for the univariate or full Brownian Motion.")
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
        clusterindex = node2belief[i]
        b = beliefs.belief[clusterindex]
        dimclus = length(b.nodelabel)
        # child ind in the cluster
        childind = findfirst(b.nodelabel .== i)
        # find parents: should be in the cluster, if node2belief is valid
        # fixit: why not use PN.getparents and PN.getparentedges below?
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
        if nodechild.leaf # tip node
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
    bestθ = (sigma2_hat, mu_hat, zeros(p, p)) # zero variance at the root: fixed
    bestmodel = evomodelfun(bestθ...)
    ## Get associated likelihood
    ## TODO: likelihood for the full BM (not implemented)
    loglikscore = NaN
    if evomodelfun == UnivariateBrownianMotion
        init_beliefs_allocate_atroot!(beliefs.belief, beliefs.factor, model) # fixit: should this be bestmodel? why do this anyway?
        init_beliefs_assignfactors!(beliefs.belief, bestmodel, tbl, taxa, prenodes)
        init_factors_frombeliefs!(beliefs.factor, beliefs.belief)
        propagate_1traversal_postorder!(beliefs, spt...)
        _, loglikscore = integratebelief!(beliefs, rootj)
    end

    return bestmodel, loglikscore
end

"""
    findClusterIndex(node::PN.Node, belief_vector)

In the belief in the vector that contains both the node and all its parents.
Throws an error if this cluster does not ex

fixit: delete, function not used, and prone to error because it uses
the belief's metadata instead of the belief's node labels.
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
