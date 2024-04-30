"""
    calibrate!(beliefs::ClusterGraphBelief, schedule, niterations=1;
        auto::Bool=false, info::Bool=false)

Propagate messages in postorder then preorder for each tree in the `schedule`
list, for `niterations`. Each schedule "tree" should be a tuple
of 4 vectors as output by [`spanningtree_clusterlist`](@ref), where each vector
provides the parent/child label/index of an edge along which to pass a message,
and where these edges are listed in preorder. For example, the parent of the
first edge is taken to be the root of the schedule tree.

Calibration is evaluated after each schedule tree is run, and said to be reached
if all message residuals have a small norm.
If `info` is true, information is sent with the iteration number and tree index
when calibration is reached.
If `auto` is true, then belief updates are stopped after calibration is found
to be reached. Otherwise belief updates continue for the full number of iterations.

Output: `true` if calibration is reached, `false` otherwise.

optional positional arguments (default value):
- `verbose` (true): log error messages about degenerate messages
- `update_residualnorm` (true)
- `update_residualkldiv` (false)

See also: [`iscalibrated_residnorm`](@ref)
and [`iscalibrated_residnorm!`](@ref) for the tolerance and norm used by default,
to declare calibration for a given sepset message (in 1 direction).
"""
function calibrate!(beliefs::ClusterGraphBelief, schedule::AbstractVector,
    niter::Integer=1; auto::Bool=false, info::Bool=false,
    verbose::Bool=true,
    update_residualnorm::Bool=true,
    update_residualkldiv::Bool=false,
)
    iscal = false
    for i in 1:niter
        for (j, spt) in enumerate(schedule)
            iscal = calibrate!(beliefs, spt, verbose, update_residualnorm, update_residualkldiv)
            if iscal
                info && @info "calibration reached: iteration $i, schedule tree $j"
                auto && return true
            end
        end
    end
    return iscal
end
function calibrate!(beliefs::ClusterGraphBelief, spt::Tuple,
    verbose::Bool=true,
    up_resnorm::Bool=true,
    up_reskldiv::Bool=false,
)
    propagate_1traversal_postorder!(beliefs, spt..., verbose, up_resnorm, up_reskldiv)
    propagate_1traversal_preorder!(beliefs, spt..., verbose, up_resnorm, up_reskldiv)
    return iscalibrated_residnorm(beliefs)
end

"""
    propagate_1traversal_postorder!(beliefs::ClusterGraphBelief, spanningtree...)
    propagate_1traversal_preorder!(beliefs::ClusterGraphBelief,  spanningtree...)

Messages are propagated along the spanning tree, from the tips to the root by
`propagate_1traversal_postorder!` and from the root to the tips by
`propagate_1traversal_preorder!`.

The "spanning tree" should be a tuple of 4 vectors as output by
[`spanningtree_clusterlist`](@ref), meant to list edges in preorder.
Its nodes (resp. edges) should correspond to clusters (resp. sepsets) in
`beliefs`: labels and indices in the spanning tree information
should correspond to indices in `beliefs`.
This condition holds if beliefs are produced on a given cluster graph and if the
tree is produced by [`spanningtree_clusterlist`](@ref) on the same graph.

optional positional arguments after spanning tree, in this order (default value):
- `verbose` (true): log error messages about degenerate messages that failed
  to be passed.
- `update_residualnorm` (true): to update each message residual's `iscalibrated_resid`
- `update_residualkldiv` (false): to update each message residual's field
  `kldiv`: KL divergence between the new and old sepset beliefs,
  normalized to be considered as (conditional) distributions.
"""
function propagate_1traversal_postorder!(
    beliefs::ClusterGraphBelief,
    pa_lab, ch_lab, pa_j, ch_j,
    verbose::Bool=true,
    update_residualnorm::Bool=true,
    update_residualkldiv::Bool=false,
)
    b = beliefs.belief
    mr = beliefs.messageresidual
    # (parent <- sepset <- child) in postorder
    for i in reverse(1:length(pa_lab))
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        sepset = b[ss_j]
        mrss = mr[(pa_lab[i], ch_lab[i])]
        flag = propagate_belief!(b[pa_j[i]], sepset, b[ch_j[i]], mrss)
        if isnothing(flag)
            update_residualnorm && iscalibrated_residnorm!(mrss)
            update_residualkldiv && residual_kldiv!(mrss, sepset)
        elseif verbose
            @error flag.msg
        end
    end
end

function propagate_1traversal_preorder!(
    beliefs::ClusterGraphBelief,
    pa_lab, ch_lab, pa_j, ch_j,
    verbose::Bool=true,
    update_residualnorm::Bool=true,
    update_residualkldiv::Bool=false,
)
    b = beliefs.belief
    mr = beliefs.messageresidual
    # (child <- sepset <- parent) in preorder
    for i in eachindex(pa_lab)
        ss_j = sepsetindex(pa_lab[i], ch_lab[i], beliefs)
        sepset = b[ss_j]
        mrss = mr[(ch_lab[i], pa_lab[i])]
        flag = propagate_belief!(b[ch_j[i]], sepset, b[pa_j[i]], mrss)
        if isnothing(flag)
            update_residualnorm && iscalibrated_residnorm!(mrss)
            update_residualkldiv && residual_kldiv!(mrss, sepset)
        elseif verbose
            @error flag.msg
        end
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
        max_iterations=100)

Same as [`calibrate_optimize_cliquetree!`](@ref) above, except that the user can
supply an arbitrary `clustergraph` (including a clique tree) for the input
network. Optimization aims to maximize the [`factored_energy`](@ref) approximation
to the ELBO for the log-likelihood of the data
(which is also the negative Bethe [`free_energy`](@ref)).
When `clustergraph` is a clique tree, the factored energy approximation is exactly
equal to the ELBO and the log-likelihood.

The calibration repeatedly loops through a minimal set of spanning trees (see
[`spanningtrees_clusterlist`](@ref)) that covers all edges in the cluster
graph, and does a postorder-preorder traversal for each tree. The loop runs till
calibration is detected or till `max_iterations` have passed, whichever occurs
first.
"""
function calibrate_optimize_clustergraph!(beliefs::ClusterGraphBelief,
        cgraph, prenodes::Vector{PN.Node},
        tbl::Tables.ColumnTable, taxa::AbstractVector,
        evomodelfun, # constructor function
        evomodelparams, maxiter::Integer=100)
    sch = spanningtrees_clusterlist(cgraph, prenodes)
    mod = evomodelfun(evomodelparams...) # model with starting values
    function score(θ)
        model = evomodelfun(params_original(mod, θ)...)
        init_beliefs_assignfactors!(beliefs.belief, model, tbl, taxa, prenodes)
        init_factors_frombeliefs!(beliefs.factor, beliefs.belief)
        # fixit: raise warning if calibration is not attained within `maxiter`?
        regularizebeliefs_bycluster!(beliefs, cgraph)
        calibrate!(beliefs, sch, maxiter, auto=true)
        return free_energy(beliefs)[3] # to be minimized
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
        evolutionarymodel_name, evomodelparams)

For a Brownian Motion with a fixed root, compute the maximum likelihood
(for the mean) and restricted maximum likelihood (REML, for the variance rate)
parameter estimates using analytical formulas relying on belief propagation.

fixit: why have argument `evomodelparams` if it's only use to estimate the mean,
and required to have infinite prior variance at the root?
Could we make evolutionarymodel_name be able to construct a default model
with variance 1, mean 0, infinite prior (if we don't have any default yet)
so that we don't need argument `evomodelparams`?

output: `(bestmodel, loglikelihood_score)`
where `bestmodel` is an evolutionary model created by `evolutionarymodel_name`,
containing the estimated model parameters.

The `clustergraph` is assumed to be a clique tree for the input tree,
whose nodes in preorder are `nodevector_preordered`.
Optimization aims to maximize the likelihood
of the data in `tbl` at leaves in the network.
The taxon names in `taxa` should appear in the same order as they come in `tbl`.
The parameters being optimized are the variance rate(s) and prior mean(s)
at the root. The prior variance at the root is fixed to zero.

The calibration does a postorder of the clique tree only,
using the optimal parameters, to get the likelihood
at the root *without* the conditional distribution at all nodes, modifying
`beliefs` in place. Therefore, if the distribution of ancestral states is sought,
an extra preorder calibration would be required.

fixit: the variance estimator requires the conditional distribution at all nodes,
so it this docstring obsolete?
fixit: as this function is not supposed to perform the calibration, should it
be renamed to avoid `calibrate_*`? perhaps to: `fit_homogeneous!()`
or `fit_analytical!()` ??
It seems that a clique tree is required to get the true ML/REML estimates,
but the function should still work if the cluster graph was not a clique tree
(would get the associated approximation to the ML/REML estimates).
If so, deleting "cliquetree" from the name and from the docstring would be good.

Warning: there is *no* check that the cluster graph is in fact a clique tree,
or that calibration has been reached.
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
    #= fixit: alternatively, pass `rootj` as an argument to this function,
       since calibration would have to be done prior to using this function.
       Or: if calibration is full (postorder + preorder), then find *any* cluster
       containing the root node, with pre-index 1.
    =#
    spt = spanningtree_clusterlist(cgraph, prenodes)
    rootj = spt[3][1] # spt[3] = indices of parents. parent 1 = root
    exp_root, _ = integratebelief!(beliefs, rootj)
    mu_hat = exp_root[(end-p+1):end]
    # fixit: use scopeindex((root_node_label), root_belief) instead?

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
    init_beliefs_allocate_atroot!(beliefs.belief, beliefs.factor, beliefs.messageresidual, bestmodel) # fixit: should this be bestmodel? why do this anyway?
    init_beliefs_assignfactors!(beliefs.belief, bestmodel, tbl, taxa, prenodes)
    init_factors_frombeliefs!(beliefs.factor, beliefs.belief)
    propagate_1traversal_postorder!(beliefs, spt...)
    _, loglikscore = integratebelief!(beliefs, rootj)

    return bestmodel, loglikscore
end
