"""
    calibrate!(beliefs::ClusterGraphBelief, schedule, niterations=1;
        auto::Bool=false, info::Bool=false,
        verbose::Bool=true,
        update_residualnorm::Bool=true,
        update_residualkldiv::Bool=false)

Propagate messages in postorder then preorder for each tree in the `schedule`
list, for `niterations`. Each schedule "tree" should be a tuple
of 4 vectors as output by [`spanningtree_clusterlist`](@ref), where each vector
provides the parent/child label/index of an edge along which to pass a message,
and where these edges are listed in preorder. For example, the parent of the
first edge is taken to be the root of the schedule tree.
Calibration is evaluated after each schedule tree is run,
and said to be reached if all message residuals have a small norm,
based on [`iscalibrated_residnorm`](@ref).

Output: `true` if calibration is reached, `false` otherwise.

Optional keyword arguments:

- `auto`: If true, then belief updates are stopped after calibration is
  found to be reached.
  Otherwise belief updates continue for the full number of iterations.
- `info`: Is true, information is logged with the iteration number and
  schedule tree index when calibration is reached.
- `verbose`: log error messages about degenerate messages
- `update_residualnorm`
- `update_residualkldiv`

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

"""
    calibrate!(beliefs::ClusterGraphBelief, scheduletree::Tuple,
        verbose::Bool=true, up_resnorm::Bool=true, up_reskldiv::Bool=false)

Propage messages along the `scheduletree`, in postorder then preorder:
see [`propagate_1traversal_postorder!`](@ref).

Output: `true` if all message residuals have a small norm,
based on [`iscalibrated_residnorm`](@ref), `false` otherwise.
"""
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

Optional positional arguments after spanning tree, in this order (default value):

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
        # reset beliefs based on factors from new model parameters
        assignfactors!(beliefs.belief, model, tbl, taxa, prenodes, beliefs.cluster2fams)
        # init_beliefs_assignfactors!(beliefs.belief, model, tbl, taxa, prenodes)
        # no need to reset factors: free_energy not used on a clique tree
        init_messagecalibrationflags_reset!(beliefs, false)
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
        # reset beliefs based on factors from new model parameters
        assignfactors!(dualBeliefs.belief, model, tbl, taxa, prenodes, dualBeliefs.cluster2fams)
        # init_beliefs_assignfactors!(dualBeliefs.belief, model, tbl, taxa, prenodes)
        # no need to reset factors: free_energy not used on a clique tree
        init_messagecalibrationflags_reset!(dualBeliefs, false)
        propagate_1traversal_postorder!(dualBeliefs, spt...)
        _, res = integratebelief!(dualBeliefs, rootj) # drop conditional mean
        return -res # score to be minimized (not maximized)
    end
    # optim using autodiff
    od = OnceDifferentiable(score, params_optimize(mod); autodiff = :forward);
    opt = Optim.optimize(od, params_optimize(mod), Optim.LBFGS())
    loglikscore = -Optim.minimum(opt)
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, loglikscore, opt
end

"""
    calibrate_optimize_clustergraph!(beliefs::ClusterGraphBelief, clustergraph,
        nodevector_preordered, tbl::Tables.ColumnTable, taxa::AbstractVector,
        evolutionarymodel_name, evolutionarymodel_startingparameters,
        max_iterations=100, regularizationfunction=regularizebeliefs_bycluster!)

Same as [`calibrate_optimize_cliquetree!`](@ref) above, except that the user can
supply an arbitrary `clustergraph` (including a clique tree) for the input
network. Optimization aims to maximize the [`factored_energy`](@ref) approximation
to the ELBO for the log-likelihood of the data
(which is also the negative Bethe [`free_energy`](@ref)).
When `clustergraph` is a clique tree, the factored energy approximation is exactly
equal to the ELBO and the log-likelihood.

Cluster beliefs are regularized using [`regularizebeliefs_bycluster!`](@ref)
by default (other options include [`regularizebeliefs_bynodesubtree!`](@ref),
[`regularizebeliefs_onschedule!`](@ref)) before calibration.
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
        evomodelparams, maxiter::Integer=100,
        regfun=regularizebeliefs_bycluster!, # regularization function
 )
    sch = spanningtrees_clusterlist(cgraph, prenodes)
    mod = evomodelfun(evomodelparams...) # model with starting values
    function score(θ)
        model = evomodelfun(params_original(mod, θ)...)
        assignfactors!(beliefs.belief, model, tbl, taxa, prenodes, beliefs.cluster2fams)
        init_factors_frombeliefs!(beliefs.factor, beliefs.belief)
        init_messagecalibrationflags_reset!(beliefs, true)
        regfun(beliefs, cgraph)
        calibrate!(beliefs, sch, maxiter, auto=true)
        return free_energy(beliefs)[3] # to be minimized
    end
    opt = Optim.optimize(score, params_optimize(mod), Optim.LBFGS())
    iscalibrated_residnorm(beliefs) ||
      @warn "calibration was not reached. increase maxiter ($maxiter) or use a different cluster graph?"
    fenergy = -Optim.minimum(opt) 
    bestθ = Optim.minimizer(opt)
    bestmodel = evomodelfun(params_original(mod, bestθ)...)
    return bestmodel, fenergy, opt
end

"""
    calibrate_exact_cliquetree!(beliefs::ClusterGraphBelief,
        schedule,
        nodevector_preordered,
        tbl::Tables.ColumnTable, taxa::AbstractVector,
        evolutionarymodel_name)

For a Brownian Motion with a fixed root, compute the maximum likelihood estimate
of the prior mean at the root and the restricted maximum likelihood (REML)
estimate of the variance/covariance rate matrix
using analytical formulas relying on belief propagation,
using the data in `tbl` at leaves in the network.
These estimates are for the model with a prior variance of 0 at the root,
that is, a root state equal to the prior mean.

output: `(bestmodel, loglikelihood_score)`
where `bestmodel` is an evolutionary model created by `evolutionarymodel_name`,
containing the estimated model parameters.

assumptions:
- `taxa` should list the taxon names in the same order in which they come in the
  rows of `tbl`.
- `schedule` should provide a schedule to transmit messages between beliefs
  in `beliefs` (containing clusters first then sepsets). This schedule is
  assumed to traverse a clique tree for the input phylogeny,
  with the root cluster containing the root of the phylogeny in its scope.
- `nodevector_preordered` should list the nodes in this phylogeny, in preorder.
- `beliefs` should be of size and scope consistent with `evolutionarymodel_name`
  and data in `tbl`.
- a leaf should either have complete data, or be missing data for all traits.

Steps:
1. Calibrate `beliefs` in place according to the `schedule`, under a model
   with an infinite prior variance at the root.
2. Estimate parameters analytically.
3. Re-calibrate `beliefs`, to calculate the maximum log-likelihood of the
   fixed-root model at the estimated optimal parameters, again modifying
   `beliefs` in place. (Except that beliefs with the root node in scope are
   re-assigned to change their scoping dimension.)

Warning: there is *no* check that the beliefs and schedule are consistent
with each other.
"""
function calibrate_exact_cliquetree!(beliefs::ClusterGraphBelief{B},
    spt, # node2belief may be needed if pre & post calibrations are moved outside
    prenodes::Vector{PN.Node},
    tbl::Tables.ColumnTable, taxa::AbstractVector,
    evomodelfun # constructor function
) where B<:AbstractBelief{T} where T
    evomodelfun ∈ (UnivariateBrownianMotion, MvFullBrownianMotion) ||
        error("Exact optimization is only implemented for the univariate or full Brownian Motion.")
    p = length(tbl)
    # check for "clean" data at the tips: data at 0 or all p traits
    function clean(v) s = sum(v); s==0 || s==p; end
    for ic in 1:nclusters(beliefs)
        b = beliefs.belief[ic]
        all(map(clean, eachslice(inscope(b), dims=2))) ||
          error("some leaf must have partial data: cluster $(b.metadata) has partial traits in scope")
    end
    ## calibrate beliefs using infinite root and identity rate variance
    calibrationparams = evomodelfun(LA.diagm(ones(p)), zeros(p), LA.diagm(repeat([Inf], p)))
    init_beliefs_allocate_atroot!(beliefs.belief, beliefs.factor, beliefs.messageresidual,
        calibrationparams, beliefs.cluster2fams) # in case root status changed
    node2belief = assignfactors!(beliefs.belief, calibrationparams, tbl, taxa, prenodes, beliefs.cluster2fams)
    # node2belief = init_beliefs_assignfactors!(beliefs.belief, calibrationparams, tbl, taxa, prenodes)
    # no need to reset factors: free_energy not used on a clique tree
    init_messagecalibrationflags_reset!(beliefs, false)
    calibrate!(beliefs, [spt])

    ## Compute μ hat from root belief
    rootj = spt[3][1] # spt[3] = indices of parents. parent 1 = root
    exp_root, _ = integratebelief!(beliefs, rootj)
    mu_hat = exp_root[scopeindex((1,), beliefs.belief[rootj])]

    ## Compute σ² hat from conditional moments
    tmp_num = zeros(T, p, p)
    tmp_den = zero(T)
    for i in 2:length(prenodes) # loop over non-root notes (1=root in pre-order)
        nodechild = prenodes[i]
        clusterindex = node2belief[i] # index of cluster to which its node family factor was assigned
        b = beliefs.belief[clusterindex]
        dimclus = length(b.nodelabel)
        childind = findfirst(lab == i for lab in b.nodelabel) # child index in cluster
        # parents should all be in the cluster, if node2belief is valid
        all_parent_edges = PN.Edge[]; parind = Int[] # parent(s) indices in cluster
        edge_length = zero(T) # parent edge length if 1 parent; sum of γ² t over all parent edges
        all_gammas = zeros(T, dimclus)
        for ee in nodechild.edge
            getchild(ee) === nodechild || continue # skip below if child edge
            push!(all_parent_edges, ee)
            pn = getparent(ee) # parent node
            pi = findfirst(prenodes[j].name == pn.name for j in b.nodelabel)
            push!(parind, pi)
            all_gammas[pi] = ee.gamma
            edge_length += ee.gamma * ee.gamma * ee.length
        end
        edge_length == 0.0 && continue # 0 length => variance parameter absent from factor
        # moments
        exp_be, _ = integratebelief!(b)
        vv = inv(b.J)
        if nodechild.leaf # tip node
            # there should be only 1 parent
            length(parind) == 1 || error("leaf $(nodechild.name) does not have 1 parent...")
            inscope_i_pa = scopeindex(parind[1], b)
            isempty(inscope_i_pa) && continue  # no data at or below: do nothing
            # find tip data
            # TODO later: replace tbl,taxa by tipvalue, to avoid re-constructing it over and over
            i_row = findfirst(isequal(nodechild.name), taxa)
            !isnothing(i_row) || error("leaf $(nodechild.name) is missing from the data's list of taxa")
            tipvalue = [tbl[v][i_row] for v in eachindex(tbl)]
            diffExp = view(exp_be, inscope_i_pa) - tipvalue
            tmp_num += diffExp * transpose(diffExp) ./ edge_length
            # assumes that vv ∝ (co)variance rate matrix R_test
            tmp_den += 1 - vv[1, 1] / edge_length
        else # internal node
            inscope_i_ch = scopeindex(childind, b)
            isempty(inscope_i_ch) && continue  # no data at or below: do nothing
            begic = inscope_i_ch[1]
            diffExp = view(exp_be, inscope_i_ch) # init with child node
            diffVar = vv[begic, begic]
            # sum over parent nodes
            inscope_i_pa = [scopeindex(j, b) for j in parind]
            for (j1, j1_ii) in zip(parind, inscope_i_pa)
                # exp and covar with child
                diffExp -= all_gammas[j1] .* view(exp_be, j1_ii)
                diffVar -= 2 * all_gammas[j1] * vv[begic, j1_ii[1]]
                # parents var covar
                for (j2, j2_ii) in zip(parind, inscope_i_pa)
                    diffVar += all_gammas[j1] * all_gammas[j2] * vv[j1_ii[1], j2_ii[1]]
                end
            end
            tmp_num += diffExp * transpose(diffExp) ./ edge_length
            tmp_den += 1 - diffVar / edge_length
        end
    end
    sigma2_hat = tmp_num ./ tmp_den

    ## Get optimal paramters
    bestθ = (sigma2_hat, mu_hat, zeros(T, p, p)) # zero variance at the root: fixed
    bestmodel = evomodelfun(bestθ...)
    ## Get associated likelihood
    loglikscore = NaN
    init_beliefs_allocate_atroot!(beliefs.belief, beliefs.factor, beliefs.messageresidual, bestmodel, beliefs.cluster2fams)
    # init_beliefs_assignfactors!(beliefs.belief, bestmodel, tbl, taxa, prenodes)
    assignfactors!(beliefs.belief, bestmodel, tbl, taxa, prenodes, beliefs.cluster2fams)
    init_messagecalibrationflags_reset!(beliefs, false)
    calibrate!(beliefs, [spt])
    _, loglikscore = integratebelief!(beliefs, rootj)

    return bestmodel, loglikscore
end
