"""
    propagate_1treetraversal!(beliefs, spanningtree, postorder=true::Bool)

Messages are propagated from the tips to the root of the tree by default,
or from the root to the tips if `postorder` is false.

fixit: `spanningtree` should be fixit,
as produced by [`spanningtree_clusterlist`](@ref)

warning: beliefs are assumed to be ordered with clusters first (sepsets last),
and indices in the spanning tree information should correspond to indices
in `beliefs`.
"""
function propagate_1treetraversal!(beliefs, spt, postorder=true::Bool)
    rootj = spt[3][1]
    # iterate over (parent <- sepset <- child) in postorder
    #           or (child <- sepset <- parent) in preorder
    # propagate_belief!(b[clu1], b[sepset], b[clu2])
    return integratebelief!(beliefs[rootj])
end

"""
    calibrate!(beliefs, clustergraph, nodevector_preordered, niterations=10)

fixit
`niterations` is ignored and set to 1 if the cluster graph is a clique tree.
"""
function calibrate!(beliefs, clustergraph, prenodes::Vector{PN.Node}, niter=10)
    niter = (is_tree(clustergraph) ? 1 : niter)
    for _ in 1:niter
        spt = spanningtree_clusterlist(clustergraph, prenodes)
        # fixit: pick a *random* spanning tree instead, for a general cluster graph
        propagate_1treetraversal!(beliefs, spt, true)  # postorder
        propagate_1treetraversal!(beliefs, spt, false) # preorder
    end
end
