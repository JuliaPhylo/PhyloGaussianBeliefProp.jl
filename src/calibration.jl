"""
    propagate_1treetraversal!(beliefs, spanningtree, postorder=true::Bool)

Messages are propagated from the tips to the root of the tree by default,
or from the root to the tips if `postorder` is false.
"""
function propagate_1treetraversal!(beliefs, spanningtree, postorder=true::Bool)
    return beliefs
end

"""
    calibrate!(beliefs, clustergraph)

"""
function calibrate!(beliefs, clustergraph, niter=10)
    # pick a spanning tree
    # propagate with 1 postorder traversal of the tree
    # propagate with 1 preorder traversal of the tree
    # re-do niter times
    return beliefs
end
