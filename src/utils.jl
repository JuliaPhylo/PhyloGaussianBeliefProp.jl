"""
    isdegenerate(node)

`true` if *all* parent edges of `node` have length 0, `false` otherwise.
Intented for hybrid nodes, as tree edges of length 0 should be suppressed
before trait evolution analysis.
"""
function isdegenerate(node::PN.Node)
    nparents = 0
    for e in node.edge # loop over parent edges
        getchild(e)===node || continue
        nparents += 1
        e.length > 0.0 && return false
    end
    return nparents > 0 # true unless it's the root
end

"""
    unscope(node)

`true` if `node` is a hybrid node with all parent edges of length 0
(see [`isdegenerate`](@ref)) and has a single child of positive length.
"""
function unscope(node::PN.Node)
    (node.hybrid && isdegenerate(node)) || return false
    nchildren = 0
    for e in node.edge # loop over child edges
        getparent(e) === node || continue
        nchildren += 1
        e.length > 0.0 || return false
    end
    return nchildren == 1 # true unless 1 edge of >0 length
end

"""
    parentinformation(node, net)

Tuple of (edge length, edge γ, index of parent node in `net.nodes_changed`)
for all parent edges of `node`. Assumes that `net` has been preordered before.
"""
function parentinformation(hyb::PN.Node, net::HybridNetwork)
    t = Float64[]
    γ = Float64[]
    i_par = Int[]
    for e in hyb.edge # loop over parent edges
        getchild(e)===hyb || continue
        push!(t, e.length)
        push!(γ, e.gamma)
        push!(i_par, findfirst(isequal(getparent(e)), net.nodes_changed))
    end
    return (t, γ, i_par)
end

"""
   shrinkdegenerate_treeedges(net::HybridNetwork)

Network obtained from `net` with any non-external tree edge of length 0 suppressed.
Returns an error if any edge length is missing or negative,
or if any γ is missing or non positive.
It is assumed that γs sum to 1 across partner hybrid edges.
"""
function shrinkdegenerate_treeedges(net::HybridNetwork)
    str = "Trait evolution models need the network to have edge lengths and γs."
    PN.check_nonmissing_nonnegative_edgelengths(net, str)
    if any(e.gamma <= 0 for e in net.edge)
        error("Branch number $(e.number) has a missing or non-positive γ.\n" * str)
    end
    net = deepcopy(net)
    redo = true
    while redo
        for e in net.edge
            e.hybrid && continue
            if e.length == 0.0
                getchild(e).leaf && error("external edge $(e.number) has length 0")
                PN.shrinkedge!(net, e) # changes net.edge
                break # of for loop over net.edge
            end
        end
        redo = false
    end
    return net
end
