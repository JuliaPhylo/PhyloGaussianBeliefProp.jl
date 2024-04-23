"""
    isdegenerate(node)

`true` if *all* parent edges of `node` have length 0, `false` otherwise.
Intended for hybrid nodes, as tree edges of length 0 should be suppressed
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
    ishybridsinglepositivechild(node)

`true` if `node` is a hybrid node with a single child edge of positive length.
If it [`isdegenerate`](@ref)) (all its parent edges have length 0) and
if its child is a tree node, then it could be removed from scope:
see [`unscope`](@ref).
"""
ishybridsinglepositivechild(v::PN.Node) = v.hybrid && hassinglechild(v) && getchildedge(v).length > 0.0

"""
    unscope(node)

`true` if `node` is a hybrid node with a single child edge of positive length,
and if its child node is a tree node.
If it [`isdegenerate`](@ref)) (all its parent edges have length 0) then it
could be removed from scope:
see [`addtreenode_belowdegeneratehybrid!`](@ref).
"""
unscope(v::PN.Node) = ishybridsinglepositivechild(v) && !(getchild(v).hybrid)

"""
    hasdegenerate(net)

`true` if degenerate nodes remain in scope, that is, if there exists a tree
edge of length 0, or if there exists a hybrid node with all parent edges of
length 0 and with 2 or more children edges, or with 1 child edge of length 0.
"""
hasdegenerate(net::HybridNetwork) = any(isdegenerate(v) && !unscope(v) for v in net.node)


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

"""
    addtreenode_belowdegeneratehybrid!(net::HybridNetwork)

If a degenerate hybrid node h1 has 1 child edge of length t>0 to a hybrid child h2:
break the edge by adding a tree node at distance t from h1 and 0 from h2.
That way, h1 may be removed from scope.
This is done iteratively, as h2 may become degenerate after this operation.
See [`shrinkdegenerate_treeedges`](@ref) to remove degenerate internal tree nodes,
and [`hasdegenerate`](@ref) to check if `net` still has degenerate nodes.
"""
function addtreenode_belowdegeneratehybrid!(net::HybridNetwork)
    restart = true
    while restart
        for hyb in net.hybrid
            (isdegenerate(hyb) && ishybridsinglepositivechild(hyb)) || continue
            che = getchildedge(hyb)
            getchild(che).hybrid || continue
            t = che.length
            _,newe = PN.breakedge!(che, net) # hyb --newe--> newv --che--> hybridchild
            newe.length = t
            che.length = 0.0 # the hybrid child may now be degenerate, so restart
            break
        end
        restart=false
    end
    return net
end
