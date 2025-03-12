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

Tuple of (edge length, edge γ, index of parent node in `net.vec_node`)
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
        push!(i_par, findfirst(isequal(getparent(e)), net.vec_node))
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
    # find prefix for naming new nodes
    m = match(r"(^\D+)\d+$", net.node[net.rooti].name)
    prefix = (isnothing(m) ? "I" : m.captures[1])
    while restart
        for hyb in net.hybrid
            (isdegenerate(hyb) && ishybridsinglepositivechild(hyb)) || continue
            che = getchildedge(hyb)
            getchild(che).hybrid || continue
            t = che.length
            _,newe = PN.breakedge!(che, net) # hyb --newe--> newv --che--> hybridchild
            newe.length = t
            che.length = 0.0 # the hybrid child may now be degenerate, so restart
            preprocessnet!(net, prefix) # name new node, update net.vec_node
            break
        end
        restart=false
    end
    return net
end

"""
    isdegenerate_extendedfamily_covered(nodeindex, clustermembers,
        node2family, node2degen, node2fixed)

Check for the absence of an intermediate node in a degenerate extended family,
for an input node `v` represented by its preorder index, looking at its ancestors
in the input cluster `C` represented by its members, a vector of node preorder indices.

Output: tuple of 2 booleans `(b1,b2)` where
- `b1` is true if `v` is degenerate conditional on its ancestors that are
  present in `C`, false otherwise (e.g. `v` is not degenerate conditional on
  its parents, or if none of its parents are in `C`)
- `b2` is true if `C` is a "good cover" for `v` in the following sense:
  either `v` is not generate given its ancestors in `C`,
  or `C` contains all of `v`'s parents.

By `v` is "degenerate" we mean that its distribution is deterministic, taking as
value a linear (or affine) combination of its ancestor(s)' value(s).
"""
function isdegenerate_extendedfamily_covered(
    nodeindex::Integer,
    clustermembers::Vector{<:Integer},
    node2family,
    node2degen,
    node2fixed,
)
    b1 = node2degen[nodeindex]
    b2 = true
    b1 || return (b1,b2) # node is not degenerate
    # if we get here: degenerate given its parents
    for ip in Iterators.drop(node2family[nodeindex], 1)
        node2fixed[ip] && continue # skip parents with a fixed value: not in scope
        ip in clustermembers && continue # skip parents present in cluster
        b1p, _ = isdegenerate_extendedfamily_covered(ip, clustermembers,
            node2family, node2degen, node2fixed)
        if b1p # this parent is degenerate given ancestors in cluster
            b2 = false
        else # parent *not* degenerate given cluster
            return (false, true)
        end
    end
    return (b1,b2)
end

"""
    isdegenerate_extendedfamily_covered(clustermembers, args...)
    isdegenerate_extendedfamily_covered(cluster2nodes, args...)

Boolean:
`true` if for each node `v` in the input cluster `C`,
`C` contains all intermediate nodes in the degenerate extended family of `v`.
`false` if this property fails for one or more node `v` in `C`.

The first method considers a single cluster, containing `clustermembers`.
The second method consider all clusters in a cluster graph,
and returns true if *all* clusters meet the condition.

We say that `C` contains all intermediate ancestors for `v` in `v`'s degenerate
extended family if:
- for any set of ancestors `A ⊆ C` such that `v` is degenerate conditional on `A`,
- for any `p` intermediate between `A` and `v` (that is `p` is a descendant of
  `A` and ancestor of `v`),
- then we have that `p ∈ C`.

We check that this condition holds for all nodes `v` in `C` by checking,
more simply, that for any `v` degenerate given its ancestors in `C`,
all of `v`'s parents are in `C`.

positional arguments `args...`: `node2family, node2degen, node2fixed`.
"""
function isdegenerate_extendedfamily_covered(
    clustermembers::Vector{<:Integer},
    args...
)
    for ni in reverse(clustermembers) # loop through nodes in pre-order
        _, b2 = isdegenerate_extendedfamily_covered(ni, clustermembers, args...)
        b2 || return false
    end
    return true
end
function isdegenerate_extendedfamily_covered(
    cgraph::MetaGraph,
    args...
)
    for lab in labels(cgraph)
        clustermembers = cgraph[lab][2] # nodes listed by their preorder index
        if !isdegenerate_extendedfamily_covered(clustermembers, args...)
            @error "cluster $lab is missing an intermediate ancestor in a generate family"
            return false
        end
    end
    return true
end
