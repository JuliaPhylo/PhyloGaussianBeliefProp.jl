@enum BeliefType bclustertype=1 bsepsettype=2

abstract type AbstractBelief end

nodelabels(b::AbstractBelief) = b.nodelabel
nvar(b::AbstractBelief)       = b.nvar
inscope(b::AbstractBelief) = b.inscope
nodedimensions(b::AbstractBelief) = map(sum, eachslice(inscope(b), dims=2))
dimension(b::AbstractBelief)  = sum(inscope(b))
mvnormcanon(b::AbstractBelief) = MvNormalCanon(b.μ, b.h, PDMat(LinearAlgebra.Symmetric(b.J)))

struct ClusterBelief{Vlabel<:AbstractVector,T<:Real,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief
    "Integer label for nodes in the cluster"
    nodelabel::Vlabel # StaticVector{N,Tlabel}
    "Total number of variables at each node"
    nvar::Int
    """Matrix inscope[i,j] is `false` if variable `i` at node `j` is / will be
    removed from scope, to avoid issues from 0 precision or infinite variance; or
    when there is no data for variable `i` below node `j` (in which case tracking
    this variable is only good for prediction, not for learning parameters).
    """
    inscope::BitArray
    """belief = exponential quadratic form, using the canonical parametrization:
       mean μ and potential h = inv(Σ)μ, both of type V,
       precision J = inv(Σ) of type P --so the normalized belief is `MvNormalCanon{T,P,V}`
       constant g to get the unnormalized belief"""
    # belief::MvNormalCanon{T,P,V},
    # downside: PDMat J not easily mutable, stores cholesky computed upon construction only
    μ::V
    h::V
    J::P
    g::MVector{1,Float64} # mutable
    "belief type: cluster (node in cluster grahp) or sepset (edge in cluster graph)"
    type::BeliefType
    "metadata, e.g. index in cluster graph"
    metadata::M
end

function Base.show(io::IO, b::ClusterBelief)
    disp = "belief for " * (b.type == bclustertype ? "Cluster" : "SepSet") * " $(b.metadata),"
    disp *= " $(nvar(b)) variables × $(length(nodelabels(b))) nodes, dimension $(dimension(b)).\n"
    disp *= "Node labels: "
    print(io, disp)
    print(io, nodelabels(b))
    print(io, "\nvariable × node matrix of non-degenerate beliefs:\n")
    show(io, inscope(b))
    print(io, "\nexponential quadratic belief, parametrized by\nμ: $(b.μ)\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    ClusterBelief(nodelabels, nvar, inscope, belieftype, metadata)

Constructor to allocate memory for one cluster, and initialize objects with 0s
to initilize the belief with the constant function exp(0)=1.
"""
function ClusterBelief(nl::AbstractVector{Tlabel}, nvar::Integer, inscope::BitArray, belief, metadata) where Tlabel<:Integer
    nnodes = length(nl)
    nodelabels = SVector{nnodes}(nl)
    size(inscope) == (nvar,nnodes) || error("inscope of the wrong size")
    cldim = sum(inscope)
    T = Float64
    μ = MVector{cldim,T}(zero(T) for _ in 1:cldim)  # zeros(T, cldim)
    h = MVector{cldim,T}(zero(T) for _ in 1:cldim)
    J = MMatrix{cldim,cldim,T}(undef) # Matrix{T}(LinearAlgebra.I, cldim, cldim)
    fill!(J, zero(T))
    g = MVector{1,Float64}(0.0)
    ClusterBelief{typeof(nodelabels),T,typeof(J),typeof(h),typeof(metadata)}(
        nodelabels,nvar,inscope,μ,h,J,g,belief,metadata)
end

function scopeindex(node_labels, belief::AbstractBelief)
    inscope = inscope(belief)
    node_j = indexin(node_labels, nodelabels(belief))
    node_dims = map(sum, eachslice(inscope, dims=2))
    node_cumsum = cumsum(node_dims)
    res = Vector{Int}(undef, node_cumsum[end])
    i0=0
    for jj in node_j
        k0 = (jj==1 ? 0 : node_cumsum[jj-1])
        for _ in 1:node_dims[jj]
            i0 += 1; k0 += 1
            res[i0] = k0
        end
    end
    return res
end

"""
    init_beliefs_allocate(tbl::Tables.ColumnTable, taxa, net, clustergraph)

Vector of beliefs, initialized to the constant function exp(0)=1,
one for each cluster then one for each sepset in `clustergraph`.
`tbl` is used to know which leaf in `net` has data for which variable,
so as to remove from the scope each variable without data below it.
`taxa` should be a vector with taxon names in the same order as they come in
the table of data `tbl`.
Also removed from scope is any hybrid node that is degenerate and who has
a single child edge of positive length.
"""
function init_beliefs_allocate(tbl::Tables.ColumnTable, taxa::AbstractVector,
        net::HybridNetwork, clustergraph)
    nvar = length(tbl)
    nnodes = length(net.nodes_changed)
    nnodes > 0 ||
        error("the network should have been pre-ordered, with indices used in cluster graph")
    #= hasdata: to know, for each node, whether that node has a descendant
                with data, for each variable.
    If not: that node can be removed from all clusters & sepsets (true??).
    If yes and the node is a tip: the evidence should be used later.
    If yes for all variables and tip: remove tip after the evidence is absorbed.
    =#
    hasdata = falses(nvar, nnodes)
    for i_node in reverse(eachindex(net.nodes_changed))
        node = net.nodes_changed[i_node]
        nodelab = node.name
        i_row = findfirst(isequal(nodelab), taxa)
        if !isnothing(i_row) # the node has data: it should be a tip!
            node.leaf || error("A node with data is internal, should be a leaf")
            for v in 1:nvar
              hasdata[v,i_node] = !ismissing(tbl[v][i_row])
            end
        end
        if node.leaf
            all(!hasdata[v,i_node] for v in 1:nvar) && @error("tip $nodelab in network without any data")
            continue
        end
        for e in node.edge
            ch = getchild(e)
            ch !== node || continue # skip parent edges
            i_child = findfirst( n -> n===ch, net.nodes_changed)
            isempty(i_child) && error("oops, child (number $(ch.number)) not found in nodes_changed")
            hasdata[:,i_node] .|= hasdata[:,i_child] # bitwise or
        end
        all(!hasdata[v,i_node] for v in 1:nvar) && @error("internal node $nodelab without any data below")
    end
    #= next: create a belief for each cluster and sepset. inscope =
    'has partial information and non-degenerate variance or precision?' =
    - 'hasdata?' at internal nodes (assumes non-degenerate transitions)
    - false at tips (assumes all data are at tips)
    - false at degenerate hybrid node with 1 tree child edge of positive length
    =#
    function build_inscope(set_nodeindices)
        inscope = falses(nvar, length(set_nodeindices)) # remove from scope by default
        for (i,i_node) in enumerate(set_nodeindices)
            node = net.nodes_changed[i_node]
            (node.leaf || (isdegenerate(node) && unscope(node))) && continue # inscope[:,i] already false
            inscope[:,i] .= hasdata[:,i_node]
        end
        return inscope
    end
    beliefs = ClusterBelief[]
    for cllab in labels(clustergraph)
        nodeindices = clustergraph[cllab][2]
        inscope = build_inscope(nodeindices)
        push!(beliefs, ClusterBelief(nodeindices, nvar, inscope, bclustertype, cllab))
    end
    for sslab in edge_labels(clustergraph)
        nodeindices = clustergraph[sslab...]
        inscope = build_inscope(nodeindices)
        push!(beliefs, ClusterBelief(nodeindices, nvar, inscope, bsepsettype, sslab))
    end
    return beliefs
end

"""
    init_beliefs_assignfactors!(fixit)

Initialize cluster beliefs prior to belief propagation, by assigning each
factor to one cluster. There is one factor for each node v in the network:
distribution of X_v conditional on its parent X_pa(v) if v is not the root,
and prior distribution for x_root.

Assumptions:
- `net` is already preordered, and belief node labels contain the index of
  each node in `net.nodes_changed`.
fixit
"""
function init_beliefs_assignfactors!(beliefs, model::EvolutionaryModel,
        tbl::Tables.ColumnTable, taxa::AbstractVector, net::HybridNetwork, clustergraph)
    prenodes = net.nodes_changed # nodes in pre-order
    visited = falses(length(prenodes))
    for (i_node,node) in enumerate(prenodes)
        visited[i_node] && continue # skip tree child of unscoped degenerate hybrid
        visited[i_node] = true
        nodelab = node.name
        if i_node == 1 # root
            i_b = findfirst(x -> 1 ∈ x, nodelabels(b) for b in beliefs)
            isnothing(i_b) && error("no cluster containing the root, number $(node.number).")
            be = beliefs[i_b]
            be.type == bclustertype || error("belief $(be.metadata) is of type $(be.type)")
            i_inscope = (1,)
            μ,h,J,g = factor_root(model)
        elseif node.hybrid
            pa = getparents(node)
            i_parents = indexin(pa, prenodes)
            if isdegenerate(node) && unscope(node)
                ch = getchild(node)
                i_child = findfirst(n -> n===ch, prenodes)
                i_inscope = (i_child, i_parents...)
                visited[i_child] = true
                # todo: do something with child to handle degenerate case
            else
                i_inscope = (i_node, i_parents...)
            end
            i_b = findfirst(x -> i_node  ∈ x && all(i_parents .∈ x), nodelabels(b) for b in beliefs)
            isnothing(i_b) && error("no cluster containing hybrid node $(node.number) and its parents.")
            # todo: get μ,h,J,g
        else
            e = getparentedge(node)
            pa = getparent(e)
            i_parent = findfirst(n -> n===pa, prenodes)
            i_b = findfirst(x -> i_parent ∈ x && i_node  ∈ x, nodelabels(b) for b in beliefs)
            isnothing(i_b) && error("no cluster containing nodes $(node.number) and $(pa.number).")
            be = beliefs[i_b]
            be.type == bclustertype || error("belief $(be.metadata) is of type $(be.type)")
            pa_vscope = view(be.inscope, :, findfirst(isequal(i_parent), be.nodelabels))
            # todo: case when node is a leaf and its parent is a hybrid removed from scope
            if node.leaf
                all(.!pa_vscope) && error("internal tree node without any data below")
                i_datarow = findfirst(isequal(nodelab), taxa)
                μ,h,J,g = factor_leaf(model, e.length, i_datarow, tbl)
                i_inscope = (i_parent,)
                # todo: check that i_parent is in fact not fully out of scope
            else # internal tree node
                μ,h,J,g = factor_treeedge(model, e.length)
                i_inscope = (i_node,i_parent)
            end
        end
        be = beliefs[i_b]
        be.type == bclustertype || error("belief $(be.metadata) is of type $(be.type)")
        canonind = scopeindex(i_inscope, be)
        view(be.h, canonind) .+= h
        view(be.J, canonind) .+= J
        be.g += g
        # todo: assign μ,h,J,g to belief 'be' with correct indices
    end
    # todo: update be.μ after all factors have been assign to the cluster?
    return beliefs
end

# todo: absorb data at tips
# todo: integrate beliefs
# todo: propagate belief
