@enum BeliefType bclustertype=1 bsepsettype=2

abstract type AbstractBelief end

nodelabels(b::AbstractBelief) = b.nodelabel
ntraits(b::AbstractBelief) = b.ntraits
inscope(b::AbstractBelief) = b.inscope
nodedimensions(b::AbstractBelief) = map(sum, eachslice(inscope(b), dims=2))
dimension(b::AbstractBelief)  = sum(inscope(b))
mvnormcanon(b::AbstractBelief) = MvNormalCanon(b.μ, b.h, PDMat(LA.Symmetric(b.J)))

struct Belief{Vlabel<:AbstractVector,T<:Real,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief
    "Integer label for nodes in the cluster"
    nodelabel::Vlabel # StaticVector{N,Tlabel}
    "Total number of traits at each node"
    ntraits::Int
    """Matrix inscope[i,j] is `false` if trait `i` at node `j` is / will be
    removed from scope, to avoid issues from 0 precision or infinite variance; or
    when there is no data for trait `i` below node `j` (in which case tracking
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
    g::MVector{1,T} # mutable
    "belief type: cluster (node in cluster graph) or sepset (edge in cluster graph)"
    type::BeliefType
    "metadata, e.g. index in cluster graph,
    of type (M) `Symbol` for clusters or Tuple{Symbol,Symbol} for edges."
    metadata::M
end

function Base.show(io::IO, b::Belief)
    disp = "belief for " * (b.type == bclustertype ? "Cluster" : "SepSet") * " $(b.metadata),"
    disp *= " $(ntraits(b)) traits × $(length(nodelabels(b))) nodes, dimension $(dimension(b)).\n"
    disp *= "Node labels: "
    print(io, disp)
    print(io, nodelabels(b))
    print(io, "\ntrait × node matrix of non-degenerate beliefs:\n")
    show(io, inscope(b))
    print(io, "\nexponential quadratic belief, parametrized by\nμ: $(b.μ)\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    Belief(nodelabels, numtraits, inscope, belieftype, metadata,T=Float64)

Constructor to allocate memory for one cluster, and initialize objects with 0s
to initilize the belief with the constant function exp(0)=1.
"""
function Belief(nl::AbstractVector{Tlabel}, numtraits::Integer,
                inscope::BitArray, belief, metadata,T::Type=Float64) where Tlabel<:Integer
    nnodes = length(nl)
    nodelabels = SVector{nnodes}(nl)
    size(inscope) == (numtraits,nnodes) || error("inscope of the wrong size")
    cldim = sum(inscope)
    μ = MVector{cldim,T}(zero(T) for _ in 1:cldim)  # zeros(T, cldim)
    h = MVector{cldim,T}(zero(T) for _ in 1:cldim)
    J = MMatrix{cldim,cldim,T}(zero(T) for _ in 1:(cldim*cldim))
    g = MVector{1,T}(0)
    Belief{typeof(nodelabels),T,typeof(J),typeof(h),typeof(metadata)}(
        nodelabels,numtraits,inscope,μ,h,J,g,belief,metadata)
end

"""
    scopeindex(node_labels, belief::AbstractBelief)

Indices in the belief's μ,h,J vectors and matrices of the variables
for nodes labelled `node_labels`. The belief's `inscope` matrix of
booleans says which node (row) and trait (column) is in the belief's scope.
These variables are vectorized by stacking up columns, that is,
listing all in-scope traits of the first node, then all in-scope traits of
the second node etc.
"""
function scopeindex(node_labels::Union{Tuple,AbstractVector}, belief::AbstractBelief)
    binscope = inscope(belief)
    node_j = indexin(node_labels, nodelabels(belief))
    any(isnothing.(node_j)) && error("some label is not in the belief's node labels")
    node_dims = map(sum, eachslice(binscope, dims=2))
    node_cumsum = cumsum(node_dims)
    res = Vector{Int}(undef, sum(node_dims[node_j]))
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
    scopeindex(sepset::AbstractBelief, cluster::AbstractBelief)

Indices `ind`, in the cluster in-scope variables (that is, in the cluster's μ,h,J
vectors and matrices) of the sepset in-scope variables, such that
`cluster.μ[ind]` correspond to the same variables as `sepset.μ`, for example.
These sepset in-scope variables can be a subset of traits for each node in the
sepset, as indicated by `inscope(sepset)`.

Warning: the labels in the sepset are assumed to be ordered as in the cluster.

An error is returned if `sepset` contains labels not in the `cluster`,
or if a variable in the `sepset`'s scope is not in scope in the `cluster`.
"""
scopeindex(sep::AbstractBelief, clu::AbstractBelief) =
    scopeindex(nodelabels(sep), inscope(sep), nodelabels(clu), inscope(clu))
function scopeindex(subset_labels::AbstractVector, subset_inscope::BitArray,
                    belief_labels::AbstractVector, belief_inscope::BitArray)
    node_index = indexin(subset_labels, belief_labels)
    issorted(node_index) || error("subset labels come in a different order in the belief")
    any(isnothing.(node_index)) && error("subset_labels not a subset of belief_labels")
    any(subset_inscope .&& .!view(belief_inscope,:,node_index)) &&
        error("some variable(s) in subset's scope yet not in full belief's scope")
    subset_inclusterscope = falses(size(belief_inscope))
    subset_inclusterscope[:,node_index] .= subset_inscope
    return findall(subset_inclusterscope[belief_inscope])
end

# fixit: add option to turn off checks, and
# add function to run these checks once between all incident sepset-cluster

"""
    init_beliefs_allocate(tbl::Tables.ColumnTable, taxa, net, clustergraph,
                          evolutionarymodel)

Vector of beliefs, initialized to the constant function exp(0)=1,
one for each cluster then one for each sepset in `clustergraph`.
`tbl` is used to know which leaf in `net` has data for which trait,
so as to remove from the scope each variable without data below it.
`taxa` should be a vector with taxon names in the same order as they come in
the table of data `tbl`.
The root is removed from scope if the evolutionary model has a fixed root: so as
to use the model's fixed root value as data if the root as zero prior variance.
Also removed from scope is any hybrid node that is degenerate and who has
a single child edge of positive length.
"""
function init_beliefs_allocate(tbl::Tables.ColumnTable, taxa::AbstractVector,
        net::HybridNetwork, clustergraph, model::EvolutionaryModel{T}) where T
    numtraits = length(tbl)
    nnodes = length(net.nodes_changed)
    nnodes > 0 ||
        error("the network should have been pre-ordered, with indices used in cluster graph")
    fixedroot = isrootfixed(model)
    #= hasdata: to know, for each node, whether that node has a descendant
                with data, for each trait.
    If not: that node can be removed from all clusters & sepsets.
    If yes and the node is a tip: the evidence should be used later,
      then the tip can be removed the evidence is absorbed.
    =#
    hasdata = falses(numtraits, nnodes)
    for i_node in reverse(eachindex(net.nodes_changed))
        node = net.nodes_changed[i_node]
        nodelab = node.name
        i_row = findfirst(isequal(nodelab), taxa)
        if !isnothing(i_row) # the node has data: it should be a tip!
            node.leaf || error("A node with data is internal, should be a leaf")
            for v in 1:numtraits
              hasdata[v,i_node] = !ismissing(tbl[v][i_row])
            end
        end
        if node.leaf
            all(!hasdata[v,i_node] for v in 1:numtraits) &&
                @error("tip $nodelab in network without any data")
            continue
        end
        for e in node.edge
            ch = getchild(e)
            ch !== node || continue # skip parent edges
            i_child = findfirst( n -> n===ch, net.nodes_changed)
            isempty(i_child) && error("oops, child (number $(ch.number)) not found in nodes_changed")
            hasdata[:,i_node] .|= hasdata[:,i_child] # bitwise or
        end
        all(!hasdata[v,i_node] for v in 1:numtraits) &&
            @error("internal node $nodelab without any data below")
    end
    #= next: create a belief for each cluster and sepset. inscope =
    'has partial information and non-degenerate variance or precision?' =
    - false at the root if "fixedroot", else:
    - 'hasdata?' at internal nodes (assumes non-degenerate transitions)
    - false at tips (assumes all data are at tips)
    - false at degenerate hybrid node with 1 tree child edge of positive length
    =#
    function build_inscope(set_nodeindices)
        inscope = falses(numtraits, length(set_nodeindices)) # remove from scope by default
        for (i,i_node) in enumerate(set_nodeindices)
            node = net.nodes_changed[i_node]
            (node.leaf || (isdegenerate(node) && unscope(node))) && continue # inscope[:,i] already false
            fixedroot && i_node==1 && continue # keep 'false' at the root if fixed
            inscope[:,i] .= hasdata[:,i_node]
        end
        return inscope
    end
    beliefs = Belief[]
    for cllab in labels(clustergraph)
        nodeindices = clustergraph[cllab][2]
        inscope = build_inscope(nodeindices)
        push!(beliefs, Belief(nodeindices, numtraits, inscope, bclustertype, cllab,T))
    end
    for sslab in edge_labels(clustergraph)
        nodeindices = clustergraph[sslab...]
        inscope = build_inscope(nodeindices)
        push!(beliefs, Belief(nodeindices, numtraits, inscope, bsepsettype, sslab,T))
    end
    return beliefs
end

"""
    update_root_inscope!(beliefs, model)

Check the status of the root, and change the scope of root clusters and sepsets
accordingly.
This function is needed to update beliefs when the root changes from fixed to non-fixed,
or vice-versa.

The function assumes that all traits at the root have at least one data below them.
"""
function update_root_inscope!(beliefs, model::EvolutionaryModel{T}) where T
    ## TODO: is that the correct way to do it ?
    ## TODO: Or should we assume that the root status will never change and make that clear ?
    ## TODO: Should this be called at the begining of "init_beliefs_assignfactors", in case the root status of the model has changed ?
    numtraits = dimension(model)
    fixedroot = isrootfixed(model)
    function update_inscope!(inscope, root_int)
        if fixedroot 
            inscope[:,root_int] .= falses(numtraits) # fixed root : 'false'
        else
            inscope[:,root_int] .= trues(numtraits) # root is assumed to have data below it
        end
    end
    for (i_b, b) in enumerate(beliefs)
        if 1 ∈ nodelabels(b) # root is in belief
            root_int = findfirst(nl -> 1 == nl, nodelabels(b))
            inscope = beliefs[i_b].inscope
            update_inscope!(inscope, root_int)
            beliefs[i_b] = Belief(beliefs[i_b].nodelabel, numtraits, inscope, beliefs[i_b].type, beliefs[i_b].metadata, T)
        end
    end
end

"""
    init_beliefs_reset!(beliefs)

Reset cluster and sepset beliefs to h=0, J=0, g=0 (μ unchanged) so they can
later be re-initialized with factors from different model parameters,
then re-calibrated.
"""
function init_beliefs_reset!(beliefs)
    for be in beliefs
        be.h .= 0.0
        be.J .= 0.0
        be.g[1] = 0.0
    end
end
function init_beliefs_reset!(beliefs::Vector{AbstractBelief},
    factors::Vector{AbstractBelief})
    # fixit: input checks?
    for (be, fa) in zip(beliefs, factors)
        be.h .= fa.h
        be.J .= fa.J
        be.g[1] = fa.g[1]
    end
end

"""
    init_beliefs_assignfactors!(beliefs, evolutionarymodel, columntable, taxa,
                                nodevector_preordered)

Initialize cluster beliefs prior to belief propagation, by assigning each factor
to one cluster. There is one factor for each node `v` in the vector of nodes:
the density of X_v conditional on its parent X_pa(v) if v is not the root,
and prior density for X_root.
- for each leaf, the factor is reduced by absorbing the evidence for that leaf,
  that is, the data found in the `columntable`, whose rows are supposed to
  correspond to taxa in the same order in which they are listed in `taxa`.
- for each leaf, missing trait values are removed from scope.
- for each internal node, any trait not in scope (e.g. if all descendant leaves
  are missing a value for this trait) is marginalized out of the factor.

Assumptions:
- In vector `nodevector_preordered`, nodes are assumed to be preordered.
  Typically, this vector is `net.nodes_changed` after the network is preordered.
- Belief node labels correspond to the index of each node in `nodevector_preordered`.
- In `beliefs`, cluster beliefs come first and sepset beliefs come last,
  as when created by [`init_beliefs_allocate`](@ref)

Output: `beliefs` vector. Each belief is modified in place.
"""
function init_beliefs_assignfactors!(beliefs, model::EvolutionaryModel,
        tbl::Tables.ColumnTable, taxa::AbstractVector, prenodes::Vector{PN.Node})
    init_beliefs_reset!(beliefs)
    numtraits = dimension(model)
    visited = falses(length(prenodes))
    for (i_node,node) in enumerate(prenodes)
        visited[i_node] && continue # skip child of unscoped degenerate hybrid
        nodelab = node.name
        if i_node == 1 # root
            isrootfixed(model) && continue # h,J,g all 0: nothing to do
            i_b = findfirst(b -> 1 ∈ nodelabels(b), beliefs)
            isnothing(i_b) && error("no cluster containing the root, number $(node.number).")
            i_inscope = (1,)
            h,J,g = factor_root(model)
        elseif node.hybrid
            pae = PN.Edge[] # collect parent edges, parent nodes, and their
            pa  = PN.Node[] # preorder indices, sorted in postorder
            i_parents = Int[]
            for e in node.edge # loop over parent edges
                getchild(e) === node || continue
                pn = getparent(e) # parent node
                pi = findfirst(n -> n===pn, prenodes) # parent index
                ii = findfirst(i_parents .< pi) # i_parents is reverse-sorted
                if isnothing(ii) ii = length(i_parents) + 1; end
                insert!(i_parents, ii, pi)
                insert!(pa, ii, pn)
                insert!(pae, ii, e)
            end
            if isdegenerate(node) && unscope(node)
                che = getchildedge(node)
                ch = getchild(che)
                i_child = findfirst(n -> n===ch, prenodes)
                visited[i_child] = true
                h,J,g = factor_tree_degeneratehybrid(model, che.length, [p.gamma for p in pae])
                if ch.leaf
                    i_datarow = findfirst(isequal(ch.name), taxa)
                    h,J,g = absorbleaf!(h,J,g, i_datarow, tbl)
                    i_inscope = (i_parents...,)
                else
                    i_inscope = (i_child, i_parents...)
                end
            else
                i_inscope = (i_node, i_parents...)
                h,J,g = factor_hybridnode(model, [e.length for e in pae], [p.gamma for p in pae])
                @debug "node $(node.name), lengths $([e.length for e in pae]), gammas $([p.gamma for p in pae])\nh=$h, J=$J, g=$g"
            end
            i_b = findfirst(b -> issubset(i_inscope, nodelabels(b)), beliefs)
            isnothing(i_b) && error("no cluster containing the scope for hybrid $(node.number).")
        else
            e = getparentedge(node)
            pa = getparent(e)
            i_parent = findfirst(n -> n===pa, prenodes)
            i_b = findfirst(x -> i_parent ∈ x && i_node ∈ x, nodelabels(b) for b in beliefs)
            isnothing(i_b) && error("no cluster containing nodes $(node.number) and $(pa.number).")
            h,J,g = factor_treeedge(model, e.length)
            if node.leaf
                i_datarow = findfirst(isequal(nodelab), taxa)
                h,J,g = absorbleaf!(h,J,g, i_datarow, tbl)
                i_inscope = (i_parent,)
            else
                i_inscope = (i_node,i_parent)
            end
        end
        be = beliefs[i_b]
        be.type == bclustertype || error("belief $(be.metadata) is of type $(be.type)")
        if isrootfixed(model) && 1 ∈ i_inscope # the node's parents include the root
            1 == i_inscope[end] || error("expected the root to be listed last (postorder)")
            i_inscope = i_inscope[1:(end-1)] # remove last entry '1'
            rootindex = (length(h) - numtraits + 1):length(h)
            h,J,g = absorbevidence!(h,J,g, rootindex, rootpriormeanvector(model))
        end
        factorind = scopeindex(i_inscope, be)
        @debug """
        factor for node $(node.name), nodes in scope have preorder $i_inscope,
        cluster $i_b with labels $(nodelabels(be)), inscope: $(inscope(be)),
        their variable belief indices $factorind.
        before marginalizing: h=$h, J=$J, g=$g
        """ 
        if length(factorind) != numtraits * length(i_inscope)
            # then marginalize variables not in scope, e.g. bc no data below
            var_inscope = view(inscope(be), :, indexin(i_inscope, nodelabels(be)))
            keep_index = LinearIndices(var_inscope)[var_inscope]
            h,J,g = marginalizebelief(h,J,g, keep_index)
        end
        view(be.h, factorind) .+= h
        view(be.J, factorind, factorind) .+= J
        be.g[1] += g
        # @show be.h,be.J,be.g[1]
    end
    #= removed because of too many cases when some clusters would be initialized to 0.
    #  example: variable-clusters in Bethe cluster graph
    for be in beliefs # sanity check: each cluster belief should be non-zero
        # unless one variable was completely removed from the scope (e.g. leaf without any data)
        be.type == bclustertype || break # sepsets untouched and listed last
        @debug "cluster $(be.metadata), node labels $(nodelabels(be)), inscope: $(inscope(be)),\nμ: $(be.μ)\nh: $(be.h)\nJ: $(be.J)\ng: $(be.g[1])"
        any(be.J .!= 0.0) || any(sum(inscope(be), dims = 1) .== 0) ||
            error("belief for nodes $(nodelabels(be)) was not assigned any non-zero factor")
        # do NOT update μ = J^{-1} h because J often singular before propagation
        # be.μ .= PDMat(LA.Symmetric(be.J)) \ be.h
    end =#
    return beliefs
end

abstract type AbstractResidual end

#= messages in ReactiveMP.jl have an `addons` field that stores computation history:
https://biaslab.github.io/ReactiveMP.jl/stable/lib/message/#ReactiveMP.Message

Here, similar "in spirit" to track progress towards calibration
or to facilitate adaptive scheduling (e.g. residual BP), we store a
message residual: difference (on log-scale) between a message *received*
by a cluster and the belief that the sepset previously had.
=#
"""
    MessageResidual{T<:Real, P<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractResidual

Structure to store the most recent computation history of a message, in the
form of the ratio: sent_message / current_sepset_belief, when a message is
sent from one cluster to another along a given sepset.
At calibration, this ratio is 1. For Gaussian beliefs, this ratio is an
exponential quadratic form, stored using its canonical parametrization,
excluding the constant.

Fields:

- `Δh`: canonical parameter vector of the message residual
- `ΔJ`: canonical parameter matrix of the message residual
- `kldiv`: kl divergence between the message that was last sent and the
   sepset belief before the last update
- `iscalibrated_resid`: true if the last message and prior sepset belief were
  approximately equal, false otherwise. see [`iscalibrated_canon!`](@ref)
- `iscalibrated_kl`: same, but in terms of the KL divergence,
  see [`iscalibrated_kl!`](@ref).
"""
struct MessageResidual{T<:Real, P<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractResidual
    Δh::V
    ΔJ::P
    kldiv::MVector{1,T}
    iscalibrated_resid::MVector{1,Bool}
    iscalibrated_kl::MVector{1,Bool}
end

"""
    MessageResidual(J::AbstractMatrix{T}, h::AbstractVector{T})

Constructor to allocate memory for a `MessageResidual` with canonical parameters
`(ΔJ, Δh)` of the same dimension and type as `J` and `h`, initialized to zeros.
`kldiv` is initalized to `[-1.0]` and the flags `iscalibrated_{resid,kl}`
are initialized to `false`.

`(ΔJ, Δh)` of zero suggest calibration, but the flags `iscalibrated_{resid,kl}`
being false indicate otherwise.
"""
function MessageResidual(J::AbstractMatrix{T}, h::AbstractVector{T}) where {T <: Real}
    Δh = fill!(similar(h), zero(T))
    ΔJ = fill!(similar(J), zero(T))
    MessageResidual{T,typeof(ΔJ),typeof(Δh)}(Δh, ΔJ, MVector(-1.0), MVector(false), MVector(false))
end
