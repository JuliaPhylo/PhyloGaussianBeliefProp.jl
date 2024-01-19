@enum BeliefType bclustertype=1 bsepsettype=2

abstract type AbstractBelief{T} end

struct FamilyFactor{T<:Real,P<:AbstractMatrix{T},V<:AbstractVector{T}} <: AbstractBelief{T}
    h::V
    J::P
    g::MVector{1,T} # mutable
    "metadata, e.g. index of cluster in cluster graph"
    metadata::Symbol # because clusters have metadata of type Symbol
end
function Base.show(io::IO, b::FamilyFactor)
    print(io, "factor for node family" * " $(b.metadata)")
    print(io, "\nexponential quadratic belief, parametrized by\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    FamilyFactor(belief::AbstractBelief{T}) where T

Constructor to allocate memory for one familty factor, with canonical parameters
and metadata initialized to be a copy of those in `belief`.
`FamilyFactor`s metadata are supposed to be symbols, so this constructor should
fail if its input is a sepset belief, whose `metadata` is a Tuple of Symbols.
"""
function FamilyFactor(belief::AbstractBelief{T}) where T
    h = deepcopy(belief.h)
    J = deepcopy(belief.J)
    g = deepcopy(belief.g)
    FamilyFactor{T,typeof(J),typeof(h)}(h,J,g,belief.metadata)
end

"""
    Belief{T<:Real,Vlabel<:AbstractVector,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief{T}

A "belief" is an exponential quadratic form, using the canonical parametrization:

    C(x | J,h,g) = exp( -(1/2)x'Jx + h'x + g )

It is a *normalized* distribution density if

    g = - (1/2) (log(2πΣ) + μ'Jμ)
      = - entropy of normalized distribution + (1/2) dim(μ) - (1/2) μ'Jμ.

- μ is the mean, of type V (stored but typically not updated)
- h = inv(Σ)μ is the potential, also of type V,
- Σ is the variance matrix (not stored)
- J = inv(Σ) is the precision matrix, of type P
- g is a scalar to get the unnormalized belief: of type `MVector{1,T}` to be mutable.

See `MvNormalCanon{T,P,V}` in
[Distributions.jl](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormalCanon)

Other fields are used to track which cluster or edge the belief corresponds to,
and which traits of which variables are in scope:
- `nodelabel` of type `Vlabel`
- `ntraits`
- `inscope`
- `type`: cluster or sepset
- `metadata` of type `M`: `Symbol` for clusters, `Tuple{Symbol,Symbol}` for sepsets.
"""
struct Belief{T<:Real,Vlabel<:AbstractVector,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief{T}
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
    μ::V
    h::V
    J::P # PDMat(J) not easily mutable, stores cholesky computed upon construction only
    g::MVector{1,T}
    "belief type: cluster (node in cluster graph) or sepset (edge in cluster graph)"
    type::BeliefType
    "metadata, e.g. index in cluster graph,
    of type (M) `Symbol` for clusters or Tuple{Symbol,Symbol} for edges."
    metadata::M
end

nodelabels(b::Belief) = b.nodelabel
ntraits(b::Belief) = b.ntraits
inscope(b::Belief) = b.inscope
nodedimensions(b::Belief) = map(sum, eachslice(inscope(b), dims=2))
dimension(b::Belief)  = sum(inscope(b))
# commented out because not used:
# mvnormcanon(b::Belief) = MvNormalCanon(b.μ, b.h, PDMat(LA.Symmetric(b.J)))

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
    Belief(nodelabels, numtraits, inscope, belieftype, metadata, T=Float64)

Constructor to allocate memory for one cluster, and initialize objects with 0s
to initilize the belief with the constant function exp(0)=1.
"""
function Belief(nl::AbstractVector{Tlabel}, numtraits::Integer,
                inscope::BitArray, belief, metadata, T::Type=Float64) where Tlabel<:Integer
    nnodes = length(nl)
    nodelabels = SVector{nnodes}(nl)
    size(inscope) == (numtraits,nnodes) || error("inscope of the wrong size")
    cldim = sum(inscope)
    μ = MVector{cldim,T}(zero(T) for _ in 1:cldim)  # zeros(T, cldim)
    h = MVector{cldim,T}(zero(T) for _ in 1:cldim)
    J = MMatrix{cldim,cldim,T}(zero(T) for _ in 1:(cldim*cldim))
    g = MVector{1,T}(0)
    Belief{T,typeof(nodelabels),typeof(J),typeof(h),typeof(metadata)}(
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

# todo perhaps: add option to turn off checks, and
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
    beliefs = Belief{T}[]
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
    init_factors_allocate(beliefs::AbstractVector{<:Belief}, nclusters::Integer)

Vector of `nclusters` factors of type [`FamilyFactor`](@ref), whose canonical
parameters and metadata are initialized to be a copy of those in `beliefs`.
Assumption: the first `nclusters` beliefs are cluster beliefs, and the next
beliefs (if any) are sepset beliefs. This is not checked.
"""
function init_factors_allocate(beliefs::AbstractVector{B}, nclusters::Integer) where B<:Belief{T} where T
    factors = FamilyFactor{T}[]
    for i in 1:nclusters
        push!(factors, FamilyFactor(beliefs[i]))
    end
    return factors
end

"""
    init_beliefs_allocate_atroot!(beliefs, factors, model)

Update the scope of cluster & sepset `beliefs` and `factors` to include or exclude
the root, depending on whether the root variable is random or fixed in `model`.
To change the dimension of canonical parameters μ,h,J, new memory is allocated
and initilized to 0.
This function can be used to update beliefs when the root model changes from
fixed to non-fixed or vice-versa.
It re-allocates less memory than [`init_beliefs_allocate`](@ref) (which would
need to be followed by [`init_factors_allocate`](@ref))
because clusters and sepsets that do not have the root are not modified.

Assumptions:
- all traits at the root have at least one descendant with non-missing data,
- beliefs were previously initialized with a model that had the same number of
  traits as the current `model`.
"""
function init_beliefs_allocate_atroot!(beliefs, factors, messageresidual, model::EvolutionaryModel{T}) where T
    ## TODO: Should this be called at the begining of "init_beliefs_assignfactors", in case the root status of the model has changed ?
    numtraits = dimension(model)
    fixedroot = isrootfixed(model)
    # root *not* in scope if fixed; else *in* scope bc we assume data below
    update_inscope!(inscope, root_ind) = inscope[:,root_ind] .= !fixedroot
    for (i_b, be) in enumerate(beliefs)
        root_ind = findfirst(nl -> 1 == nl, nodelabels(be))
        isnothing(root_ind) && continue # skip: root ∉ belief
        iscluster = be.type == bclustertype
        be_insc = be.inscope
        update_inscope!(be_insc, root_ind)
        beliefs[i_b] = Belief(be.nodelabel, numtraits, be_insc, be.type, be.metadata, T)
        if iscluster # re-allocate the corresponding factor. if sepset: nothing to do
            factors[i_b] = FamilyFactor(beliefs[i_b])
        end
        issepset = beliefs[i_b].type == bsepsettype
        if issepset # re-allocate the corresponding messages for sepsets
            (clustlab1, clustlab2) = beliefs[i_b].metadata
            messageresidual[(clustlab1, clustlab2)] = MessageResidual(beliefs[i_b].J, beliefs[i_b].h)
            messageresidual[(clustlab2, clustlab1)] = MessageResidual(beliefs[i_b].J, beliefs[i_b].h)
        end
    end
end

"""
    init_beliefs_reset!(beliefs)

Reset all beliefs (which may be cluster & sepset beliefs or factors, which are
initial cluster beliefs) to h=0, J=0, g=0 (μ unchanged).
They can later be re-initialized for different model parameters and
re-calibrated, without re-allocating memory.
"""
function init_beliefs_reset!(beliefs)
    for be in beliefs
        be.h .= 0.0
        be.J .= 0.0
        be.g[1] = 0.0
    end
end

"""
    init_factors_frombeliefs!(factors, beliefs, checkmetadata::Bool=false)

Reset all `factors` by copying h,J,g from `beliefs`.
Assumption: the cluster beliefs match the factors exactly: for a valid factor
index `i`, `beliefs[i]` is of cluster type and has the same dimension as
`factors[i]`.

Set `checkmetadata` to true to check that `beliefs[i]` and `factors[i]` have
the same metadata.
"""
function init_factors_frombeliefs!(factors, beliefs, checkmetadata::Bool=false)
    for (fa,be) in zip(factors,beliefs)
        if checkmetadata
            fa.metadata == be.metadata ||
                error("factor $(fa.metadata) mismatched with belief $(be.metadata)")
        end
        fa.h .= be.h
        fa.J .= be.J
        fa.g[1] = be.g[1]
    end
end

"""
    init_beliefs_assignfactors!(beliefs,
                                evolutionarymodel, columntable, taxa,
                                nodevector_preordered)

Initialize cluster beliefs prior to belief propagation, by assigning
each factor to one cluster. Sepset beliefs are reset to 0.
There is one factor for each node `v` in the vector of nodes:
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
- In `beliefs`, cluster beliefs (those of type `bclustertype`) come first
  and sepset beliefs (of type `bclustertype`) come last,
  as when created by [`init_beliefs_allocate`](@ref)

Output: vector `node2belief` such that, if `i` is the preorder index of a node
in the network, `node2belief[i]` is the index of the belief that the node family
was assigned to.

Output: `beliefs` vector. Each belief & factor is modified in place.
"""
function init_beliefs_assignfactors!(
        beliefs,
        model::EvolutionaryModel,
        tbl::Tables.ColumnTable, taxa::AbstractVector, prenodes::Vector{PN.Node})
    init_beliefs_reset!(beliefs)
    numtraits = dimension(model)
    visited = falses(length(prenodes))
    node2belief = Vector{Int}(undef, length(prenodes)) # node preorder index → belief index
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
                h,J,g = factor_tree_degeneratehybrid(model, pae, che)
                if ch.leaf
                    i_datarow = findfirst(isequal(ch.name), taxa)
                    h,J,g = absorbleaf!(h,J,g, i_datarow, tbl)
                    i_inscope = (i_parents...,)
                else
                    i_inscope = (i_child, i_parents...)
                end
            else
                i_inscope = (i_node, i_parents...)
                h,J,g = factor_hybridnode(model, pae)
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
            h,J,g = factor_treeedge(model, e)
            if node.leaf
                i_datarow = findfirst(isequal(nodelab), taxa)
                h,J,g = absorbleaf!(h,J,g, i_datarow, tbl)
                i_inscope = (i_parent,)
            else
                i_inscope = (i_node,i_parent)
            end
        end
        node2belief[i_node] = i_b
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
            h,J,g = marginalizebelief(h,J,g, keep_index, be.metadata)
        end
        view(be.h, factorind) .+= h
        view(be.J, factorind, factorind) .+= J
        be.g[1] += g
    end
    return node2belief
end

#= messages in ReactiveMP.jl have an `addons` field that stores computation history:
https://biaslab.github.io/ReactiveMP.jl/stable/lib/message/#ReactiveMP.Message

Here, similar "in spirit" to track progress towards calibration
or to facilitate adaptive scheduling (e.g. residual BP), we store a
message residual: difference (on log-scale) between a message *received*
by a cluster and the belief that the sepset previously had.
=#

abstract type AbstractResidual{T} end

"""
    MessageResidual{T<:Real, P<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractResidual{T}

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
  approximately equal, false otherwise. see [`iscalibrated_residnorm!`](@ref)
- `iscalibrated_kl`: same, but in terms of the KL divergence,
  see [`iscalibrated_kl!`](@ref).
"""
struct MessageResidual{T<:Real, P<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractResidual{T}
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
    Δh = zero(h)
    ΔJ = zero(J)
    MessageResidual{T,typeof(ΔJ),typeof(Δh)}(Δh, ΔJ, MVector(-1.0), MVector(false), MVector(false))
end

"""
    init_messageresidual_allocate(beliefs::Vector{B}, nclusters)

Dictionary of `2k` residuals of type [`MessageResidual`](@ref), whose canonical
parameters (Δh,ΔJ) are initialized using [`MessageResidual`](@ref), to be of
the same size as sepsets in `beliefs`, where `k` is `length(beliefs) - nclusters`.
Assumption: the first `nclusters` beliefs are cluster beliefs, and the next
`k` beliefs are sepset beliefs. This is not checked.

The sepset for edge `(label1,label2)` is associated with 2 messages, for the
2 directions in which beliefs can be propagated along the edge. The keys for
these messages are `(label1,label2)` and `(label2,label1)`.
"""
function init_messageresidual_allocate(beliefs::Vector{B}, nclusters) where B<:Belief{T} where T<:Real
    messageresidual = Dict{Tuple{Symbol,Symbol}, MessageResidual{T}}()
    for j in (nclusters+1):length(beliefs)
        ssbe = beliefs[j] # sepset belief
        (clustlab1, clustlab2) = ssbe.metadata
        messageresidual[(clustlab1, clustlab2)] = MessageResidual(ssbe.J, ssbe.h)
        messageresidual[(clustlab2, clustlab1)] = MessageResidual(ssbe.J, ssbe.h)
    end
    return messageresidual
end

iscalibrated_residnorm(res::AbstractResidual) = res.iscalibrated_resid[1]
iscalibrated_kl(res::AbstractResidual) = res.iscalibrated_kl[1]

"""
    iscalibrated_residnorm!(res::AbstractResidual, atol=1e-5, p::Real=Inf)

True if the canonical parameters `res.Δh` and `res.ΔJ` of the message residual
have `p`-norm within `atol` of 0; false otherwise.
`res.iscalibrated_resid` is updated accordingly.

With `p` infinite, the max norm is used by default, meaning that
`res.Δh` and `res.ΔJ` should be close to 0 element-wise.
"""
function iscalibrated_residnorm!(res::AbstractResidual{T}, atol=T(1e-5), p::Real=Inf) where T
    res.iscalibrated_resid[1] =
        isapprox(LA.norm(res.Δh, p), zero(T), atol=atol) &&
        isapprox(LA.norm(res.ΔJ, p), zero(T), atol=atol)
end

"""
    iscalibrated_kl!(res::AbstractResidual, atol=1e-5)

True if the KL divergence stored in `res.kldiv` is within `atol` of 0;
false otherwise. `res.iscalibrated_kl` is modified accordingly.
This KL divergence should have been previously calculated: between a sepset
belief, equal to the message that was passed most recently, and its belief just
prior to passing that message.
"""
function iscalibrated_kl!(res::AbstractResidual{T}, atol=T(1e-5)) where T
    res.iscalibrated_kl[1] = isapprox(res.kldiv[1], zero(T), atol=atol)
end

"""
    approximate_kl!(residual::AbstractResidual, sepset::AbstractBelief,
        canonicalparams::Tuple)

Update `residual.kldiv` with an approximation to the KL divergence between
a message sent through a sepset (normalized to a probability distribution),
and the sepset belief before the belief propagation (also normalized).
`sepset` should contain the updated belief, and `residual` the difference
in the `J` and `h` parameters due to the belief update (after - before).
As a side product, `sepset.μ` is updated.

Output: true if the approximated KL divergence is close to 0, false otherwise.

If the `sepset` belief is degenerate, in the sense that its precision matrix is
not positive definite and the belief cannot be normalized to a proper distribution,
then a warning is sent, `residual` and `sepset` are not updated, and
`false` is returned.

## Calculation:

This approximation computes the negative average energy of the residual canonical
parameters, with the `g` parameter set to 0, with respect to the message
canonical parameters.

message sent: f(x) = C(x | Jₘ, hₘ, _) ≡ x ~ 𝒩(μ=Jₘ⁻¹hₘ, Σ=Jₘ⁻¹)  
sepset (before belief-update): C(.| Jₛ, hₛ, gₛ)  
sepset (after belief-update): C(.| Jₘ, hₘ, gₘ)  
residual: ΔJ = Jₘ - Jₛ, Δh = hₘ - hₛ  
Below, we use the nodation Δg for the change in constants to normalize each
message, which is *not* gₘ-gₛ because the stored beliefs are not normalized.

    KL(C(Jₘ, hₘ, _) || C(Jₛ, hₛ, _))
    = E[log C(x | Jₘ,hₘ,_)/C(x | Jₛ,hₛ,_)] where x ∼ C(Jₘ,hₘ,_)
    = E[-(1/2) x'ΔJx + Δh'x + Δg)]
    ≈ E[-(1/2) x'ΔJx + Δh'x], Δg dropped
    = -average_energy(C(Jₘ, hₘ, _), C(ΔJ, Δh, 0))

See also: [`average_energy!`](@ref)

## Note:
E[Δg] = Δg
      = +(1/2) log|Jₘ Jₛ⁻¹|
      = +(1/2) log|I - (I - Jₘ Jₛ⁻¹)|
      = +(1/2) log|I - P(I-D)P⁻¹|, where JₘJₛ⁻¹ = PDP⁻¹ is the eigendecomposition
      ≈ +(1/2) (1 + tr(P(I-D)P⁻¹)) + O(‖vec(I-D)‖²), where ‖⋅‖ is the 1-norm
      = +(1/2) (1 + tr(I-D)), first-order approx wrt ‖vec(I-D)‖
      = +(1/2) (1 + k - tr(D)), where k = dim(I)
*todo: Discuss if above should be added to current output of approximate_kl!
"""
function approximate_kl!(res::AbstractResidual{T}, sepset::AbstractBelief{T}) where {T <: Real}
    #= Check if empty because `isposdef` returns true for empty matrices, e.g.
    both `Real[;;] |> isposdef` and `MMatrix{0,0}(Real[;;]) |> isposdef` return
    `true`. =#
    isempty(sepset.J) && return true
    # fixit: should an "empty" sepset have it iscalibrated_* fields initialized to true?
    (Jchol, μ) = try getcholesky_μ!(sepset)
    catch
        # @warn "cannot approximate KL divergence between messages: degenerate sepset belief"
        # TODO 
        return false
    end
    #=  Check that diagonal entries of lower factor are positive, to make sure
        that J is psd. This is needed because the more generic cholesky method,
        which is applied for example to matrices of Dual numbers, or Float64
        matrices that are wrapped within another type (e.g. MMatrix) currently
        does not throw an error for psd matrices.
        This bug has been fixed here: https://github.com/JuliaLang/julia/pull/49417/commits,
        though it has not been incorporated into the lastest stable release (1.9.3).
        But: this bug does not seem to affect PDMat as used by getcholesky_μ!
    =#
    # if !LA.issuccess(Jchol) || any(Jchol.L[i,i] <= 0 for i in 1:dimension(sepset))
    # then: warn and return false?
    res.kldiv[1] = - average_energy(Jchol, μ, res.ΔJ, res.Δh, zero(T))
    iscalibrated_kl!(res)
end
