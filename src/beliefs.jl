@enum BeliefType bclustertype=1 bsepsettype=2

abstract type AbstractFactor{T} end
abstract type AbstractBelief{T} <: AbstractFactor{T} end

struct ClusterFactor{
    T<:Real,
    P<:AbstractMatrix{T},
    V<:AbstractVector{T}
} <: AbstractFactor{T}
    h::V
    J::P
    g::MVector{1,T} # mutable
    "metadata, e.g. index of cluster in cluster graph"
    metadata::Symbol # because clusters have metadata of type Symbol
end
function Base.show(io::IO, b::ClusterFactor)
    print(io, "factor for node family" * " $(b.metadata)")
    print(io, "\nexponential quadratic belief, parametrized by\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    CanonicalBelief{
        T<:Real,
        Vlabel<:AbstractVector,
        P<:AbstractMatrix{T},
        V<:AbstractVector{T},
        M,
    } <: AbstractBelief{T}

A canonical belief is an exponential quadratic form with canonical parameters J, h, g
(see [`MvNormalCanon{T,P,V}`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormalCanon)
in Distributions.jl):

    C(x | J,h,g) = exp(-(1/2)xᵀJx + hᵀx + g)

If J is positive-definite (i.e. J ≻ 0), then C(x | J,h,g) can be interpreted as a
non-degenerate, though possibly unnormalized, distribution density for x.

# Fields
- `μ::V`: mean (of x); well-defined only when J ≻ 0; not always updated
- `h::V`: potential; interpretable as Jμ when J ≻ 0
- `J::P`: precision; interpretable as inv(Σ), where Σ is the variance of x, when J ≻ 0
- `g::MVector{1,T}`: `g[1]` is interpretable as the normalization constant for the belief
when J ≻ 0

    The belief is normalized if:

            g = - (1/2) (log(2πΣ) + μ'Jμ)
            = - entropy of normalized distribution + (1/2) dim(μ) - (1/2) μ'Jμ.

Other fields used to track which cluster or edge the belief corresponds to, and
which traits of which nodes are in scope:
- `nodelabel::Vlabel`: vector of labels (e.g. preorder indices) for nodes from the phylogeny
that this belief corresponds to (i.e. whose traits distribution it describes)
- `ntraits::Integer`: maximum number of traits per node
- `inscope::BitArray`: matrix of booleans (row i: trait i, column j: node j) indicating
which traits are present for which nodes
- `type::BeliefType`: cluster or sepset
- `metadata::M`: index (in the cluster graph) of the cluster or sepset that this belief
corresponds to (e.g. ::Symbol for a cluster, ::Tuple{Symbol,Symbol} for a sepset)

# Methods
- `nodelabels(b)`: `b.nodelabel`
- `ntraits(b)`: `b.ntraits`
- `inscope(b)`: `b.inscope`, a `ntraits(b)`×`nodelabels(b)` matrix
- `nodedimensions(b)`: vector of integers, with jth value giving the dimension
  (number of traits in scope) of node j.
- `dimension(b)`: total dimension of the belief, that is, total number of traits
  in scope. Without any missing data, that would be `ntraits(b)`*`nodelabels(b)`.
"""
struct CanonicalBelief{
    T<:Real,
    Vlabel<:AbstractVector,
    P<:AbstractMatrix{T},
    V<:AbstractVector{T},
    M,
} <: AbstractBelief{T}
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
    """
    metadata, e.g. index in cluster graph,
    of type (M) `Symbol` for clusters or Tuple{Symbol,Symbol} for edges.
    """
    metadata::M
end

"""
    CanonicalBelief(nodelabels, numtraits, inscope, belieftype, metadata, T=Float64)

Constructor to allocate memory for one cluster, and initialize objects with 0s
to initialize the belief with the constant function exp(0)=1.
"""
function CanonicalBelief(
    nl::AbstractVector{Tlabel},
    numtraits::Integer,
    inscope::BitArray,
    belief,
    metadata,
    T::Type=Float64,
) where Tlabel<:Integer
    nnodes = length(nl)
    # nodelabels = SVector{nnodes}(nl)
    nodelabels = nl
    size(inscope) == (numtraits,nnodes) || error("inscope of the wrong size")
    cldim = sum(inscope)
    # μ = MVector{cldim,T}(zero(T) for _ in 1:cldim)  # zeros(T, cldim)
    μ = zeros(T, cldim)
    # h = MVector{cldim,T}(zero(T) for _ in 1:cldim)
    h = zeros(T, cldim)
    # J = MMatrix{cldim,cldim,T}(zero(T) for _ in 1:(cldim*cldim))
    J = zeros(T, cldim, cldim)
    g = MVector{1,T}(0)
    # CanonicalBelief{T,typeof(nodelabels),typeof(J),typeof(h),typeof(metadata)}(
    #     nodelabels,numtraits,inscope,μ,h,J,g,belief,metadata)
    CanonicalBelief{T,typeof(nodelabels),typeof(J),typeof(h),typeof(metadata)}(
        nodelabels,numtraits,inscope,μ,h,J,g,belief,metadata)
end

"""
    GeneralizedBelief{
        T<:Real,
        Vlabel<:AbstractVector,
        P<:AbstractMatrix{T},
        V<:AbstractVector{T},
        M,
    } <: AbstractBelief{T}
     
A generalized belief is the product of an exponential quadratic form and a Dirac measure
(generalizing the [`CanonicalBelief`](@ref)) with generalized parameters Q, R, Λ, h, c, g:

    𝒟(x | Q,R,Λ,h,c,g) = exp(-(1/2)xᵀ(QΛQᵀ)x + (Qh)ᵀx + g) • δ(Rᵀx - c)
    
For m-dimensional x (i.e. x ∈ ℝᵐ), [Q R] is assumed to be m × m orthonormal
(i.e. [Q R]ᵀ[Q R] = Iₘ) such that Q and R are orthogonal (i.e. QᵀR = 0).

Generally, Q is m × (m-k) and R is m × k, where 0 ≤ k ≤ m. We refer to R as the constraint
matrix (R represents linear dependencies among the variables of x) and to k as the
constraint rank. Λ is (m-k) × (m-k) diagonal, h and c are respectively m × 1 and k × 1, and
g is scalar.

When k=0, a generalized belief is equivalent to a canonical belief with canonical parameters
J₁=QΛQᵀ, h₁=Qh, g₁=g.

# Fields
- `μ::V`: m × 1, where `μ[1:(m-k)]` stores (QΛQᵀ)⁻¹Qh = QΛ⁻¹h, the mean of x; well-defined
only when QΛQᵀ is positive-definite (i.e. Q is square and Λ has no zero-entries on its
diagonal); not always updated
- `h::V`: m × 1 vector, where `h[1:(m-k)]` stores generalized parameter h. Note that is
**not** the potential for a canonical belief.
- `Q::P`: m × m matrix, where `Q[:,1:(m-k)]` stores generalized parameter Q
- `Λ::V`: m × 1 vector, where `Diagonal(Λ[1:(m-k)])` gives generalized parameter Λ
- `g::MVector{1,T}`: `g[1]` stores the generalized parameter g
- `k::MVector{1,Int}`: `k[1]` stores the constraint rank k
- `R::P`: m × m matrix, where `R[:,1:k]` stores the generalized parameter R (i.e. the
constraint matrix)
- `c::V`: m × 1 matrix, where `c[1:k]` stores the the generalized parameter c

`hmsg`, `Qmsg`, `Λmsg`, `gmsg`, `kmsg`, `Rmsg`, `cmsg` are matrices/vectors of the same
dimensions as `h`, `Q`, `Λ`, `g`, `k`, `R`, `c` and are meant to store (1) the outgoing
message for a sending cluster, (2) the incoming message for a receiving cluster, or (3) the
message quotient (from dividing an outgoing message by a sepset belief) for a mediating
sepset. We refer to these as the *message fields* of the generalized belief.

Other fields (defined identically as for [`CanonicalBelief`](@ref)) used to track which
cluster or edge the belief corresponds to, and which traits of which nodes are in scope:
- `nodelabel::Vlabel`
- `ntraits::Integer`
- `inscope::BitArray`
- `type::BeliefType`
- `metadata::M`
"""
struct GeneralizedBelief{
    T<:Real,
    Vlabel<:AbstractVector,
    P<:AbstractMatrix{T},
    V<:AbstractVector{T},
    M,
} <: AbstractBelief{T}
    "Integer label for nodes in the cluster"
    nodelabel::Vlabel
    "Total number of traits at each node"
    ntraits::Int
    """
    Matrix inscope[i,j] is `false` if trait `i` at node `j` is / will be
    removed from scope, to avoid issues from 0 precision or infinite variance; or
    when there is no data for trait `i` below node `j` (in which case tracking
    this variable is only good for prediction, not for learning parameters).
    """
    inscope::BitArray
    μ::V
    h::V
    hmsg::V
    "eigenbasis for precision"
    Q::P
    Qmsg::P
    "eigenvalues of precision"
    Λ::V
    Λmsg::V
    g::MVector{1,T}
    gmsg::MVector{1,T}
    "constraint rank"
    k::MVector{1,Int}
    kmsg::MVector{1,Int}
    "offset"
    c::V
    cmsg::V
    type::BeliefType
    """metadata, e.g. index in cluster graph, of type (M) `Symbol` for clusters or
    Tuple{Symbol, Symbol} for edges"""
    metadata::M
end

"""
    GeneralizedBelief(nodelabels, numtraits, inscope, belieftype, metadata, T=Float64)

Constructor to allocate memory for one cluster, and initialize objects with zeros
to initialize the belief with the constant function exp(0)=1.
"""
function GeneralizedBelief(
    nl::AbstractVector{Tlabel},
    numtraits::Integer,
    inscope::BitArray,
    belief,
    metadata,
    T::Type=Float64,
) where Tlabel<:Integer
    nnodes = length(nl)
    nodelabels = SVector{nnodes}(nl)
    size(inscope) == (numtraits,nnodes) || error("inscope of the wrong size")
    cldim = sum(inscope)
    # canonical part
    Q = MMatrix{cldim,cldim,T}(0 for _ in 1:(cldim*cldim))
    Λ = MVector{cldim,T}(0 for _ in 1:cldim)
    k = MVector{1,Int}(0)
    h = MVector{cldim,T}(0 for _ in 1:cldim)
    μ = MVector{cldim,T}(0 for _ in 1:cldim)
    g = MVector{1,T}(0)
    # constraint part
    c = MVector{cldim,T}(0 for _ in 1:cldim)
    GeneralizedBelief{T,typeof(nodelabels),typeof(Q),typeof(h),typeof(metadata)}(
        nodelabels,numtraits,inscope,
        μ,h,similar(h),Q,similar(Q),Λ,similar(Λ),g,similar(g),
        k,similar(k),
        c,similar(c),
        belief,metadata)
end

"""
    GeneralizedBelief(b::CanonicalBelief)

Constructor from a canonical belief `b`.

Precision `b.J` is eigendecomposed into `Q Λ transpose(Q)`, where `Q` and `Λ`
are square with the same dimensions as `b.J`, and `Λ` is positive semidefinite.
"""
function GeneralizedBelief(b::CanonicalBelief{T,Vlabel,P,V,M}) where {T,Vlabel,P,V,M}
    # J = SArray{Tuple{size(b.J)...}}(b.J) # `eigen` cannot be called on
    Q, Λ = LA.svd(b.J) # todo: should this call svd(...; full=true)?
    m = size(b.J,1) # dimension
    k = MVector{1,Int}(0) # constraint rank 0
    # R = MMatrix{m,m,T}(undef)
    c = MVector{m,T}(undef)
    GeneralizedBelief{T,Vlabel,P,V,M}(
        b.nodelabel,b.ntraits,b.inscope,
        b.μ,Q*b.h,similar(Q*b.h),Q,similar(Q),Λ,similar(Λ),b.g,similar(b.g),
        k,similar(k),
        # R,similar(R),
        c,similar(c),
        b.type,b.metadata)
end

nodelabels(b::AbstractBelief) = b.nodelabel
ntraits(b::AbstractBelief) = b.ntraits
inscope(b::AbstractBelief) = b.inscope
nodedimensions(b::AbstractBelief) = map(sum, eachslice(inscope(b), dims=2))
dimension(b::AbstractBelief) = sum(inscope(b))

function show_name_scope(io::IO, b::AbstractBelief)
    disp = showname(b) * " for "
    disp *= (b.type == bclustertype ? "Cluster" : "SepSet") * " $(b.metadata),"
    disp *= " $(ntraits(b)) traits × $(length(nodelabels(b))) nodes, dimension $(dimension(b)).\n"
    disp *= "Node labels: "
    print(io, disp)
    print(io, nodelabels(b))
    print(io, "\ntrait × node matrix of non-degenerate beliefs:\n")
    show(io, inscope(b))
end

showname(::CanonicalBelief) = "canonical belief"
function Base.show(io::IO, b::CanonicalBelief)
    show_name_scope(io, b)
    print(io, "\nexponential quadratic belief, parametrized by\nμ: $(b.μ)\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

showname(::GeneralizedBelief) = "generalized belief"
function Base.show(io::IO, b::GeneralizedBelief)
    show_name_scope(io, b)
    # todo: show major fields of b
    # print(io, "\nexponential quadratic belief, parametrized by\nμ: $(b.μ)\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    inscope_onenode(node_label, b:AbstractBelief)

AbstractVector: view of the row vector in `b`'s inscope matrix corresponding to
node `node_label`, indicating whether a trait at that node is in scope or not.
"""
function inscope_onenode(node_label, belief::AbstractBelief)
    node_j = findfirst(isequal(node_label), nodelabels(belief))
    return view(inscope(belief), :, node_j)
end

"""
    scopeindex(j::Integer, belief::AbstractBelief)

Indices in the belief's μ,h,J vectors and matrices of the traits in scope
for node indexed `j` in `nodelabels(belief)`.
"""
function scopeindex(j::Integer, belief::AbstractBelief)
    binscope = inscope(belief)
    # subset_inscope = falses(size(belief_inscope))
    # subset_inscope[:,j] .= binscope[:,j]
    # return findall(subset_inscope[binscope])
    node_dims = map(sum, eachslice(binscope, dims=2))
    k0 = sum(node_dims[1:(j-1)]) # 0 if j=1
    collect(k0 .+ (1:node_dims[j]))
end

"""
    scopeindex(node_labels::Union{Tuple,AbstractVector}, belief::AbstractBelief)

Indices in the belief's μ,h,J vectors and matrices of the variables
for nodes labeled `node_labels`. The belief's `inscope` matrix of
booleans says which node (column) and trait (row) is in the belief's scope.
These variables are vectorized by stacking up columns, that is,
listing all in-scope traits of the first node, then all in-scope traits of
the second node etc.
"""
function scopeindex(
    node_labels::Union{Tuple,AbstractVector},
    belief::AbstractBelief
)
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
function scopeindex(
    subset_labels::AbstractVector,
    subset_inscope::BitArray,
    belief_labels::AbstractVector,
    belief_inscope::BitArray
)
    node_index = indexin(subset_labels, belief_labels)
    issorted(node_index) || error("subset labels come in a different order in the belief")
    any(isnothing.(node_index)) && error("subset_labels not a subset of belief_labels")
    any(subset_inscope .&& .!view(belief_inscope,:,node_index)) &&
        error("some variable(s) in subset's scope yet not in full belief's scope")
    subset_inclusterscope = falses(size(belief_inscope))
    subset_inclusterscope[:,node_index] .= subset_inscope
    return findall(subset_inclusterscope[belief_inscope])
end

"""
    scopeindex(node_label, sepset::AbstractBelief, cluster::AbstractBelief)

Tuple of 2 index vectors `(ind_in_sepset, ind_in_cluster)` in the sepset and in the
cluster in-scope variables (that is, in the cluster's μ,h,J vectors and matrices)
of the *shared* in-scope traits for node `node_label`, such that
`sepset.μ[ind_in_sepset]` correspond to all the node's traits in the sepset scope
and `cluster.μ[ind_in_cluster]` correspond to the same traits in the cluster scope,
which may be a subset of all the node's traits in scope for that cluster.
If not, an error is thrown.
"""
function scopeindex(node_lab, sep::AbstractBelief, clu::AbstractBelief)
    s_j = findfirst(isequal(node_lab), nodelabels(sep))
    isnothing(s_j) && error("$node_lab not in sepset")
    sep_inscope = inscope(sep)
    s_insc_node = view(sep_inscope, :,s_j) # column for node in sepset inscope
    c_j = findfirst(isequal(node_lab), nodelabels(clu))
    isnothing(c_j) && error("$node_lab not in cluster")
    clu_inscope = inscope(clu)
    c_insc_node = view(clu_inscope, :,c_j) # column for node in cluster inscope
    any(s_insc_node .& .!c_insc_node) &&
        error("some traits are in sepset's but not in cluster's scope for node $node_lab")
    s_insc = falses(size(sep_inscope))
    s_insc[:,s_j] .= s_insc_node
    ind_sep = findall(s_insc[sep_inscope])
    c_insc = falses(size(clu_inscope))
    c_insc[:,c_j] .= s_insc_node # !! not c_insc_node: to get *shared* traits
    ind_clu = findall(c_insc[clu_inscope])
    return (ind_sep, ind_clu)
end

# todo perhaps: add option to turn off checks, and
# add function to run these checks once between all incident sepset-cluster

"""
    allocatebeliefs(tbl, taxa, nodevector_preordered, clustergraph,
                     evolutionarymodel)

Tuple `(beliefs, (node2cluster, node2family, node2degen, cluster2nodes))` with:
- `beliefs`: vector of beliefs, canonical or generalized as appropriate,
  initialized to the constant function exp(0)=1,
  one for each cluster then one for each sepset in `clustergraph`.
- `node2cluster`: vector mapping each node to the cluster it is assigned to:
  `node2cluster[i]` is the index of the cluster to which was assigned the family
  for node of preorder index `i`.
- `node2family`: vector mapping each node to its node family. Each node is
  represented by its preorder index. A family is represented as a vector of
  preorder indices: [child, parent1, ...].
- `node2fixed`: vector of booleans, indicating for each node if its value is
  fixed, either because it's the root and is assumed fixed by the model,
  or because it's a tip with data (and any missing values removed from scope).
- `node2degen`: vector mapping each node to a boolean indicating if its
  distribution is deterministic given its parents (e.g. if all of its parent
  edges 0 length under a BM with no extra variance).
- `cluster2nodes`: vector mapping each cluster to the nodes (represented by
  their preorder index) whose families are assigned to the cluster.

`tbl` is used to know which leaf in `net` has data for which trait, so as to
remove from the scope each variable without data below it.
`taxa` should be a vector with taxon names in the same order as they come in
the table of data `tbl`.
The root is removed from scope if the evolutionary model has a fixed root: so as
to use the model's fixed root value as data if the root as zero prior variance.

Warnings:
- This function might need to be re-run to re-do allocation if:
    - the data changed: different number of traits, or different pattern of
    missing data at the tips
    - the model changed: with the root changed from fixed to random, see
    [`init_beliefs_allocate_atroot!`](@ref) in that case
"""
function allocatebeliefs(
    tbl::Tables.ColumnTable,
    taxa::AbstractVector,
    prenodes::Vector{PN.Node},
    clustergraph::MetaGraph{T1},
    model::EvolutionaryModel{T2},
) where {T1,T2}
    numtraits = length(tbl)
    nnodes = length(prenodes)
    nnodes > 0 ||
        error("the network should have been pre-ordered, with indices used in cluster graph")
    fixedroot = isrootfixed(model)
    #= hasdata: to know for each node, whether that node has a descendant
                with data, for each trait.
    If not: that node can be removed from all clusters & sepsets
    If yes and the node is a tip: the evidence should be used later, and the tip
    can be removed when the evidence is absorbed =#
    clusterlabs = labels(clustergraph) # same order as cluster beliefs
    node2cluster = zeros(T1, nnodes)
    node2family = Vector{Vector{T1}}(undef, nnodes)
    node2fixed = falses(nnodes)
    node2degen = falses(nnodes)
    cluster2nodes = [Vector{T1}() for _ in clusterlabs]
    #= cluster2degen[i]: cluster i is assigned some degenerate node family, excluding the
    singleton of a fixed root =#
    cluster2degen = falses(length(clusterlabs))
    hasdata = falses(numtraits, nnodes)
    for ni in reverse(1:nnodes)
        node = prenodes[ni]
        nodelab = node.name
        if node.leaf
            i_row = findfirst(isequal(nodelab), taxa)
            isnothing(i_row) && error("tip $nodelab in network without any data")
            for v in 1:numtraits
                hasdata[v,ni] = !ismissing(tbl[v][i_row])
            end
        end
        i_parents = T1[] # preorder indices of parent nodes, sorted in postorder
        degen = true
        for e in node.edge 
            ch = getchild(e)
            if ch === node # parent edge
                degen && (e.length > 0) && (degen = false)
                pi = findfirst(n -> n === getparent(e), prenodes) # parent index
                ii = findfirst(i_parents .< pi) # i_parents is reverse-sorted
                if isnothing(ii) ii = length(i_parents) + 1; end
                insert!(i_parents, ii, pi)
            else # child edge
                i_child = findfirst(n -> n === ch, prenodes)
                isempty(i_child) && error("oops, child (number $(ch.number)) not
                    found in prenodes")
                hasdata[:,ni] .|= hasdata[:,i_child] # bitwise or
            end
        end
        all(!hasdata[v,ni] for v in 1:numtraits) &&
            (node.leaf ? @error("tip $nodelab in network without any data") :
                         @error("internal node $nodelab without any data below"))
        nf = [ni, i_parents...] # node family
        ci = findfirst(cl -> issubset(nf, clustergraph[cl][2]), clusterlabs)
        isnothing(ci) && error("no cluster containing the node family for $(node.number).")
        node2cluster[ni] = ci
        node2family[ni] = nf
        (node.leaf || (ni == 1) && fixedroot) && (node2fixed[ni] = true)
        node2degen[ni] = degen
        push!(cluster2nodes[ci], ni)
        (ni > 1) && !cluster2degen[ci] && degen && (cluster2degen[ci] = degen)
    end
    #= next: create a belief for each cluster and sepset. inscope =
    'has partial information and non-degenerate variance or precision?' =
    - false at the root if "fixedroot", else:
    - 'hasdata?' at internal nodes (assumes non-degenerate transitions)
    - false at tips (assumes all data are at tips)
    - false at degenerate hybrid node with 1 child tree edge of positive length =#
    function build_inscope(set_nodeindices)
        inscope = falses(numtraits, length(set_nodeindices))
        for (i,ni) in enumerate(set_nodeindices)
            node = prenodes[ni]
            (node.leaf || (ni == 1) && fixedroot) && continue # keep 'false' at the root if fixed
            inscope[:,i] = view(hasdata,:,ni)
        end
        return inscope
    end
    beliefs = AbstractBelief{T2}[]
    for (ci, cllab) in enumerate(clusterlabs)
        nodeindices = clustergraph[cllab][2]
        inscope = build_inscope(nodeindices)
        #= Current rule: assign a generalized cluster if the cluster contains a degenerate
        hybrid node family, else assign a canonical cluster
        Possible misassignment: the above rule is not sufficient to conclude that a belief
        should be canonical or generalized. E.g. its possible to have a cluster that contains
        a degenerate hybrid, none of its parents, but all its grandparents. If both its
        parents are also degenerate hybrids, then the child and its grandparents will be
        deterministically related =#
        beliefconstructor = cluster2degen[ci] ? GeneralizedBelief : CanonicalBelief
        push!(beliefs, beliefconstructor(nodeindices, numtraits, inscope, bclustertype,
            cllab, T2))
    end
    for (cllab1, cllab2) in edge_labels(clustergraph)
        nodeindices = clustergraph[cllab1, cllab2]
        inscope = build_inscope(nodeindices)
        beliefconstructor = (cluster2degen[code_for(clustergraph, cllab1)] &&
            cluster2degen[code_for(clustergraph, cllab2)]) ?
            GeneralizedBelief : CanonicalBelief
        #= Current rule: assign a generalized sepset if both adjacent cluster beliefs are
        generalized, else assign a canonical sepset
        Possible cases:
        (1) sender and receiver are generalized => generalized sepset
        (2) sender and receiver are canonical => canonical sepset
        (3) canonical sender, generalized receiver => canonical sepset
        (4) generalized sender, canonical receiver => canonical sepset
        Possible misassignment (valid but less efficient): case 1, where a canonical sepset
        may have sufficed =#
        push!(beliefs, beliefconstructor(nodeindices, numtraits, inscope, bsepsettype,
            (cllab1, cllab2), T2))
    end
    return beliefs, (node2cluster, node2family, node2fixed, node2degen, cluster2nodes)
end

"""
    ClusterFactor(belief::AbstractBelief{T}) where T

Constructor to allocate memory for one cluster factor, with canonical parameters
and metadata initialized to be a copy of those in `belief`.
`ClusterFactor`s metadata are supposed to be symbols, so this constructor should
fail if its input is a sepset belief, whose `metadata` is a Tuple of Symbols.
"""
function ClusterFactor(belief::CanonicalBelief{T}) where T
    h = deepcopy(belief.h)
    J = deepcopy(belief.J)
    g = deepcopy(belief.g)
    ClusterFactor{T,typeof(J),typeof(h)}(h,J,g,belief.metadata)
end
function ClusterFactor(belief::GeneralizedBelief{T}) where T
    k1 = belief.k[1]
    m1 = size(belief.Q)[1]
    J = view(belief.Q,:,1:(m1-k1))*LA.Diagonal(view(belief.Λ,1:(m1-k1)))*
        transpose(view(belief.Q,:,1:(m1-k1)))
    h = view(belief.Q,:,1:(m1-k1))*view(belief.h,1:(m1-k1))
    g = deepcopy(belief.g)
    ClusterFactor{T,typeof(J),typeof(h)}(h,J,g,belief.metadata)
end

"""
    init_factors_allocate(beliefs::AbstractVector{<:AbstractBelief}, nclusters::Integer)

Vector of `nclusters` factors of type [`ClusterFactor`](@ref), whose canonical
parameters and metadata are initialized to be a copy of those in `beliefs`.
Assumption: `beliefs[1:nclusters]` are cluster beliefs, and
`beliefs[nclusters+1:end]` (if any) are sepset beliefs. This is not checked.
"""
function init_factors_allocate(
    beliefs::AbstractVector{B},
    nclusters::Integer
) where B<:AbstractBelief{T} where T
    factors = Vector{ClusterFactor}(undef, nclusters)
    for i in 1:nclusters
        factors[i] = ClusterFactor(beliefs[i])
    end
    return factors
end

"""
    init_beliefs_allocate_atroot!(beliefs, factors, messageresiduals, model,
        node2family, cluster2nodes)

Update the scope and re-allocate memory for cluster & sepset `beliefs`, `factors`
and `messageresiduals` to include or exclude the root,
depending on whether the root variable is random or fixed in `model`.
To change the dimension of canonical parameters μ,h,J, new memory is allocated
and initialized to 0.
This function can be used to update beliefs when the root model changes from
fixed to non-fixed or vice-versa.
It re-allocates less memory than [`allocatebeliefs`](@ref) (which would
need to be followed by [`init_factors_allocate`](@ref))
because clusters and sepsets that do not have the root are not modified.

Assumptions:
- all traits at the root have at least one descendant with non-missing data,
- beliefs were previously initialized with a model that had the same number of
  traits as the current `model`.
"""
function init_beliefs_allocate_atroot!(
    beliefs,
    factors,
    messageresidual,
    model::EvolutionaryModel{T},
    node2family,
    node2fixed,
    cluster2nodes,
) where T
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
        beliefs[i_b] = CanonicalBelief(be.nodelabel, numtraits, be_insc, be.type, be.metadata, T)
        if iscluster # re-allocate the corresponding factor. if sepset: nothing to do
            factors[i_b] = ClusterFactor(beliefs[i_b])
            for ni in cluster2nodes[i_b]
                nf = node2family[ni]
                # if node family contains root node, then update whether it is inscope
                (nf[end] == 1) && (node2fixed[1] = fixedroot)
            end
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
    init_beliefs_reset!(beliefs::Vector{<:AbstractBelief})

Reset all beliefs (which can be cluster and/or sepset beliefs) so that they can
later be re-initialized for different model parameters and re-calibrated, without
re-allocating memory.
For CanonicalBeliefs, all entries of h, J, g are zeroed, with μ left unchanged.
For GeneralizedBeliefs, all entries of h, hmsg, Λ, Λmsg, g, gmsg, k, kmsg are
zeroed. Q and Qmsg are set to the identity. μ, R, Rmsg, c, cmsg are left unchanged.
"""
function init_beliefs_reset!(beliefs::AbstractVector{B}) where B<:AbstractBelief{T} where T
    for be in beliefs
        init_belief_reset!(be)
    end
    return nothing
end
function init_belief_reset!(be::CanonicalBelief{T}) where T
    be.h .= zero(T)
    be.J .= zero(T)
    be.g[1] = zero(T)
    return nothing
end
function init_belief_reset!(be::GeneralizedBelief{T}) where T
    be.h .= zero(T)
    be.hmsg .= zero(T)
    be.Q[:,:] = LA.UniformScaling(one(T))(size(be.Q,1))
    be.Qmsg[:,:] = LA.UniformScaling(one(T))(size(be.Qmsg,1))
    be.Λ .= zero(T)
    be.Λmsg .= zero(T)
    be.g[1] = zero(T)
    be.gmsg[1] = zero(T)
    be.k[1] = Int(0)
    be.kmsg[1] = Int(0)
    return nothing
end

"""
    init_factors_frombeliefs!(
        factors,
        beliefs::AbstractVector{<:CanonicalBelief},
        checkmetadata::Bool=false
    )

Reset all `factors` by copying h,J,g from `beliefs`.
Assumption: the cluster beliefs match the factors exactly: for a valid factor
index `i`, `beliefs[i]` is of cluster type and has the same dimension as
`factors[i]`.

Set `checkmetadata` to true to check that `beliefs[i]` and `factors[i]` have
the same metadata.
"""
function init_factors_frombeliefs!(
    factors,
    beliefs::AbstractVector{B},
    checkmetadata::Bool=false,
) where B <: AbstractBelief
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
    assignfactors!(beliefs,
                   evolutionarymodel, columntable, taxa,
                   nodevector_preordered, node2cluster, node2family)

Initialize cluster beliefs prior to belief propagation, by assigning each factor
to one cluster. Sepset beliefs are reset to 1.
There is one factor for each node v: the density of X\\_v conditional on its
parent X\\_pa(v) if v is not the root, or the prior density for X_root.

- For each leaf, the factor is reduced by absorbing the evidence for that leaf,
that is, the data found in `columntable`, whose rows should be ordered
consistent with `taxa`
- For each leaf, missing trait values are removed from scope.
- For each internal node, any trait not in scope (e.g. if all descendant leaves
are missing a value for this trait) is marginalized out of the factor.

Assumptions:
- In `beliefs`, cluster beliefs come first and sepset beliefs come last, as when
created by [`allocatebeliefs`](@ref)

`beliefs` is modified in place.
"""
function assignfactors!(
    beliefs::AbstractVector{B},
    model::EvolutionaryModel,
    tbl::Tables.ColumnTable,
    taxa::AbstractVector,
    prenodes::Vector{PN.Node},
    node2cluster,
    node2family,
    node2fixed,
) where B <: AbstractBelief{T} where T
    init_beliefs_reset!(beliefs)
    numtraits = dimension(model)
    for (ni, ci) in enumerate(node2cluster)
        be = beliefs[ci]
        nf = node2family[ni] # nodefamily: [child, parent1, ...]
        nfsize = length(nf)
        ch = prenodes[ni] # child node
        if nfsize == 1
            ni != 1 && error("only the root node can belong to a family of size 1")
            # root is inscope iff root is not fixed (i.e. isrootfixed(model) == false)
            node2fixed[1] && continue
            ϕ = factor_root(model)
        else
            if nfsize == 2
                ϕ = factor_treeedge(model, getparentedge(ch))
            else
                pae = PN.Edge[]
                for pi in nf[2:end] # parent indices
                    for e in prenodes[pi].edge
                        if getchild(e) === ch
                            push!(pae, e)
                            break
                        end
                    end
                end
                ϕ = factor_hybridnode(model, pae)
            end
            # absorb evidence (assume that only at leaves or root)
            if node2fixed[ni] # `ch` is a leaf
                i_datarow = findfirst(isequal(ch.name), taxa)
                ϕ = absorbleaf!(ϕ..., i_datarow, tbl)
            end
            if any(node2fixed[nf[2:end]]) # node's parents include a fixed root
                rootindex = (size(ϕ[1],1) - numtraits + 1):size(ϕ[1],1) # ϕ[1] is h or R
                ϕ, _ = absorbevidence!(ϕ..., rootindex, rootpriormeanvector(model))
            end
        end
        i_inscope = nf[.!node2fixed[nf]]
        factorind = scopeindex(i_inscope, be)
        if length(factorind) != numtraits * length(i_inscope)
            var_inscope = view(inscope(be), :, indexin(i_inscope, nodelabels(be)))
            keep_index = LinearIndices(var_inscope)[var_inscope]
            if !node2fixed[ni]
                #= if child node is not leaf, integrate out non-inscope traits of child node
                first then those of parent(s), rather than all at once 
                fixit: this works for BM, but not for general linear Gaussian? =#
                # todo: clean up
                integrate_index_ch = setdiff(1:numtraits, keep_index[keep_index .≤ numtraits])
                keep_index_ch = setdiff(1:size(ϕ[1],1), integrate_index_ch)
                ϕ = marginalize(ϕ..., keep_index_ch, be.metadata)
                if any(.!node2fixed[nf[2:end]])
                    keep_index_pa = keep_index[keep_index .> numtraits]
                    integrate_index_pa = setdiff((numtraits+1):(numtraits * length(i_inscope)), keep_index_pa)
                    keep_index_pa .-= (numtraits - sum(keep_index .≤ numtraits))
                    integrate_index_pa .-= (numtraits - sum(keep_index .≤ numtraits))
                    ϕ = marginalize(ϕ..., vcat(1:sum(keep_index .≤ numtraits), keep_index_pa), integrate_index_pa, be.metadata)
                end
            else
                # if child node is leaf, integrate out non-inscope traits of parent(s)
                ϕ = marginalize(ϕ..., keep_index, be.metadata)
            end
        end
        mult!(be, factorind, ϕ...)
    end
    return
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
form of the ratio: sent\\_message / current\\_sepset\\_belief, when a message is
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
are initialized to `false` if the message is of positive dimension.
If the message is empty (ΔJ and Δh of dimension 0) then the message is initialized
as being calibrated: `kldiv` is set to 0 and `iscalibrated` flags set to true.

`(ΔJ, Δh)` of zero suggest calibration, but the flags `iscalibrated_{resid,kl}`
being false indicate otherwise.
"""
function MessageResidual(J::AbstractMatrix{T}, h::AbstractVector{T}) where {T <: Real}
    Δh = zero(h)
    ΔJ = zero(J)
    kldiv, iscal_res, iscal_kl = (isempty(h) ?
        (MVector(zero(T)), MVector(true),  MVector(true) ) :
        (MVector(-one(T)), MVector(false), MVector(false))
    )
    MessageResidual{T,typeof(ΔJ),typeof(Δh)}(Δh, ΔJ, kldiv, iscal_res, iscal_kl)
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
function init_messageresidual_allocate(
    beliefs::Vector{AbstractBelief{T}},
    nclusters,
) where T<:Real
    messageresidual = Dict{Tuple{Symbol,Symbol}, MessageResidual{T}}()
    for j in (nclusters+1):length(beliefs)
        ssbe = beliefs[j] # sepset belief
        (clustlab1, clustlab2) = ssbe.metadata      
        if isa(ssbe, CanonicalBelief)
            messageresidual[(clustlab1, clustlab2)] = MessageResidual(ssbe.J, ssbe.h)
            messageresidual[(clustlab2, clustlab1)] = MessageResidual(ssbe.J, ssbe.h)
        else
            isa(ssbe, GeneralizedBelief) || error() # todo: error message
            k1 = ssbe.k[1]
            m1 = size(ssbe.Q)[1]
            J = view(ssbe.Q,:,1:(m1-k1))*LA.Diagonal(view(ssbe.Λ,1:(m1-k1)))*
                transpose(view(ssbe.Q,:,1:(m1-k1)))
            h = view(ssbe.Q,:,1:(m1-k1))*view(ssbe.h,1:(m1-k1))
            messageresidual[(clustlab1, clustlab2)] = MessageResidual(J, h)
            messageresidual[(clustlab2, clustlab1)] = MessageResidual(J, h)
        end
    end
    return messageresidual
end

"""
    init_messagecalibrationflags_reset!(mr::AbstractResidual, reset_kl::Bool)

For a non-empty message residual `mr`, reset its `iscalibrated_*` flags to false,
and if `reset_kl` is true, reset its `kldiv` to -1.
Its `ΔJ` and `Δh` fields are *not* reset here, because they are overwritten
during a belief propagation step.
Nothing is done for empty messages.
"""
function init_messagecalibrationflags_reset!(mr::AbstractResidual{T}, resetkl::Bool) where T
    if isempty(mr.Δh) return nothing; end
    if resetkl   mr.kldiv[1] = - one(T); end
    mr.iscalibrated_resid[1] = false
    mr.iscalibrated_kl[1] = false
    return nothing
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
        isapprox(LA.norm(res.Δh ./ sqrt(length(res.Δh)), p), zero(T), atol=atol) &&
        isapprox(LA.norm(res.ΔJ ./ sqrt(length(res.ΔJ)), p), zero(T), atol=atol)
        # isapprox(LA.norm(res.Δh, p)/sqrt(length(res.Δh)), zero(T), atol=atol) &&
        # isapprox(LA.norm(res.ΔJ, p)/sqrt(length(res.ΔJ)), zero(T), atol=atol)
        # isapprox(LA.norm(res.Δh, p), zero(T), atol=atol) &&
        # isapprox(LA.norm(res.ΔJ, p), zero(T), atol=atol)
        # TODO: discuss adjusted threshold
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
    residual_kldiv!(residual::AbstractResidual, sepset::AbstractFactor,
        canonicalparams::Tuple)

Update `residual.kldiv` with the
[Kullback-Leibler](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
divergence between
a message sent through a sepset (normalized to a probability distribution),
and the sepset belief before the belief update (also normalized).
`sepset` should contain the updated belief, and `residual` the difference
in the `J` and `h` parameters due to the belief update (after - before),
such that the previous belief is: `sepset` belief - `residual`.
As a side product, `sepset.μ` is updated.

Output: `true` if the KL divergence is close to 0, `false` otherwise.
See [`iscalibrated_kl!`](@ref) for the tolerance.

If the current or previous `sepset` belief is degenerate,
in the sense that its precision matrix is not positive definite and the
belief cannot be normalized to a proper distribution, then
`residual` and `sepset` are not updated, and `false` is returned.
No warning and no error is sent, because sepset beliefs are initialized at 0
and this case is expected to be frequent before enough messages are sent.

## Calculation:

sepset after belief-update (i.e. message sent): C(x | Jₘ, hₘ, gₘ) ∝ density for
    X ~ 𝒩(μ=Jₘ⁻¹hₘ, Σ=Jₘ⁻¹)  
sepset before belief-update: C(x | Jₛ, hₛ, gₛ)  
residual: ΔJ = Jₘ - Jₛ, Δh = hₘ - hₛ  
p: dimension of X (number of variables: number of nodes * number of traits).
Below, we use the notation Δg for the change in constants to normalize each
message, which is *not* gₘ-gₛ because the stored beliefs are not normalized.

    KL(C(Jₘ, hₘ, _) || C(Jₛ, hₛ, _))
    = Eₘ[log C(x | Jₘ,hₘ,_)/C(x | Jₛ,hₛ,_)] where x ∼ C(Jₘ,hₘ,_)
    = Eₘ[-(1/2) x'ΔJx + Δh'x + Δg)]
    = (  tr(JₛJₘ⁻¹) - p + (μₛ-μₘ)'Jₛ(μₛ-μₘ) + log(det(Jₘ)/det(Jₛ))  ) /2

See also: [`average_energy!`](@ref), which only requires the sepset belief
to be positive definite.
"""
function residual_kldiv!(res::AbstractResidual{T}, sepset::AbstractFactor{T}) where {T <: Real}
    # isposdef returns true for empty matrices e.g. isposdef(Real[;;]) and isposdef(MMatrix{0,0}(Real[;;]))
    isempty(sepset.J) && return true
    (J0, μ0) = try getcholesky_μ!(sepset) # current (m): message that was passed
    catch
        return false
    end
    (J1, μ1) = try getcholesky_μ(sepset.J .- res.ΔJ, sepset.h .- res.Δh) # before (s)
    catch
        return false
    end
    res.kldiv[1] = ( - LA.tr(J0 \ res.ΔJ) + # tr(JₛJₘ⁻¹-I) = tr((Jₛ-Jₘ)Jₘ⁻¹) = tr(-ΔJ Jₘ⁻¹)
        quad(J1, μ1-μ0) + # (μ1-μ0)' J1 (μ1-μ0)
        LA.logdet(J0) - LA.logdet(J1) )/2
    iscalibrated_kl!(res)
end