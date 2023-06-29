@enum BeliefType bclustertype=1 bsepsettype=2

abstract type AbstractBelief end

# generic methods
nodelabels(b::AbstractBelief) = b.nodelabel
nvar(b::AbstractBelief)       = b.nvar
nonmissing(b::AbstractBelief) = b.nonmissing
nodedimensions(b::AbstractBelief) = map(sum, eachslice(nonmissing(b), dims=2))
dimension(b::AbstractBelief)  = sum(nonmissing(b))

struct ClusterBelief{Vlabel<:AbstractVector,T<:Real,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief
    "Integer label for nodes in the cluster"
    nodelabel::Vlabel # StaticVector{N,Tlabel}
    "Total number of variables at each node"
    nvar::Int
    """Matrix nonmissing[i,j] is `false` if variable `i` at node `j` is
       missing, that is: has 0 precision or infinite variance"""
    nonmissing::BitArray
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

"""
    ClusterBelief(nodelabels, nvar, nonmissing, metadata)

Constructor to allocate memory for one cluter, without initializing it.
"""
function ClusterBelief(nl::AbstractVector{Tlabel}, nvar::Integer, nonmissing::BitArray, belief, metadata) where Tlabel<:Integer
    nnodes = length(nl)
    nodelabels = SVector{nnodes}(nl)
    size(nonmissing) == (nvar,nnodes) || error("nonmissing of the wrong size")
    cldim = sum(nonmissing)
    T = Float64
    μ = MVector{cldim,T}(undef)       # zeros(T, cldim)
    h = MVector{cldim,T}(undef)
    J = MMatrix{cldim,cldim,T}(undef) # Matrix{T}(LinearAlgebra.I, cldim, cldim)
    g = MVector{1,Float64}(0.0)
    ClusterBelief{typeof(nodelabels),T,typeof(J),typeof(h),typeof(metadata)}(
        nodelabels,nvar,nonmissing,μ,h,J,g,belief,metadata)
end

function init_beliefs_allocate(taxa::AbstractVector, tbl::Tables.ColumnTable,
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
    hasdata = BitArray(false for v in 1:nvar, n in 1:nnodes)
    for i_node in reverse(eachindex(net.nodes_changed))
        node = net.nodes_changed[i_node]
        nodelab = node.name
        i_row = findfirst(isequal(nodelab), taxa)
        if !isnothing(i_row) # the node has data: it should be a tip!
            node.leaf || error("A node with data is internal, should be a leaf")
            for v in 1:nvar
              hasdata[v,i_node] = !ismissing(tbl[v][i_row])
            end
        else
            # todo: loop over children node then do bit-wise OR
        end
    end
    # todo: change the meaning of booleans in 'hasdata':
    # from 'has data at descendant'
    # to 'has partial information and non-degenerate variance or precision'.
    # So: change to 'false' at tips
    for cllab in labels(clustergraph)
        clnodeindices = clustergraph[cllab][2]
        nonmissing = BitArray(undef, nvar,length(clnodeindices))
        for (i,i_node) in enumerate(clnodeindices)
            nonmissing[:,i] .= hasdata[:,i_node]
        end
    end
    # todo:
    # build nonmissing for each cluster and each sepset
    # construct a ClusterBelief for each cluster and each sepset
    return hasdata
end
