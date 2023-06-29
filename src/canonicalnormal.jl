abstract type AbstractBelief end

# generic methods
nodelabels(b::AbstractBelief) = b.nodelabel
nvar(b::AbstractBelief)       = b.nvar
nonmissing(b::AbstractBelief) = b.nonmissing
nodedimensions(b::AbstractBelief) = map(sum, eachslice(nonmissing(b), dims=2))
dimension(b::AbstractBelief)  = sum(nonmissing(b))

struct ClusterBelief{Vlabel<:AbstractVector,T<:Real,P<:AbstractMatrix{T},V<:AbstractVector{T}} <: AbstractBelief
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
end

"""
    ClusterBelief(nodelabels, nvar)

Constructor to allocate memory for one cluter, initialized with:
nothing missing, mean 0, precision I, g=0.
"""
function ClusterBelief(nl::AbstractVector{Tlabel}, nvar::Integer) where Tlabel<:Integer
    nnodes = length(nl)
    nodelabels = SVector{nnodes}(nl)
    cldim = nnodes * nvar
    nonmissing = BitArray(undef, cldim,cldim)
    fill!(nonmissing, true)
    T = Float64
    μ = zeros(T, cldim) # fixit: use MVector & MMatrix if same dimension always, despite missings
    h = zeros(T, cldim)
    J = Matrix{T}(LinearAlgebra.I, cldim, cldim)
    g = MVector{1,Float64}(0.0)
    ClusterBelief{SVector{nnodes,Tlabel},T,Matrix{T},Vector{T}}(nodelabels, nvar, nonmissing, μ, h, J, g)
end

#function init_beliefs(clustergraph, tbl::Tables.ColumnTable)
#    return nothing
#end
