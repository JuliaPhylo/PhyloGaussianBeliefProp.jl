"""
    PaintedParameter{T}

Type with 2 fields:
- `parameter`: vector whose elements are of type `T`
- `color`: `DefaultDict` dictionary mapping integers to integers,
  with a default value of 1.

`ncolors(pp)` returns the number of parameters, that is, the length
of `pp.parameter`
This type is meant to store several values for a given evolutionary parameter
(say, Brownian motion variance rate), each one being used on some edges or
nodes of a phylogenetic network. The default parameter value is the first one.
For an edge number `i`, color `j=pp.color[i]` indexes its parameter value,
that is, evolution along edge number `i` should use `pp.parameter[j]`.
This parameter value is obtained with `getparameter`.
"""
struct PaintedParameter{T}
    parameter::Vector{T}
    color::DefaultDict{Int,Int}
end

function PaintedParameter(parameter::Vector{T}, d::Dict=Dict{Int,Int}()) where T
    PaintedParameter{T}(parameter, DefaultDict(1, d))
end

ncolors(pp::PaintedParameter) = length(pp.parameter)

getparameter(pp::PaintedParameter, e::PN.Edge) = getparameter(pp, e.number)
getparameter(pp::PaintedParameter, number::Int) = pp.parameter[pp.color[number]]

function Base.show(io::IO, obj::PaintedParameter)
    disp = "Painted parameter on a network with $(ncolors(obj)) different parameters:"
    for (ind, pp) in enumerate(obj.parameter)
        disp *= "\n$ind: $pp"
    end
    disp *= "\nmapping of edge/node number to parameter color:\n$(obj.color)"
    print(io, disp)
end

################################################################
## Heterogeneous Model
################################################################

abstract type HeterogeneousEvolutionaryModel{T} <: EvolutionaryModel{T} end

"""
    HeterogeneousBrownianMotion{T,U,V,W} <: HeterogeneousEvolutionaryModel{T}

Type for a heterogeneous Brownian motion model, univariate or multivariate.
Along each edge, evolution follows a Brownian motion.
Each edge can have its own variance rate.
This model has no shifts, and no extra hybrid variance.
By default, the root is fixed with prior variance 0.

`T` is the scalar type, `U` is the type for the root mean (vector of length d,
where `d` is the trait dimension, even if univariate), `V` is the type for
the root variance, and `W` the type of each variance rate, one per color.
For a univariate BM, we may have `W=T` and `V=Vector{T}`.
For a multivariate BM, we may have `W=V<:Matrix{T}`.
This is such that each field is mutable, so we can update model parameters
in place within the model object, itself immutable.
"""
struct HeterogeneousBrownianMotion{T<:Real, U<:AbstractVector{T}, V<:AbstractMatrix{T}, W<:Union{T, V, PDMats.PDMat{T}}} <: HeterogeneousEvolutionaryModel{T}
    "variance rate"
    variancerate::PaintedParameter{W}
    "inverse variance (precision) rate"
    inverserate::PaintedParameter{W}
    "prior mean at the root"
    μ::U
    "prior variance at the root"
    v::V
    "g0: -log(2π variancerate)/2"
    g0::PaintedParameter{T}
    # TODO: we do not use g0 right now. Should we ? Or delete ? Or maybe will be used if we do not rely on fall backs anymore.
end

modelname(m::HeterogeneousBrownianMotion) = "Heterogeneous Brownian motion"
variancename(m::HeterogeneousBrownianMotion) = "evolutionary variance rates"
varianceparam(m::HeterogeneousBrownianMotion) = m.variancerate
function HeterogeneousBrownianMotion(R::AbstractMatrix, μ, v=nothing)
    HeterogeneousBrownianMotion([R], Dict{Int,Int}(), μ, v)
end
function HeterogeneousBrownianMotion(Rvec, colors::AbstractDict, μ, v=nothing)
    if !isa(μ, Array) μ = [μ]; end
    numt = length(μ)
    length(Rvec) >= 1 || error("Rvec must have at list one component")
    T = promote_type(Float64, eltype(Rvec[1]), eltype(μ))
    v = getrootvariancemultivariate(T, numt, v)
    all(size(R) == (numt,numt) for R in Rvec) || error("R and μ have conflicting sizes")
    all(LA.issymmetric(R) for R in Rvec) || error("R should be symmetric")
    Rvec = [PDMat(R) for R in Rvec]
    Jvec = inv.(Rvec) # uses cholesky. fails if not symmetric positive definite
    gvec = [-(numt * log2π + LA.logdet(R))/2 for R in Rvec]
    HeterogeneousBrownianMotion{T, typeof(μ), typeof(v), typeof(Rvec[1])}(
        PaintedParameter(Rvec, colors), PaintedParameter(Jvec, colors), μ, v,
        PaintedParameter(gvec, colors)
    )
end
function HeterogeneousBrownianMotion(paintedrates::PaintedParameter, μ, v=nothing)
    HeterogeneousBrownianMotion(paintedrates.parameter, paintedrates.color, μ, v)
end
# params(m::HeterogeneousBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)

function branch_actualization(obj::HeterogeneousBrownianMotion{T}, ::PN.Edge) where T
    ScalMat(dimension(obj), one(T))
end
function branch_actualization!(q::AbstractMatrix, ::HeterogeneousBrownianMotion, ::PN.Edge)
    # below: would error on a ScalMat, but used on a Matrix created by branch_actualization
    q[LA.diagind(q)] .= 1.0
    LA.tril!(q)
    LA.triu!(q)
end
function branch_precision(obj::HeterogeneousBrownianMotion, edge::PN.Edge)
    getparameter(obj.inverserate, edge) ./ edge.length
end
function branch_variance(obj::HeterogeneousBrownianMotion, edge::PN.Edge)
    edge.length .* getparameter(obj.variancerate, edge)
end
function branch_logdet(obj::HeterogeneousBrownianMotion, edge::PN.Edge)
    getparameter(obj.g0, edge) - dimension(obj) * log(edge.length)/2
end
function factor_treeedge(m::HeterogeneousBrownianMotion, edge::PN.Edge)
    ntraits = dimension(m)
    q = branch_actualization(m, edge)
    j = branch_precision(m, edge)
    g = branch_logdet(m, edge)
    factor_treeedge(q,j,1,ntraits,g)
end
function factor_hybridnode(m::HeterogeneousBrownianMotion{T}, pae::AbstractVector{PN.Edge}) where T
    ntraits = dimension(m)
    nparents = length(pae)
    v = zeros(T, ntraits, ntraits) # no extra node variance
    q = Matrix{T}(undef, ntraits, nparents * ntraits) # init actualisation
    for (k, edge) in enumerate(pae)
        qe = view(q, :, ((k-1) * ntraits + 1):(k*ntraits))
        ve = branch_variance(m, edge)
        branch_actualization!(qe, m, edge)
        qe .*= edge.gamma
        v .+= edge.gamma^2 .* ve
    end
    j = inv(v) # block variance
    g0 = (- ntraits * log2π + LA.logdet(j))/2
    factor_treeedge(q,j,nparents,ntraits,g0)
end

"""
    HeterogeneousShiftedBrownianMotion{T,U,V,W} <: HeterogeneousEvolutionaryModel{T}

Type for a heterogeneous Brownian motion model like
[`HeterogeneousBrownianMotion`](@ref) but with a possible
shift in the mean along each edge.
"""
struct HeterogeneousShiftedBrownianMotion{T<:Real, U<:AbstractVector{T}, V<:AbstractMatrix{T}, W<:Union{T, V, PDMats.PDMat{T}}} <: HeterogeneousEvolutionaryModel{T}
    "variance rate"
    variancerate::PaintedParameter{W}
    "inverse variance (precision) rate"
    inverserate::PaintedParameter{W}
    "shift in the mean along edges"
    shiftmean::PaintedParameter{U}
    "prior mean at the root"
    μ::U
    "prior variance at the root"
    v::V
    "g0: -log(2π variancerate)/2"
    g0::PaintedParameter{T}
    # TODO: we do not use g0 right now. Should we ? Or delete ? Or maybe will be used if we do not rely on fall backs anymore.
end

modelname(m::HeterogeneousShiftedBrownianMotion) = "Heterogeneous Brownian motion with mean shifts"
variancename(m::HeterogeneousShiftedBrownianMotion) = "evolutionary variance rates"
varianceparam(m::HeterogeneousShiftedBrownianMotion) = m.variancerate
# fixit: write a constructor

function branch_actualization(obj::HeterogeneousShiftedBrownianMotion{T}, ::PN.Edge) where T
    ScalMat(dimension(obj), one(T))
end
function branch_actualization!(q::AbstractMatrix, ::HeterogeneousShiftedBrownianMotion, ::PN.Edge)
    # below: would error on a ScalMat, but used on a Matrix created by branch_actualization
    q[LA.diagind(q)] .= 1.0
    LA.tril!(q)
    LA.triu!(q)
end
function branch_displacement(obj::HeterogeneousShiftedBrownianMotion, ::PN.Edge)
    getparameter(obj.shiftmean, edge)
end
function branch_precision(obj::HeterogeneousShiftedBrownianMotion, edge::PN.Edge)
    getparameter(obj.inverserate, edge) ./ edge.length
end
function branch_variance(obj::HeterogeneousShiftedBrownianMotion, edge::PN.Edge)
    edge.length .* getparameter(obj.variancerate, edge)
end
function branch_logdet(obj::HeterogeneousShiftedBrownianMotion, edge::PN.Edge, j)
    getparameter(obj.g0, edge) - dimension(obj) * log(edge.length)/2
end
function hybdridnode_displacement(obj::HeterogeneousShiftedBrownianMotion{T}, parentedges::AbstractVector{PN.Edge}) where T
    zeros(T, dimension(obj))
end
function hybridnode_variance(obj::HeterogeneousShiftedBrownianMotion{T}, parentedges::AbstractVector{PN.Edge}) where T
    ntraits = dimension(obj)
    zeros(T, ntraits, ntraits)
end
