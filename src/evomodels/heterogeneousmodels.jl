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

function PaintedParameter(parameter::Vector{T}, d::Dict=DefaultDict{Int,Int}(1)) where T
    PaintedParameter{T}(parameter, DefaultDict(1, d))
end

ncolors(pp::PaintedParameter) = length(pp.parameter)

getparameter(pp::PaintedParameter, e::PN.Edge) = getparameter(pp, e.number)
getparameter(pp::PaintedParameter, number::Int) = pp.parameter[pp.color[number]]

function Base.show(io::IO, obj::PaintedParameter)
    disp = "Painted parameter on a network with $(ncolors(obj)) different parameters:\n"
    for pp in obj.parameter
        disp *= " $pp"
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
end

modelname(m::HeterogeneousBrownianMotion) = "Heterogeneous Brownian motion"
variancename(m::HeterogeneousBrownianMotion) = "evolutionary variance rates"
varianceparam(m::HeterogeneousBrownianMotion) = m.variancerate
# TODO: extend this constructor to take instead of vector of rates and a color map
function HeterogeneousBrownianMotion(R, μ, v=nothing)
    HeterogeneousBrownianMotion([R], DefaultDict{Int,Int}(1), μ, v)
end
function HeterogeneousBrownianMotion(Rvec, colors, μ, v=nothing)
    if !isa(μ, Array) μ = [μ]; end
    numt = length(μ)
    T = promote_type(Float64, eltype(R), eltype(μ))
    all(size(R) == (numt,numt) for R in Rvec) || error("R and μ have conflicting sizes")
    all(LA.issymmetric(R) for R in Rvec) || error("R should be symmetric")
    Rvec = [PDMat(R) for R in Rvec]
    Jvec = inv.(Rvec) # uses cholesky. fails if not symmetric positive definite
    if isnothing(v)
        v = LA.Symmetric(zeros(T, numt, numt))
    else
        size(v) == (numt,numt)       || error("v and μ have conflicting sizes")
        LA.issymmetric(v)            || error("v should be symmetric")
        v = LA.Symmetric(v)
        LA.isposdef(v)               || error("v is not positive semi-definite")
    end
    HeterogeneousBrownianMotion{T, typeof(μ), typeof(v), typeof(R)}(
        PaintedParameter(Rvec, colors), PaintedParameter(Rvec, colors), μ, v,
        PaintedParameter([-(numt * log2π + LA.logdet(R))/2], colors)
    )
end
# params(m::HeterogeneousBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)

function branch_actualization(obj::HeterogeneousBrownianMotion{T}, ::PN.Edge) where T
    ScalMat(dimension(obj), one(T))
    # LA.Diagonal{T, Vector{T}}(LA.I(dimension(obj)))
end
function branch_actualization!(q::AbstractMatrix, ::HeterogeneousBrownianMotion, ::PN.Edge)
    # below: would error on a ScalMat, but used on a Matrix created by branch_actualization
    q[LA.diagind(q)] .= 1.0
    LA.tril!(q)
    LA.triu!(q)
end
function branch_displacement(obj::HeterogeneousBrownianMotion{T}, ::PN.Edge) where T
    zeros(T, dimension(obj))
end
function branch_precision(obj::HeterogeneousBrownianMotion, edge::PN.Edge)
    getparameter(obj.inverserate, edge) ./ edge.length
end
function branch_variance(obj::HeterogeneousBrownianMotion, edge::PN.Edge)
    edge.length .* getparameter(obj.variancerate, edge)
end
function hybdridnode_displacement(obj::HeterogeneousBrownianMotion{T}, parentedges::AbstractVector{PN.Edge}) where T
    zeros(T, dimension(obj))
end
function hybridnode_variance(obj::HeterogeneousBrownianMotion{T}, parentedges::AbstractVector{PN.Edge}) where T
    zeros(T, dimension(obj), dimension(obj))
end
