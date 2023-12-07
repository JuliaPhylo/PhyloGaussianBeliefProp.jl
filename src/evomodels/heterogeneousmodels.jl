################################################################
## Painted Parameter
################################################################

struct PaintedParameter{T}
    parameter::Vector{T}
    color::DefaultDict{Int,Int}
end

function PaintedParameter(parameter::Vector{T}) where T
    PaintedParameter{T}(parameter, DefaultDict{Int,Int}(1))
end

function PaintedParameter(parameter::Vector{T}, d::Dict) where T
    PaintedParameter{T}(parameter, DefaultDict(1, d))
end

ncolors(pp::PaintedParameter) = length(pp.parameter)

################################################################
## Heterogeneous Model
################################################################

abstract type HeterogeneousEvolutionaryModel{T} <: EvolutionaryModel{T} end

"""
    HeterogeneousBrownianMotion{T} <: HeterogeneousEvolutionaryModel{T}

The heterogeneous Brownian motion.
TODO
"""
struct HeterogeneousBrownianMotion{T<:Real, U<:AbstractVector{T}, V<:AbstractMatrix{T}, W<:Union{T, V}} <: HeterogeneousEvolutionaryModel{T}
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
variancename(m::HeterogeneousBrownianMotion) = "evolutionary variance rate matrix"
varianceparam(m::HeterogeneousBrownianMotion) = m.variancerate
function HeterogeneousBrownianMotion(R, μ, v=nothing)
    if !isa(μ, Array) μ = [μ]; end
    numt = length(μ)
    T = promote_type(Float64, eltype(R), eltype(μ))
    size(R) == (numt,numt)       || error("R and μ have conflicting sizes")
    LA.issymmetric(R) || error("R should be symmetric")
    R = PDMat(R)
    J = inv(R) # uses cholesky. fails if not symmetric positive definite
    if isnothing(v)
        v = LA.Symmetric(zeros(T, numt, numt))
    else
        size(v) == (numt,numt)       || error("v and μ have conflicting sizes")
        LA.issymmetric(v)            || error("v should be symmetric")
        v = LA.Symmetric(v)
        LA.isposdef(v)               || error("v is not positive semi-definite")
    end

    HeterogeneousBrownianMotion{T, typeof(μ), typeof(v), typeof(R)}(
        PaintedParameter([R]), PaintedParameter([J]), μ, v, PaintedParameter([-(numt * log2π + LA.logdet(R))/2])
    )
end
# params(m::HeterogeneousBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)
