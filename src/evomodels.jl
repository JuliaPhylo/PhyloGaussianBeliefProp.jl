abstract type EvolutionaryModel end

# generic methods
modelname(obj::EvolutionaryModel) = string(typeof(obj))
variancename(obj::EvolutionaryModel) = "variance"
varianceparam(obj::EvolutionaryModel) = "error: 'varianceparam' not implemented"

function Base.show(io::IO, obj::EvolutionaryModel)
    disp = modelname(obj) * "\n" * variancename(obj) * " = $(varianceparam(obj))"
    disp *= "\nroot mean: μ = $(obj.μ)\nroot variance: v = $(obj.v)"
    print(io, disp)
end

struct UnivariateBrownianMotion{T<:Real} <: EvolutionaryModel
    "variance rate"
    σ2::T
    "inverse variance (precision) rate"
    J::T
    "prior mean at the root"
    μ::T
    "prior variance at the root"
    v::T
    "g0: -log(2π σ2)/2"
    g0::T
end
modelname(m::UnivariateBrownianMotion) = "Univariate Brownian motion"
variancename(m::UnivariateBrownianMotion) = "evolutionary variance rate σ2"
varianceparam(m::UnivariateBrownianMotion) = m.σ2
function UnivariateBrownianMotion(σ2, μ, v=Float64(0.0))
    σ2 > 0.0 || error("evolutionary variance rate σ2 = $(σ2) must be positive")
    v >= 0.0 || error("root variance v=$v must be non-negative")
    UnivariateBrownianMotion{Float64}(σ2, 1/σ2, μ, v, -(log2π + log(σ2))/2)
end
function factor_treenode(m::UnivariateBrownianMotion, x0, x1, len, delta)
    J = SMatrix{2,2}(m.J,-m.J, -m.J,m.J)
    h0 = m.J * delta
    h = SVector{2}(h0, -h0)
    g = m.g0 + x1^2 * m.J/2
end

struct MvDiagBrownianMotion{T<:Real, V<:AbstractVector{T}} <: EvolutionaryModel
    "diagonal entries of the diagonal variance rate matrix"
    R::V
    "inverse variance rates (precision) on the diagonal inverse rate matrix"
    J::V
    "prior mean vector at the root"
    μ::V
    "prior variance vector at the root"
    v::V
    "g0: -log(det(2πR))/2"
    g0::T
end
modelname(m::MvDiagBrownianMotion) = "Multivariate Diagonal Brownian motion"
variancename(m::MvDiagBrownianMotion) = "evolutionary variance rates (diagonal values in the rate matrix): R"
varianceparam(m::MvDiagBrownianMotion) = m.R
function MvDiagBrownianMotion(R, μ, v=nothing)
    nvar = length(μ)
    length(R) == nvar || error("R and μ have different lengths")
    T = Float64 # promote_type(eltype(R), eltype(μ))
    SV = SVector{nvar, T}
    all(R .> 0.0) || error("evolutionary variance rates R = $R must all be positive")
    if isnothing(v)
        v = SV(zero(T) for _ in 1:nvar)
    else
        length(v) == nvar || error("v and μ have different lengths")
        all(v .>= 0.0) || error("root variances v=$v must all be non-negative")
    end
    R = SV(R)
    J = 1 ./R
    MvDiagBrownianMotion{T, SV}(R, J, SV(μ), SV(v), -(nvar * log2π + sum(log.(R)))/2)
end

struct MvFullBrownianMotion{T<:Real, P<:AbstractMatrix{T}, V<:AbstractVector{T}} <: EvolutionaryModel
    "variance rate matrix"
    R::P
    "inverse variance (precision) rate matrix"
    J::P
    "prior mean vector at the root"
    μ::V
    "prior variance/covariance matrix at the root"
    v::P
    "g0: -log(det(2πR))/2"
    g0::T
end
modelname(m::MvFullBrownianMotion) = "Multivariate Brownian motion"
variancename(m::MvFullBrownianMotion) = "evolutionary variance rate matrix: R"
varianceparam(m::MvFullBrownianMotion) = m.R
function MvFullBrownianMotion(R, μ, v=nothing)
    nvar = length(μ)
    T = Float64 # promote_type(eltype(R), eltype(μ))
    SV = SVector{nvar, T}
    SP = SMatrix{nvar,nvar,T}
    size(R) == (nvar,nvar) || error("R and μ have conflicting sizes")
    R = SP(R)
    J = inv(R)
    # to do: check that R is symmetric positive definite. perhaps store PDMat instead
    if isnothing(v)
        v = SP(zero(T) for i in 1:nvar for j in 1:nvar)
    else
        size(v) == (nvar,nvar) || error("v and μ have conflicting sizes")
        # to do: check that v is symmetric positive semi-definite
    end
    MvFullBrownianMotion{T, SP, SV}(R, J, SV(μ), SP(v), -(nvar * log2π + log(LinearAlgebra.det(R)))/2)
end

