################################################################
## Homogeneous OU Model
################################################################

"""
UnivariateOrnsteinUhlenbeck{T} <: EvolutionaryModel{T}

The univariate Ornstein-Uhlenbeck
TODO
"""
struct UnivariateOrnsteinUhlenbeck{T<:Real} <: EvolutionaryModel{T}
    "stationary variance rate"
    γ2::T
    "inverse stationary variance (precision) rate"
    J::T
    "selection strength"
    α::T
    "optimal value"
    θ::T
    "prior mean at the root"
    μ::T
    "prior variance at the root"
    v::T
    "g0: -log(2π γ2)/2"
    g0::T
end
UnivariateType(::Type{<:UnivariateOrnsteinUhlenbeck}) = IsUnivariate()
modelname(m::UnivariateOrnsteinUhlenbeck) = "homogeneous univariate Ornstein-Uhlenbeck"
variancename(m::UnivariateOrnsteinUhlenbeck) = "stationary evolutionary variance γ2"
varianceparam(m::UnivariateOrnsteinUhlenbeck) = m.γ2
notrootparamnames(m::UnivariateOrnsteinUhlenbeck) = (variancename(m), "selection strength α", "optimal value θ")
function UnivariateOrnsteinUhlenbeck(σ2::U1, α::U2, θ::U3, μ::U4, v=nothing) where {U1<:Number, U2<:Number, U3<:Number, U4<:Number}
    T = promote_type(Float64, typeof(σ2), typeof(α), typeof(θ), typeof(μ))
    v = getrootvarianceunivariate(T, v)
    σ2 > 0 || error("evolutionary variance rate σ2 = $(σ2) must be positive")
    α > 0 || error("selection strength α = $(α) must be positive")
    γ2 = σ2 / (2 * α)
    UnivariateOrnsteinUhlenbeck{T}(γ2, 1/γ2, α, θ, μ, v, -(log2π + log(γ2))/2)
end
params(m::UnivariateOrnsteinUhlenbeck) = isrootfixed(m) ? (m.γ2, m.α, m.θ, m.μ) : (m.γ2, m.α, m.θ, m.μ, m.v)
params_optimize(m::UnivariateOrnsteinUhlenbeck) = [-2*m.g0 - log2π, log(m.α), m.θ, m.μ]
params_original(m::UnivariateOrnsteinUhlenbeck, transparams::AbstractArray) = (exp(transparams[1]), exp(transparams[2]), transparams[3], transparams[4], m.v)

function branch_transition_qωjg(obj::UnivariateOrnsteinUhlenbeck, edge::PN.Edge)
    q = exp(-obj.α * edge.length)
    facvar = (1 - q^2)
    j = 1 / obj.γ2 / facvar
    ω = (1 - q) * obj.θ
    g = obj.g0 - log(facvar)/2
    return (reshape([q],1,1),[ω],reshape([j],1,1),g)
end
function branch_transition_qωv!(q::AbstractMatrix, obj::UnivariateOrnsteinUhlenbeck, edge::PN.Edge)
    actu = exp(-obj.α * edge.length)
    facvar = (1 - actu^2)
    v = obj.γ2 * facvar
    ω = (1 - actu) * obj.θ
    q[1,1] = actu
    return ([ω],reshape([v],1,1))
end
