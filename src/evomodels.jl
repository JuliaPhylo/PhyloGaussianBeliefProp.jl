abstract type EvolutionaryModel end

# generic methods
modelname(obj::EvolutionaryModel) = string(typeof(obj))
variancename(obj::EvolutionaryModel) = "variance"
varianceparam(obj::EvolutionaryModel) = "error: 'varianceparam' not implemented"
# requires all models to have a field named μ
dimension(obj::EvolutionaryModel) = length(obj.μ)

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

# factor for [X_child,X_parent] from a
# univariate Brownian motion along an edge of length t, q=I, ω=0
function factor_treeedge(m::UnivariateBrownianMotion, t::Real)
    j = m.J/t
    J = SMatrix{2,2}(j,-j, -j,j)
    μ = SVector{2, Float64}(0.0, 0.0)
    h = SVector{2, Float64}(0.0, 0.0)
    g = m.g0 - dimension(m) * log(t)/2
    return(μ,h,J,g)
end

# factor for [X_hybrid,X_parents...] if hybrid node has
# at least 1 parent edge of positive length
function factor_hybridnode(m::UnivariateBrownianMotion, t::AbstractVector, γ::AbstractVector)
    t0 = sum(γ.^2 .* t) # >0 if hybrid node is not degenerate
    factor_tree_degeneratehybrid(m, t0, γ)
end
function factor_tree_degeneratehybrid(m::UnivariateBrownianMotion, t0::Real, γ::AbstractVector)
    j = m.J/t0
    nparents = length(γ); nn = 1 + nparents
    γj = -γ .* j; pushfirst!(γj, j)
    J = SMatrix{nn,nn, Float64}(x*y for x in γj for y in γj)
    μ = SVector{nn, Float64}(0.0 for _ in 1:nn)
    h = SVector{nn, Float64}(0.0 for _ in 1:nn)
    g = m.g0 - dimension(m) * log(t0)/2
    return(μ,h,J,g)
end

function factor_root(m::UnivariateBrownianMotion)
    j = 1/m.v # 0 with improper prior v=Inf, Inf with fixed root v=0
    g = (j == 0.0 || j == Inf ? 0.0 : -(log2π + log(m.v) + m.μ^2 * j)/2)
    # todo: how to handle h = m.μ*j under a fixed-root model v=0, j=Inf?
    return(m.μ, m.μ*j, j, g)
end

# absorb evidence at leaf: tbl[v][row] for variable v
function factor_leaf(m::EvolutionaryModel, t::Real, row::Integer, tbl)
    # fixit
    factor_treeedge(m, t)
end
