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

struct MvFullBrownianMotion{T<:Real, P1<:AbstractMatrix{T}, V<:AbstractVector{T}, P2<:AbstractMatrix{T}} <: EvolutionaryModel
    "variance rate matrix"
    R::P1
    "inverse variance (precision) rate matrix"
    J::P1
    "prior mean vector at the root"
    μ::V
    "prior variance/covariance matrix at the root"
    v::P2
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
    size(R) == (nvar,nvar) || error("R and μ have conflicting sizes")
    R = PDMat(R)
    J = inv(R) # uses cholesky. fails if not symmetric positive definite
    if isnothing(v)
        v = LinearAlgebra.Symmetric(SMatrix{nvar,nvar,T}(zero(T) for _ in 1:(nvar*nvar)))
    else
        size(v) == (nvar,nvar) || error("v and μ have conflicting sizes")
        v = LinearAlgebra.Symmetric(SMatrix{nvar,nvar,T}(v))
        # to do: check that v is symmetric (Symmetric doesn't check) positive semi-definite
    end
    MvFullBrownianMotion{T, typeof(R), SV, typeof(v)}(R, J, SV(μ), v, -(nvar * log2π + LinearAlgebra.logdet(R))/2)
end

"""
    factor_treeedge(evolutionarymodel, edge_length)

Canonical parameters `h,J,g` of factor ϕ(X0,X1) from the given evolutionary model
along one edge, where X0 is the state of the child node and X1 the state of the
parent node. In `h` and `J`, the first p coordinates are for the child and the
last p for the parent, where p is the number of traits (determined by the model).

Under the most general linear Gaussian model, X0 given X1 is Gaussian with
conditional mean q X1 + ω and conditional variance independent of X1.

Under a Brownian motion, se have q=I, ω=0, and conditional variance tR
where R is the model's variance rate.
"""
function factor_treeedge(m::UnivariateBrownianMotion, t::Real)
    j = m.J / t
    J = LinearAlgebra.Symmetric(SMatrix{2,2}(j,-j, -j,j))
    # μ = SVector{2, Float64}(0.0, 0.0)
    h = SVector{2, Float64}(0.0, 0.0)
    g = m.g0 - dimension(m) * log(t)/2
    return(h,J,g)
end

"""
    factor_hybridnode(evolutionarymodel, ts::AbstractVector, γs)
    factor_tree_degeneratehybrid(model,  t0::Real,           γs)

Canonical parameters `h,J,g` of factor ϕ(X0, X1,X2,...) from the evolutionary model
for a hybrid node: where X0 is the state at the hybrid node and X1,X2,... the
states of the parent nodes.
It is assumed that the conditional mean is a simple weighted average:

``X0 = \\sum_k \\gamma_k Xk = q vec(X1,X2,...) + \\omega``

where q has one block for each parent, and each block is diagonal scalar:
``\\gamma_k I_p``.
More complex models could consider adding a shift ω.

If all the parent hybrid edges edges have length 0, then it is assumed that
the model gives a degenerate distribution, with 0 conditional variance.
More complex models could consider adding a hybrid conditional variance.

- The first form assumes that at least 1 parent edge length is positive,
  with conditional variance ``\\sum_k \\gamma_k^2 V_k`` where ``V_k`` is
  the conditional variance from the kth parent edge.
- The second form can be used in case all parent edges have 0 length,
  to integrate out the hybrid node state and the factor ϕ(X0, X1,X2,...)
  when X0 is its **child** state, along an edge of length `t0` between
  the hybrid node and its child. This second form is appropriate when
  this hybrid's child is a tree node, and `t0>0`.`

In `h` and `J`, the first p coordinates are for the hybrid (or its child) and
the last coordinates for the parents, in the same order in which
the edge lengths and γs are given.
"""
function factor_hybridnode(m::UnivariateBrownianMotion, t::AbstractVector, γ::AbstractVector)
    t0 = sum(γ.^2 .* t) # >0 if hybrid node is not degenerate
    factor_tree_degeneratehybrid(m, t0, γ)
end
function factor_tree_degeneratehybrid(m::UnivariateBrownianMotion, t0::Real, γ::AbstractVector)
    j = m.J / t0
    nparents = length(γ); nn = 1 + nparents
    γj = -γ .* j; pushfirst!(γj, j)
    J = LinearAlgebra.Symmetric(SMatrix{nn,nn, Float64}(x*y for x in γj for y in γj))
    # μ = SVector{nn, Float64}(0.0 for _ in 1:nn)
    h = SVector{nn, Float64}(0.0 for _ in 1:nn)
    g = m.g0 - dimension(m) * log(t0)/2
    return(h,J,g)
end

function factor_root(m::UnivariateBrownianMotion)
    m.v > 0.0 || error("fixed root: absorb the evidence instead please")
    # todo: for a fixed-root model (v=0, j=Inf), absorb the evidence instead
    j = 1/m.v # 0 with improper prior v=Inf, Inf with fixed root v=0
    g = (j == 0.0 || j == Inf ? 0.0 : -(log2π + log(m.v) + m.μ^2 * j)/2)
    # todo: is that corect?
    return(m.μ*j, j, g) # m.μ
end
