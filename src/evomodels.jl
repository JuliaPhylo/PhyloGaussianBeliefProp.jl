abstract type EvolutionaryModel{T} end

# generic methods
modelname(obj::EvolutionaryModel) = string(typeof(obj))
variancename(obj::EvolutionaryModel) = "variance"
varianceparam(obj::EvolutionaryModel) = "error: 'varianceparam' not implemented"
# requires all models to have a field named μ
rootpriormeanvector(obj::EvolutionaryModel) = obj.μ
dimension(obj::EvolutionaryModel) = length(rootpriormeanvector(obj))

"""
    params(d::EvolutionaryModel)

Tuple of parameters, the same that can be used to construct the evolutionary model.
"""
params(d::EvolutionaryModel) # extends StatsAPI.params

function Base.show(io::IO, obj::EvolutionaryModel)
    disp = modelname(obj) * "\n" * variancename(obj) * " = $(varianceparam(obj))"
    disp *= "\nroot mean: μ = $(obj.μ)\nroot variance: v = $(obj.v)"
    print(io, disp)
end

struct UnivariateBrownianMotion{T<:Real} <: EvolutionaryModel{T}
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
rootpriormeanvector(m::UnivariateBrownianMotion) = [m.μ]
isrootfixed(m::UnivariateBrownianMotion) = m.v == 0
function UnivariateBrownianMotion(σ2, μ, v=nothing)
    T = promote_type(Float64, typeof(σ2), typeof(μ))
    if isnothing(v) v = zero(T); end
    σ2 > 0 || error("evolutionary variance rate σ2 = $(σ2) must be positive")
    v >= 0 || error("root variance v=$v must be non-negative")
    UnivariateBrownianMotion{T}(σ2, 1/σ2, μ, v, -(log2π + log(σ2))/2)
end
params(m::UnivariateBrownianMotion) = isrootfixed(m) ? (m.σ2, m.μ) : (m.σ2, m.μ, m.v)
params_optimize(m::UnivariateBrownianMotion) = [-2*m.g0 - log2π, m.μ] # log(σ2),μ
params_original(m::UnivariateBrownianMotion, logσ2, μ) = (exp(logσ2), μ, m.v)

struct MvDiagBrownianMotion{T<:Real, V<:AbstractVector{T}} <: EvolutionaryModel{T}
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
isrootfixed(m::MvDiagBrownianMotion) = all(m.v .== 0)
function MvDiagBrownianMotion(R, μ, v=nothing)
    numt = length(μ) # number of traits
    length(R) == numt || error("R and μ have different lengths")
    T = promote_type(Float64, eltype(R), eltype(μ))
    SV = SVector{numt, T}
    all(R .> 0.0) || error("evolutionary variance rates R = $R must all be positive")
    if isnothing(v)
        v = SV(zero(T) for _ in 1:numt)
    else
        length(v) == numt || error("v and μ have different lengths")
        all(v .>= 0.0) || error("root variances v=$v must all be non-negative")
    end
    R = SV(R)
    J = 1 ./R
    MvDiagBrownianMotion{T, SV}(R, J, SV(μ), SV(v), -(numt * log2π + sum(log.(R)))/2)
end
params(m::MvDiagBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)
params_optimize(m::MvDiagBrownianMotion) = [log.(m.R), m.μ]
params_original(m::MvDiagBrownianMotion, logR, μ) = (exp.(logR), μ, m.v)

struct MvFullBrownianMotion{T<:Real, P1<:AbstractMatrix{T}, V<:AbstractVector{T}, P2<:AbstractMatrix{T}} <: EvolutionaryModel{T}
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
isrootfixed(m::MvFullBrownianMotion) = all(m.v .== 0)
function MvFullBrownianMotion(R, μ, v=nothing)
    numt = length(μ)
    T = promote_type(Float64, eltype(R), eltype(μ))
    SV = SVector{numt, T}
    size(R) == (numt,numt)       || error("R and μ have conflicting sizes")
    LA.issymmetric(R) || error("R should be symmetric")
    R = PDMat(R)
    J = inv(R) # uses cholesky. fails if not symmetric positive definite
    if isnothing(v)
        v = LA.Symmetric(SMatrix{numt,numt,T}(zero(T) for _ in 1:(numt*numt)))
    else
        size(v) == (numt,numt)       || error("v and μ have conflicting sizes")
        LA.issymmetric(v) || error("v should be symmetric")
        v = LA.Symmetric(SMatrix{numt,numt,T}(v))
        λ = LA.eigvals(v)
        all(λ .>= 0)                 || error("v is not positive semi-definite")
    end
    MvFullBrownianMotion{T, typeof(R), SV, typeof(v)}(R, J, SV(μ), v, -(numt * log2π + LA.logdet(R))/2)
end
params(m::MvFullBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)

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
function factor_treeedge(m::UnivariateBrownianMotion{T}, t::Real) where T
    j = T(m.J / t)
    J = LA.Symmetric(SMatrix{2,2}(j,-j, -j,j))
    h = SVector{2,T}(zero(T), zero(T))
    g = m.g0 - dimension(m) * log(t)/2
    return(h,J,g)
end

"""
    factor_hybridnode(evolutionarymodel, ts::AbstractVector, γs)
    factor_tree_degeneratehybrid(model,  t0::Real,           γs)

Canonical parameters `h,J,g` of factor ϕ(X0, X1,X2,...) from the evolutionary model
for a hybrid node: where X0 is the state at the hybrid node and X1,X2,... the
states of the parent nodes.
**Warning:** `γs` is modified in placed, changed to `[1 -γs]`.

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
function factor_hybridnode(m::UnivariateBrownianMotion{T}, t::AbstractVector, γ::AbstractVector) where T
    t0 = T(sum(γ.^2 .* t)) # >0 if hybrid node is not degenerate
    factor_tree_degeneratehybrid(m, t0, γ)
end
function factor_tree_degeneratehybrid(m::UnivariateBrownianMotion{T}, t0::Real, γ::AbstractVector) where T
    j = T(m.J / t0)
    nparents = length(γ); nn = 1 + nparents
    # modifies γ in place below, to get longer vector: [1 -γ]
    γ .= -γ; pushfirst!(γ, one(eltype(γ)))
    J = LA.Symmetric(SMatrix{nn,nn, T}(j*x*y for x in γ, y in γ))
    h = SVector{nn,T}(zero(T) for _ in 1:nn)
    g = m.g0 - dimension(m) * log(t0)/2
    return(h,J,g)
end

"""
    factor_root(m::EvolutionaryModel)

Canonical parameters `h,J,g` of the prior density at the root, from model `m`.
Assumes that `isrootfixed(m)` returns `false` (in which case the root value
should be absorbed as evidence and the root removed from scope).
More strongly, the root variance is assumed to be invertible, in particular,
traits are all non-fixed at the root.

The prior is improper if the prior variance is infinite. In this case this prior
is not a distribution (total probability ≠ 1) but is taken as the constant
function 1, which corresponds to h,J,g all 0 (and an irrelevant mean).
"""
function factor_root(m::UnivariateBrownianMotion{T}) where T
    j = T(1/m.v) # improper prior: j=0, v=Inf, factor ≡ 1: h,J,g all 0
    g = (j == 0.0 ? zero(T) : -(log2π + log(m.v) + m.μ^2 * j)/2)
    return(m.μ*j, j, g) # m.μ
end
