################################################################
## Model Definitions
################################################################

## Abstract BM type
abstract type HomogeneousBrownianMotion{T} <: EvolutionaryModel{T} end

## Univariate BM
"""
    UnivariateBrownianMotion{T} <: HomogeneousBrownianMotion{T}

The univariate Brownian motion.
TODO
"""
struct UnivariateBrownianMotion{T<:Real} <: HomogeneousBrownianMotion{T}
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
UnivariateType(::Type{<:UnivariateBrownianMotion}) = IsUnivariate()
modelname(m::UnivariateBrownianMotion) = "Univariate Brownian motion"
variancename(m::UnivariateBrownianMotion) = "evolutionary variance rate σ2"
varianceparam(m::UnivariateBrownianMotion) = m.σ2
function UnivariateBrownianMotion(σ2::U1, μ::U2, v=nothing) where {U1<:Number, U2<:Number}
    T = promote_type(Float64, typeof(σ2), typeof(μ))
    v = getrootvarianceunivariate(T, v)
    σ2 > 0 || error("evolutionary variance rate σ2 = $(σ2) must be positive")
    UnivariateBrownianMotion{T}(σ2, 1/σ2, μ, v, -(log2π + log(σ2))/2)
end
function UnivariateBrownianMotion(σ2::Union{U1,V1}, μ::Union{U2,V2}, v=nothing) where {U1<:Number, U2<:Number, V1<:AbstractArray{U1}, V2<:AbstractArray{U2}}
    if (isnothing(v))
        (length(σ2) == 1 && length(μ) == 1) || error("UnivariateBrownianMotion can only take scalars as entries.")
        UnivariateBrownianMotion(σ2[1], μ[1])
    else 
        (length(σ2) == 1 && length(μ) == 1 && length(v) == 1) || error("UnivariateBrownianMotion can only take scalars as entries.")
        UnivariateBrownianMotion(σ2[1], μ[1], v[1])
    end
end
params(m::UnivariateBrownianMotion) = isrootfixed(m) ? (m.σ2, m.μ) : (m.σ2, m.μ, m.v)
params_optimize(m::UnivariateBrownianMotion) = [-2*m.g0 - log2π, m.μ] # log(σ2),μ
params_original(m::UnivariateBrownianMotion, logσ2μ::AbstractArray) = (exp(logσ2μ[1]), logσ2μ[2], m.v)

## Diagonal BM
struct MvDiagBrownianMotion{T<:Real, V<:AbstractVector{T}} <: HomogeneousBrownianMotion{T}
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
    numt = length(μ) # number of traits
    length(R) == numt || error("R and μ have different lengths")
    T = promote_type(Float64, eltype(R), eltype(μ))
    v = getrootvariancediagonal(T, numt, v)
    all(R .> 0.0) || error("evolutionary variance rates R = $R must all be positive")
    SV = SVector{numt, T}
    R = SV(R)
    J = 1 ./R
    MvDiagBrownianMotion{T, SV}(R, J, SV(μ), SV(v), -(numt * log2π + sum(log.(R)))/2)
end
params(m::MvDiagBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)
params_optimize(m::MvDiagBrownianMotion) = [log.(m.R)..., m.μ...]
params_original(m::MvDiagBrownianMotion, logRμ::AbstractArray) = (exp.(logRμ[1:dimension(m)]), logRμ[(dimension(m)+1):end], m.v)
rootpriorvariance(obj::MvDiagBrownianMotion) = LA.Diagonal(obj.v)

## Full BM
struct MvFullBrownianMotion{T<:Real, P1<:AbstractMatrix{T}, V<:AbstractVector{T}, P2<:AbstractMatrix{T}} <: HomogeneousBrownianMotion{T}
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
    numt = length(μ)
    T = promote_type(Float64, eltype(R), eltype(μ))
    v = getrootvariancemultivariate(T, numt, v)
    SV = SVector{numt, T}
    size(R) == (numt,numt)       || error("R and μ have conflicting sizes")
    LA.issymmetric(R) || error("R should be symmetric")
    R = PDMat(R)
    J = inv(R) # uses cholesky. fails if not symmetric positive definite
    MvFullBrownianMotion{T, typeof(R), SV, typeof(v)}(R, J, SV(μ), v, -(numt * log2π + LA.logdet(R))/2)
end
MvFullBrownianMotion(Uvec::AbstractVector, μ, v=nothing) = begin
    # R is specified by its upper cholesky factor that has been vectorized
    # todo: this can be absorbed into the original constructor above
    numt = length(μ)
    div(numt*(1+numt),2) == length(Uvec) || error("Uvec and μ have conflicting sizes")
    R = zeros(numt, numt)
    for (k, (i,j)) in enumerate(((i,j) for i in 1:numt for j in i:numt))
        R[i,j] = Uvec[k]
    end
    R = R' * R
    MvFullBrownianMotion(R, μ, v)
end
params(m::MvFullBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)
params_optimize(m::MvFullBrownianMotion) = begin
    U = LA.cholesky(m.R).U
    # todo: fix below to allow for non-positive covariances
    [log.(U[i,j] for d in dimension(m) for i in 1:d for j in i:d)..., m.μ...]
end
params_original(m::MvFullBrownianMotion, logRμ::AbstractArray) = begin
    numut = (x -> div(x*(1+x),2))(dimension(m)) # no. of ◹ elements in m.R
    # todo: fix below to allow for non-positive covariances
    (exp.(logRμ[1:numut]), logRμ[(numut+1):end], m.v)
end

################################################################
## factor_treeedge
################################################################

factor_treeedge(m::HomogeneousBrownianMotion, edge::PN.Edge) = factor_treeedge(m, edge.length)

function factor_treeedge(m::UnivariateBrownianMotion{T}, t::Real) where T
    j = T(m.J / t)
    J = LA.Symmetric(SMatrix{2,2}(j,-j, -j,j))
    h = SVector{2,T}(zero(T), zero(T))
    g = m.g0 - dimension(m) * log(t)/2
    return(h,J,g)
end
function factor_treeedge(m::MvDiagBrownianMotion{T,V}, t::Real) where {T,V}
    numt = dimension(m); ntot = numt * 2
    j = m.J ./ T(t) # diagonal elements
    # J = [diag(j) -diag(j); -diag(j) diag(j)]
    gen = ((u,tu,v,tv) for u in 1:2 for tu in 1:numt for v in 1:2 for tv in 1:numt)
    Juv = (u,tu,v,tv) -> (tu==tv ? (u==v ? j[tu] : -j[tu]) : 0)
    J = LA.Symmetric(SMatrix{ntot,ntot}(Juv(x...) for x in gen))
    h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
    g = m.g0 - numt * log(t)/2
    return(h,J,g)
end
function factor_treeedge(m::MvFullBrownianMotion{T,P1,V,P2}, t::Real) where {T,P1,V,P2}
    numt = dimension(m); ntot = numt * 2
    j = m.J ./ T(t)
    # J = [j -j; -j j]
    gen = ((u,tu,v,tv) for u in 1:2 for tu in 1:numt for v in 1:2 for tv in 1:numt)
    Juv = (u,tu,v,tv) -> (u==v ? j[tu,tv] : -j[tu,tv])
    J = LA.Symmetric(SMatrix{ntot,ntot}(Juv(x...) for x in gen))
    h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
    g = m.g0 - numt * log(t)/2
    return(h,J,g)
end

################################################################
## factor_hybridnode
################################################################

factor_hybridnode(m::HomogeneousBrownianMotion, pae::AbstractVector{PN.Edge}) = 
factor_hybridnode(m, [e.length for e in pae], [p.gamma for p in pae])

factor_tree_degeneratehybrid(m::HomogeneousBrownianMotion, pae::AbstractVector{PN.Edge}, che::PN.Edge) =
factor_tree_degeneratehybrid(m, che.length, [p.gamma for p in pae])

function factor_hybridnode(m::HomogeneousBrownianMotion{T}, t::AbstractVector, γ::AbstractVector) where T
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
function factor_tree_degeneratehybrid(m::MvDiagBrownianMotion{T,V}, t0::Real, γ::AbstractVector) where {T,V}
    j = m.J ./ T(t0) # diagonal elements. Dj = diag(j)
    nparents = length(γ); nn = 1 + nparents
    numt = dimension(m); ntot = nn * numt
    # J = [Dj -γ1Dj -γ2Dj; -γ1Dj γ1γ1Dj γ1γ2Dj; -γ2Dj γ1γ2Dj γ2γ2Dj]
    gen = ((u,tu,v,tv) for u in 0:nparents for tu in 1:numt for v in 0:nparents for tv in 1:numt)
    Juv = (u,tu,v,tv) -> (tu==tv ?
            (u==0 ? (v==0 ? j[tu] : -γ[v] * j[tu]) :
                    (v==0 ? -γ[u] * j[tu] : γ[u] * γ[v] * j[tu])) : zero(T))
    J = LA.Symmetric(SMatrix{ntot,ntot, T}(Juv(x...) for x in gen))
    h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
    g = m.g0 - numt * log(t0)/2
    return(h,J,g)
end
function factor_tree_degeneratehybrid(m::MvFullBrownianMotion{T,P1,V,P2}, t0::Real, γ::AbstractVector) where {T,P1,V,P2}
    j = m.J ./ T(t0)
    nparents = length(γ); nn = 1 + nparents
    numt = dimension(m); ntot = nn * numt
    # J = [j -γ1j -γ2j; -γ1j γ1γ1j γ1γ2j; -γ2j γ1γ2j γ2γ2j]
    gen = ((u,tu,v,tv) for u in 0:nparents for tu in 1:numt for v in 0:nparents for tv in 1:numt)
    Juv = (u,tu,v,tv) -> (u==0 ? (v==0 ? j[tu,tv] : -γ[v] * j[tu,tv]) :
                                 (v==0 ? -γ[u] * j[tu,tv] : γ[u] * γ[v] * j[tu,tv]))
    J = LA.Symmetric(SMatrix{ntot,ntot, T}(Juv(x...) for x in gen))
    h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
    g = m.g0 - numt * log(t0)/2
    return(h,J,g)
end