################################################################
## Model Definitions
################################################################

## Abstract BM type
abstract type HomogeneousBrownianMotion{T} <: EvolutionaryModel{T} end

"""
    UnivariateBrownianMotion{T} <: HomogeneousBrownianMotion{T}

The univariate Brownian motion, homogeneous across the phylogeny, that is,
with the same variance rate `σ2` across all edges.
`μ` is the prior mean at the root.
`v` the prior variance at the root, 0 by default.
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
function UnivariateBrownianMotion(σ2::Number, μ::Number, v=nothing)
    T = promote_type(Float64, typeof(σ2), typeof(μ))
    v = getrootvarianceunivariate(T, v)
    σ2 > 0 || DomainError(σ2, "evolutionary variance rate σ2 must be positive")
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

"""
    MvDiagBrownianMotion{T,V} <: HomogeneousBrownianMotion{T}

The multivariate Brownian motion with diagonal variance rate matrix, that is,
traits with independent evolution. It is homogeneous across the phylogeny.
`R` is the variance rate (stored as a vector of type `V`),
`μ` is the prior mean at the root and
`v` the prior variance at the root, 0 by default (and both also of type `V`)
"""
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
    # SV = SVector{numt, T}
    # R = SV(R)
    J = 1 ./R
    # MvDiagBrownianMotion{T, SV}(R, J, SV(μ), SV(v), -(numt * log2π + sum(log.(R)))/2)
    MvDiagBrownianMotion{T, typeof(R)}(R, J, μ, v, -(numt * log2π + sum(log.(R)))/2)
end
params(m::MvDiagBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)
params_optimize(m::MvDiagBrownianMotion) = [log.(m.R)..., m.μ...]
params_original(m::MvDiagBrownianMotion, logRμ::AbstractArray) = (exp.(logRμ[1:dimension(m)]), logRμ[(dimension(m)+1):end], m.v)
rootpriorvariance(obj::MvDiagBrownianMotion) = LA.Diagonal(obj.v)

"""
    MvFullBrownianMotion{T,P1,V,P2} <: HomogeneousBrownianMotion{T}

The full multivariate Brownian motion. It is homogeneous across the phylogeny.
`R` is the variance rate (of matrix type `P1`),
`μ` is the prior mean at the root (of vector type `V`) and
`v` the prior variance at the root, 0 by default (of matrix type `P2`).
"""
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
function MvFullBrownianMotion(R::AbstractMatrix, μ, v=nothing)
    numt = length(μ)
    T = promote_type(Float64, eltype(R), eltype(μ))
    v = getrootvariancemultivariate(T, numt, v)
    # SV = SVector{numt, T}
    size(R) == (numt,numt)       || error("R and μ have conflicting sizes")
    LA.issymmetric(R) || error("R should be symmetric")
    # R = PDMat(R) # todo: discuss precision issues
    J = inv(R) # uses cholesky. fails if not symmetric positive definite
    # MvFullBrownianMotion{T, typeof(R), SV, typeof(v)}(R, J, SV(μ), v, branch_logdet_variance(numt, R))
    MvFullBrownianMotion{T, typeof(R), typeof(μ), typeof(v)}(R, J, μ, v, branch_logdet_variance(numt, R))
end
params(m::MvFullBrownianMotion) = isrootfixed(m) ? (m.R, m.μ) : (m.R, m.μ, m.v)
function params_optimize(m::MvFullBrownianMotion)
    #=
    Based on log-cholesky parametrization.
    See https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factors-of-covariance-matrices
    =#
    numt = dimension(m)
    U = LA.cholesky(m.R).U # upper cholesky factor of covariance matrix
    idx = LinearIndices(U)
    idx_diag = (idx[i,i] for i in 1:numt)
    idx_abovediag = (idx[i,j] for j in 2:numt for i in 1:(j-1))
    # log-transform the diagonal elements of U
    return [[log.(U[k]) for k in idx_diag]..., [U[k] for k in idx_abovediag]..., m.μ...]
end
function params_original(m::MvFullBrownianMotion{T}, Uμ::AbstractArray) where T
    numt = dimension(m)
    U = zeros(T, numt, numt) # upper Cholesky factor for R
    k = 1
    for i in 1:numt
        U[i,i] = exp(Uμ[k])
        k += 1
    end
    for j in 2:numt
        for i in 1:(j-1)
            U[i,j] = Uμ[k]
            k += 1
        end
    end
    R = PDMat(LA.Symmetric(transpose(U)*U))
    return (R, Uμ[k:end], m.v)
end
# function params_optimize(m::MvFullBrownianMotion)
#     #=
#     Based on spherical parametrization (for correlation matrices), aka LKJ transformation by
#     sampling spheres.
#     See:
#         - Unconstrained parametrizations for variance-covariance matrices
#         (https://doi.org/10.1007/BF00140873), Section 2
#         - The Spherical Parametrization for Correlation Matrices and its Computational
#         Advantages (https://doi.org/10.1007/s10614-023-10467-3), Section 2
#         - Efficient Bayesian inference of general Gaussian models on large phylogenetic
#         trees (https://doi.org/10.1214/20-AOAS1419), Appendix C.1.1
#     =#
#     numt = dimension(m)
#     σ = sqrt.(LA.diag(m.R)) # vector of standard deviations
#     # upper cholesky factor of correlation matrix
#     ρchol = LA.cholesky(LA.symmetric(LA.Diagonal(1 ./ σ)*m.R*LA.Diagonal(1 ./ σ))).U
#     # store coordinates in unconstrained space that coordinates in ρchol are mapped to
#     ρcholtrans = zeros(binomial(numt, 2))
#     k = 1
#     for i in 2:numt
#         #= multiply by h2b to get from ρchol[i,j] ∈ S^{pos}_n (half Euclidean sphere) to
#         x ∈ B^{inf}_{n-1} (infinite norm ball) =#
#         inf2e = 1
#         for j in 1:(i-1)
#             x = ρchol[j,i] / inf2e # coordinate in B^{inf}_{n-1}
#             ρcholtrans[k] = atanh(x) # coordinate in R^{n-1}
#             inf2e *= sqrt(1 - x^2)
#             k += 1
#         end
#     end
#     return [ρcholtrans..., log.(σ)..., m.μ...]
# end
# function params_original(m::MvFullBrownianMotion{T}, ρσμ::AbstractArray) where T
#     numt = dimension(m)
#     C = zeros(T, numt, numt); C[1,1] = 1
#     numρ = binomial(numt, 2)
#     ρcholtrans = ρσμ[1:numρ] # ρ (i.e. correlation) related parameters
#     σ = exp.(ρσμ[numρ+1:numρ+numt]) # standard deviations of R
#     k = 1
#     for i in 2:numt
#         #= multiply by inf2e to get from tanh(ρcholtrans[k]) ∈ B^{inf}_{n-1} (infinite norm
#         ball) to B_{n-1} (Euclidean ball) =#
#         inf2e = 1
#         norm = 0
#         for j in 1:(i-1)
#             x = tanh(ρcholtrans[k]) # R^{n-1} → B^{inf}_{n-1}
#             l = x * inf2e # B^{inf}_{n-1} → B_{n-1}
#             C[j,i] = l
#             norm += l^2
#             inf2e *= sqrt(1-x^2)
#             k += 1
#         end
#         C[i,i] = sqrt(1 - norm) # B_{n-1} → S^{pos}_n
#     end
#     R = LA.Diagonal(σ)*transpose(C)*C*LA.Diagonal(σ)
#     return (LA.Symmetric(R), ρσμ[numρ+numt+1:end], m.v)
# end

################################################################
## factor_treeedge
################################################################

factor_treeedge(m::HomogeneousBrownianMotion, edge::PN.Edge) = factor_treeedge(m, edge.length)

function factor_treeedge(m::UnivariateBrownianMotion{T}, t::Real) where T
    if iszero(t) # degenerate tree-edge factor, todo: set threshold
        R = T[-1/sqrt(2); 1/sqrt(2) ;;]
        g = T(-0.5*log(2))
        c = T[0]
        return(R,c,g)
    else
        j = T(m.J / t)
        # todo: discuss not enforcing symmetry (e.g. J = LA.Symmetric(SMatrix...))
        # J = SMatrix{2,2,T}(j,-j,-j,j)
        J = T[j -j; -j j]
        # h = SVector{2,T}(0,0)
        h = zeros(T,2)
        g = m.g0 - dimension(m) * log(t)/2
        return(h,J,g)
    end
end
function factor_treeedge(m::MvDiagBrownianMotion{T,V}, t::Real) where {T,V}
    numt = dimension(m)
    if iszero(t)
        R = kron(T[-1/sqrt(2); 1/sqrt(2) ;;], LA.I(numt))
        g = T(-numt*log(2)/2)
        c = zeros(T, numt)
        return(R,c,g)
    else
        ntot = numt * 2
        j = m.J ./ T(t) # diagonal elements
        # J = [diag(j) -diag(j); -diag(j) diag(j)]
        gen = ((u,tu,v,tv) for u in 1:2 for tu in 1:numt for v in 1:2 for tv in 1:numt)
        Juv = (u,tu,v,tv) -> (tu==tv ? (u==v ? j[tu] : -j[tu]) : 0)
        # J = LA.Symmetric(SMatrix{ntot,ntot}(Juv(x...) for x in gen))
        J = LA.Symmetric(reshape(collect(T,Juv(x...) for x in gen), (ntot,ntot)))
        # h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
        h = zeros(T,ntot)
        g = m.g0 - numt * log(t)/2
        return(h,J,g)
    end
end
function factor_treeedge(m::MvFullBrownianMotion{T,P1,V,P2}, t::Real) where {T,P1,V,P2}
    numt = dimension(m)
    if iszero(t)
        R = kron(T[-1/sqrt(2); 1/sqrt(2) ;;], LA.I(numt))
        g = T(-numt*log(2)/2)
        c = zeros(T, numt)
        return(R,c,g)
    else
        ntot = numt * 2
        j = m.J ./ T(t)
        # J = [j -j; -j j]
        gen = ((u,tu,v,tv) for u in 1:2 for tu in 1:numt for v in 1:2 for tv in 1:numt)
        Juv = (u,tu,v,tv) -> (u==v ? j[tu,tv] : -j[tu,tv])
        # J = LA.Symmetric(SMatrix{ntot,ntot}(Juv(x...) for x in gen))
        J = LA.Symmetric(reshape(collect(T,Juv(x...) for x in gen), (ntot,ntot)))
        # h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
        h = zeros(T,ntot)
        g = m.g0 - numt * log(t)/2
        return(h,J,g)
    end
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
    numt = dimension(m)
    if iszero(t0) # degenerate hybrid factor
        R = kron(T[-1; γ ;;], LA.I(numt))
        LA.ldiv!(sqrt(sum(γ.^2) + 1), R) # normalize
        g = T(-numt*log(sum(γ.^2) + 1)/2)
        c = zeros(T, numt)
        return(R,c,g)
    else
        return factor_tree_degeneratehybrid(m, t0, γ)
    end
end
function factor_tree_degeneratehybrid(m::UnivariateBrownianMotion{T}, t0::Real, γ::AbstractVector) where T
    j = T(m.J / t0)
    nparents = length(γ); nn = 1 + nparents
    # modifies γ in place below, to get longer vector: [1 -γ]
    γ .= -γ; pushfirst!(γ, 1)
    # γ .= -γ; pushfirst!(γ, one(eltype(γ)))
    # todo: discuss not enforcing symmetry (e.g. J = LA.Symmetric(SMatrix...))
    # J = SMatrix{nn,nn,T}(j*x*y for x in γ, y in γ)
    J = collect(T,j*x*y for x in γ, y in γ)
    # h = SVector{nn,T}(0 for _ in 1:nn)
    h = zeros(T,nn)
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
    # J = LA.Symmetric(SMatrix{ntot,ntot, T}(Juv(x...) for x in gen))
    J = LA.Symmetric(reshape(collect(T,Juv(x...) for x in gen), (ntot,ntot)))
    # h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
    h = zeros(T,ntot)
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
    # J = LA.Symmetric(SMatrix{ntot,ntot, T}(Juv(x...) for x in gen))
    J = LA.Symmetric(reshape(collect(T,Juv(x...) for x in gen), (ntot,ntot)))
    # h = SVector{ntot,T}(zero(T) for _ in 1:ntot)
    h = zeros(T,ntot)
    g = m.g0 - numt * log(t0)/2
    return(h,J,g)
end