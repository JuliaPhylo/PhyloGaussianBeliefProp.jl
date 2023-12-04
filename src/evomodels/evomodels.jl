"""
    EvolutionaryModel{T}

Evolutionary model type,
with `T` the element type in all parameter vector and matrices.
Implemented models include the [`UnivariateBrownianMotion`](@ref).

An object of this type must contain at least the following elements:
* μ: the mean of the trait at the root.
* v: the variance of the trait at the root. Can be zero (fixed root) or infinite.

New evoloutionary models must implement the following interfaces:
```julia
params(obj::EvolutionaryModel)
params_optimize(obj::EvolutionaryModel)
params_original(obj::EvolutionaryModel, transformedparams::AbstractArray)
```
"""
abstract type EvolutionaryModel{T} end

# Trait for univariate / multivariate models
abstract type UnivariateType end
struct IsUnivariate <: UnivariateType end
struct IsMultivariate <: UnivariateType end
# Default to multivariate models
UnivariateType(::Type) = IsMultivariate()

# generic methods
modelname(obj::EvolutionaryModel) = string(typeof(obj))
variancename(obj::EvolutionaryModel) = "variance"
varianceparam(obj::EvolutionaryModel) = "error: 'varianceparam' not implemented"
# requires all models to have a field named μ
rootpriormeanvector(obj::T) where {T <: EvolutionaryModel} = rootpriormeanvector(UnivariateType(T), obj)
rootpriormeanvector(::IsMultivariate, obj) = obj.μ
rootpriormeanvector(::IsUnivariate, obj) = [obj.μ]
# requires all models to have a field named v
isrootfixed(obj::EvolutionaryModel) = all(obj.v .== 0)
rootpriorvariance(obj::EvolutionaryModel) = obj.v
rootpriorprecision(obj::EvolutionaryModel) = inv(rootpriorvariance(obj))

"""
    dimension(m::EvolutionaryModel)

Number of traits, e.g. 1 for univariate models.
"""
dimension(obj::EvolutionaryModel) = length(rootpriormeanvector(obj))
"""
    params(m::EvolutionaryModel)

Tuple of parameters, the same that can be used to construct the evolutionary model.
"""
params(d::EvolutionaryModel) # extends StatsAPI.params

function Base.show(io::IO, obj::EvolutionaryModel)
    disp = modelname(obj) * "\n" * variancename(obj) * " = $(varianceparam(obj))"
    disp *= "\nroot mean: μ = $(obj.μ)\nroot variance: v = $(obj.v)"
    print(io, disp)
end

"""
    branch_actualization(obj::EvolutionaryModel, edge::PN.Edge) 
    branch_displacement(obj::EvolutionaryModel, edge::PN.Edge)
    branch_precision(obj::EvolutionaryModel, edge::PN.Edge)

Under the most general linear Gaussian model, X₀ given X₁ is Gaussian with
conditional mean q X₁ + ω and conditional variance Σ independent of X₁.
`branch_precision` should return a matrix of symmetric type.

Under a Brownian motion, we have q=I, ω=0, and conditional variance tR
where R is the model's variance rate.
"""
function branch_actualization(obj::EvolutionaryModel, edge::PN.Edge)
    error("branch_actualization not implemented for type $(typeof(obj)).")
end
@doc (@doc branch_actualization) branch_displacement
function branch_displacement(obj::EvolutionaryModel, edge::PN.Edge)
    error("`branch_displacement` not implemented for type $(typeof(obj)).")
end
@doc (@doc branch_actualization) branch_precision
function branch_precision(obj::EvolutionaryModel, edge::PN.Edge)
    error("`branch_precision` not implemented for type $(typeof(obj)).")
end

"""
    factor_treeedge(evolutionarymodel, edge)

Canonical parameters `h,J,g` of factor ϕ(X0,X1) from the given evolutionary model
along one edge, where X₀ is the state of the child node and X₁ the state of the
parent node. In `h` and `J`, the first p coordinates are for the child and the
last p for the parent, where p is the number of traits (determined by the model).

Under the most general linear Gaussian model, X₀ given X₁ is Gaussian with
conditional mean q X₁ + ω and conditional variance Σ independent of X₁.
The generic fallback method uses functions 
[`branch_actualization`](@ref) for q, 
[`branch_displacement`](@ref) for ω, 
[`branch_precision`](@ref) for Σ⁻¹.

Under a Brownian motion, we have q=I, ω=0, and Σ=tR
where R is the model's variance rate ant t the length of the branch.
In that case, a specific (more efficient) method is implemented,
and the default fallback is not used.
"""
function factor_treeedge(m::EvolutionaryModel{T}, edge::PN.Edge) where T
    numt = dimension(m); ntot = numt * 2
    j = branch_precision(m, edge)  # bloc precision
    q = - branch_actualization(m, edge)
    ω = branch_displacement(m, edge)
    jq = j * q
    qjq = transpose(q) * jq
    # J = [j -jq; -q'j q'jq]
    gen = ((u,tu,v,tv) for u in 1:2 for tu in 1:numt for v in 1:2 for tv in 1:numt)
    Juv = (u,tu,v,tv) -> (u==0 ? (v==0 ? j[tu,tv] : jq[tu,tv]) :
                                 (v==0 ? jq[tv,tu] : qjq[tu,tv]))
    J = LA.Symmetric(SMatrix{ntot,ntot,T}(Juv(x...) for x in gen))
    qjomega = transpose(jq) * ω
    jomega = j * ω
    gen = ((u,tu) for u in 1:2 for tu in 1:numt)
    huv = (u,tu) -> (u==0 ? jomega[tu] : qjomega[tu])
    h =  SVector{ntot,T}(huv(x...) for x in gen)
    g = (- numt * log2π + LA.logdet(j) - LA.dot(ω, jomega)) / 2
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
function factor_hybridnode(m::EvolutionaryModel{T}, pae::AbstractVector{PN.Edge}) where T
    "error: 'factor_hybridnode' not implemented"
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

If the root variance is not invertible (e.g., fixed root),
this function fails and should never be called
(see `isrootfix`)
"""
factor_root(obj::T) where {T <: EvolutionaryModel} = factor_root(UnivariateType(T), obj)

function factor_root(::IsUnivariate, m::EvolutionaryModel{T}) where T
    j = T(1/m.v) # improper prior: j=0, v=Inf, factor ≡ 1: h,J,g all 0
    g = (j == 0.0 ? zero(T) : -(log2π + log(m.v) + m.μ^2 * j)/2)
    return(m.μ*j, j, g)
end
function factor_root(::IsMultivariate, m::EvolutionaryModel{T}) where T
    j = rootpriorprecision(m)
    μ = rootpriormeanvector(m)
    h = j * μ
    improper = any(diag(j) .== 0.0) # then assumes that *all* are 0
    g = (improper ? zero(T) : (-dimension(m) * log2π + LA.logdet(j) - LA.dot(m.μ, h))/2)
    return(h, j, g)
end
