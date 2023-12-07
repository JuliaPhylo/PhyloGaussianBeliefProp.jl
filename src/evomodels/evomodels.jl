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

################################################################
## factor_treeedge
################################################################

"""
    branch_actualization(obj::EvolutionaryModel, edge::PN.Edge) 
    branch_displacement(obj::EvolutionaryModel, edge::PN.Edge)
    branch_precision(obj::EvolutionaryModel, edge::PN.Edge)
    branch_variance(obj::EvolutionaryModel, edge::PN.Edge)

Under the most general linear Gaussian model, X₀ given X₁ is Gaussian with
conditional mean q X₁ + ω and conditional variance Σ independent of X₁.
`branch_precision` and `branch_variance`should return a matrix of symmetric type.
`branch_variance` defaults to the inverse of `branch_precision`.

Under a Brownian motion, we have q=I, ω=0, and conditional variance tR
where R is the model's variance rate.
"""
function branch_actualization(obj::EvolutionaryModel{T}, edge::PN.Edge) where T
    p = dimension(obj)
    M = Matrix{T}(undef, p, p)
    branch_actualization!(M, obj, edge)
end
function branch_actualization!(M::AbstractMatrix, obj::EvolutionaryModel, edge::PN.Edge)
    error("branch_actualization! not implemented for type $(typeof(obj)).")
end
@doc (@doc branch_actualization) branch_displacement
function branch_displacement(obj::EvolutionaryModel, edge::PN.Edge)
    error("`branch_displacement` not implemented for type $(typeof(obj)).")
end
@doc (@doc branch_actualization) branch_precision
function branch_precision(obj::EvolutionaryModel, edge::PN.Edge)
    error("`branch_precision` not implemented for type $(typeof(obj)).")
end
@doc (@doc branch_actualization) branch_variance
function branch_variance(obj::EvolutionaryModel, edge::PN.Edge)
    return inv(branch_precision(obj, edge))
end
function branch_transition_qωj(obj::EvolutionaryModel{T}, edge::PN.Edge) where T
    j = branch_precision(obj, edge)
    ω = branch_displacement(obj, edge)
    q = branch_actualization(obj, edge)
    return (q,ω,j)
end
function branch_transition_qωv!(q::AbstractMatrix, obj::EvolutionaryModel{T}, edge::PN.Edge) where T
    v = branch_variance(obj, edge)
    ω = branch_displacement(obj, edge)
    branch_actualization!(q,obj, edge)
    return (ω,v)
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
    (q,ω,j) = branch_transition_qωj(m, edge)
    factor_treeedge(q, ω, j, 1, dimension(m))
end

# factor from precision, actualization, displacement
function factor_treeedge(q::AbstractMatrix{T}, ω::AbstractVector{T}, j::AbstractMatrix{T},
                         nparents::Int, ntraits::Int) where T
    nn = 1 + nparents; ntot = ntraits * nn
    jq = - j * q
    qjq = - transpose(q) * jq
    # J = [j -jq; -q'j q'jq]
    gen = ((u,tu,v,tv) for u in 0:nparents for tu in 1:ntraits for v in 0:nparents for tv in 1:ntraits)
    Juv = (u,tu,v,tv) -> (u==0 ? (v==0 ? j[tu,tv] : jq[tu,tv]) :
                                 (v==0 ? jq[tv,tu] : qjq[tu,tv]))
    J = LA.Symmetric(SMatrix{ntot,ntot,T}(Juv(x...) for x in gen))
    qjomega = transpose(jq) * ω
    jomega = j * ω
    gen = ((u,tu) for u in 0:nparents for tu in 1:ntraits)
    huv = (u,tu) -> (u==0 ? jomega[tu] : qjomega[tu])
    h =  SVector{ntot,T}(huv(x...) for x in gen)
    g = (- ntraits * log2π + LA.logdet(j) - LA.dot(ω, jomega)) / 2
    return(h,J,g)
end

################################################################
## factor_hybridnode
################################################################

"""
    hybdridnode_displacement(obj::EvolutionaryModel, parentedges::AbstractVector{PN.Edge})
    hybdridnode_precision(obj::EvolutionaryModel, parentedges::AbstractVector{PN.Edge})
    hybdridnode_variance(obj::EvolutionaryModel, parentedges::AbstractVector{PN.Edge})

Under the most general weighted average Gaussian model, X₀ given its parents X₁, X₂, ...
is Gaussian with conditional mean the weighted average of the parents
plus a displacement vecror ω and conditional variance Σ independent of X₁, X₂, ... .
`hybdridnode_variance` and `hybdridnode_precision`should return a matrix of symmetric type.
`hybdridnode_precision` defaults to the inverse of `hybdridnode_variance`
"""
function hybdridnode_displacement(obj::EvolutionaryModel, parentedges::AbstractVector{PN.Edge})
    error("hybdridnode_displacement not implemented for type $(typeof(obj)).")
end
@doc (@doc hybdridnode_displacement) hybdridnode_variance
function hybdridnode_variance(obj::EvolutionaryModel, parentedges::AbstractVector{PN.Edge})
    error("`hybdridnode_variance` not implemented for type $(typeof(obj)).")
end
@doc (@doc hybdridnode_displacement) hybdridnode_precision
function hybdridnode_precision(obj::EvolutionaryModel, parentedges::AbstractVector{PN.Edge})
    return inv(hybdridnode_variance(obj, parentedges))
end

"""
    factor_hybridnode(evolutionarymodel, ts::AbstractVector, γs)
    factor_tree_degeneratehybrid(model,  t0::Real,           γs)

Canonical parameters `h,J,g` of factor ϕ(X₀, X₁, X₂, ...) from the evolutionary model
for a hybrid node: where X₀ is the state at the hybrid node and X₁, X₂, ... the
states of the parent nodes.
**Warning:** `γs` is modified in placed, changed to `[1 -γs]`.

It is assumed that the conditional mean is a simple weighted average:

``E[X_0 | X_1, X_2, ...] = \\sum_k \\gamma_k X_k = q vec(X_1,X_2,...) + \\omega``

where q has one block for each parent, and each block is diagonal scalar:
``\\gamma_k I_p``.
More complex models could consider adding a shift ω to the conditional mean.

If all the parent hybrid edges edges have length 0, then it is assumed that
the model gives a degenerate distribution, with 0 conditional variance.
More complex models could consider adding a hybrid conditional variance Σ.

- The first form assumes that at least 1 parent edge length is positive,
  with conditional variance ``\\sum_k \\gamma_k^2 V_k`` where ``V_k`` is
  the conditional variance from the kth parent edge.
- The second form can be used in case all parent edges have 0 length,
  to integrate out the hybrid node state and the factor ϕ(X₀, X₁, X₂, ...)
  when X₀ is its **child** state, along an edge of length `t0` between
  the hybrid node and its child. This second form is appropriate when
  this hybrid's child is a tree node, and `t0>0`.`

In `h` and `J`, the first p coordinates are for the hybrid (or its child) and
the last coordinates for the parents, in the same order in which
the edge lengths and γs are given.
"""
function factor_hybridnode(m::EvolutionaryModel{T}, pae::AbstractVector{PN.Edge}) where T
    ntraits = dimension(m)
    nparents = length(pae)
    v = hybridnode_variance(m, pae) # extra node variance
    ω = hybdridnode_displacement(m, pae) # extra node displacement
    q = Matrix{T}(undef, ntraits, nparents * ntraits) # init actualisation
    for (k, edge) in enumerate(pae)
        (ωe , ve) = branch_transition_qωv!(view(q, :, ((k-1) * ntraits + 1):(k*ntraits)), m, edge)
        v .+= edge.gamma^2 .* ve
        ω .+= edge.gamma .* ωe
    end
    j = inv(v) # bloc variance
    factor_treeedge(q, ω, j, nparents, ntraits)
end

# j = Sigma_child^{-1}
# omega = q_child * (sum_k gamma_k omega_k + omega_hybrid)
# q = q_child [gamma_k q_k]
# TODO: is this necessary ?
function factor_tree_degeneratehybrid(m::EvolutionaryModel{T}, pae::AbstractVector{PN.Edge}, che::PN.Edge) where T
    ntraits = dimension(m)
    nparents = length(pae)
    # hybridnode_variance(m, pae) is zero if degenerate, as well as branch_variance(m, edge) for all edge in pae
    j = branch_precision(m, che)
    # hybrid displacement and actualisation
    ωh = hybdridnode_displacement(m, pae)
    qh = Matrix{T}(undef, ntraits, nparents * ntraits)
    for (k, edge) in enumerate(pae)
        ωh .+= edge.gamma .* qche * branch_displacement(m, edge)
        branch_actualization!(view(qh, :, ((k-1) * ntraits + 1):(k*ntraits)), m, edge)
    end
    # child displacement and actualization
    # TODO: can we avoid re-allocation here ?
    qche =  branch_actualization(m, che)
    ω = branch_displacement(m, che) + qche * ωh
    q = qche * qh
    factor_treeedge(q, ω, j, nparents, ntraits)
end

################################################################
## factor_root
################################################################

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
    improper = any(LA.diag(j) .== 0.0) # then assumes that *all* are 0
    g = (improper ? zero(T) : (-dimension(m) * log2π + LA.logdet(j) - LA.dot(m.μ, h))/2)
    return(h, j, g)
end
