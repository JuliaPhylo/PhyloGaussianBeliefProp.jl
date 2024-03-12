"""
    getcholesky(J::AbstractMatrix)

Cholesky decomposition of J, assumed to be symmetric *positive* definite,
stored as a `PDMat` object.
Warning: PDMat is not a subtype of Cholesky.
[PDMats.jl](https://github.com/JuliaStats/PDMats.jl) is efficient for
structured matrices (e.g diagonal or sparse) and has efficient methods for
linear algebra, e.g. `\`, `invquad`, `X_invA_Xt` etc.
"""
function getcholesky(J::AbstractMatrix)
    return PDMat(J) # LA.cholesky(b.J)
end
"""
    getcholesky_μ(J::AbstractMatrix, h)
    getcholesky_μ!(belief::Belief)

Tuple `(Jchol, μ)` where `Jchol` is a cholesky representation of `J` or `belief.J`
and `μ` is J⁻¹h, used to update `belief.μ` (by the second method).
"""
function getcholesky_μ(J::AbstractMatrix, h)
    Jchol = getcholesky(J)
    μ = Jchol \ h
    return (Jchol, μ)
end
@doc (@doc getcholesky_μ) getcholesky_μ!
function getcholesky_μ!(b::Belief)
    (Jchol, μ) = getcholesky_μ(b.J, b.h)
    b.μ .= μ
    return (Jchol, μ)
end

"""
    entropy(J::Cholesky)
    entropy(J::AbstractMatrix)
    entropy(belief::AbstractBelief)

Entropy of a multivariate Gaussian distribution with precision matrix `J`,
assumed to be square and symmetric (not checked).
It is 0 if `J` is empty (of size 0×0). It may be `Inf` if `J` is semi-definite.
The second version applies the first to the belief precision `belief.J`.

`entropy` is defined for discrete distributions in
[StatsBase.jl](https://juliastats.org/StatsBase.jl/stable/scalarstats/#StatsBase.entropy)
and extended to Gaussian distributions in Distributions.jl around
[here](https://github.com/JuliaStats/Distributions.jl/blob/master/src/multivariate/mvnormal.jl#L95)
"""
function entropy(J::Union{LA.Cholesky{T},PDMat{T}}) where T<:Real
    n = size(J,2)
    n == 0 && return zero(T)
    (n * (T(log2π) + 1) - LA.logdet(J)) / 2
end
function entropy(J::AbstractMatrix{T}) where T<:Real
    n = size(J,2)
    n == 0 && return zero(T)
    (n * (T(log2π) + 1) - LA.logdet(LA.Symmetric(J))) / 2
end
entropy(cluster::AbstractBelief) = entropy(cluster.J)

"""
    average_energy!(ref::Belief, target::AbstractBelief, dropg::Bool=false)
    average_energy!(ref::Belief, Jₜ, hₜ, gₜ)
    average_energy(Jᵣ::Union{LA.Cholesky,PDMat}, μᵣ, Jₜ, hₜ, gₜ)

Average energy (i.e. negative expected log) of a `target` canonical form with
parameters `(Jₜ, hₜ, gₜ)` with respect to a normalized non-degenerate reference
canonical form `ref` with parameters `(Jᵣ, hᵣ)`. The reference distribution
is normalized, so specifying `gᵣ` is unnecessary.
When the target canonical form is also normalized and non-degenerate,
this is equal to their cross-entropy:

    H(fᵣ, fₜ) = - Eᵣ(log fₜ) = - ∫ fᵣ log fₜ .

If `dropg=true`, then average energy is computed assuming that `gₜ=0`.
`ref` is assumed to be non-degenerate, that is, `Jᵣ` should be positive definite.

`average_energy!` modifies the reference belief by updating `ref.μ` to J⁻¹h.
It calls `average_energy` after a cholesky decomposition of `ref.J`,
stored in `Jᵣ`: see [`getcholesky_μ!`](@ref).

## Calculation:

ref: f(x) = C(x | Jᵣ, hᵣ, _) is the density of 𝒩(μ=Jᵣ⁻¹hᵣ, Σ=Jᵣ⁻¹)  
target: C(x | Jₜ, hₜ, gₜ) = exp( - (1/2)x'Jₜx - hₜ'x - gₜ )

    E[-log C(X | Jₜ, hₜ, gₜ)] where X ∼ C(Jᵣ, hᵣ, _)
    = 0.5 (μᵣ'Jₜ μᵣ + tr(Jᵣ⁻¹Jₜ)) - hₜ'μᵣ - gₜ

With empty vectors and matrices (J's of dimension 0×0 and h's of length 0),
the result is simply: - gₜ.
"""
function average_energy!(ref::Belief, target::AbstractBelief, dropg::Bool=false)
    gₜ = (dropg ? zero(target.g[1]) : target.g[1])
    average_energy!(ref, target.J, target.h, gₜ)
end
function average_energy!(ref::Belief, Jₜ, hₜ, gₜ)
    (Jᵣ, μᵣ) = getcholesky_μ!(ref)
    average_energy(Jᵣ, μᵣ, Jₜ, hₜ, gₜ)
end
@doc (@doc average_energy) average_energy!
function average_energy(Jᵣ::Union{LA.Cholesky,PDMat}, μᵣ, Jₜ, hₜ, gₜ)
    isempty(Jₜ) && return -gₜ # dot(x,A,x) fails on empty x & A
    (LA.tr(Jᵣ \ Jₜ) + LA.dot(μᵣ, Jₜ, μᵣ)) / 2 - LA.dot(hₜ, μᵣ) - gₜ
end

"""
    factored_energy(beliefs::ClusterGraphBelief)

Factored energy functional for general cluster graphs (Koller & Friedman 2009),
which approximates the evidence lower bound (ELBO), a lower bound for the
log-likelihood. It is
also called the (negative) Bethe free energy in the context of factor graphs
It is the sum of the cluster average energies and entropies,
minus the sepset entropies.
It is assumed but not checked that `beliefs` are calibrated
(neighbor clusters and sepset beliefs are consistent, used as local marginals).

For a calibrated clique tree, the factored energy is equal to the
log-likelihood. For a calibrated cluster graph, it can serve as as approximation.

output: tuple of 3 values, the 3rd being the factored energy:
(average energy, approximate entropy, factored energy = -energy + entropy).

See also: [`free_energy`](@ref),
[`entropy`](@ref),
[`average_energy!`](@ref),
[`iscalibrated`](@ref)

## References

D. Koller and N. Friedman.
*Probabilistic graphical models: principles and techniques*.
MIT Press, 2009. ISBN 9780262013192.

D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational inference: A Review
for Statisticians, Journal of the American statistical Association, 112:518,
859-877, 2017, doi: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773).
"""
function factored_energy(b::ClusterGraphBelief)
    res = free_energy(b)
    return (res[1], res[2], -res[3])
end

"""
    free_energy(beliefs::ClusterGraphBelief)

negative [`factored_energy`](@ref) to approximate the negative log-likelihood.
The approximation is exact on a clique tree after calibration.
"""
function free_energy(beliefs::ClusterGraphBelief{B}) where B<:Belief{T} where T<:Real
    b = beliefs.belief
    init_b = beliefs.factor
    nclu = nclusters(beliefs)
    ave_energy = zero(T)
    approx_entropy = zero(T)
    for i in 1:nclu
        fac = init_b[i]
        if isempty(fac.J) # then b[i].J should be empty too
            ave_energy -= fac.g[1]
        else # do 1 cholesky of b[i], use it twice
            (Jclu, μclu) = getcholesky_μ!(b[i])
            ave_energy += average_energy(Jclu, μclu, fac.J, fac.h, fac.g[1])
            approx_entropy += entropy(Jclu)
        end
    end
    for i in (nclu+1):length(b)
        approx_entropy -= entropy(b[i])
    end
    return (ave_energy, approx_entropy, ave_energy - approx_entropy)
end
