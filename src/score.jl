"""
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
function entropy(J::AbstractMatrix{T}) where T<:Real
    n = size(J,2)
    if n == 0
        return zero(T)
    end
    (n * (T(log2π) + 1) - LA.logdet(LA.Symmetric(J))) / 2
end
entropy(cluster::AbstractBelief) = entropy(cluster.J)

"""
    average_energy(ref::Tuple, target::Tuple, dropg::Bool=false)
    average_energy(ref::AbstractBelief, target::AbstractBelief, dropg::Bool=false)
    average_energy(ref::AbstractBelief, target::Tuple, dropg::Bool=false)

Average energy (i.e. negative expected log) of a `target` canonical form with
parameters `(Jₜ, hₜ, gₜ)` with respect to a normalized non-degenerate reference
(`ref`) canonical form with parameters `(Jᵣ, hᵣ)` (specifying `gᵣ` is
unnecessary to compute this quantity). When the target canonical form is also
normalized and non-degenerate, this is equal to their cross-entropy. If
`dropg=true`, then average energy is computed assuming that `gₜ=0`.

## Calculation:
    `ref`: f(x) = C(Jᵣ, hᵣ, _) ≡ x ~ 𝒩(μ=Jᵣ⁻¹hᵣ, Σ=Jᵣ⁻¹)
    `target`: C(Jₜ, hₜ, gₜ)

        E[-log C(Jₜ, hₜ, gₜ)]
    = E[(1/2)x'*Jₜ*x - hₜ'x - gₜ)] where x ∼ C(Jᵣ, hᵣ, _)
    = (1/2)*(μᵣ'*Jₜ*μᵣ + tr(Jₜ*Jᵣ⁻¹)) - hₜ'*μᵣ - gₜ
    = (1/2)*(tr(Jₜ*μᵣ*μᵣ') + tr(Jₜ*Jᵣ⁻¹)) - hₜ'*μᵣ - gₜ

The second version takes two possible beliefs (`ref`, `target`) for a given
cluster/sepset and computes the average energy of `target` with respect to `ref`
by applying the first version to their canonical parameters. `ref` is assumed to
be non-degenerate (i.e. `Jᵣ` is positive-definite).

The third version is similar to the second one, except that `target` is specified
by its canonical parameters.
"""
function average_energy(refcanon::Tuple, targetcanon::Tuple, dropg::Bool=false)
    # fixit: review. If reference is constant, then return `g` param of target.
    isempty(refcanon[1]) && return -targetcanon[3]
    Jᵣ = LA.cholesky(refcanon[1])
    μᵣ = Jᵣ \ refcanon[2]
    (Jₜ, hₜ, gₜ) = targetcanon
    # fixit: check for more efficient order of operations
    if dropg gₜ = zero(gₜ) end
    0.5*LA.tr(Jₜ*(μᵣ*μᵣ' + LA.inv(Jᵣ))) - hₜ'*μᵣ - gₜ
end
average_energy(reference::AbstractBelief, target::AbstractBelief, drop::Bool=false) =
    average_energy((reference.J, reference.h), (target.J, target.h, target.g[1]), drop)
average_energy(reference::AbstractBelief, targetcanon::Tuple, drop::Bool=false) =
    average_energy((reference.J, reference.h), targetcanon, drop)

"""
    free_energy(beliefs::ClusterGraphBelief)

Bethe free energy from `beliefs`. This is computed by adding up cluster average
energies and entropies, and subtracting sepset entropies. It is assumed but not
checked that `beliefs` are calibrated.

For a calibrated clique tree, -(Bethe free energy) is equal to the
log-likelihood. For a calibrated cluster graph, -(Bethe free energy) approximates
the evidence lower bound (ELBO) for the log-likelihood. Thus, minimizing the
Bethe free energy maximizes an approximation to the ELBO for the log-likelihood.

See also: [`entropy`](@ref), [`average_energy`](@ref), [`iscalibrated`](@ref)

## References
D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational inference: A Review
for Statisticians, Journal of the American statistical Association, 112:518,
859-877, 2017, doi: [10.1080/01621459.2017.1285773](https://doi/org/10.1080/01621459.2017.1285773).
"""
function free_energy(beliefs::ClusterGraphBelief{B}) where B<:Belief{T} where T<:Real
    b = beliefs.belief
    init_b = beliefs.factor
    nbeliefs = length(b)
    nclusters = beliefs.nclusters
    ave_energy = zero(T)
    approx_entropy = zero(T)
    for i in 1:nclusters
        ave_energy += average_energy(b[i], init_b[i])
        approx_entropy += entropy(b[i])
    end
    for i in (nclusters+1):nbeliefs
        approx_entropy -= entropy(b[i])
    end
    return (ave_energy, approx_entropy, ave_energy - approx_entropy)
end