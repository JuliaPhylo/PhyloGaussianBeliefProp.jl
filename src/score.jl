"""
    entropy(J::AbstractMatrix)
    entropy(belief::AbstractBelief)

Entropy value for a multivariate Gaussian with positive-definite precision `J`.

The second version applies the first version to the precision of `belief` (i.e.
`belief.J`, which must be positive definite). 

See implementation in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl/blob/e407fa5fd098e50df51801c6d062946eac7a7d0f/src/multivariate/mvnormal.jl#L95).
"""
function entropy(J::AbstractMatrix)
    0.5*(size(J,2) * (log2π + 1) - LA.logdet(LA.cholesky(J)))
end
function entropy(cluster::AbstractBelief)
    entropy(cluster.J)
end

"""
    average_energy(refcanon::Tuple{AbstractMatrix, AbstractVector},
        targetcanon::Tuple{AbstractMatrix, AbstractVector})
    average_energy(reference::AbstractBelief, target::AbstractBelief)
    average_energy(reference::AbstractBelief,
        targetcanon::Tuple{AbstractMatrix, AbstractVector})

Average energy (also known as cross-entropy) of a cluster/sepset belief
(specified by canonical parameters `targetcanon`) with respect to a
non-degenerate multivariate Gaussian (specified by canonical parameters `canon`).

The second version takes two possible beliefs (`reference`, `target`) for a given
cluster/sepset and computes the average energy of `target` with respect to
`reference` by applying the first version to their canonical parameters.
`reference` is assumed to correspond to a non-degenerate multivariate Gaussian.

The third version is similar to the second one, except that `target` is replaced
by its canonical parameters `targetcanon`.
"""
function average_energy(refcanon::Tuple{AbstractMatrix, AbstractVector},
    targetcanon::Tuple{AbstractMatrix, AbstractVector, AbstractFloat})
    #= `canon`: 𝒞(J, h, _) ≡ 𝒩(μ=J⁻¹h, Σ=J⁻¹), `belief`: 𝒞(Jₜ, hₜ, gₜ)
    E[-(1/2)x'*Jₜ*x + hₜ'x + gₜ] where x ∼ 𝒞(J,h,_)
    = -(1/2)*(μ'*Jₜ*μ + tr(Jₜ*J⁻¹)) + hₜ'*μ + gₜ
    = -(1/2)*(tr(Jₜ*μ*μ') + tr(Jₜ*J⁻¹)) + ... =#
    J = LA.cholesky(refcanon[1])
    μ = J \ refcanon[2]
    (Jₜ, hₜ, gₜ) = targetcanon
    # fixit: check for more efficient order of operations
    0.5*LA.tr(Jₜ*(μ*μ' + LA.inv(J))) - hₜ'*μ - gₜ
end
average_energy(reference::AbstractBelief, target::AbstractBelief) =
    average_energy((reference.J, reference.h), (target.J, target.h, target.g[1]))
average_energy(reference::AbstractBelief,
    targetcanon::Tuple{AbstractMatrix, AbstractVector, AbstractFloat}) =
    average_energy((reference.J, reference.h), targetcanon)

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
function free_energy(beliefs::ClusterGraphBelief)
    b = beliefs.belief
    init_b = beliefs.factors
    nbeliefs = length(b)
    nclusters = beliefs.nclusters
    ave_energy = 0.0
    approx_entropy = 0.0
    for i in 1:nclusters
        ave_energy += average_energy(b[i], init_b[i])
        approx_entropy += entropy(b[i])
    end
    for i in (nclusters+1):nbeliefs
        approx_entropy -= entropy(b[i])
    end
    return (ave_energy, approx_entropy, ave_energy - approx_entropy)
end