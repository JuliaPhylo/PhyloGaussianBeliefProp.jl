"""
getcholesky: warning as it returns PDMat object, not a subtype of Cholesky.
getcholesky_μ
getcholesky_μ!
fixit: write docstring, or delete these functions if not used
"""
function getcholesky(J::AbstractMatrix)
    return PDMat(J) # LA.cholesky(b.J)
end
function getcholesky_μ(J::AbstractMatrix,h)
    Jchol = getcholesky(J)
    μ = Jchol \ h
    return (Jchol, μ)
end
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
    average_energy(ref::Tuple, target::Tuple, dropg::Bool=false)
    average_energy(ref::AbstractBelief, target::AbstractBelief, dropg::Bool=false)
    average_energy(ref::AbstractBelief, target::Tuple, dropg::Bool=false)

Average energy (i.e. negative expected log) of a `target` canonical form with
parameters `(Jₜ, hₜ, gₜ)` with respect to a normalized non-degenerate reference
(`ref`) canonical form with parameters `(Jᵣ, hᵣ)`. The reference distribution
is normalized, so specifying `gᵣ` is unnecessary.
When the target canonical form is also normalized and non-degenerate,
this is equal to their cross-entropy:

    H(fᵣ, fₜ) = - Eᵣ(log fₜ) = - ∫ fᵣ log fₜ .

If `dropg=true`, then average energy is computed assuming that `gₜ=0`.
`ref` is assumed to be non-degenerate, that is, `Jᵣ` should be positive definite.

The second method takes `AbstractBelief`s, which contain `h` and `J` fields,
and computes the average energy of `target` with respect to `ref` by applying
the first method to their canonical parameters.

The third method is similar to the second one, except that `target` is specified
by its canonical parameters.

## Calculation:

ref: f(x) = C(x | Jᵣ, hᵣ, _) is the density of 𝒩(μ=Jᵣ⁻¹hᵣ, Σ=Jᵣ⁻¹)  
target: C(x | Jₜ, hₜ, gₜ) = exp( - (1/2)x'Jₜx - hₜ'x - gₜ )

    E[-log C(X | Jₜ, hₜ, gₜ)] where X ∼ C(Jᵣ, hᵣ, _)
     = (1/2)*(μᵣ'*Jₜ*μᵣ + tr(Jₜ*Jᵣ⁻¹)) - hₜ'*μᵣ - gₜ
     = (1/2)*(tr(Jₜ*μᵣ*μᵣ') + tr(Jₜ*Jᵣ⁻¹)) - hₜ'*μᵣ - gₜ

"""
function average_energy(ref::Belief, target::AbstractBelief, dropg::Bool=false)
    gₜ = (dropg ? zero(target.g[1]) : target.g[1])
    average_energy(ref, target.J, target.h, gₜ)
end
function average_energy(ref::Belief, Jₜ, hₜ, gₜ)
    (Jᵣ, μᵣ) = getcholesky_μ!(ref)
    average_energy(Jᵣ, μᵣ, Jₜ, hₜ, gₜ)
end
function average_energy(Jᵣ::Union{LA.Cholesky,PDMat}, μᵣ, Jₜ, hₜ, gₜ)
    # fixit: invquad is still causing PosDefException issues
    (LA.tr(Jᵣ \ Jₜ) + quad(Jₜ, μᵣ)) / 2 - LA.dot(hₜ, μᵣ) - gₜ
end


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
    nclu = nclusters(beliefs)
    ave_energy = zero(T)
    approx_entropy = zero(T)
    for i in 1:nclu
        (Jclu, μclu) = getcholesky_μ!(b[i]) # do 1 cholesky of b[i], use it twice
        ss = init_b[i]
        ave_energy += average_energy(Jclu, μclu, ss.J, ss.h, ss.g[1])
        approx_entropy += entropy(Jclu)
    end
    for i in (nclu+1):nbeliefs
        approx_entropy -= entropy(b[i])
    end
    return (ave_energy, approx_entropy, ave_energy - approx_entropy)
end