```@setup evolutionary_models
using PhyloGaussianBeliefProp
const PGBP = PhyloGaussianBeliefProp
```

# Evolutionary models
## Specifying a process
Each trait evolutionary model is specified as a BM, or some extension of it,
that evolves along the phylogeny.

Minimally, the user provides a variance rate ``\Sigma``, and a prior mean
``\mu`` and variance ``\bm{V}_{\!\!\rho}`` for the root state ``X_\rho``.
For example, ``\bm{V}_{\!\!\rho}=0`` treats ``X_\rho=\mu`` as known, while
``\bm{V}_{\!\!\rho}=\infty`` disregards all prior beliefs about ``X_\rho``.
```@repl evolutionary_models
PGBP.UnivariateBrownianMotion(1, 0) # v = 0
PGBP.UnivariateBrownianMotion(1, 0, Inf)
```

The multivariate BM is available to model multivariate traits.
If the components of a multivariate trait evolve in an uncorrelated manner,
then ``\Sigma`` is a diagonal matrix and is specified its diagonal entries
(e.g. `MvDiagBrownianMotion`). Otherwise, ``\Sigma`` is potentially dense and
is passed in whole (e.g. `MvFullBrownianMotion`).
```@repl evolutionary_models
PGBP.MvDiagBrownianMotion([1, 0.5], [-1, 1]) # v = [0, 0]
PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [-1,1], [10^10 0; 0 10^10])
```

``\Sigma`` can vary along the phylogeny. If path length ``t\ge 0`` from the root
represents evolutionary time, then the Early Burst (EB) model and Accelerating
Rate (AC) model respectively allow ``\Sigma`` to decay (``b<0``) and grow
(``b>0``) along a *time-consistent* phylogeny:
```math
\Sigma(t) = \Sigma_0\exp(bt), \text{ where } \Sigma_0 = \Sigma(0)
```

Selection can be additionally modeled by the
[Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process)
(OU) process, which allows a trait to diffuse with variance rate ``\Sigma`` yet
drift towards some optimal value ``\theta`` (with selection "strength"
``\bm{A}``).
```@repl evolutionary_models
PGBP.UnivariateOrnsteinUhlenbeck(2, 3, -2, 0, 0.4) # σ2 = 2, γ2 = σ2/2α
```

## Edge factors
After specifying the evolutionary model
(e.g. `m = PGBP.UnivariateBrownianMotion(1, 0)`), it is eventually passed to
`init_beliefs_assignfactors!`
(see [4\. Initialize cluster graph beliefs](@ref)), which infers the conditional
distribution for each node and assigns it to a cluster.

We refer to these conditional distributions as *edge factors* since they relate
the states of a parent node and its child.

## Hybrid factors
Each reticulation node ``X_h`` has multiple parents ``X_{p_1},\dots,X_{p_k}``,
and is thus potentially associated with multiple edge factors.
We reconcile these by modeling ``X_h`` as a weighted-average of its
"immediate parents", the child states for each of these edge factors.

Imagine that we introduce ``k`` copies ``X_{(p_1,h)},\dots,X_{(p_k,h)}`` of
``X_h``, each of which descends from the corresponding ``X_{p_i}``. Then ``X_h``
is modeled as a weighted-average of ``X_{(p_1,h)},\dots,X_{(p_k,h)}``:
```math
X_h = \sum\gamma_{(p_i,h)} X_{(p_i,h)}
```
where the inheritance weights ``\gamma_{(p_i,h)}`` are positive and sum to 1.