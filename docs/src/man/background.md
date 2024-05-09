# Background

## Trait evolution on a phylogeny
The evolution of molecular and phenotypic traits is commonly modeled using
[Markov processes](https://en.wikipedia.org/wiki/Markov_chain) along a rooted
phylogeny.

For example, most models of continuous trait evolution on a phylogenetic tree
are extensions of the
[Brownian motion](https://en.wikipedia.org/wiki/Wiener_process) (BM) to capture
features such as:
- evolutionary trends
- adaptation
- rate variation across lineages

## Factoring the joint model
A Markov process along the phylogeny induces a *joint distribution*
``p_\theta(x_1,\dots,x_m)``, with parameters ``\theta``, over all nodes
(trait vectors) ``X_1,\dots,X_m``.

``p_\theta(x_1,\dots,x_m)`` can also be factored as the product of *conditional
distributions* ``\phi_v`` for each node ``X_v``, where
``\phi_v=p_\theta(x_v\mid x_{\mathrm{pa}(v)})`` is the distribution of ``X_v``
given its parent(s) ``X_{\mathrm{pa}(v)}``, e.g.
``X_{\mathrm{pa}(v)}=\begin{bmatrix} X_{p_1} \\ X_{p_2}\end{bmatrix}``:
```math
p_\theta(x_1,\dots,x_m) = \prod_{v=1}^n \phi_v
```

We focus on the case where all conditional distributions are *linear Gaussian*.
That is, for each node ``X_v``:
```math
\begin{aligned}
X_v\mid X_{\mathrm{pa}(v)} &\sim \mathcal{N}(\omega_v+
\bm{q}_v X_{\mathrm{pa}(v)},\bm{V}_v) \\
\phi_v &= (|2\pi\bm{V}_v|)^{-1/2}\exp(-||x_v-(\omega_v+\bm{q}_v x_{\mathrm{pa}(v)})||_{\bm{V}_v^{-1}}/2)
\end{aligned}
```
with trend vector ``\omega_v``, actualization matrix ``\bm{q}_v``, and
covariance matrix ``\bm{V}_v``. For example, the BM (and most of its extensions)
on a phylogeny satisfies this characterization [mitov2020fast](@cite).

## Parameter inference
Typically, we observe the tips of the phylogeny
``X_1=\mathrm{x}_1,\dots,X_n=\mathrm{x}_n`` and use this data for parameter
inference by optimizing the log-likelihood
``\mathrm{LL}(\theta)=\log\int p_\theta(\mathrm{x}_1,\dots,\mathrm{x}_n,x_{n+1},
\dots x_m)dx_{n+1}\dots dx_m``:
```math
\widehat{\theta} = \argmax_{\theta} \ \mathrm{LL}(\theta)
```

For simpler models, it is possible to derive a closed-form expression for
``\widehat{\theta}`` by solving:
```math
\nabla_\theta \ [\mathrm{LL}(\theta)]|_{\theta=\widehat{\theta}}=0
```
for the zero of the log-likelihood gradient, and to compute it directly.

For more complicated models however, ``\widehat{\theta}`` must be obtained by
iterative methods that evaluate ``\mathrm{LL}(\theta)`` over different
parameter values.

In general, evaluating ``\mathrm{LL}(\theta)`` is costly as the size and
complexity of the phylogeny grows.

## BP for exact inference
[Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation) (BP) is a
framework for efficiently computing various
marginals of a joint distribution ``p_\theta`` that can be factored into
conditional distributions ``\phi_v\in\Phi``, where ``\Phi`` denotes the full set
of conditional distributions.
We refer to the reference book by Koller and Friedman (2009) [koller2009probabilistic](@cite)
for more background on BP, and only sketch the main steps involved here:

1. Construct a tree data structure called a [*clique tree*](https://en.wikipedia.org/wiki/Tree_decomposition) (also known by *junction tree*, *join tree*, or *tree decomposition*), whose nodes ``\mathcal{C}_i`` (called *clusters*) are subsets of ``\{X_1,\dots,X_m\}``.
2. Each conditional distribution is assigned (``\mapsto``) to a cluster of the clique tree, and the product of all conditional distributions assigned to a cluster ``\mathcal{C}_i`` initializes its *belief* ``\beta_i = \prod_{\phi_v\mapsto\mathcal{C}_i,\ \phi_v\in\Phi}\phi_v``
3. Each cluster computes messages from its belief, and propagates these to its neighbor clusters to update their beliefs.

``\mathrm{LL}(\theta)`` can be computed by passing messages according to a
single postorder traversal of the clique tree,

An additional preorder traversal guarantees that every cluster belief is the
corresponding marginal of ``p_\theta``. That is, every cluster belief reflects
the conditional distribution of the cluster given the data.


## Loopy BP for approximate inference
A clique tree is a special case of a graph data structure called a
*cluster graph*. In general, cluster graphs can be non-treelike with cycles.

BP on a *loopy cluster graph* (i.e. a cluster graph with cycles), abbreviated
as *loopy BP*, can approximate the likelihood and conditional distributions of
the unobserved, ancestral nodes, and be more computationally efficient than BP.


## References

```@bibliography
```