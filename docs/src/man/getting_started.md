```@setup getting_started
using PhyloNetworks, PhyloGaussianBeliefProp
const PGBP = PhyloGaussianBeliefProp
net = readTopology("lazaridis_2014.phy")
ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
m = PGBP.UnivariateBrownianMotion(1, 0)
sched = PGBP.spanningtrees_clusterlist(ct, net.nodes_changed)
using DataFrames, Tables
df = DataFrame(taxon=tipLabels(net),
        x=[1.343, 0.841, -0.623, -1.483, 0.456, -0.081, 1.311])
tbl_x = columntable(select(df, :x))
b = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, m)
PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed)
ctb = PGBP.ClusterGraphBelief(b)
```

# Getting started

## Exact likelihood for fixed parameters

### 1\. Read in the network and the tip data
```@repl
using PhyloNetworks # `readTopology`, `tipLabels`
using DataFrames # `DataFrame`
net = readTopology("lazaridis_2014.phy") # from `example_networks/` folder
df = DataFrame(taxon=tipLabels(net),
        x=[1.343, 0.841, -0.623, -1.483, 0.456, -0.081, 1.311])
```
In this example, the trait `x` observed for the tip species is univariate.
We have mapped the observed data to the corresponding species in the dataframe `df`.

`net`, which reproduces
[Lazaridis et al. (2014), Figure 3](https://doi.org/10.1038/nature13673), is
displayed below:

![](../assets/lazaridis_2014_trim.png)

### 2\. Choose an evolutionary model
Models available are: `UnivariateBrownianMotion`, `MvDiagBrownianMotion`, `MvFullBrownianMotion`
```@repl getting_started
using PhyloGaussianBeliefProp
const PGBP = PhyloGaussianBeliefProp
m = PGBP.UnivariateBrownianMotion(1, 0) # σ2 = 1.0, μ = 0.0
```
We specify a univariate Brownian motion with mean ``\mu=0`` and variance rate
``\sigma^2=1``. We want to compute the likelihood for these particular values,
though other values may better fit the data.

### 3\. Build a cluster graph from the network
Methods available are: `Bethe`, `LTRIP`, `JoinGraphStructuring`, `Cliquetree`
```@repl getting_started
ct = PGBP.clustergraph!(net, PGBP.Cliquetree())
PGBP.labels(ct) |> collect # cluster labels
```
We choose `Cliquetree` to compute the likelihood exactly. Other methods may
return a loopy cluster graph, which gives an approximate likelihood.

See that each cluster's label is derived by concatenating the labels of the
nodes it contains.

### 4\. Initialize cluster graph beliefs
`ct` describes the topology of our cluster graph, but does not track the beliefs
for each cluster. Next, we:
- allocate memory for these beliefs
- initialize their values using the evolutionary model
- wrap them within another data structure to facilitate message passing.
```@repl getting_started
using Tables # `columntable`
tbl_x = columntable(select(df, :x)) # extract trait `x` from `df` as a column table
b = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, m); # allocate memory for beliefs
length(b) # no. of beliefs
b[1] # belief for cluster {H1, EasternNorthAfrican, AncientNorthEurasian} before factor assignment

PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed); # initialize beliefs from evolutionary model
b[1] # belief for cluster {H1, EasternNorthAfrican, AncientNorthEurasian} after factor assignment

ctb = PGBP.ClusterGraphBelief(b); # wrap beliefs to facilitate message passing
PGBP.nclusters(ctb) # no. of cluster beliefs
PGBP.nsepsets(ctb) # no. of edge/sepset beliefs
```
`b` is a vector of all beliefs, one for each cluster and edge (also known as
*sepset*) in the cluster graph. The edge beliefs store the most recent messages
passed between neighboring clusters.

Recall that each cluster or edge is associated with a set of nodes. The *scope*
``x`` of its belief comes from stacking the trait vectors for these nodes.
A belief with scope ``x`` is parametrized by ``(\bm{J},h,g)`` as follows:
```math
\exp(-x^{\top}\bm{J}x/2 + h^{\top}x + g)
```

We show belief `b[1]` before and after factor (i.e. conditional distribution)
assignment. Note that its `J` and `g` parameters are changed.

`ctb` contains `b` with added information to locate specific beliefs in `b` from
their corresponding cluster/edge labels in `ct`, and added storage to log
information during message passing.

### 5\. Propose a schedule from the cluster graph
A message schedule can be described by a sequence of cluster pairs.
Each pairing tells us to send a message between these clusters (which must be
neighbors), while the order within the pair indicates the sender and the
recipient.

We build a message schedule `sched` from `ct` by finding a minimal set of
spanning trees for the cluster graph that together cover all its edges (i.e.
neighbor cluster pairs). Each spanning tree is represented as a sequence of
edges following some preorder traversal of `ct`.

Since `ct` is a clique tree, there is a single spanning tree (`sched[1]`). We
extract and display the preorder sequence of edges from `sched[1]`. For example,
`NonAfricanI3` is the root cluster of `ct`, and `KaritianaH1` is a leaf cluster.
```@repl getting_started
sched = PGBP.spanningtrees_clusterlist(ct, net.nodes_changed);
length(sched) # no. of spanning trees needed
DataFrame(parent=sched[1][1], child=sched[1][2]) # edges of spanning tree in preorder
```

### 6\. Calibrate beliefs with the schedule
We apply one iteration of belief propagation on `ctb` following the schedule
`sched`. Since `ct` is a clique tree, the resulting beliefs are guaranteed to be
calibrated. 
```@repl getting_started
PGBP.calibrate!(ctb, sched);
```

### 7\. Extract the log-likelihood
On a calibrated clique tree, there are two ways to obtain the log-likelihood:
- integrate any belief over its scope to get its normalization constant (`norm`)
- compute the factored energy, which approximates the log-likelihood on loopy cluster graphs but is exact on a clique tree
```@repl getting_started
PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed) # hide
PGBP.calibrate!(ctb, sched) # hide
(_, norm) = PGBP.integratebelief!(b[1]); # `norm` is the integral of `b[1]` over its scope
norm
(_, _, fe) = PGBP.factored_energy(ctb); # `fe` is the factored energy from the cluster/edge beliefs
fe
```
The first approach is more efficient (it uses only one belief, rather than all
beliefs), but only works for a clique tree. The normalization constant of a
belief from a calibrated loopy cluster graph cannot be similarly interpreted.

We see that both approaches return the same value, modulo rounding error.

## Parameter inference
In the section above, we computed the log-likelihood for ``\mu=0``, ``\sigma^2=1``.

Now we find ``\mu=\widehat{\mu}`` and ``\sigma^2=\widehat{\sigma}^2`` that
maximize the log-likelihood. There are two options:
- iterative optimization
- exact computation using belief propagation
```@repl getting_started
mod, _, _ = PGBP.calibrate_optimize_cliquetree!( # iterative optimization
        ctb, # beliefs
        ct, # clique tree
        net.nodes_changed,
        tbl_x, # trait data
        df.taxon, # tip labels
        PGBP.UnivariateBrownianMotion, # type of evolutionary model
        (1.0, 0)); # starting parameters: σ2 = 1.0, μ = 0.0
mod # ML estimates

mod, _ = PGBP.calibrate_exact_cliquetree!( # exact computation
        ctb,
        sched[1], # spanning tree
        net.nodes_changed,
        tbl_x,
        df.taxon,
        PGBP.UnivariateBrownianMotion);
mod # REML estimate for σ2, ML estimate for μ
```
Both options return the maximum-likelihood (ML) estimate for ``\mu``, though
the latter returns the restricted maximum-likelihood (REML) estimate for
``\sigma^2``.

Strictly speaking, the estimates from the latter option do not jointly maximize
the log-likelihood. However, the REML estimate for ``\sigma^2`` is generally
less biased than its ML counterpart.