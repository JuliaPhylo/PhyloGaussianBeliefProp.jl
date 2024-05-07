```@meta
CurrentModule = PhyloGaussianBeliefProp
```

```@setup clustergraphs
using PhyloNetworks, PhyloGaussianBeliefProp
using StatsBase
const PGBP = PhyloGaussianBeliefProp
net = readTopology(pkgdir(PhyloGaussianBeliefProp, "test/example_networks", "muller_2022.phy"))
jg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(10));
```

# Cluster graphs
A cluster graph groups the nodes ``X_1,\dots,X_m`` of the phylogeny into
several (possibly intersecting) clusters/subsets, such that for any node:
- there is ``\ge 1`` cluster that contains it and its parents (i.e. the clusters are (node) *family-preserving*)
- the clusters that contain it are always joined as a tree, whose edge labels contain that node
Each edge between two clusters ``\mathcal{C}_i,\mathcal{C}_j`` is labeled with
a node subset ``\mathcal{S}_{i,j}\subseteq\mathcal{C}_i\cap\mathcal{C}_j``
based on the second property.
These labels are referred to as sepsets (i.e a "sep(arating) sets"). 

Intuitively, if the factors of a joint distribution ``p_\theta`` over the nodes
are distributed among the clusters, then the topology of the cluster graph
implies the possible computational pathways in which these factors may be
sequentially "combined" by product or marginalization.

For example, a cluster's belief is the cumulative result of computations that
follow some walk along the cluster graph, ending at that cluster. This is then
interpreted as an estimate of its conditional distribution given the data.

## Clique tree
A cluster graph whose topology is a tree is known as a clique tree. We provide
the option to construct a clique tree, and 3 further options (below) for
constructing (potentially) loopy cluster graphs.

A clique tree tends to have more clusters of larger size than a loopy cluster
graph. The time-complexity of message passing on a cluster graph is parametrized
by maximum cluster size, and so clique trees allow for exact inference but at a
greater cost than approximate inference on a loopy cluster graph.
```@repl clustergraphs
net = readTopology(pkgdir(PhyloGaussianBeliefProp, "test/example_networks", "muller_2022.phy")) # a virus recombination network from Fig 1a of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9297283/
ct = PGBP.clustergraph!(net, PGBP.Cliquetree());
PGBP.labels(ct) |> length # no. of vertices/clusters in clique tree
PGBP.edge_labels(ct) |> length # no. of edges in clique tree, one less than no. of clusters
clusters = PGBP.labels(ct) |> collect; # vector of cluster labels
clusters[1] # cluster label is the concatenation of node labels
ct[clusters[1]] # access metadata for `cluster[1]`: (node labels, preorder indices), nodes are arranged by decreasing preorder index
using StatsBase # `summarystats`
(length(ct[cl][1]) for cl in PGBP.labels(ct)) |> collect |> summarystats # distribution of cluster sizes
```

## Bethe / Factor graph
The Bethe cluster graph has a cluster for each node family (*factor clusters*)
and for each node (*variable clusters*). Each factor cluster is joined to the
variable cluster for any node it contains.
```@repl clustergraphs
fg = PGBP.clustergraph!(net, PGBP.Bethe());
(PGBP.labels(fg) |> length, PGBP.edge_labels(fg) |> length) # (no. of clusters, no. of edges)
(length(fg[cl][1]) for cl in PGBP.labels(fg)) |> collect |> summarystats
```
If each hybrid node has 2 parents (as for `net`), then the maximum cluster size
is 3.

## Join-graph structuring
[Join-graph structuring](https://doi.org/10.1613/jair.2842) allows the user to
specify the maximum cluster size ``k^*``. See [`JoinGraphStructuring`](@ref)
for more details on the algorithm.
```@repl clustergraphs
jg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(10));
(PGBP.labels(jg) |> length, PGBP.edge_labels(jg) |> length)
(length(jg[cl][1]) for cl in PGBP.labels(jg)) |> collect |> summarystats
```
Since the set of clusters has to be family-preserving (see above), ``k^*``
cannot be smaller than the largest node family (i.e. a node and its parents).
For example, if the network has hybrid nodes, then ``k^*\ge 3`` necessarily.
```@repl clustergraphs
jg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(2));
```
On the other extreme, suppose we set ``k^*=54``, the maximum cluster size of the
clique tree above:
```@repl clustergraphs
jg2 = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(54));
(PGBP.labels(jg2) |> length, PGBP.edge_labels(jg2) |> length)
(length(jg2[cl][1]) for cl in PGBP.labels(jg2)) |> collect |> summarystats
```
then it turns out the `jg2` is a clique tree (since the number of clusters and
edges differ by 1), though not the same one as `ct`. Generally, a cluster graph
with larger clusters is less likely to be loopy than one with smaller clusters.

## LTRIP
For [LTRIP](https://doi.org/10.1145/3132711.3132717), the user provides the set
of clusters, which are assumed to be family-preserving (see above).
1. For each node, the clusters that contain it are joined as a tree, prioritizing edges formed with clusters that intersect heavily with others. See [`LTRIP{T<:Integer}`](@ref) for details on the spanning tree algorithm applied.
2. The trees for each node are layered on one another (the sepsets for an edge are merged) to produce the cluster graph.
As an example, we use the clusters from join-graph structuring:
```@repl clustergraphs
clusters = (jg[cl][2] for cl in PGBP.labels(jg)) |> collect; # vector of clusters, each given as a vector of preorder indices in decreasing order
lg = PGBP.clustergraph!(net, PGBP.LTRIP(clusters, net));
(PGBP.labels(lg) |> length, PGBP.edge_labels(lg) |> length)
```
The summary statistics would be the same as for `jg`'s clusters, though it
appears that `lg` is more densely connected than `jg`.

If the user does not provide the clusters, then the set of node families
(see [`nodefamilies(net)`](@ref)) is used by default:
```@repl clustergraphs
lg = PGBP.clustergraph!(net, PGBP.LTRIP(net));
(PGBP.labels(lg) |> length, PGBP.edge_labels(lg) |> length)
```
There are 801 clusters, 1 per node (family), as expected.