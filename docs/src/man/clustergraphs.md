```@meta
CurrentModule = PhyloGaussianBeliefProp
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

For example, a cluster's belief is the cumulative result of computations that might
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
Below is an example using a virus recombination network from
[Müller et al. (2022, Fig 1a)](https://doi.org/10.1038/s41467-022-31749-8)
[muller2022bayesian](@cite), with inheritance probabilities estimated from the
inferred recombination breakpoints (see [muller2022_nexus2newick.jl](https://github.com/bstkj/graphicalmodels_for_phylogenetics_code/blob/5f61755c4defe804fd813113e883d49445971ade/real_networks/muller2022_nexus2newick.jl)).

```jldoctest clustergraphs; setup = :(using PhyloNetworks, PhyloGaussianBeliefProp; const PGBP = PhyloGaussianBeliefProp)
julia> net = readTopology(pkgdir(PhyloGaussianBeliefProp, "test/example_networks", "muller_2022.phy")); # 1161 edges, 801 nodes: 40 tips, 361 hybrid nodes, 400 internal tree nodes.

julia> preorder!(net)

julia> ct = PGBP.clustergraph!(net, PGBP.Cliquetree());

julia> PGBP.labels(ct) |> length # no. of vertices/clusters in clique tree
664

julia> PGBP.edge_labels(ct) |> length # no. of edges in clique tree, one less than no. of clusters
663

julia> clusters = PGBP.labels(ct) |> collect; # vector of cluster labels

julia> clusters[1] # cluster label is the concatenation of node labels
:I300I301I302I189

julia> ct[clusters[1]] # access metadata for `cluster[1]`: (node labels, preorder indices), nodes are arranged by decreasing preorder index
([:I300, :I301, :I302, :I189], Int16[722, 719, 717, 487])

julia> using StatsBase # `summarystats`

julia> (length(ct[cl][1]) for cl in PGBP.labels(ct)) |> collect |> summarystats # distribution of cluster sizes
Summary Stats:
Length:         664
Missing Count:  0
Mean:           6.728916
Std. Deviation: 6.120608
Minimum:        2.000000
1st Quartile:   4.000000
Median:         5.000000
3rd Quartile:   7.000000
Maximum:        54.000000
```

## Bethe / Factor graph
The Bethe cluster graph has a cluster for each node family (*factor clusters*)
and for each node (*variable clusters*). Each factor cluster is joined to the
variable cluster for any node it contains.

```jldoctest clustergraphs
julia> fg = PGBP.clustergraph!(net, PGBP.Bethe());

julia> (PGBP.labels(fg) |> length, PGBP.edge_labels(fg) |> length) # (no. of clusters, no. of edges)
(1557, 1914)

julia> (length(fg[cl][1]) for cl in PGBP.labels(fg)) |> collect |> summarystats
Summary Stats:
Length:         1557
Missing Count:  0
Mean:           1.743738
Std. Deviation: 0.809151
Minimum:        1.000000
1st Quartile:   1.000000
Median:         2.000000
3rd Quartile:   2.000000
Maximum:        3.000000
```
If each hybrid node has 2 parents (as for `net`), then the maximum cluster size
is 3.

## Join-graph structuring
[Join-graph structuring](https://doi.org/10.1613/jair.2842) allows the user to
specify the maximum cluster size ``k^*``. See [`JoinGraphStructuring`](@ref)
for more details on the algorithm.

```jldoctest clustergraphs
julia> jg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(10));

julia> (PGBP.labels(jg) |> length, PGBP.edge_labels(jg) |> length)
(1001, 1200)

julia> (length(jg[cl][1]) for cl in PGBP.labels(jg)) |> collect |> summarystats
Summary Stats:
Length:         1001
Missing Count:  0
Mean:           6.036963
Std. Deviation: 2.177070
Minimum:        1.000000
1st Quartile:   4.000000
Median:         6.000000
3rd Quartile:   8.000000
Maximum:        10.000000
```
Since the set of clusters has to be family-preserving (see above), ``k^*``
cannot be smaller than the largest node family (i.e. a node and its parents).
For example, if the network has hybrid nodes, then ``k^*\ge 3`` necessarily.

```jldoctest clustergraphs
julia> jg = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(2));
ERROR: maxclustersize 2 is smaller than the size of largest node family 3.
```
On the other extreme, suppose we set ``k^*=54``, the maximum cluster size of the
clique tree above:

```jldoctest clustergraphs
julia> jg2 = PGBP.clustergraph!(net, PGBP.JoinGraphStructuring(54));

julia> (PGBP.labels(jg2) |> length, PGBP.edge_labels(jg2) |> length)
(801, 800)

julia> (length(jg2[cl][1]) for cl in PGBP.labels(jg2)) |> collect |> summarystats
Summary Stats:
Length:         801
Missing Count:  0
Mean:           9.539326
Std. Deviation: 9.953078
Minimum:        1.000000
1st Quartile:   4.000000
Median:         5.000000
3rd Quartile:   10.000000
Maximum:        54.000000
```
then it turns out the `jg2` is a clique tree (since the number of clusters and
edges differ by 1), though not the same one as `ct`. Generally, a cluster graph
with larger clusters is less likely to be loopy than one with smaller clusters.

## LTRIP
For [LTRIP](https://doi.org/10.1145/3132711.3132717), the user provides the set
of clusters, which are assumed to be family-preserving (see above).
1. For each node, the clusters that contain it are joined as a tree, prioritizing edges formed with clusters that intersect heavily with others. See [`LTRIP`](@ref) for details.
2. The trees for each node are layered on one another (the sepsets for an edge are merged) to produce the cluster graph.
As an example, we use the clusters from join-graph structuring:

```jldoctest clustergraphs
julia> clusters = (jg[cl][2] for cl in PGBP.labels(jg)) |> collect; # vector of clusters, each given as a vector of preorder indices in decreasing order

julia> lg = PGBP.clustergraph!(net, PGBP.LTRIP(clusters, net));

julia> (PGBP.labels(lg) |> length, PGBP.edge_labels(lg) |> length)
(1001, 1249)
```
The summary statistics would be the same as for `jg`'s clusters, though it
appears that `lg` is more densely connected than `jg`.

If the user does not provide the clusters, then the set of node families
(see [`nodefamilies`](@ref)) is used by default:

```jldoctest clustergraphs
julia> lg = PGBP.clustergraph!(net, PGBP.LTRIP(net));

julia> (PGBP.labels(lg) |> length, PGBP.edge_labels(lg) |> length)
(801, 1158)
```
There are 801 clusters, 1 per node (family), as expected.