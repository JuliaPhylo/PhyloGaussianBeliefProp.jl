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

For example, a cluster's belief can be thought of as the cumulative result at
some stage of a computational pathway, which is then interpreted as an estimate
of its conditional distribution given the data.

A cluster graph whose topology is a tree is known as a clique tree. We provide
the option to construct a clique tree, and further options for constructing
loopy cluster graphs.

## Clique tree
To build a clique tree, we:
1. [Moralize](https://en.wikipedia.org/wiki/Moral_graph) the phylogeny
2. [Triangulate](https://en.wikipedia.org/wiki/Chordal_graph) the undirected graph from 1.
3. Extract the maximum cliques of the chordal graph from 2.
4. Join the cliques/clusters from 3. in a tree that maximizes sepset sizes
For clique trees, the sepset for neighbor clusters
``\mathcal{C}_i,\mathcal{C}_j`` is
``\mathcal{S}_{i,j}=\mathcal{C}_i\cap\mathcal{C}_j``.

## Bethe / Factor graph
The Bethe cluster graph has a cluster for each factor (*factor clusters*) and
for each node (*variable clusters*). Each factor cluster is joined to the
variable cluster for any node it contains.

## LTRIP
For [LTRIP](https://doi.org/10.1145/3132711.3132717), the user provides the set
of clusters, which are assumed to be family-preserving (see above).
1. For each node, the clusters that contain it are joined as a tree, prioritizing edges formed with clusters that intersect heavily with others
2. The trees for each node are layered on one another (the sepsets for an edge are merged) to produce the cluster graph
Different spanning tree algorithms can be applied for step 1.

## Join-graph structuring
[Join-graph structuring](https://doi.org/10.1613/jair.2842), whose steps are
more opaque than those for LTRIP or the Bethe cluster graph, allows the user to
specify the maximum cluster size ``k^*``.