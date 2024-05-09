```@meta
CurrentModule = PhyloGaussianBeliefProp
```

# Regularization

## Ill-defined messages
Propagating a message ``\tilde{\mu}_{i\rightarrow j}`` from a cluster
``\mathcal{C}_i`` to its neighbor ``\mathcal{C}_j`` involves 3 steps:

1\. ``\mathcal{C}_i``'s belief ``\beta_i`` is marginalized over the sepset nodes
``\mathcal{S}_{i,j}``:
```math
\tilde{\mu}_{i\rightarrow j} = \int\beta_i d(\mathcal{C}_i\setminus
\mathcal{S}_{i,j})
```
2\. The message ``\tilde{\mu}_{i\rightarrow j}`` is divided by the current
edge belief ``\mu_{i,j}``, and the result is multiplied into ``\mathcal{C}_j``'s
belief ``\beta_j``:
```math
\beta_j \leftarrow \beta_j\tilde{\mu}_{i\rightarrow j}/\mu_{i,j}
```
3\. The edge belief ``\mu_{i,j}`` is updated to the message just passed:
```math
\mu_{i,j} \leftarrow \tilde{\mu}_{i\rightarrow j}
```

In the linear Gaussian setting, where each belief has the form
``\exp(-x^{\top}\bm{J}x/2 + h^{\top}x + g)``, where ``x`` denotes its scope,
these steps can be concisely expressed in terms of the ``(\bm{J},h,g)``
parameters of the beliefs involved.

Crucially, the precision matrix ``\bm{J}`` for ``\beta_i`` has to be
full-rank / invertible with respect to the nodes to be integrated out.

For example, if ``\mathcal{C}_i=\{X_1,X_2,X_3\}``,
``\mathcal{S}_{i,j}=\{X_1\}`` and:
```math
\beta_i = \exp\left(-\begin{bmatrix}x_1 \\ x_2 \\ x_3\end{bmatrix}^{\top}\bm{J}
\begin{bmatrix}x_1 \\ x_2 \\ x_3\end{bmatrix}/2 +
h^{\top}\begin{bmatrix}x_1 \\ x_2 \\ x_3\end{bmatrix} + g\right), \text{ where }
\bm{J} = \begin{matrix}x_1 \\ x_2 \\ x_3\end{matrix}\!\!
\begin{bmatrix}1 & -1/2 & -1/2 \\ -1/2 & 1/4 & 1/4 \\
-1/2 & 1/4 & 1/4\end{bmatrix}
```
then the ``2\times 2`` submatrix of ``\bm{J}`` for
``\mathcal{C}_i\setminus\mathcal{S}_{i,j}=\{X_2,X_3\}`` (annotated above)
consists only of ``1/4``s, is not full-rank, and thus
``\tilde{\mu}_{i\rightarrow j}`` is ill-defined / cannot be computed.

On a clique tree, a schedule of messages that follows a postorder traversal of
the tree (from the tip clusters to a root cluster) can always be computed.

On a loopy cluster graph however, it may be unclear how to find a schedule such
that each message is well-defined, or if such a schedule even exists.
This is a problem since a loopy cluster graph typically requires multiple
traversals to reach convergence.

## A heuristic
One approach to deal with ill-defined messages is to skip their computation and
proceed on with the schedule.

A more robust, yet simple, alternative is to *regularize* cluster beliefs by
increasing some diagonal elements of their precision matrix so that the relevant
submatrices are full-rank:
```math
 \begin{matrix}x_1 \\ x_2 \\ x_3\end{matrix}\!\!
\begin{bmatrix}1 & -1/2 & -1/2 \\ -1/2 & 1/4 & 1/4 \\
-1/2 & 1/4 & 1/4\end{bmatrix} \longrightarrow
\begin{bmatrix}1 & -1/2 & -1/2 \\ -1/2 & 1/4\textcolor{red}{+\epsilon} & 1/4 \\
-1/2 & 1/4 & 1/4\textcolor{red}{+\epsilon}\end{bmatrix}
```
To maintain the probability model, the product of all cluster beliefs divided
by the product of all edge beliefs must remain equal to the joint distribution 
``p_\theta`` (this is satisfied after factor assignment, and everytime a message
is passed).

Thus, each time a cluster belief is regularized, we "balance" this change by a
similar modification to one or more associated edge beliefs. For example, if
``\mathcal{C}_i`` above was connected to another sepset
``\mathcal{S}_{i,k}=\{X_2,X_3\}`` with ``\bm{J}=\bm{0}`` then we might do:
```math
 \begin{matrix}x_2 \\ x_3\end{matrix}\!\!
\begin{bmatrix}0 & 0 \\ 0 & 0\end{bmatrix} \longrightarrow
\begin{bmatrix}0\textcolor{red}{+\epsilon} & 0 \\ 0 &
\textcolor{red}{+\epsilon}\end{bmatrix}
```
We provide several options for regularization below. A typical usage of these
methods is after the initial assignment of factors.

### Along node subtrees
[`regularizebeliefs_bynodesubtree!`](@ref) performs regularization separately
from message passing.

A *node subtree* of a cluster graph is the subtree induced by all clusters that
contain that node.

The algorithm loops over each node subtree. For all edges and all but one
cluster in a given node subtree, it adds ``\epsilon`` to the diagonal entries of
the precision matrix that correspond to that node.

### On a schedule
[`regularizebeliefs_onschedule!`](@ref) interleaves regularization with message
passing.

The algorithm loops over each cluster and tracks which messages have been sent:
- Each cluster ``\mathcal{C}_i`` is regularized only if it has not received a message from ``\ge 1`` of its neighbors.
- Regularization proceeds by adding ``\epsilon`` to the diagonal entries of ``\mathcal{C}_i``'s precision that correspond to the nodes in ``\mathcal{S}_{i,j}``, and to all diagonal entries of ``\mathcal{S}_{i,j}``'s precision, if neighbor ``\mathcal{C}_j`` has not sent a message to ``\mathcal{C}_i``.
- After being regularized, ``\mathcal{C}_i`` sends a message to each neighbor for which it has not already done so.

The example below shows how both regularization methods can help to minimize
ill-defined messages. We follow the steps in [Exact likelihood for fixed parameters](@ref):

```jldoctest regularization
julia> using DataFrames, Tables, PhyloNetworks, PhyloGaussianBeliefProp

julia> const PGBP = PhyloGaussianBeliefProp;

julia> net = readTopology(pkgdir(PGBP, "test/example_networks", "lipson_2020b.phy")); #  54 edges; 44 nodes: 12 tips, 11 hybrid nodes, 21 internal tree nodes.

julia> preorder!(net)

julia> df = DataFrame(taxon=tipLabels(net),
               x=[0.431, 1.606, 0.72, 0.944, 0.647, 1.263, 0.46, 1.079, 0.877, 0.748, 1.529, -0.469]); # read in tip data

julia> m = PGBP.UnivariateBrownianMotion(1, 0); # choose model: σ2 = 1.0, μ = 0.0 

julia> fg = PGBP.clustergraph!(net, PGBP.Bethe()); # build factor graph

julia> tbl_x = columntable(select(df, :x)); # trait data as column table

julia> b = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, fg, m); # allocate memory for beliefs

julia> PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed); # assign factors based on model

julia> fgb = PGBP.ClusterGraphBelief(b); # wrap beliefs for message passing

julia> sched = PGBP.spanningtrees_clusterlist(fg, net.nodes_changed); # generate schedule
```

Without regularization, errors indicating ill-defined messages (which are skipped)
are returned when we run a single iteration of calibration:

```jldoctest regularization
julia> PGBP.calibrate!(fgb, sched); # there are ill-defined messages (which are skipped)
┌ Error: belief H5I5I16, integrating [2, 3]
└ @ PhyloGaussianBeliefProp ~/Work/Research/BeliefPropagation/PhyloGaussianBeliefProp.jl/src/calibration.jl:101
┌ Error: belief H6I10I15, integrating [2, 3]
└ @ PhyloGaussianBeliefProp ~/Work/Research/BeliefPropagation/PhyloGaussianBeliefProp.jl/src/calibration.jl:101
```

However, with regularization, there are no ill-defined messages for a single
iteration of calibration:

```jldoctest regularization
julia> PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed); # reset to initial beliefs

julia> PGBP.regularizebeliefs_bynodesubtree!(fgb, fg); # regularize by node subtree

julia> PGBP.calibrate!(fgb, sched); # no ill-defined messages

julia> PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed); # reset to initial beliefs

julia> PGBP.regularizebeliefs_onschedule!(fgb, fg); # regularize by on schedule

julia> PGBP.calibrate!(fgb, sched); # no ill-defined messages
```
Note that this does not necessarily guarantee that subsequent iterations avoid
ill-defined messages.