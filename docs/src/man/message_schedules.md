```@meta
CurrentModule = PhyloGaussianBeliefProp
DocTestSetup  = quote
  using DataFrames, Tables, PhyloNetworks, PhyloGaussianBeliefProp;
  const PGBP = PhyloGaussianBeliefProp;
end
```

# Message schedules
As described in [5. Propose a schedule from the cluster graph](@ref), we build a
message schedule by calling [`spanningtrees_clusterlist`](@ref) on our given
cluster graph (output by [`clustergraph!`](@ref)).

A schedule of messages can be visualized as a sequence of edge traversals
(from sender to recipient) on the cluster graph.

Since the calibration of a cluster graph requires neighbor clusters to reach
some state of agreement with each other, it is reasonable to expect that
multiple messages may need to be sent back and forth on each edge.
Thus, *proper* message schedules require that each edge is traversed in both directions, infinitely often (until stopping criteria are met).

`spanningtrees_clusterlist` satisfies the requirements of a proper message
schedule by specifying a finite sequence of edge traversals that together
account for all possible messages on the cluster graph. This sequence can then
be repeated as needed. Specifically:
- the sequence of edge traversals is returned as a collection of edge sets for different spanning trees of the cluster graph
- each edge set is ordered as a preorder traversal of the spanning tree

Each time [`calibrate!`](@ref) is called with a particular spanning tree, it
passes messages according to a postorder then preorder traversal of the tree.

Returning to the last few edges of the spanning tree schedule from
[5. Propose a schedule from the cluster graph](@ref):

```jldoctest; setup = :(net = readTopology(pkgdir(PGBP, "test/example_networks", "lazaridis_2014.phy")); preorder!(net); ct = PGBP.clustergraph!(net, PGBP.Cliquetree()); sched = PGBP.spanningtrees_clusterlist(ct, net.nodes_changed);)
julia> DataFrame(parent=sched[1][1], child=sched[1][2])[13:16,:] # edges of spanning tree in preorder
4×2 DataFrame
 Row │ parent                             child                             
     │ Symbol                             Symbol                            
─────┼──────────────────────────────────────────────────────────────────────
   1 │ AncientNorthEurasianI1I2NonAfric…  EasternNorthAfricanAncientNorthE…
   2 │ EasternNorthAfricanAncientNorthE…  H1EasternNorthAfricanAncientNort…
   3 │ H1EasternNorthAfricanAncientNort…  OngeEasternNorthAfrican
   4 │ H1EasternNorthAfricanAncientNort…  KaritianaH1
```
The first message to be sent is from `KaritianaH1` to
`H1EasternNorthAfricanAncientNort…`, followed by `OngeEasternNorthAfrican` to
`H1EasternNorthAfricanAncientNort…` and so on.

An *iteration* of calibration refers to `calibrate!` being called once for each
spanning tree in the collection.

Continuing with the code example from [A heuristic](@ref), we:
- increase the number of iterations of calibration to 100 (the default is 1)
- tell `calibrate!` to return once calibration is detected (`auto=true`)
- log information on when calibration was detected (`info=true`)

```jldoctest; setup = :(net = readTopology(pkgdir(PGBP, "test/example_networks", "lipson_2020b.phy")); preorder!(net); df = DataFrame(taxon=tipLabels(net), x=[0.431, 1.606, 0.72, 0.944, 0.647, 1.263, 0.46, 1.079, 0.877, 0.748, 1.529, -0.469]); m = PGBP.UnivariateBrownianMotion(1, 0); fg = PGBP.clustergraph!(net, PGBP.Bethe()); tbl_x = columntable(select(df, :x)); b = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, fg, m); fgb = PGBP.ClusterGraphBelief(b); sched = PGBP.spanningtrees_clusterlist(fg, net.nodes_changed);)
julia> PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed); # reset to initial beliefs

julia> PGBP.regularizebeliefs_bynodesubtree!(fgb, fg); # regularize by node subtree

julia> PGBP.calibrate!(fgb, sched, 100; auto=true, info=true)
[ Info: calibration reached: iteration 20, schedule tree 1
true
```
Similarly, during iterative optimization
(e.g [`calibrate_optimize_clustergraph!`](@ref)), multiple iterations of
calibration are run for each set of candidate parameter values (till beliefs
are calibrated) to determine the associated factored energy.