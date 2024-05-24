# todo: convert to proper test suite as design finalizes
using DataFrames
using LinearAlgebra
using PhyloGaussianBeliefProp, PhyloNetworks
const PGBP = PhyloGaussianBeliefProp
using StaticArrays
using Tables, Test

#################################
# Example from SM-F of manuscript
#################################
netstr = "((x1:1.0,(x2:1.0)#H1:0.0::0.5)x4:1.0,(#H1:0.0::0.5,x3:1.0)x6:1.0)x0;";
net = readTopology(netstr);
preorder!(net);
ct = PGBP.clustergraph!(net, PGBP.Cliquetree());
m = PGBP.UnivariateBrownianMotion(1,0);
df = DataFrame(taxon=tipLabels(net), x=[1.0, 1.0, 1.0]);
tbl_x = columntable(select(df, :x));
b = PGBP.init_beliefs_allocate(tbl_x, df.taxon, net, ct, m);
PGBP.init_beliefs_assignfactors!(b, m, tbl_x, df.taxon, net.nodes_changed);
ctb = PGBP.ClusterGraphBelief(b);

######################################
# Set up beliefs / generalized beliefs
######################################
b_x1x4 = b[PGBP.clusterindex(:x1x4, ctb)] # cluster :x1x4
gb_x1x4 = PGBP.generalizedBelief(b_x1x4)
b_x4 = b[PGBP.sepsetindex(:x1x4, :H1x4x6, ctb)] # sepset (:x1x4, :H1x4x6)
gb_x4 = PGBP.generalizedBelief(b_x4)

b_x3x6 = b[PGBP.clusterindex(:x3x6, ctb)] # cluster :x3x6
gb_x3x6 = PGBP.generalizedBelief(b_x1x4)
b_x6 = b[PGBP.sepsetindex(:x3x6, :H1x4x6, ctb)] # sepset (:x3x6, :H1x4x6)
gb_x6 = PGBP.generalizedBelief(b_x6)

#= Construct the belief directly because currently, clusters containing
degenerate hybrids may have them removed from the scope =#
b_x2H1 = PGBP.Belief([6, 5], 1, BitArray([0 1]),
    PGBP.bclustertype, :x2H1) # cluster :x2H1
b_x2H1.J .= MMatrix{1,1}([1.0;;])
b_x2H1.h .= MVector{1}([1.0])
b_x2H1.g[1] = -0.5+log(1/sqrt(2π))
gb_x2H1 = PGBP.generalizedBelief(b_x2H1)
b_H1 = PGBP.Belief([5], 1, BitArray([1;;]), PGBP.bsepsettype,
    (:x2H1, :H1x4x6)) # sepset (:x2H1, :H1x4x6)
gb_H1 = PGBP.generalizedBelief(b_H1)

b_H1x4x6 = PGBP.Belief([5, 4, 2], 1, BitArray([1 1 1]),
    PGBP.bclustertype, :H1x4x6) # cluster :H1x4x6
gb_H1x4x6 = PGBP.generalizedBelief(b_H1x4x6, [-1;0.5;0.5;;]) # assigned Dirac measure
# todo: update constructor for fully degenerate generalized belief
Q = nullspace(transpose([-1;0.5;0.5;;]))
gb_H1x4x6.Q[:,1:2] .= Q
gb_H1x4x6.Λ[1:2] .= diag(transpose(Q)*b_H1x4x6.J*Q)
b_x4x6 = b[PGBP.sepsetindex(:H1x4x6, :x4x6x0, ctb)] # sepset (:H1x4x6, :x4x6x0)
gb_x4x6 = PGBP.generalizedBelief(b_x4x6)

##############################################################
# trivial marginalization (no nodes need to be integrated out)
##############################################################
PGBP.marg!(gb_x1x4, PGBP.scopeindex(b_x4, b_x1x4))
PGBP.marg!(gb_x2H1, PGBP.scopeindex(b_H1, b_x2H1))
PGBP.marg!(gb_x3x6, PGBP.scopeindex(b_x6, b_x3x6))
@test gb_x1x4.Qbuf == gb_x1x4.Q
@test gb_x1x4.kbuf == gb_x1x4.k
@test gb_x1x4.hbuf == gb_x1x4.h
@test gb_x1x4.gbuf == gb_x1x4.g
@test gb_x1x4.Λbuf == gb_x1x4.Λ

##########################################################
# trivial division (sepset beliefs are constant initially)
##########################################################
PGBP.div!(gb_x4, gb_x1x4)
PGBP.div!(gb_H1, gb_x2H1)
PGBP.div!(gb_x6, gb_x3x6)
@test gb_x4.Qbuf == gb_x4.Q
@test gb_x4.kbuf == gb_x4.k
@test gb_x4.hbuf == gb_x4.h
@test gb_x4.gbuf == gb_x4.g
@test gb_x4.Λbuf == gb_x4.Λ

################
# multiplication
################
# cluster :H1x4x6 receives messages from clusters :x1x4, :x2H1, :x3x6
PGBP.mult!(gb_H1x4x6, gb_x4)
PGBP.mult!(gb_H1x4x6, gb_H1)
PGBP.mult!(gb_H1x4x6, gb_x6)

m = size(gb_H1x4x6.Q)[1] # cluster dimension
@test m == 3
k = gb_H1x4x6.k[1] # degrees of degeneracy
@test k == 1
Q = gb_H1x4x6.Q[:,1:(m-k)]
Λ = Diagonal(gb_H1x4x6.Λ[1:(m-k)])
# Q is not unique, though QΛQᵀ is
@test Q*Λ*transpose(Q) ≈ [1/3 1/3 1/3; 1/3 5/6 -1/6; 1/3 -1/6 5/6]
R = gb_H1x4x6.R[:,1:k]
@test R == [-1; 0.5; 0.5;;]
@test transpose(Q)*R ≈ [0;0;;] atol=1e-10 # Qᵀ*R == 0
h = gb_H1x4x6.h[1:(m-k)]
# h is not unique, though Qh is
@test Q*h ≈ [1;1;1;;]
@test gb_H1x4x6.g[1] == -4.2568155996140185 # -4.2568155996140185 == 3*(-0.5+log(1/sqrt(2π)))

#################
# marginalization
#################
# send message to cluster :x4x6x0
PGBP.marg!(gb_H1x4x6, PGBP.scopeindex(b_x4x6, b_H1x4x6))
m = size(gb_x4x6.Q)[1] # message dimension
@test m == 2
k = gb_H1x4x6.kbuf[1] # message degrees of degeneracy
@test k == 0
Q = gb_H1x4x6.Qbuf[1:(m-k),1:(m-k)]
Λ = Diagonal(gb_H1x4x6.Λbuf[1:(m-k)])
@test Q*Λ*transpose(Q) ≈ [5/4 1/4; 1/4 5/4]
h = gb_H1x4x6.hbuf[1:(m-k)]
@test Q*h ≈ [1.5, 1.5]
@test gb_H1x4x6.gbuf[1] == -4.2568155996140185