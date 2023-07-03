@testset "evolutionary models" begin
m = PGBP.UnivariateBrownianMotion(2, 3)
PGBP.MvDiagBrownianMotion([1,0.5], [-1,1], [0,1])
PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [-1,1], [0 0; 0 1])
PGBP.factor_treenode(m, 0,1,2,0)
end
