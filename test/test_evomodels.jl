@testset "evolutionary models" begin
PGBP.MvDiagBrownianMotion([1,0.5], [-1,1], [0,1])
PGBP.MvFullBrownianMotion([1 0.5; 0.5 1], [-1,1], [0 0; 0 1])
m = PGBP.UnivariateBrownianMotion(2, 3)
h,J,g = PGBP.factor_treeedge(m, 1)
@test h == [0.0,0]
@test J == [.5 -.5; -.5 .5]
@test g ≈ -1.2655121234846454
end
