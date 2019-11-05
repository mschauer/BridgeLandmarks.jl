using BridgeLandmarks
using Test
using Bridge

const LM = BridgeLandmarks

@testset "state" begin

q, p = zeros(PointF, 5), rand(PointF, 5)

x = LM.NState(q, p)
@test x.p == p
@test x.q == q

@test x.x[1, :] == q
@test x.x[2, :] == p

U = rand(UncF, 5,5)
U * x.p

end
