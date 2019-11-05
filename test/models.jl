
using Test
using BridgeLandmarks
using Bridge
using Random

const LM = BridgeLandmarks

@testset "models" begin

Paux.

Hend⁺ = [rand(UncF) for i in 1:2Paux.n, j in 1:2Paux.n]
t0 = 2.0
BB = Bridge.B(t0,Paux) * Hend⁺
out = deepcopy(Hend⁺)
Bridge.B!(t0,Hend⁺,out,Paux)
@test out==BB
differ = Bridge.b(t0,Paux.xT,P) - Bridge.B(t0,Paux) * Paux.xT - Bridge.β(t0,Paux)
@test Bridge.inner(vec(differ.q)) < 10^(-6) # at the endpoint, drift should match for q coordinates

end
