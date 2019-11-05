using Random
using BridgeLandmarks
using Bridge
using Test
using LinearAlgebra

const LM = BridgeLandmarks

@testset "patches" begin

A = reshape(rand(Unc,15),5,3)
B = LM.conj2(A)
@test norm(LM.deepmat(A)' - LM.deepmat(B))<10^(-10)

end
