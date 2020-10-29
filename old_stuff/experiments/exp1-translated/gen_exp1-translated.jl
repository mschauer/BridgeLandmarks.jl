# ellipse to multiple forward simulated shapes
# only final landmark positions observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2

Random.seed!(23)

workdir = @__DIR__
println(workdir)
cd(workdir)

n = 5
nshapes = 1
T = 1.0

a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 0.7     # Noise level

Ptrue = MarslandShardlow(a, c, γ, 0.0, n)

xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]

θ, ψ =  π/10, 0.1
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [1.5, 1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n ]

JLD2.@save "data_exp1-translated.jld2" xobs0 xobsT n nshapes
