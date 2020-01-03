# ellipse to multiple forward simulated shapes
# only final landmark positions observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD

workdir = @__DIR__
println(workdir)
cd(workdir)

n = 4
nshapes = 1
T = 1.0

a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 1.0     # Noise level

Ptrue = MarslandShardlow(a, c, γ, 0.0, n)

q0 = [PointF(i*1.0/n) for i in 1:n]
p0 = zeros(PointF,n)
x0 = State(q0, p0)



xobs0 = x0.q

θ, ψ =  π/4, 0.25
pb = Lmplotbounds(-3.0,3.0,-3.0,3.0)
xobsT = [2.0*xobs0[i] + [0.8]  for i in 1:n ]

save("data_exp1-1D.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb, "nshapes", nshapes)
