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

n = 3
nshapes = 1
T = 1.0

a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 1.0     # Noise level

Ptrue = MarslandShardlow(a, c, γ, 0.0, n)

#q0 = [PointF(i*1.0/n) for i in 1:n]# 0.1*rand(PointF,n)#
q0 = [PointF(.5), PointF(1.0), PointF(1.3)]
p0 = zeros(PointF,n)
x0 = State(q0, p0)

xobs0 = x0.q
xobsT = [exp.(xobs0[i])  for i in 1:n ]
#xobsT = [PointF(1.0), PointF(0.4)]

save("data_exp1-1D.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "nshapes", nshapes)
