# ellipse to multiple forward simulated shapes
# only final landmark positions observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD

Random.seed!(23)

workdir = @__DIR__
println(workdir)
cd(workdir)

n = 5#18  # in the final experiment we'll take n = 18
nshapes = 1
T = 1.0

a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 1.0     # Noise level

Ptrue = MarslandShardlow(a, c, γ, 0.0, n)

q0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]
#p0 = [PointF(1.0, -3.0) for i in 1:n]/n
p0 = zeros(PointF,n)
x0 = State(q0, p0)

#Wf, Xf = landmarksforward(0.0:0.001:T, x0, Ptrue)

xobs0 = x0.q

θ, ψ =  π/10, 0.1
pb = Lmplotbounds(-1.0,3.0,-1.0,3.0)
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [1.5, 1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n ]

save("data_exp1-translated.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb, "nshapes", nshapes)
