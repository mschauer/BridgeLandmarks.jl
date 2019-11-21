using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD

workdir = @__DIR__
println(workdir)
cd(workdir)

n = 20

a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 1.0     # Noise level

Ptrue = MarslandShardlow(a, c, γ, 0.0, n)

q0 = [PointF(2.0cos(t), sin(t))  for t in (0:(2pi/n):2pi)[1:n]]
p0 = [PointF(1.0, -3.0) for i in 1:n]/n
x0 = State(q0, p0)

Wf, Xf = landmarksforward(0.0:0.001:T, x0, Ptrue)

xobs0 = x0.q + σobs * randn(PointF,n)

θ, η =  π/5, 0.4
pb = Lmplotbounds(-3.0,3.0,-3.0,3.0)
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + η, 0.0, 0.0, 1.0 - η)
xobsT = [rot * stretch * xobs0[i]  for i in 1:Ptrue.n ] + σobs * randn(PointF,n)

save("./figs/data_exp1.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb)
