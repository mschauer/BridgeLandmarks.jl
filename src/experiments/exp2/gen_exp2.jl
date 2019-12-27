# ellipse to multiple forward simulated shapes
# only final configurations observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD

workdir = @__DIR__
println(workdir)
cd(workdir)

Random.seed!(19)

σobs

n = 15
nshapes = 8

a0 = 2.0
c0 = 0.1
γ0 = 0.7

q0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]]
#p0 = [PointF(1.0, -3.0) for i in 1:n]/n
p0 = zeros(PointF,n)
x0 = State(q0, p0)
xobsT = Vector{PointF}[]
Ptrue = MarslandShardlow(a0, c0, γ0, 0.0, n)

for k in 1:nshapes
    # Ptrue = MarslandShardlow(a0 * exp(0.5*randn()), c0 * exp(0.5*randn()),
    #                 γ0 *  exp(0.5*randn()), 0.0, n)
    x0 = State(q0, randn(PointF,n))
    Wf, Xf = landmarksforward(t, x0, Ptrue)
    push!(xobsT, [Xf.yy[end].q[i] for i in 1:n ] + σobs * randn(PointF,n))
end
xobs0 = []
pb = Lmplotbounds(-3.0,3.0,-3.0,3.0)


save("data_exp2.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb, "nshapes", nshapes)
