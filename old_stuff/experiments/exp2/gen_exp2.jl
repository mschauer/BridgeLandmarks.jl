# ellipse to multiple forward simulated shapes
# only final configurations observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2

workdir = @__DIR__
cd(workdir)

Random.seed!(9)

σobs = 0.01

n = 15
nshapes = 10

a0 = 2.0
c0 = 0.1
γ0 = 0.7

q0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]]
xobsT = Vector{PointF}[]
Ptrue = MarslandShardlow(a0, c0, γ0, 0.0, n)

for k in 1:nshapes
    # Ptrue = MarslandShardlow(a0 * exp(0.5*randn()), c0 * exp(0.5*randn()),
    #                 γ0 *  exp(0.5*randn()), 0.0, n)
    x0 = State(q0, randn(PointF,n))
    T = 1.0; dt = 0.01; t = 0.0:dt:T
    Wf, Xf = landmarksforward(t, x0, Ptrue)
    push!(xobsT, [Xf.yy[end].q[i] for i in 1:n ] + σobs * randn(PointF,n))
end
xobs0 = []


JLD2.@save "data_exp2.jld2" xobs0 xobsT n nshapes q0
