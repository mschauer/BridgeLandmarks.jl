# ellipse to multiple forward simulated shapes
# only final landmark positions observed

using Revise
using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2

workdir = @__DIR__
println(workdir)
cd(workdir)

n = 40  # in the final experiment we'll take n = 18
nshapes = 1
T = 1.0

a = 1.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 0.7     # Noise level

xobs0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]]

θ, ψ =  π/3, 0.2
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [0.2, 0.0]
#xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n ]
#xobsT = α*[rot*stretch*PointF(cos(t)^3, sin(t)^3)  + shift for t in collect(0:(2pi/n):2pi)[2:end]]

# α = 1.2
# xobs0 = α*[rot*stretch*PointF(cos(t)^3, sin(t)^3)  + shift for t in collect(0:(2pi/n):2pi)[2:end]]
#
 P = MarslandShardlow(1.0, 2.0, 0.5, 0.0, n)
 Random.seed!(2)
 initmom = 0.5*randn(PointF,length(xobs0))

 x0 = State(xobs0, initmom)
 W,X = BridgeLandmarks.landmarksforward(collect(0.0:0.001:0.5), x0, P)
 xobsT = [BridgeLandmarks.q(X[end][2])[j] for j ∈ 1:n]

# template matching
# x0 = []
# nshapes = 2
# xobsT = [xobs0, xobsT]

JLD2.@save "data_exp1_hard.jld2" xobs0 xobsT n nshapes
