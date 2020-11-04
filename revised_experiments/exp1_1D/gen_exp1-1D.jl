# important: set d=1 in BridgeLandmarks.jl
using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2

workdir = @__DIR__
println(workdir)
cd(workdir)

n = 3
nshapes = 1
T = 1.0




xobs0 = [PointF(-0.5), PointF(0.0), PointF(0.1)]
#xobsT = [exp.(q0[i])  for i in 1:n ] .- PointF(3.0)
xobsT = [PointF(-0.5), PointF(0.2), PointF(1.0)]

JLD2.@save "data_exp1-1D.jld2" xobs0 xobsT n nshapes
