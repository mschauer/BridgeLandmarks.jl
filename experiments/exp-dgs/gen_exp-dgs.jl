# dgs = data generated stefan
# only final landmark positions observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD
using NPZ

workdir = @__DIR__
println(workdir)
cd(workdir)

nshapes = 1
T = 1.0

testshapes = npzread("match.npy.npz")
xobs0vec =  get(testshapes,"q0",0)
xobsTvec =  get(testshapes,"v",0)
p0vec = get(testshapes,"p",0)
nb = div(length(xobs0vec),2)

subs = 1:5:nb
xobs0 = [PointF(xobs0vec[2i-1],xobs0vec[2i]) for i in subs]
xobsT = [PointF(xobsTvec[2i-1],xobsTvec[2i]) for i in subs]
p0 = [PointF(p0vec[2i-1],p0vec[2i]) for i in subs]
n = length(subs)

pb = Lmplotbounds(-2.0,2.0,-1.5,1.5) # verified


x0 = State(xobs0, p0)


save("data_exp-dgs.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb, "nshapes", nshapes)
#save("data_exp-dgs.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "pb", pb, "nshapes", nshapes)
